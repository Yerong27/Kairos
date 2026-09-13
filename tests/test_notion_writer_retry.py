from types import SimpleNamespace
from unittest.mock import patch

import requests

from backend.notion import writer


class _Response:
    def __init__(self, status_code, *, body="", url=None, headers=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = body
        self.headers = headers or {}
        self._url = url

    def json(self):
        return {"url": self._url}


def _post(**overrides):
    args = {
        "notion_token": "secret",
        "notion_database_id": "database",
        "notion_version": "2022-06-28",
        "properties": {"Job Title": {}},
        "children": [],
    }
    args.update(overrides)
    return writer._post_to_notion(**args)


def test_429_honors_retry_after_then_succeeds():
    responses = [
        _Response(429, body="rate limited", headers={"Retry-After": "2"}),
        _Response(200, url="https://notion.so/page"),
    ]
    sleeps = []
    timeouts = []

    def fake_post(*_args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return responses.pop(0)

    with patch.object(writer.requests, "post", side_effect=fake_post), patch.object(
        writer.time, "sleep", side_effect=sleeps.append
    ):
        result = _post()

    assert result.url == "https://notion.so/page"
    assert sleeps == [2.0]
    assert timeouts == [(10, 45), (10, 45)]


def test_timeout_is_not_blindly_retried_because_outcome_is_unknown():
    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(1)
        raise requests.ReadTimeout("read timed out")

    with patch.object(writer.requests, "post", side_effect=fake_post):
        result = _post(max_attempts=3)

    assert result.url is None
    assert result.error_kind == "transport_unknown"
    assert len(calls) == 1


def test_validation_error_uses_title_and_reduced_core_blocks():
    calls = []

    def fake_notion_post(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return writer._NotionPostResult(status_code=400, error_kind="api", detail="validation")
        return writer._NotionPostResult(url="https://notion.so/safe", status_code=200)

    req = SimpleNamespace(title="Cloud Engineer", url="https://example.com/job")
    resp = SimpleNamespace(
        job_title="Cloud Engineer",
        company="Example",
        final_score=70,
        should_apply="Maybe",
        seniority_gap="small",
        missing_skills=[],
        required_skills=[],
        summary="Summary",
        cap=80,
        raw_json={},
        uncapped_score=75,
        distance_score=70,
    )
    contract = {
        "display_required_domains": [f"Domain {i}" for i in range(30)],
        "display_domain_evidence": {},
        "score": {},
    }

    with patch.object(writer, "_post_to_notion", side_effect=fake_notion_post):
        url = writer.create_notion_page(
            req,
            resp,
            None,
            public_contract=contract,
            notion_token="secret",
            notion_database_id="database",
            notion_version="2022-06-28",
        )

    assert url == "https://notion.so/safe"
    assert len(calls[0]["children"]) > 24
    assert len(calls[1]["children"]) == 24
    assert set(calls[1]["properties"]) == {"Job Title"}


def test_requirement_matrix_keeps_all_compact_actions():
    actions = [f"Complete action {index}" for index in range(12)]
    contract = {
        "requirements": {
            "items": [
                {
                    "name": "Example requirement",
                    "importance": "should",
                    "status": "matched",
                    "jd_evidence": "Example evidence",
                }
            ],
            "counts": {"total": 1, "matched": 1},
        },
        "analysis_quality": {"score_reliable": True},
        "actions": actions,
    }
    resp = SimpleNamespace(
        distance_score=80,
        final_score=80,
        seniority_gap="none",
        cap=92,
    )

    blocks = writer._render_requirement_matrix_to_notion(contract, resp)
    rendered = [
        block["bulleted_list_item"]["rich_text"][0]["text"]["content"]
        for block in blocks
        if block.get("type") == "bulleted_list_item"
    ]

    assert actions == [item for item in rendered if item.startswith("Complete action")]
