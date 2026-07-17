from types import SimpleNamespace
from unittest.mock import patch

from backend.ir.schema_v3 import AnalyzeIRv3, DomainRequirement, ToolEvidence
from backend.llm import analyze_v3 as analyzer
from backend.notion import writer
from backend.scoring.scoring_engine_v3 import score_ir_v3, score_to_public_dict


def test_top_level_gemini_schema_survives_normalization():
    page_text = "Requirements\nBuild production APIs with Python and own reliable services."
    raw = {
        "job_title": "Software Engineer",
        "company": "Example",
        "job_seniority_signal": "mid",
        "candidate_seniority_signal": "junior",
        "candidate_skills": ["Python", "REST APIs"],
        "domain_requirements": [
            {
                "name": "API Development",
                "importance": "must",
                "evidence_quote": "Build production APIs with Python",
                "evidence_summary": "Build production APIs.",
                "examples": [
                    {"name": "Python", "importance": "must", "evidence_quote": "APIs with Python"}
                ],
                "evidence_level": "anchored",
                "anchors": ["Python", "APIs"],
            }
        ],
        "ownership_and_scope": {
            "ownership": {"present": True, "level_val": 2, "evidence": ["own reliable services"]},
            "scope": {"level": "team", "level_val": 2, "evidence": []},
            "leadership": {"present": False, "level_val": 0, "evidence": []},
        },
        "evidence_hints": {"years_experience": "2 years"},
    }

    with patch.object(analyzer, "_extract_with_gemini_v3", return_value=raw), patch.object(
        analyzer, "adjudicate_domains", side_effect=lambda domains, **_kwargs: domains
    ):
        result = analyzer.analyze_v3(
            page_text=page_text,
            title="Software Engineer | Example | LinkedIn",
            candidate_profile={
                "candidate_skills": ["Python", "REST APIs"],
                "candidate_seniority_signal": "junior",
                "evidence_claims": [
                    {
                        "resume_quote": "Built REST APIs with Python.",
                        "skills": ["Python", "REST APIs"],
                        "domains": ["API Development"],
                    }
                ],
            },
        )

    assert [d.name for d in result.domain_requirements] == ["API Development"]
    assert "Python" in result.candidate_skills
    assert result.ownership_and_scope.ownership.level_val == 2
    assert result.analysis_status == "success"


def _public_contract():
    ir = AnalyzeIRv3(
        job_title="Software Engineer",
        company="Example",
        location="Melbourne",
        job_seniority_signal="junior",
        candidate_seniority_signal="junior",
        candidate_skills=["Built and deployed Python REST APIs for production services."],
        tools_in_jd=["Python", "Docker"],
        domain_requirements=[
            DomainRequirement(
                name="API Development",
                importance="must",
                evidence_quote="Build production APIs with Python",
                evidence_level="anchored",
                anchors=["Python", "REST APIs"],
                examples=[ToolEvidence(name="Python", importance="must", evidence_quote="Python")],
            ),
            DomainRequirement(
                name="Observability",
                importance="should",
                evidence_quote="Instrument systems with OpenTelemetry",
                evidence_level="anchored",
                anchors=["OpenTelemetry"],
            ),
        ],
        evidence_hints={},
    )
    return score_to_public_dict(score_ir_v3(ir))


def test_public_contract_contains_every_requirement_with_evidence_status():
    contract = _public_contract()
    items = contract["requirements"]["items"]

    assert {item["name"] for item in items} == {"API Development", "Observability"}
    by_name = {item["name"]: item for item in items}
    assert by_name["API Development"]["status"] in {"matched", "partial"}
    assert by_name["API Development"]["jd_evidence"] == "Build production APIs with Python"
    assert by_name["Observability"]["status"] == "missing"
    assert contract["analysis_quality"]["score_reliable"] is True
    tools = {item["name"]: item["status"] for item in contract["tools"]["items"]}
    assert tools["Python"] == "matched"
    assert tools["Docker"] == "missing"
    assert [tool["name"] for tool in by_name["Observability"]["tools"]] == []


def test_notion_uses_compact_requirement_matrix_instead_of_repeated_sections():
    contract = _public_contract()
    captured = {}

    def fake_post(**kwargs):
        captured.update(kwargs)
        return writer._NotionPostResult(url="https://notion.so/page", status_code=200)

    req = SimpleNamespace(title="Software Engineer", url="https://linkedin.com/jobs/view/123")
    score = contract["score"]
    resp = SimpleNamespace(
        job_title="Software Engineer",
        company="Example",
        final_score=score["final_score"],
        should_apply=score["should_apply"],
        seniority_gap=score["seniority_gap"],
        missing_skills=contract["missing_skills"],
        required_skills=contract["required_skills"],
        summary=score["summary"],
        cap=score["cap"],
        raw_json={},
        distance_score=score["distance_score"],
    )

    with patch.object(writer, "_post_to_notion", side_effect=fake_post):
        url = writer.create_notion_page(
            req,
            resp,
            None,
            public_contract=contract,
            notion_token="secret",
            notion_database_id="database",
            notion_version="2022-06-28",
        )

    assert url == "https://notion.so/page"
    blocks = captured["children"]
    serialized = str(blocks)
    assert "JD ↔ Resume Match Coverage" in serialized
    assert "API Development" in serialized
    assert "Observability" in serialized
    assert "Layered Breakdown" not in serialized
    assert len(blocks) < 30
