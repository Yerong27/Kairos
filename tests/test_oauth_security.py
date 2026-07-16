import tempfile
from pathlib import Path
from unittest.mock import patch

import main


class _SearchResponse:
    ok = True
    text = ""

    def json(self):
        return {
            "results": [
                {
                    "id": "database-id",
                    "title": [{"plain_text": '<script>alert("x")</script>'}],
                }
            ]
        }


def test_oauth_state_expires():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        now = 1_800_000_000
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main.time, "time", return_value=now
        ):
            main._store_oauth_state("state")

        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main.time, "time", return_value=now + main.OAUTH_STATE_TTL_SECONDS + 1
        ):
            assert main._consume_oauth_state("state") is False


def test_notion_database_title_is_html_escaped():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path):
            main._store_oauth_session("session", "notion-token")
            with patch.object(main.requests, "post", return_value=_SearchResponse()):
                response = main.notion_select_database(session="session")

        body = response.body.decode("utf-8")
        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert "Refresh" in body
