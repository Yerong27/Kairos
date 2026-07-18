import tempfile
import json
import sqlite3
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from backend.ir.candidate_profile import CandidateEvidenceClaim, CandidateProfile
from backend.ir.schema_v3 import AnalyzeIRv3, DomainRequirement
from backend.llm import analyze_v3 as jd_analyzer
from backend.llm import analyze_resume as resume_analyzer


def _profile(skill: str = "Python") -> CandidateProfile:
    return CandidateProfile(
        candidate_skills=[skill],
        candidate_domains=["API Development"],
        candidate_seniority_signal="junior_to_mid",
        evidence_claims=[
            CandidateEvidenceClaim(
                evidence_id="ev_profile",
                resume_quote=f"Built production services with {skill}.",
                skills=[skill],
                domains=["API Development"],
            )
        ],
    )


def test_resume_parser_keeps_only_resume_grounded_claims():
    now = time.gmtime()
    start_year = now.tm_year - 2
    resume = f"""Cloud Support Associate\nJAN {start_year} – PRESENT\nBuilt production services with Python."""
    payload = {
        "candidate_skills": ["Python", "Kubernetes"],
        "candidate_domains": ["API Development"],
        "candidate_seniority_signal": "senior",
        "seniority_reason": "model guess",
        "evidence_claims": [
            {
                "resume_quote": "Built production services with Python.",
                "claim_type": "capability",
                "skills": ["Python"],
                "domains": ["API Development"],
            },
            {
                "resume_quote": "Led Kubernetes platform strategy.",
                "skills": ["Kubernetes"],
                "domains": ["Platform Engineering"],
            },
        ],
        "roles": [],
    }

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, *_args, **_kwargs):
            return SimpleNamespace(text=json.dumps(payload))

    with patch.object(resume_analyzer, "_ensure_configured"), patch.object(
        resume_analyzer.genai, "GenerativeModel", FakeModel
    ):
        profile = resume_analyzer.analyze_resume(resume)

    assert [claim.resume_quote for claim in profile.evidence_claims] == [
        "Built production services with Python."
    ]
    assert "Python" in profile.candidate_skills
    assert "Kubernetes" not in profile.candidate_skills
    assert profile.evidence_claims[0].claim_type == "capability"
    assert profile.evidence_claims[0].evidence_id.startswith("ev_")
    assert profile.candidate_seniority_signal == "unknown"
    assert profile.seniority_reason == "deferred_to_job_family_analysis"
    evidence_text = resume_analyzer.candidate_profile_evidence_text(profile)
    assert "Built production services with Python." in evidence_text
    assert "Candidate skills:" not in evidence_text
    assert "Candidate domains:" not in evidence_text


def test_resume_parser_preserves_structured_facts_omitted_by_model():
    resume = """PROFESSIONAL EXPERIENCE
Support Associate | Example
JAN 2024 – PRESENT
Supported customer workloads.

CERTIFICATIONS
Certified Platform Practitioner
Certified Solutions Professional

EDUCATION
Diploma of Applied Technology | Example Institute
JAN 2022 – JAN 2024
"""
    payload = {
        "candidate_skills": [],
        "candidate_domains": [],
        "evidence_claims": [
            {
                "resume_quote": "Supported customer workloads.",
                "claim_type": "experience",
                "skills": [],
                "domains": [],
            }
        ],
        "roles": [],
    }

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, *_args, **_kwargs):
            return SimpleNamespace(text=json.dumps(payload))

    with patch.object(resume_analyzer, "_ensure_configured"), patch.object(
        resume_analyzer.genai, "GenerativeModel", FakeModel
    ):
        profile = resume_analyzer.analyze_resume(resume)

    claims = {claim.resume_quote: claim.claim_type for claim in profile.evidence_claims}
    assert claims["Certified Platform Practitioner"] == "credential"
    assert claims["Certified Solutions Professional"] == "credential"
    assert claims["Diploma of Applied Technology | Example Institute"] == "education"


def test_local_profile_fallback_is_grounded_in_resume_sections():
    resume = """SKILLS
Incident investigation and stakeholder communication

PROFESSIONAL EXPERIENCE
Support Associate | Example
JAN 2024 – PRESENT
Resolved complex customer incidents.

CERTIFICATIONS
Certified Platform Practitioner
"""

    profile = resume_analyzer.build_local_resume_profile(resume)

    assert profile.analysis_status == "degraded"
    assert profile.candidate_seniority_signal == "unknown"
    quotes = {claim.resume_quote for claim in profile.evidence_claims}
    assert "Incident investigation and stakeholder communication" in quotes
    assert "Resolved complex customer incidents." in quotes
    assert "Certified Platform Practitioner" in quotes


def test_resume_parser_retries_truncated_json_with_compact_prompt():
    resume = "Cloud Engineer\nBuilt production services with Python and FastAPI."
    payload = {
        "candidate_skills": ["Python", "FastAPI"],
        "candidate_domains": ["API Development"],
        "candidate_seniority_signal": "junior",
        "seniority_reason": "Resume title signal",
        "evidence_claims": [
            {
                "resume_quote": "Built production services with Python and FastAPI.",
                "skills": ["Python", "FastAPI"],
                "domains": ["API Development"],
            }
        ],
        "roles": [],
    }
    calls = []

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, prompt, **_kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                return SimpleNamespace(text='{"candidate_skills":["Python"],"evidence_claims":[{"resume_quote":"unterminated')
            return SimpleNamespace(text=json.dumps(payload))

    with patch.object(resume_analyzer, "_ensure_configured"), patch.object(
        resume_analyzer.genai, "GenerativeModel", FakeModel
    ):
        profile = resume_analyzer.analyze_resume(resume)

    assert profile.candidate_skills == ["Python", "FastAPI"]
    assert len(calls) == 2
    assert "compact retry" in calls[1]
    assert "at most 12 evidence claims" in calls[1]


def test_resume_parser_reports_friendly_error_after_two_incomplete_responses():
    resume = "Cloud Engineer\nBuilt production services with Python and FastAPI."

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, *_args, **_kwargs):
            return SimpleNamespace(text='{"candidate_skills":["Python')

    with patch.object(resume_analyzer, "_ensure_configured"), patch.object(
        resume_analyzer.genai, "GenerativeModel", FakeModel
    ):
        try:
            resume_analyzer.analyze_resume(resume)
        except ValueError as exc:
            assert "automatic compact retry" in str(exc)
        else:
            raise AssertionError("Expected incomplete JSON to fail after retry")


def test_resume_profile_is_reused_until_resume_content_changes():
    client = TestClient(main.app)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main, "kairos_analyze_resume", side_effect=[_profile("Python"), _profile("Go")]
        ) as parser:
            main._store_notion_user(
                user_token="user",
                access_token="notion-token",
                database_id="database",
                database_name="Jobs",
            )
            headers = {"Authorization": "Bearer user"}

            first = client.post(
                "/resume/upload",
                files={"file": ("resume.txt", b"Built production services with Python.", "text/plain")},
                headers=headers,
            )
            repeated = client.post(
                "/resume/upload",
                files={"file": ("renamed.txt", b"Built production services with Python.", "text/plain")},
                headers=headers,
            )
            changed = client.post(
                "/resume/upload",
                files={"file": ("resume-v2.txt", b"Built production services with Go.", "text/plain")},
                headers=headers,
            )

            assert first.status_code == repeated.status_code == changed.status_code == 200
            assert first.json()["candidate_profile_reused"] is False
            assert repeated.json()["candidate_profile_reused"] is True
            assert changed.json()["resume_changed"] is True
            assert parser.call_count == 2

            user = main._get_notion_user("user")
            record = main._get_candidate_profile_record("user")
            assert record["resume_hash"] == user["resume_hash"]
            assert '"Go"' in record["profile_json"]


def test_ai_timeout_creates_usable_local_profile_and_allows_later_retry():
    client = TestClient(main.app)
    resume = b"""SKILLS
Incident investigation and stakeholder communication

PROFESSIONAL EXPERIENCE
Support Associate | Example
JAN 2024 - PRESENT
Resolved complex customer incidents.

CERTIFICATIONS
Certified Platform Practitioner
"""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main, "kairos_analyze_resume", side_effect=TimeoutError("504 Deadline Exceeded")
        ) as parser:
            main._store_notion_user(
                user_token="user",
                access_token="notion-token",
                database_id="database",
                database_name="Jobs",
            )
            headers = {"Authorization": "Bearer user"}
            first = client.post(
                "/resume/upload",
                files={"file": ("resume.txt", resume, "text/plain")},
                headers=headers,
            )
            status = client.get("/status", headers=headers)
            retried = client.post(
                "/resume/upload",
                files={"file": ("resume.txt", resume, "text/plain")},
                headers=headers,
            )

    assert first.status_code == retried.status_code == status.status_code == 200
    assert first.json()["candidate_profile_status"] == "degraded"
    assert "analyze jobs now" in first.json()["warning"]
    assert status.json()["candidate_profile_current"] is True
    assert status.json()["candidate_profile_status"] == "degraded"
    assert retried.json()["candidate_profile_reused"] is False
    assert parser.call_count == 2


def test_docx_resume_upload_extracts_body_tables_and_header():
    content = BytesIO()
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{word_ns}"><w:body>
  <w:p><w:r><w:t>Built production services with Python and AWS.</w:t></w:r></w:p>
  <w:tbl><w:tr>
    <w:tc><w:p><w:r><w:t>Certification</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>AWS Solutions Architect</w:t></w:r></w:p></w:tc>
  </w:tr></w:tbl>
</w:body></w:document>"""
    header_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:hdr xmlns:w="{word_ns}">
  <w:p><w:r><w:t>Candidate Name | Melbourne</w:t></w:r></w:p>
</w:hdr>"""
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/header1.xml", header_xml)

    captured = {}

    def parse_resume(text):
        captured["text"] = text
        return _profile("Python")

    client = TestClient(main.app)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main, "kairos_analyze_resume", side_effect=parse_resume
        ):
            main._store_notion_user(
                user_token="user",
                access_token="notion-token",
                database_id="database",
                database_name="Jobs",
            )
            response = client.post(
                "/resume/upload",
                files={
                    "file": (
                        "resume.docx",
                        content.getvalue(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                headers={"Authorization": "Bearer user"},
            )

    assert response.status_code == 200
    assert response.json()["candidate_profile_status"] == "ready"
    assert "Built production services with Python and AWS." in captured["text"]
    assert "Certification" in captured["text"]
    assert "AWS Solutions Architect" in captured["text"]
    assert "Candidate Name | Melbourne" in captured["text"]


def test_invalid_docx_is_rejected_without_calling_ai_parser():
    client = TestClient(main.app)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main, "kairos_analyze_resume"
        ) as parser:
            main._store_notion_user(
                user_token="user",
                access_token="notion-token",
                database_id="database",
                database_name="Jobs",
            )
            response = client.post(
                "/resume/upload",
                files={
                    "file": (
                        "broken.docx",
                        b"this is not a DOCX package",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                headers={"Authorization": "Bearer user"},
            )

    assert response.status_code == 400
    assert response.json()["detail"] == "The file is not a valid DOCX document."
    parser.assert_not_called()


def test_same_resume_is_reparsed_when_candidate_profile_logic_version_changes():
    client = TestClient(main.app)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main, "kairos_analyze_resume", return_value=_profile("Python")
        ) as parser:
            main._store_notion_user(
                user_token="user",
                access_token="notion-token",
                database_id="database",
                database_name="Jobs",
            )
            headers = {"Authorization": "Bearer user"}
            resume = b"Built production services with Python."

            first = client.post(
                "/resume/upload",
                files={"file": ("resume.txt", resume, "text/plain")},
                headers=headers,
            )
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE candidate_profiles SET prompt_version = ? WHERE user_token = ?",
                    ("old-parser-version", "user"),
                )
                conn.commit()
            after_upgrade = client.post(
                "/resume/upload",
                files={"file": ("resume.txt", resume, "text/plain")},
                headers=headers,
            )

            assert first.status_code == after_upgrade.status_code == 200
            assert first.json()["candidate_profile_reused"] is False
            assert after_upgrade.json()["resume_changed"] is False
            assert after_upgrade.json()["candidate_profile_reused"] is False
            assert parser.call_count == 2
            assert main._get_candidate_profile_record("user")["prompt_version"] == (
                main.CANDIDATE_PROFILE_PROMPT_VERSION
            )


def test_reconnecting_notion_preserves_resume_and_candidate_profile():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with patch.object(main, "NOTION_OAUTH_DB", db_path):
            main._store_notion_user(
                user_token="user", access_token="old", database_id="db-1", database_name="Jobs"
            )
            resume_hash = main._sha256_text("kairos_resume:1", "Resume text")
            main._store_resume_for_user("user", "Resume text", "resume.txt", resume_hash)
            main._store_candidate_profile_record(
                user_token="user",
                resume_hash=resume_hash,
                status="ready",
                profile=_profile().model_dump(),
            )

            main._store_notion_user(
                user_token="user", access_token="new", database_id="db-2", database_name="New Jobs"
            )

            user = main._get_notion_user("user")
            assert user["access_token"] == "new"
            assert user["resume_text"] == "Resume text"
            assert main._candidate_profile_record_is_current(
                main._get_candidate_profile_record("user"), resume_hash
            )


def test_existing_database_is_migrated_without_losing_resume():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE notion_users (
                    user_token TEXT PRIMARY KEY, access_token TEXT NOT NULL,
                    database_id TEXT, database_name TEXT, created_at INTEGER,
                    resume_text TEXT, resume_filename TEXT, resume_uploaded_at INTEGER
                )
                """
            )
            conn.execute(
                "INSERT INTO notion_users VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("user", "token", "db", "Jobs", 1, "Existing resume", "resume.txt", 1),
            )
            conn.commit()

        with patch.object(main, "NOTION_OAUTH_DB", db_path):
            main._init_notion_oauth_db()
            user = main._get_notion_user("user")
            assert user["resume_text"] == "Existing resume"
            assert user["resume_hash"] is None
            with sqlite3.connect(db_path) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "candidate_profiles" in tables


def test_jd_gemini_prompt_uses_compact_cached_profile_not_raw_resume():
    captured = {}

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return SimpleNamespace(
                text=(
                    '{"job_title":"Engineer","company":"Example",'
                    '"job_seniority_signal":"mid","domain_requirements":[]}'
                )
            )

    with patch.object(jd_analyzer, "_ensure_gemini_configured"), patch.object(
        jd_analyzer.genai, "GenerativeModel", FakeModel
    ):
        jd_analyzer._extract_with_gemini_v3(
            "Requirements: build APIs and reliable services.",
            title="Engineer",
            output_language="en",
            candidate_profile=_profile("Python").model_dump(),
        )

    assert "<candidate_profile>" in captured["prompt"]
    assert "Built production services with Python." in captured["prompt"]
    assert "candidate_seniority_signal" in captured["prompt"]
    assert "every component" in captured["prompt"]
    assert "worth applying for" in captured["prompt"]
    assert "raw_llm_json" not in captured["prompt"]
    assert "PRIVATE FULL RESUME" not in captured["prompt"]


def test_job_analysis_uses_stored_profile_without_sending_raw_resume_to_jd_parser():
    client = TestClient(main.app)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "oauth.db"
        cache_dir = Path(tmp) / "cache"
        profile = _profile("Python")
        ir = AnalyzeIRv3(
            job_title="API Engineer",
            company="Example",
            job_seniority_signal="mid",
            candidate_seniority_signal=profile.candidate_seniority_signal,
            candidate_skills=profile.candidate_skills,
            domain_requirements=[
                DomainRequirement(
                    name="API Development",
                    importance="must",
                    evidence_quote="Build reliable Python APIs",
                    evidence_level="anchored",
                    anchors=["Python", "APIs"],
                )
            ],
        )
        with patch.object(main, "NOTION_OAUTH_DB", db_path), patch.object(
            main, "V3_CACHE_DIR", cache_dir
        ), patch.object(main, "kairos_analyze_v3", return_value=ir) as jd_parser, patch.object(
            main, "create_notion_page", return_value="https://notion.so/page"
        ):
            main._store_notion_user(
                user_token="user", access_token="notion", database_id="db", database_name="Jobs"
            )
            resume_text = "Built production services with Python."
            resume_hash = main._sha256_text("kairos_resume:1", resume_text)
            main._store_resume_for_user("user", resume_text, "resume.txt", resume_hash)
            main._store_candidate_profile_record(
                user_token="user", resume_hash=resume_hash, status="ready", profile=profile.model_dump()
            )

            response = client.post(
                "/analyze_and_save",
                headers={"Authorization": "Bearer user"},
                json={
                    "title": "API Engineer",
                    "page_text": "Requirements: Build reliable Python APIs for production services and own delivery.",
                    "use_v3": True,
                },
            )

        assert response.status_code == 200, response.text
        kwargs = jd_parser.call_args.kwargs
        assert "user_profile" not in kwargs
        assert kwargs["candidate_profile"]["candidate_skills"] == ["Python"]
        assert kwargs["candidate_resume_text"] == resume_text
