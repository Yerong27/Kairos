from types import SimpleNamespace
from unittest.mock import patch

from backend.ir.schema_v3 import AnalyzeIRv3, DomainRequirement, ToolEvidence
from backend.ir.domain_catalog import DomainCatalog, adjudicate_domains
from backend.llm import analyze_v3 as analyzer
from backend.notion import writer
from backend.scoring.scoring_engine_v3 import (
    _dedupe_domains_by_canon,
    _gap_bucket,
    score_ir_v3,
    score_to_public_dict,
)


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


def test_candidate_seniority_uses_job_family_relevant_resume_experience():
    page_text = """Junior Software / DevOps Engineer
    Build Python services, AWS infrastructure, and CI/CD automation."""
    raw = {
        "job_title": "Junior Software / DevOps Engineer",
        "company": "Example",
        "job_seniority_signal": "junior",
        "job_seniority_evidence": "Junior Software / DevOps Engineer",
        "domain_requirements": [],
    }
    resume = """Cloud Support Associate – AI/ML
    JUL 2024 – PRESENT
    Troubleshoot AWS, Linux, Docker, Terraform, and CI/CD workloads.

    Commercial Analyst
    JAN 2018 – JUN 2024
    Managed budgeting, forecasting, and financial reporting."""

    with patch.object(analyzer, "_extract_with_gemini_v3", return_value=raw), patch.object(
        analyzer, "adjudicate_domains", side_effect=lambda domains, **_kwargs: domains
    ):
        result = analyzer.analyze_v3(
            page_text=page_text,
            title="Junior Software / DevOps Engineer",
            candidate_profile={
                "candidate_skills": ["AWS", "Linux", "Docker", "Terraform"],
                "candidate_seniority_signal": "mid",
                "seniority_reason": "global career history",
                "evidence_claims": [
                    {
                        "resume_quote": "Troubleshoot AWS, Linux, Docker, Terraform, and CI/CD workloads.",
                        "skills": ["AWS", "Linux", "Docker", "Terraform"],
                        "domains": ["Cloud Computing"],
                    }
                ],
            },
            candidate_resume_text=resume,
        )

    assert result.candidate_seniority_signal == "junior_to_mid"
    assert result.raw_llm_json["_debug_meta"]["candidate_seniority_job_family"] == "tech_swe"
    assert result.raw_llm_json["_debug_meta"]["candidate_seniority_exp_reason"].startswith(
        "relevant_experience_"
    )


def test_apprenticeship_progresses_to_junior_after_sustained_relevant_experience():
    resume = """Cloud Support Apprentice — AWS
    JUL 2024 – PRESENT
    Troubleshoot production cloud workloads, implement automation, and resolve incidents.

    Finance / Accounting Roles
    JAN 2019 – JUN 2024
    Managed budgeting, forecasting, and financial reporting. Python appears in a later section."""

    level, reason, entries = analyzer._derive_candidate_level_from_experience(
        resume, job_family="tech_swe"
    )

    assert level == "junior"
    assert reason == "apprenticeship_progressed_by_experience"
    assert all("Finance" not in entry["title"] for entry in entries)


def test_jd_tool_extraction_drops_sentence_fragments_and_questions():
    page_text = """Experience with a wide range of technology in mission critical environments. Additionally, learn quickly.
    Are you experienced in?
    Exposure to concepts such as virtualization, containers, configuration management, source management, and application deployment.
    Infrastructure as code tools such as YAML, JSON, Ansible, CloudFormation or Terraform."""

    tools = analyzer.extract_tools_from_jd(page_text)

    assert "virtualization" not in tools
    assert "containers" not in tools
    assert "YAML" not in tools
    assert "JSON" not in tools
    assert "learn quickly" not in [tool.lower() for tool in tools]
    assert "?" not in tools
    assert "are you experienced in?" not in [tool.lower() for tool in tools]
    assert not any("wide range of technology" in tool.lower() for tool in tools)
    assert not any(tool.lower().startswith("and ") for tool in tools)


def test_jd_tool_extraction_deduplicates_databases_and_drops_low_signal_concepts():
    page_text = """Infrastructure as code and configuration technologies such as YAML, JSON, Ansible, CloudFormation or Terraform.
    Exposure to concepts such as virtualization, containers, configuration management, and application deployment.
    Experience with Relational Database Management Systems (RDBMS), and NoSQL databases."""

    tools = analyzer.extract_tools_from_jd(page_text)

    assert "Ansible" in tools
    assert "CloudFormation or Terraform" in tools
    assert tools.count("Relational Databases") == 1
    assert tools.count("NoSQL Databases") == 1
    assert "YAML" not in tools
    assert "JSON" not in tools
    assert "virtualization" not in tools
    assert "containers" not in tools


def test_grounded_requirement_not_in_catalog_is_preserved_and_verified():
    requirement = DomainRequirement(
        name="Operating System Management",
        importance="should",
        evidence_quote="Linux installation and management, including working from a shell",
        anchors=["Linux", "shell"],
    )
    catalog = DomainCatalog(
        domains={},
        aliases_global={},
        allowed_must_domains=[],
        max_must_count=None,
    )

    result = adjudicate_domains(
        [requirement],
        jd_text="Experience with Linux installation and management, including working from a shell.",
        catalog=catalog,
    )

    assert len(result) == 1
    assert result[0].domain_id.startswith("custom:")
    assert result[0].evidence_level == "exact"
    assert result[0].evidence_status == "verified"


def test_uncatalogued_requirements_do_not_collapse_into_one_other_bucket():
    domains = [
        DomainRequirement(name="Scripting", importance="should", domain_id="other_info"),
        DomainRequirement(name="Database Management", importance="should", domain_id="other_info"),
        DomainRequirement(name="System Integration", importance="should", domain_id="other_info"),
    ]

    assert [item.name for item in _dedupe_domains_by_canon(domains)] == [
        "Scripting",
        "Database Management",
        "System Integration",
    ]


def test_one_level_more_experienced_is_not_labeled_overqualified():
    assert _gap_bucket(-1.0) == "none"
    assert _gap_bucket(-2.0) == "overqualified"


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


def test_notion_uses_user_facing_summary_without_evidence_matrix():
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
    assert "JD ↔ Resume Match Coverage" not in serialized
    assert "JD evidence:" not in serialized
    assert "Resume evidence" not in serialized
    assert "API Development" in serialized
    assert "Observability" in serialized
    assert "🎯 JD Requirements Match" in serialized
    assert "❌ Missing" in serialized
    assert "✅ Matched" in serialized
    assert "🧰 Tool Match" in serialized
    assert "⚠️ Not shown in resume" in serialized
    assert "Extracted Tools (JD)" not in serialized
    assert "Layered Breakdown" not in serialized
    assert "🔒 Decision Constraints" in serialized
    assert "Your level: junior" in serialized
