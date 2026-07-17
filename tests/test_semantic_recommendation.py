import json
from types import SimpleNamespace
from unittest.mock import patch

from backend.ir.candidate_profile import CandidateEvidenceClaim, CandidateProfile
from backend.ir.schema_v3 import (
    AnalyzeIRv3,
    ApplicationRecommendation,
    DomainRequirement,
)
from backend.llm import analyze_v3 as analyzer
from backend.scoring.scoring_engine_v3 import score_ir_v3, score_to_public_dict


def test_semantic_comparison_controls_transferability_and_apply_recommendation():
    claim = CandidateEvidenceClaim(
        evidence_id="ev_cloud",
        resume_quote="Supported AWS services and CloudFormation-based workflows.",
        skills=["AWS", "CloudFormation"],
        domains=["Cloud Computing"],
    )
    ir = AnalyzeIRv3(
        job_title="Platform Engineer",
        company="Example",
        job_seniority_signal="mid",
        candidate_seniority_signal="junior_to_mid",
        candidate_evidence_claims=[claim],
        domain_requirements=[
            DomainRequirement(
                name="Platform Engineering",
                importance="must",
                evidence_quote="Experience in platform engineering",
                evidence_level="exact",
                evidence_status="verified",
                match_status="partial",
                resume_evidence_ids=["ev_cloud"],
                match_reason="AWS infrastructure experience is strongly transferable.",
            ),
            DomainRequirement(
                name="SOC2 compliance",
                importance="should",
                evidence_quote="Support SOC2 compliance work",
                evidence_level="exact",
                evidence_status="verified",
                match_status="missing",
                match_reason="No security compliance evidence.",
            ),
        ],
        application_recommendation=ApplicationRecommendation(
            should_apply="Yes",
            confidence="high",
            rationale="Strong adjacent cloud experience; remaining gaps are learnable.",
            strongest_matches=["AWS infrastructure"],
            key_gaps=["SOC2"],
        ),
    )

    contract = score_to_public_dict(score_ir_v3(ir))
    items = {item["name"]: item for item in contract["requirements"]["items"]}

    assert items["Platform Engineering"]["status"] == "partial"
    assert items["SOC2 compliance"]["status"] == "missing"
    assert contract["decision"]["verdict"] == "Yes"
    assert contract["decision"]["reason"] == "semantic_candidate_assessment"
    assert "Strong adjacent cloud experience" in contract["decision"]["explanation"]
    assert 70 <= contract["score"]["final_score"] <= 92
    assert contract["score"]["summary"].startswith("Strong adjacent cloud experience")
    assert contract["analysis_quality"]["matcher_mode"] == "semantic_profile_comparison"


def test_reversed_or_choices_collapse_to_one_requirement():
    page_text = "Requirements: Familiarity with CDK, CloudFormation, or similar."
    profile = CandidateProfile(
        evidence_claims=[
            CandidateEvidenceClaim(
                evidence_id="ev_iac",
                resume_quote="Used CloudFormation for AWS infrastructure.",
            )
        ]
    )
    raw = {
        "job_title": "Platform Engineer",
        "company": "Example",
        "job_seniority_signal": "mid",
        "domain_requirements": [
            {
                    "name": "CDK",
                "importance": "must",
                "requirement_type": "tool",
                "alternatives": ["CloudFormation"],
                "evidence_quote": "CDK, CloudFormation, or similar",
                "match_status": "matched",
                "resume_evidence_ids": ["ev_iac"],
                "match_reason": "CloudFormation satisfies the stated OR choice.",
            },
            {
                    "name": "CloudFormation",
                "importance": "must",
                "requirement_type": "tool",
                "alternatives": ["CDK"],
                "evidence_quote": "CDK, CloudFormation, or similar",
                "match_status": "matched",
                "resume_evidence_ids": ["ev_iac"],
                "match_reason": "CloudFormation is directly evidenced.",
            },
        ],
        "application_recommendation": {
            "should_apply": "Yes",
            "confidence": "high",
            "rationale": "The candidate satisfies the stated IaC tool choice.",
        },
    }

    with patch.object(analyzer, "_extract_with_gemini_v3", return_value=raw):
        result = analyzer.analyze_v3(
            page_text=page_text,
            title="Platform Engineer",
            candidate_profile=profile.model_dump(),
        )

    assert len(result.domain_requirements) == 1
    assert "CDK" in result.domain_requirements[0].name
    assert "CloudFormation" in result.domain_requirements[0].name


def test_normalization_does_not_truncate_complete_requirement_set_to_fourteen():
    clauses = [f"Capability {index} is required" for index in range(1, 17)]
    page_text = "Requirements: " + ". ".join(clauses) + "."
    raw = {
        "job_title": "Generalist",
        "company": "Example",
        "job_seniority_signal": "mid",
        "domain_requirements": [
            {
                "name": f"Capability {index}",
                "importance": "must",
                "requirement_type": "capability",
                "evidence_quote": f"Capability {index} is required",
                "match_status": "missing",
                "resume_evidence_ids": [],
                "match_reason": "No evidence in the stored profile.",
            }
            for index in range(1, 17)
        ],
    }

    with patch.object(analyzer, "_extract_with_gemini_v3", return_value=raw):
        result = analyzer.analyze_v3(
            page_text=page_text,
            title="Generalist",
            candidate_profile=CandidateProfile().model_dump(),
        )

    assert len(result.domain_requirements) == 16


def test_jd_evidence_quote_alias_is_kept_separate_from_resume_evidence():
    requirement = DomainRequirement.model_validate(
        {
            "domain": "Infrastructure as Code",
            "importance": "must",
            "jd_evidence_quote": "Experience with Infrastructure as Code",
            "match_status": "partial",
            "resume_evidence_ids": ["ev_iac"],
        }
    )

    assert requirement.evidence_quote == "Experience with Infrastructure as Code"
    assert requirement.resume_evidence_ids == ["ev_iac"]


def test_large_requirement_loss_marks_analysis_degraded():
    valid_clauses = [f"Capability {index} is required" for index in range(1, 6)]
    page_text = "Requirements: " + ". ".join(valid_clauses) + "."
    raw = {
        "job_title": "Generalist",
        "company": "Example",
        "job_seniority_signal": "mid",
        "domain_requirements": [
            {
                "name": f"Capability {index}",
                "importance": "must",
                "requirement_type": "capability",
                "evidence_quote": (
                    f"Capability {index} is required"
                    if index <= 5
                    else f"This quote is not in the JD {index}"
                ),
                "match_status": "missing",
                "resume_evidence_ids": [],
                "match_reason": "No evidence in the stored profile.",
            }
            for index in range(1, 9)
        ],
    }

    with patch.object(analyzer, "_extract_with_gemini_v3", return_value=raw):
        result = analyzer.analyze_v3(
            page_text=page_text,
            title="Generalist",
            candidate_profile=CandidateProfile().model_dump(),
        )

    assert len(result.domain_requirements) == 5
    assert result.analysis_status == "degraded"
    assert result.raw_llm_json["_debug_meta"]["normalization_incomplete"] is True
    contract = score_to_public_dict(score_ir_v3(result))
    assert contract["analysis_quality"]["score_reliable"] is False


def test_paraphrased_jd_evidence_is_repaired_to_an_exact_source_passage():
    jd = (
        "Strong hands-on experience with AWS cloud-native services is required. "
        "Exposure to AI and generative AI services on AWS, such as Amazon Bedrock, "
        "is desirable. Ability to analyse customer requirements and communicate "
        "clearly with technical stakeholders."
    )

    repaired = analyzer._repair_jd_evidence_quote(
        page_text=jd,
        requirement_name="AI and Generative AI Services on AWS",
        proposed_quote="Built AI solutions using Amazon Bedrock.",
        evidence_summary="Exposure to generative AI services in AWS.",
    )

    assert repaired
    assert repaired.lower() in jd.lower()
    assert "generative ai services on aws" in repaired.lower()


def test_evidence_repair_refuses_an_unrelated_requirement():
    repaired = analyzer._repair_jd_evidence_quote(
        page_text="The role requires clear communication and stakeholder collaboration.",
        requirement_name="Registered Nurse Licence",
        proposed_quote="Current nursing registration is mandatory.",
        evidence_summary="Must hold a nursing licence.",
    )

    assert repaired == ""


def test_evidence_repair_supports_a_semantic_label_not_written_verbatim():
    jd = (
        "Ability to analyse customer requirements, translate them into effective "
        "technical solutions, and guide customers through delivery decisions."
    )

    repaired = analyzer._repair_jd_evidence_quote(
        page_text=jd,
        requirement_name="Communication and Stakeholder Management",
        proposed_quote="Translate customer requirements and guide stakeholder decisions.",
        evidence_summary="Customer-facing requirements discovery and technical guidance.",
    )

    assert repaired
    assert repaired.lower() in jd.lower()


def test_jd_passages_are_source_grounded_and_preserve_requirement_content():
    jd = (
        "About the role\n"
        "• Design reliable cloud services for regulated customers.\n"
        "• Experience with infrastructure as code and CI/CD is required.\n"
        "• Communicate technical decisions clearly to business stakeholders."
    )

    passages = analyzer._build_jd_passages(jd)
    normalized_jd = analyzer._normalize_whitespace(jd).lower()

    assert len(passages) >= 3
    assert len({item["id"] for item in passages}) == len(passages)
    assert all(item["text"].lower() in normalized_jd for item in passages)
    assert any("infrastructure as code" in item["text"].lower() for item in passages)
    assert any("business stakeholders" in item["text"].lower() for item in passages)


def test_gemini_jd_passage_id_is_mapped_to_backend_owned_source_text():
    jd = (
        "The role designs reliable cloud services. "
        "Experience with infrastructure as code and CI/CD is required."
    )
    passages = analyzer._build_jd_passages(jd)
    source = next(
        item for item in passages if "infrastructure as code" in item["text"].lower()
    )

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, prompt, **_kwargs):
            assert '"id": "' + source["id"] + '"' in prompt
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "job_title": "Cloud Engineer",
                        "company": "Example",
                        "job_seniority_signal": "mid",
                        "domain_requirements": [
                            {
                                "domain": "Infrastructure as Code and CI/CD",
                                "importance": "must",
                                "requirement_type": "capability",
                                "match_status": "missing",
                                "jd_evidence_ids": [source["id"]],
                                "resume_evidence_ids": [],
                                "match_reason": "No supporting profile evidence.",
                            }
                        ],
                        "application_recommendation": {
                            "should_apply": "Maybe",
                            "confidence": "medium",
                            "rationale": "One important requirement is not evidenced.",
                        },
                    }
                )
            )

    with patch.object(analyzer, "_ensure_gemini_configured"), patch.object(
        analyzer.genai, "GenerativeModel", FakeModel
    ):
        result = analyzer._extract_with_gemini_v3(
            jd,
            title="Cloud Engineer",
            output_language="en",
            candidate_profile=CandidateProfile().model_dump(),
        )

    requirement = result["domain_requirements"][0]
    assert requirement["jd_evidence_ids"] == [source["id"]]
    assert requirement["evidence_quote"] == source["text"]
    assert requirement["evidence_quote"].lower() in analyzer._normalize_whitespace(jd).lower()


def test_invalid_gemini_jd_passage_id_degrades_instead_of_accepting_text():
    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate_content(self, _prompt, **_kwargs):
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "job_title": "Cloud Engineer",
                        "company": "Example",
                        "job_seniority_signal": "mid",
                        "domain_requirements": [
                            {
                                "domain": "Invented requirement",
                                "importance": "must",
                                "match_status": "missing",
                                "jd_evidence_ids": ["jd_999"],
                            }
                        ],
                    }
                )
            )

    with patch.object(analyzer, "_ensure_gemini_configured"), patch.object(
        analyzer.genai, "GenerativeModel", FakeModel
    ):
        result = analyzer._extract_with_gemini_v3(
            "The role requires clear stakeholder communication.",
            title="Cloud Engineer",
            output_language="en",
            candidate_profile=CandidateProfile().model_dump(),
        )

    assert result["analysis_status"] == "degraded"
    assert "valid JD passage ID" in result["evidence_hints"]["error"]
