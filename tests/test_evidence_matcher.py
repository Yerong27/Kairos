from backend.ir.candidate_profile import CandidateEvidenceClaim
from backend.ir.schema_v3 import AnalyzeIRv3, DomainRequirement
from backend.scoring.evidence_matcher import match_requirement_to_evidence
from backend.scoring.scoring_engine_v3 import score_ir_v3, score_to_public_dict


def _claim(
    quote: str,
    *,
    evidence_id: str = "ev_1",
    skills=None,
    domains=None,
    claim_type: str = "experience",
):
    return CandidateEvidenceClaim(
        evidence_id=evidence_id,
        claim_type=claim_type,
        resume_quote=quote,
        skills=skills or [],
        domains=domains or [],
    )


def _requirement(
    name: str,
    *,
    requirement_type: str = "capability",
    alternatives=None,
):
    return DomainRequirement(
        name=name,
        importance="should",
        requirement_type=requirement_type,
        alternatives=alternatives or [],
        evidence_quote=f"Requirement: {name}",
        evidence_level="exact",
        evidence_status="verified",
    )


def test_broad_domain_does_not_prove_specific_requirement():
    requirement = _requirement("Infrastructure as Code")
    claim = _claim(
        "Supported production workloads on AWS.",
        skills=["AWS"],
        domains=["Cloud Computing"],
    )

    decision = match_requirement_to_evidence(requirement, [claim])

    assert decision.status == "missing"
    assert decision.evidence_ids == []


def test_specific_grounded_claim_matches_capability():
    requirement = _requirement("Infrastructure as Code")
    claim = _claim(
        "Built infrastructure as code modules with Terraform.",
        skills=["Terraform"],
        domains=["Cloud Computing"],
    )

    decision = match_requirement_to_evidence(requirement, [claim])

    assert decision.status == "matched"
    assert decision.evidence_ids == ["ev_1"]
    assert decision.reason == "exact_grounded_phrase"


def test_inferred_skill_without_quote_is_only_partial():
    requirement = _requirement("Stakeholder Management")
    claim = _claim(
        "Prepared monthly financial reports.",
        skills=["Stakeholder Management"],
        domains=["Business Analysis"],
    )

    decision = match_requirement_to_evidence(requirement, [claim])

    assert decision.status == "partial"
    assert decision.reason == "structured_related_evidence"


def test_strict_credential_requires_explicit_resume_evidence():
    requirement = _requirement("CPA", requirement_type="credential")
    matched = match_requirement_to_evidence(
        requirement,
        [_claim("Certified Practising Accountant (CPA).", claim_type="credential")],
    )
    missing = match_requirement_to_evidence(
        requirement,
        [_claim("Completed a Bachelor of Commerce.", claim_type="education")],
    )

    assert matched.status == "matched"
    assert missing.status == "missing"


def test_literal_alternative_can_satisfy_one_requirement():
    requirement = _requirement(
        "Statistical programming language",
        requirement_type="tool",
        alternatives=["Python", "R"],
    )
    decision = match_requirement_to_evidence(
        requirement,
        [_claim("Built forecasting models in Python.", skills=["Python"])],
    )

    assert decision.status == "matched"


def test_scoring_ignores_ungrounded_global_skill_and_returns_evidence_ids():
    requirement = _requirement("Infrastructure as Code")
    unrelated = _claim(
        "Supported production workloads on AWS.",
        skills=["AWS"],
        domains=["Cloud Computing"],
    )
    ir = AnalyzeIRv3(
        job_title="Cloud Engineer",
        company="Example",
        job_seniority_signal="junior",
        candidate_seniority_signal="junior",
        candidate_skills=["Infrastructure as Code"],
        candidate_evidence_claims=[unrelated],
        domain_requirements=[requirement],
    )

    first = score_to_public_dict(score_ir_v3(ir))
    second = score_to_public_dict(score_ir_v3(ir))

    item = first["requirements"]["items"][0]
    assert item["status"] == "missing"
    assert item["resume_evidence_ids"] == []
    assert first["requirements"] == second["requirements"]


def test_scoring_exposes_supporting_evidence_id_for_a_match():
    requirement = _requirement("Infrastructure as Code")
    supporting = _claim(
        "Built infrastructure as code modules with Terraform.",
        evidence_id="ev_terraform",
        skills=["Terraform"],
    )
    ir = AnalyzeIRv3(
        job_title="Cloud Engineer",
        company="Example",
        job_seniority_signal="junior",
        candidate_seniority_signal="junior",
        candidate_evidence_claims=[supporting],
        domain_requirements=[requirement],
    )

    contract = score_to_public_dict(score_ir_v3(ir))
    item = contract["requirements"]["items"][0]

    assert item["status"] == "matched"
    assert item["resume_evidence_ids"] == ["ev_terraform"]
    assert contract["analysis_quality"]["matcher_mode"] == "evidence_grounded_v2"
    assert contract["analysis_quality"]["unsupported_matches"] == 0
    assert contract["analysis_quality"]["score_reliable"] is True


def test_match_from_legacy_profile_without_evidence_id_is_not_marked_reliable():
    requirement = _requirement("Financial Reporting")
    legacy_claim = CandidateEvidenceClaim(
        resume_quote="Owned monthly financial reporting and variance analysis.",
    )
    ir = AnalyzeIRv3(
        job_title="Financial Accountant",
        company="Example",
        job_seniority_signal="mid",
        candidate_seniority_signal="mid",
        candidate_evidence_claims=[legacy_claim],
        domain_requirements=[requirement],
    )

    contract = score_to_public_dict(score_ir_v3(ir))

    assert contract["requirements"]["items"][0]["status"] == "matched"
    assert contract["analysis_quality"]["unsupported_matches"] == 1
    assert contract["analysis_quality"]["score_reliable"] is False
