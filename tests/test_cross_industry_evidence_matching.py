import pytest

from backend.ir.candidate_profile import CandidateEvidenceClaim
from backend.ir.schema_v3 import DomainRequirement
from backend.scoring.evidence_matcher import match_requirement_to_evidence


@pytest.mark.parametrize(
    ("requirement", "requirement_type", "quote", "expected"),
    [
        ("Financial Reporting", "capability", "Owned monthly financial reporting and variance analysis.", "matched"),
        ("Salesforce", "tool", "Managed the sales pipeline in Salesforce.", "matched"),
        ("Registered Nurse licence", "credential", "Current Registered Nurse licence in Victoria.", "matched"),
        ("Process Improvement", "capability", "Improved the monthly close process and reduced delays.", "matched"),
        ("Curriculum Development", "capability", "Led curriculum development for secondary students.", "matched"),
        ("CPA", "credential", "Completed a Bachelor of Commerce.", "missing"),
    ],
)
def test_same_evidence_contract_works_across_professions(
    requirement,
    requirement_type,
    quote,
    expected,
):
    decision = match_requirement_to_evidence(
        DomainRequirement(
            name=requirement,
            importance="should",
            requirement_type=requirement_type,
            evidence_quote=f"Requirement: {requirement}",
            evidence_level="exact",
        ),
        [
            CandidateEvidenceClaim(
                evidence_id="ev_cross_industry",
                claim_type="experience",
                resume_quote=quote,
            )
        ],
    )

    assert decision.status == expected
    if expected != "missing":
        assert decision.evidence_ids == ["ev_cross_industry"]
