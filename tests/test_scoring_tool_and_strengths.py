from backend.ir.schema_v3 import AnalyzeIRv3, DomainRequirement, ToolEvidence
from backend.scoring.scoring_engine_v3 import score_ir_v3, score_to_public_dict


def _make_ir(domains):
    return AnalyzeIRv3(
        job_title="Test Role",
        company="TestCo",
        location="Test",
        job_seniority_signal="junior",
        candidate_seniority_signal="junior",
        candidate_skills=[],
        domain_requirements=domains,
        evidence_hints={},
        raw_llm_json=None,
    )


def test_tool_hit_from_candidate_text():
    domains = [
        DomainRequirement(
            name="Generative AI Development",
            importance="should",
            evidence_quote="Experience with Gemini",
            evidence_level="anchored",
            examples=[
                ToolEvidence(
                    name="Gemini",
                    importance="should",
                    evidence_quote="Experience with Gemini",
                )
            ],
        )
    ]
    ir = _make_ir(domains)
    result = score_ir_v3(ir, candidate_text="Built an AI agent using Gemini for LLM inference.")
    assert result.debug_breakdown.per_tool_should.get("Gemini") == "hit"

    public = score_to_public_dict(result)
    assert "Gemini" not in public.get("missing_tools", [])


def test_version_control_in_strengths():
    domains = [
        DomainRequirement(
            name="Version Control",
            importance="should",
            evidence_quote="",
            evidence_level="anchored",
            examples=[ToolEvidence(name="Git", importance="should", evidence_quote="")],
        )
    ]
    ir = _make_ir(domains)
    result = score_ir_v3(ir, candidate_skills=["GitHub"])
    assert result.debug_breakdown.per_domain.get("Version Control") in ("hit", "soft_hit")

    public = score_to_public_dict(result)
    strengths = public.get("strengths", [])
    domain_items = []
    for s in strengths:
        if s.get("type") == "domains":
            domain_items = s.get("items", [])
            break
    assert "Version Control" in domain_items


def test_multiword_named_tool_requires_the_complete_phrase():
    domains = [
        DomainRequirement(
            name="CI/CD Delivery",
            importance="should",
            evidence_quote="Build pipelines using Azure DevOps",
            evidence_level="exact",
            examples=[
                ToolEvidence(
                    name="Azure DevOps",
                    importance="should",
                    evidence_quote="Build pipelines using Azure DevOps",
                )
            ],
        )
    ]
    ir = _make_ir(domains)

    aws_only = score_ir_v3(
        ir,
        candidate_text="AWS Certified DevOps Engineer with cloud delivery experience.",
    )
    exact_tool = score_ir_v3(
        ir,
        candidate_text="Built and maintained deployment pipelines using Azure DevOps.",
    )

    assert aws_only.debug_breakdown.per_tool_should["Azure DevOps"] == "missing"
    assert exact_tool.debug_breakdown.per_tool_should["Azure DevOps"] == "hit"


def test_distinctive_compound_suffix_and_simple_plural_match_without_alias_table():
    domains = [
        DomainRequirement(
            name="Specialized Platform",
            importance="should",
            evidence_quote="Experience with Vendor NovaEngine and ABCs",
            evidence_level="exact",
            examples=[
                ToolEvidence(
                    name="Vendor NovaEngine",
                    importance="should",
                    evidence_quote="Experience with Vendor NovaEngine and ABCs",
                ),
                ToolEvidence(
                    name="ABCs",
                    importance="should",
                    evidence_quote="Experience with Vendor NovaEngine and ABCs",
                ),
            ],
        )
    ]
    result = score_ir_v3(
        _make_ir(domains),
        candidate_text="Built production workflows with NovaEngine and ABC.",
    )

    assert result.debug_breakdown.per_tool_should["Vendor NovaEngine"] == "hit"
    assert result.debug_breakdown.per_tool_should["ABCs"] == "hit"
