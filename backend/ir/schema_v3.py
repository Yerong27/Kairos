# backend/ir/schema_v3.py
"""
Kairos IR Schema v3 (Domain-first).

Goals:
- Separate "portable capabilities" (DomainRequirement) from "tool examples" (ToolEvidence).
- Keep scoring deterministic and domain-agnostic.
- Add optional anchors/aliases to DomainRequirement so matching can be robust
  across industries WITHOUT hard-coded synonym dictionaries in scoring.
- USES PYDANTIC for Validation.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import AliasChoices, BaseModel, Field, ConfigDict, field_validator
from backend.ir.candidate_profile import CandidateEvidenceClaim


# -----------------------------
# Type aliases
# -----------------------------
Importance = Literal["must", "should", "nice_to_have", "nice", "unknown"]
SeniorityLabel = Literal["intern", "junior", "mid", "senior", "lead", "principal", "unknown"]

# lightweight domain facet for explainability / safer importance rules
DomainFacet = Literal["technical", "process", "people", "unknown"]
SupportLevel = Literal["strong", "weak", "none"]
MatchStatus = Literal["matched", "partial", "missing", "unknown"]
RequirementType = Literal[
    "capability",
    "tool",
    "credential",
    "education",
    "experience",
    "work_condition",
    "responsibility",
    "other",
]

# Phase 3 Protocol: 4-Level Evidence
EvidenceLevel = Literal["exact", "anchored", "weak", "none"]


# -----------------------------
# Atomic structures
# -----------------------------
class ToolEvidence(BaseModel):
    """
    ToolEvidence is a specific tool/platform/library mentioned in the JD.
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="tool", description="Name of the tool/skill example (e.g. 'Python', 'Jira')")
    importance: Importance
    evidence_quote: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("jd_evidence_quote", "evidence_quote"),
    )
    jd_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Backend-owned JD passage IDs that literally contain this keyword",
    )
    keyword_type: Literal[
        "tool",
        "platform",
        "language",
        "method",
        "credential",
        "standard",
        "domain_term",
        "other",
    ] = "other"

    # Internal / Optional
    cost: Optional[str] = None
    link: Optional[str] = None
    description: Optional[str] = None


class DomainRequirement(BaseModel):
    """
    DomainRequirement is a portable capability (the scoring backbone).
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="domain", description="Portable, industry-agnostic domain name (e.g. 'Project Management')")
    importance: Importance
    requirement_type: RequirementType = "capability"
    alternatives: List[str] = Field(
        default_factory=list,
        description="Literal OR alternatives that can satisfy the same requirement",
    )
    match_status: MatchStatus = "unknown"
    jd_evidence_ids: List[str] = Field(default_factory=list)
    resume_evidence_ids: List[str] = Field(default_factory=list)
    match_reason: Optional[str] = None

    # Phase 3: Canonical ID & Evidence Protocol
    domain_id: Optional[str] = Field(default=None, description="Canonical Domain ID from Catalog")

    evidence_quote: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("jd_evidence_quote", "evidence_quote"),
    )
    evidence_summary: Optional[str] = Field(default=None, description="Concise summary of the full requirement context")
    examples: List[ToolEvidence] = Field(default_factory=list, description="Concrete tools or examples")

    # Enrichment / Internal (LLM does not output these)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    facet: Optional[str] = None
    proficiency: Optional[str] = None
    support_level: Optional[str] = None

    # Replaced is_verified with evidence_level protocol
    evidence_level: EvidenceLevel = "none"
    evidence_status: Optional[str] = None
    span_meta: Optional[Dict[str, Any]] = Field(default=None, description="{method, found, text}")

    anchors: List[str] = Field(default_factory=list)


# -----------------------------
# Ownership & Scope (Phase 3: Structured Integers)
# -----------------------------
class OwnershipScope(BaseModel):
    """
    Extracted signals for seniority determination.
    """
    present: bool = False
    level_val: int = 0  # 0=None, 1=Self, 2=Team, 3=Org/Multi
    evidence: List[str] = Field(default_factory=list)

class ScopeLevel(BaseModel):
    level: Literal["self", "team", "multi_team", "unknown"] = "unknown"
    level_val: int = 0 # 0=Unknown, 1=Self, 2=Team, 3=Multi
    evidence: List[str] = Field(default_factory=list)

class LeadershipSignal(BaseModel):
    present: bool = False
    level_val: int = 0 # 0=None, 1=Mentor, 2=Lead, 3=Manager/Dir
    evidence: List[str] = Field(default_factory=list)

class ExtractedOwnership(BaseModel):
    ownership: OwnershipScope = Field(default_factory=OwnershipScope)
    scope: ScopeLevel = Field(default_factory=ScopeLevel)
    leadership: LeadershipSignal = Field(default_factory=LeadershipSignal)


class ApplicationRecommendation(BaseModel):
    should_apply: Literal["Yes", "Maybe", "No"] = "Maybe"
    confidence: Literal["low", "medium", "high"] = "low"
    rationale: str = ""
    hard_blockers: List[str] = Field(default_factory=list)
    strongest_matches: List[str] = Field(default_factory=list)
    key_gaps: List[str] = Field(default_factory=list)


# -----------------------------
# Top-level IR
# -----------------------------
class AnalyzeIRv3(BaseModel):
    """
    Output of analyze_v3.py (LLM extraction + validation).
    """
    job_title: str
    company: str
    location: Optional[str] = None

    job_seniority_signal: SeniorityLabel
    candidate_seniority_signal: Optional[str] = Field(description="Verbatim title-like string from candidate profile", default="N/A")

    candidate_skills: List[str] = Field(default_factory=list, description="Extracted candidate skills")
    candidate_evidence_claims: List[CandidateEvidenceClaim] = Field(
        default_factory=list,
        description="Resume-grounded claims produced once per resume hash",
    )

    domain_requirements: List[DomainRequirement] = Field(default_factory=list, description="List of extracted domain requirements")
    jd_keywords: List[ToolEvidence] = Field(
        default_factory=list,
        description="Evidence-grounded explicit JD keywords for ATS review",
    )
    tools_in_jd: List[str] = Field(default_factory=list, description="Tools explicitly mentioned in the JD")

    # New Phase 2 Field
    ownership_and_scope: ExtractedOwnership = Field(default_factory=ExtractedOwnership)
    application_recommendation: ApplicationRecommendation = Field(
        default_factory=ApplicationRecommendation
    )

    evidence_hints: Dict[str, Optional[str]] = Field(default_factory=dict)

    # Debug / Raw data storage (for transparency)
    raw_llm_json: Optional[Dict[str, Any]] = Field(default=None, description="The raw JSON response from the LLM")

    # Phase 3: Degraded Status
    analysis_status: Literal["success", "degraded"] = "success"

    @field_validator("evidence_hints", mode="before")
    @classmethod
    def allow_empty_list_for_dict(cls, v: Any) -> Any:
        if isinstance(v, list) and not v:
            return {}
        return v


__all__ = [
    "Importance",
    "SeniorityLabel",
    "DomainFacet",
    "RequirementType",
    "MatchStatus",
    "EvidenceLevel",
    "ToolEvidence",
    "DomainRequirement",
    "AnalyzeIRv3",
    "ExtractedOwnership",
    "OwnershipScope",
    "ScopeLevel",
    "LeadershipSignal",
    "ApplicationRecommendation",
]
