"""Versioned candidate profile produced once per resume content hash."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


CandidateProfileStatus = Literal["ready", "degraded"]


class CandidateEvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resume_quote: str
    skills: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    role: Optional[str] = None


class CandidateRole(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    organization: Optional[str] = None
    date_range: Optional[str] = None
    evidence_quote: Optional[str] = None


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_skills: List[str] = Field(default_factory=list)
    candidate_domains: List[str] = Field(default_factory=list)
    candidate_seniority_signal: str = "unknown"
    seniority_reason: Optional[str] = None
    evidence_claims: List[CandidateEvidenceClaim] = Field(default_factory=list)
    roles: List[CandidateRole] = Field(default_factory=list)
    analysis_status: CandidateProfileStatus = "ready"
    raw_llm_json: Optional[Dict[str, Any]] = None


__all__ = ["CandidateEvidenceClaim", "CandidateProfile", "CandidateRole"]
