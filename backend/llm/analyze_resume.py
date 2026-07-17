"""Gemini-backed resume parser, run once for each distinct resume hash."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from backend.ir.candidate_profile import CandidateEvidenceClaim, CandidateProfile, CandidateRole
from backend.ir.canonicalize import canon_domain, canon_tool
from backend.llm.analyze_v3 import _derive_candidate_level_from_experience, _norm_seniority_label


CANDIDATE_PROFILE_SCHEMA_VERSION = "1.0"
CANDIDATE_PROFILE_PROMPT_VERSION = "1.0"
CANDIDATE_PROFILE_MODEL = (os.getenv("GEMINI_RESUME_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _quote_is_grounded(quote: str, resume_flat_lower: str) -> bool:
    normalized = _normalize(quote).lower()
    return len(normalized) >= 8 and normalized in resume_flat_lower


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = _normalize(value)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _ensure_configured() -> None:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment/.env")
    genai.configure(api_key=key)


def analyze_resume(resume_text: str) -> CandidateProfile:
    """Create a reusable, job-independent candidate profile from resume text."""
    text = (resume_text or "").strip()
    if len(text) < 30:
        raise ValueError("Resume text is too short to analyze")

    _ensure_configured()
    model = genai.GenerativeModel(CANDIDATE_PROFILE_MODEL)
    prompt = f"""You are the Resume Parser for Kairos.

Extract a job-independent candidate profile from the resume. Do not compare it
with any job description. Capture broad transferable capabilities so the same
profile can be reused for many roles.

STRICT RULES:
1. Every evidence claim and role evidence_quote must be copied verbatim from the resume.
2. Do not invent skills, employers, dates, scope, metrics, or seniority.
3. candidate_skills may include tools and portable capabilities supported by a valid evidence claim.
4. Keep evidence claims atomic and concise. Return at most 40 claims and 80 skills.
5. candidate_seniority_signal is only a preliminary label; Kairos also applies deterministic date/title rules.

<resume>
{text[:50000]}
</resume>

OUTPUT JSON ONLY:
{{
  "candidate_skills": ["string"],
  "candidate_domains": ["string"],
  "candidate_seniority_signal": "intern|apprentice|junior|junior_to_mid|mid|senior|lead|unknown",
  "seniority_reason": "short string",
  "evidence_claims": [
    {{
      "resume_quote": "verbatim resume text",
      "skills": ["string"],
      "domains": ["string"],
      "role": "string or null"
    }}
  ],
  "roles": [
    {{
      "title": "string",
      "organization": "string or null",
      "date_range": "string or null",
      "evidence_quote": "verbatim resume text or null"
    }}
  ]
}}
"""

    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
        request_options={"timeout": 30},
    )
    raw: Dict[str, Any] = json.loads(response.text)
    resume_flat_lower = _normalize(text).lower()

    claims: List[CandidateEvidenceClaim] = []
    grounded_skills: List[str] = []
    grounded_domains: List[str] = []
    for item in raw.get("evidence_claims") or []:
        if not isinstance(item, dict):
            continue
        quote = _normalize(item.get("resume_quote"))
        if not _quote_is_grounded(quote, resume_flat_lower):
            continue
        skills = _dedupe([canon_tool(str(x)) or str(x) for x in (item.get("skills") or [])])[:20]
        domains = _dedupe([canon_domain(str(x)) or str(x) for x in (item.get("domains") or [])])[:12]
        claims.append(
            CandidateEvidenceClaim(
                resume_quote=quote,
                skills=skills,
                domains=domains,
                role=_normalize(item.get("role")) or None,
            )
        )
        grounded_skills.extend(skills)
        grounded_domains.extend(domains)
        if len(claims) >= 40:
            break

    roles: List[CandidateRole] = []
    for item in raw.get("roles") or []:
        if not isinstance(item, dict):
            continue
        title = _normalize(item.get("title"))
        evidence_quote = _normalize(item.get("evidence_quote"))
        if not title:
            continue
        if evidence_quote and not _quote_is_grounded(evidence_quote, resume_flat_lower):
            evidence_quote = ""
        roles.append(
            CandidateRole(
                title=title,
                organization=_normalize(item.get("organization")) or None,
                date_range=_normalize(item.get("date_range")) or None,
                evidence_quote=evidence_quote or None,
            )
        )
        if len(roles) >= 20:
            break

    direct_skills = []
    for value in raw.get("candidate_skills") or []:
        skill = canon_tool(str(value)) or _normalize(value)
        if skill and _normalize(skill).lower() in resume_flat_lower:
            direct_skills.append(skill)
    direct_domains = []
    for value in raw.get("candidate_domains") or []:
        domain = canon_domain(str(value)) or _normalize(value)
        if domain and _normalize(domain).lower() in resume_flat_lower:
            direct_domains.append(domain)

    candidate_skills = _dedupe(direct_skills + grounded_skills)[:80]
    candidate_domains = _dedupe(direct_domains + grounded_domains)[:40]
    if not claims and not candidate_skills:
        raise ValueError("Gemini returned no resume-grounded candidate evidence")

    deterministic_level, deterministic_reason, _entries = _derive_candidate_level_from_experience(
        text, job_family="general"
    )
    llm_level = _norm_seniority_label(raw.get("candidate_seniority_signal"))
    seniority = deterministic_level or llm_level or "unknown"
    reason = (
        f"experience_inference:{deterministic_reason}"
        if deterministic_level
        else _normalize(raw.get("seniority_reason")) or "llm_resume_signal"
    )

    return CandidateProfile(
        candidate_skills=candidate_skills,
        candidate_domains=candidate_domains,
        candidate_seniority_signal=seniority,
        seniority_reason=reason,
        evidence_claims=claims,
        roles=roles,
        analysis_status="ready",
        raw_llm_json=raw,
    )


def candidate_profile_evidence_text(profile: CandidateProfile | Dict[str, Any]) -> str:
    parsed = profile if isinstance(profile, CandidateProfile) else CandidateProfile.model_validate(profile)
    lines = [claim.resume_quote for claim in parsed.evidence_claims]
    if parsed.candidate_skills:
        lines.append("Candidate skills: " + ", ".join(parsed.candidate_skills))
    if parsed.candidate_domains:
        lines.append("Candidate domains: " + ", ".join(parsed.candidate_domains))
    return "\n".join(_dedupe(lines))


__all__ = [
    "CANDIDATE_PROFILE_MODEL",
    "CANDIDATE_PROFILE_PROMPT_VERSION",
    "CANDIDATE_PROFILE_SCHEMA_VERSION",
    "analyze_resume",
    "candidate_profile_evidence_text",
]
