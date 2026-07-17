"""Gemini-backed resume parser, run once for each distinct resume hash."""

from __future__ import annotations

import json
import hashlib
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from backend.ir.candidate_profile import CandidateEvidenceClaim, CandidateProfile, CandidateRole
from backend.ir.canonicalize import canon_domain, canon_tool
CANDIDATE_PROFILE_SCHEMA_VERSION = "2.0"
CANDIDATE_PROFILE_PROMPT_VERSION = "2.1"
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


def _resume_prompt(text: str, *, compact: bool = False) -> str:
    claim_limit = 12 if compact else 20
    skill_limit = 32 if compact else 50
    role_limit = 10 if compact else 15
    quote_limit = 160 if compact else 240
    retry_note = (
        "This is a compact retry because the previous JSON response was incomplete. "
        if compact
        else ""
    )
    return f"""You are the Resume Parser for Kairos.

{retry_note}Extract a job-independent candidate profile from the resume. Do not compare it
with any job description. Capture broad transferable capabilities so the same
profile can be reused for many roles.

STRICT RULES:
1. Every evidence claim and role evidence_quote must be one continuous verbatim substring from the resume.
2. Keep every evidence quote under {quote_limit} characters. Never copy a whole section or paragraph.
3. Do not invent skills, employers, dates, scope, metrics, or seniority.
4. candidate_skills may include tools and portable capabilities supported by a valid evidence claim.
5. Return at most {claim_limit} evidence claims, {skill_limit} candidate skills, and {role_limit} roles.
6. Return at most 8 skills and 4 domains per evidence claim.
7. Do not assign one global seniority level. Seniority depends on the target job
   family and is calculated locally when a job is analyzed.
8. Classify every evidence claim using one general claim_type. Do not invent
   evidence IDs; Kairos creates them deterministically after validation.
9. `domains` must contain only portable capabilities directly demonstrated by
   the quote. A named product may support its conventional capability (for
   example, a container tool may support Containerization), but do not infer
   unrelated or broader expertise.

<resume>
{text[:50000]}
</resume>

OUTPUT JSON ONLY:
{{
  "candidate_skills": ["string"],
  "candidate_domains": ["string"],
  "evidence_claims": [
    {{
      "resume_quote": "short verbatim resume substring",
      "claim_type": "achievement|capability|credential|education|experience|work_condition|other",
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
      "evidence_quote": "short verbatim resume substring or null"
    }}
  ]
}}
"""


def _parse_json_response(response: Any) -> Dict[str, Any]:
    response_text = str(getattr(response, "text", "") or "").strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\s*```$", "", response_text)
    parsed = json.loads(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini resume response must be a JSON object")
    return parsed


def _generate_profile_json(model: Any, text: str) -> Dict[str, Any]:
    last_error: Exception | None = None
    for compact, max_tokens in ((False, 6144), (True, 4096)):
        response = model.generate_content(
            _resume_prompt(text, compact=compact),
            generation_config=GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
            request_options={"timeout": 30},
        )
        try:
            return _parse_json_response(response)
        except json.JSONDecodeError as exc:
            last_error = exc
            if compact:
                break

    raise ValueError(
        "Gemini returned incomplete resume JSON after an automatic compact retry. "
        "Please re-upload the resume."
    ) from last_error


def analyze_resume(resume_text: str) -> CandidateProfile:
    """Create a reusable, job-independent candidate profile from resume text."""
    text = (resume_text or "").strip()
    if len(text) < 30:
        raise ValueError("Resume text is too short to analyze")

    _ensure_configured()
    model = genai.GenerativeModel(CANDIDATE_PROFILE_MODEL)
    raw = _generate_profile_json(model, text)
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
        claim_type = _normalize(item.get("claim_type")).lower()
        if claim_type not in {
            "achievement",
            "capability",
            "credential",
            "education",
            "experience",
            "work_condition",
            "other",
        }:
            claim_type = "other"
        evidence_key = f"{quote.lower()}|{_normalize(item.get('role')).lower()}"
        evidence_id = "ev_" + hashlib.sha256(evidence_key.encode("utf-8")).hexdigest()[:12]
        claims.append(
            CandidateEvidenceClaim(
                evidence_id=evidence_id,
                claim_type=claim_type,
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

    return CandidateProfile(
        candidate_skills=candidate_skills,
        candidate_domains=candidate_domains,
        # A career changer does not have one meaningful global seniority.
        # The job analyzer derives a role-family-specific level from the
        # locally stored resume without sending it to Gemini again.
        candidate_seniority_signal="unknown",
        seniority_reason="deferred_to_job_family_analysis",
        evidence_claims=claims,
        roles=roles,
        analysis_status="ready",
        raw_llm_json=raw,
    )


def candidate_profile_evidence_text(profile: CandidateProfile | Dict[str, Any]) -> str:
    parsed = profile if isinstance(profile, CandidateProfile) else CandidateProfile.model_validate(profile)
    lines = [
        f"[{claim.evidence_id or 'legacy'}] {claim.resume_quote}"
        for claim in parsed.evidence_claims
    ]
    return "\n".join(_dedupe(lines))


__all__ = [
    "CANDIDATE_PROFILE_MODEL",
    "CANDIDATE_PROFILE_PROMPT_VERSION",
    "CANDIDATE_PROFILE_SCHEMA_VERSION",
    "analyze_resume",
    "candidate_profile_evidence_text",
]
