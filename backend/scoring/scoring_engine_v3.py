# backend/scoring/scoring_engine_v3.py
"""
Kairos Scoring Engine v3.5.4 (Evidence-weighted) (Domain-first) + IR adapter

Changes vs v3.5.3:

1) (2026-01-04, output formatting)
- Make actions output bullet-style (multi-line string with "- " steps),
  instead of packing many steps into a single long sentence.

2) (2026-01-04, scoring policy)
- Tools are treated as *bonus evidence* only (should), not hard blockers.
  This avoids "missing tool" turning into MUST-missing penalties.
- Still allow a small "hard tool" set to appear in required_skills (public output),
  but it does NOT affect penalties or gate risk.
  Heuristic: only keep tool marked "must" AND non-redundant with its parent domain.
  Example: Version Control -> (Git/GitHub) are redundant and excluded.

3) Kept from v3.5.3:
- candidate_text support: extract evidence claims and merge into candidate skills.
- anchors_used debug capture stable dict.
- coverage penalty cliff fix and softened keyword stuffing penalty.

Fix in this revision (2026-01-05):
- Restore proper control-flow in score_ir_v3 (no early returns of str).
- Add _decide_should_apply(...) returning ShouldApply.
- Make "apprentice" map to label "apprentice" (numeric 0.5), not "junior".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from backend.ir.schema_v3 import AnalyzeIRv3, DomainRequirement, ToolEvidence, SeniorityLabel
from backend.ir.candidate_profile import CandidateEvidenceClaim
from backend.ir.canonicalize import canon_tool
from backend.scoring.evidence_matcher import EvidenceMatchDecision, match_requirement_to_evidence

ShouldApply = Literal["Yes", "Maybe", "No"]
RiskLevel = Literal["none", "low", "medium", "high"]

SeniorityBucket = Literal["overqualified", "none", "small", "medium", "cliff"]

_WS = re.compile(r"\s+")
_RE_MUST = re.compile(r"\bmust\b", re.IGNORECASE)
_RE_SHOULD = re.compile(r"\bshould\b", re.IGNORECASE)

# Acronym in parentheses e.g. "Identity and Access Management (IAM)"
_PAREN_ACRONYM_RE = re.compile(r"\b([A-Za-z][A-Za-z \-\/]{2,80})\s*\(\s*([A-Za-z0-9]{2,12})\s*\)")

_STOP_TOKENS = {
    "and",
    "or",
    "with",
    "to",
    "in",
    "of",
    "for",
    "on",
    "into",
    "across",
    "the",
    "a",
    "an",
    "skills",
    "skill",
    "experience",
    "experienced",
    "knowledge",
    "development",
    "engineering",
    "principles",
    "systems",
    "system",
    "platform",
    "platforms",
    "framework",
    "frameworks",
    "tools",
    "tool",
    "methodologies",
    "methodology",
    "hands",
    "hand",
    "strong",
    "solid",
    "foundation",
    "proficiency",
    "modern",
    "advanced",
    "applications",
    "application",
    "service",
    "services",
    "team",
    "teams",
    "environment",
    "environments",
    "using",
    "use",
    "used",
    "practices",
    "practice",
    "principle",
    "principles",
    "approach",
    "approaches",
    "method",
    "methods",
}

_VENDOR_PREFIXES = {
    "aws",
    "amazon",
    "amazon web services",
    "azure",
    "microsoft",
    "ms",
    "gcp",
    "google",
    "google cloud",
    "ibm",
    "oracle",
    "oci",
}

# Short-but-strong tokens that often appear as standalone claims
_STRONG_SHORT_TOKENS = {
    # AWS-ish
    "iam",
    "s3",
    "ec2",
    "ecs",
    "eks",
    "emr",
    "glue",
    "athena",
    "lambda",
    "cloudwatch",
    "api",
    "apigateway",
    "sts",
    "kms",
    "vpc",
    "sqs",
    "sns",
    "dynamodb",
    "rds",
    "aurora",
    # data/eng
    "sql",
    "etl",
    "rag",
    "grpc",
    "rest",
    "restapi",
    "json",
    "yaml",
    "parquet",
    "avro",
    "spark",
    "airflow",
    "dbt",
    # security/auth
    "jwt",
    "oauth",
    "oidc",
    "tls",
    "ssl",
    # llm
    "gpt4",
    "gpt-4",
    "gemini",
}

# Tool normalization helpers
_TOOL_SPLIT_RE = re.compile(r"\s*(?:/|\+|,|;|\band\b|\bor\b)\s*", re.IGNORECASE)
_TOOL_IGNORE = {
    "llm",
    "llms",
    "agentic system",
    "agentic systems",
}

# Common aliases to reduce misses from tool/name variants
_SKILL_ALIAS_MAP = {
    "github actions": ["github", "git", "actions"],
    "gitlab ci": ["gitlab", "git"],
    "gitlab": ["gitlab", "git"],
    "bitbucket": ["bitbucket", "git"],
    "source control": ["version control", "git"],
    "version control": ["git"],
    "git hub": ["github"],
}

# Very small set of empty-vibe phrases (avoid over-penalizing; only weaken strong claims)
_WEAK_PHRASES = {
    "team player",
    "fast learner",
    "self starter",
    "self-starter",
    "hard worker",
    "proactive",
    "enthusiastic",
}

# ============================================================
# IR adapter helpers
# ============================================================
def _safe_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return []


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to access field whether obj is a dict or an object/Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)



def _get_raw_llm(job_ir: Any) -> Dict[str, Any]:
    raw = _get_field(job_ir, "raw_llm_json", None)
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_domains_from_raw(raw: Dict[str, Any]) -> List[DomainRequirement]:
    job = raw.get("job") if isinstance(raw.get("job"), dict) else {}
    dr_list = job.get("domain_requirements") if isinstance(job.get("domain_requirements"), list) else []

    out: List[DomainRequirement] = []
    for item in dr_list:
        if isinstance(item, DomainRequirement):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue

        examples_raw = item.get("examples")
        examples_list = examples_raw if isinstance(examples_raw, list) else []
        examples: List[ToolEvidence] = []
        for ex in examples_list:
            if isinstance(ex, ToolEvidence):
                examples.append(ex)
                continue
            if isinstance(ex, dict):
                examples.append(
                    ToolEvidence(
                        name=str(ex.get("name") or ""),
                        importance=cast(Any, ex.get("importance") or "should"),
                        evidence_quote=ex.get("evidence_quote"),
                    )
                )

        anchors_raw = item.get("anchors")
        anchors = [str(a) for a in anchors_raw] if isinstance(anchors_raw, list) else []

        evidence_level = item.get("evidence_level")
        evidence_status = item.get("evidence_status")
        span_meta = item.get("span_meta") if isinstance(item.get("span_meta"), dict) else None

        out.append(
            DomainRequirement(
                name=str(item.get("name") or ""),
                importance=cast(Any, item.get("importance") or "should"),
                requirement_type=cast(Any, item.get("requirement_type") or "capability"),
                alternatives=[
                    str(value)
                    for value in _safe_list(item.get("alternatives"))
                    if str(value).strip()
                ],
                match_status=cast(Any, item.get("match_status") or "unknown"),
                jd_evidence_ids=[
                    str(value)
                    for value in _safe_list(item.get("jd_evidence_ids"))
                    if str(value).strip()
                ],
                resume_evidence_ids=[
                    str(value)
                    for value in _safe_list(item.get("resume_evidence_ids"))
                    if str(value).strip()
                ],
                match_reason=(
                    str(item.get("match_reason"))
                    if item.get("match_reason") is not None
                    else None
                ),
                evidence_quote=item.get("evidence_quote"),
                evidence_summary=item.get("evidence_summary"),
                facet=cast(Any, item.get("facet") or "unknown"),
                anchors=anchors,
                examples=examples,
                domain_id=item.get("domain_id"),
                evidence_level=cast(Any, evidence_level or "none"),
                evidence_status=str(evidence_status) if isinstance(evidence_status, str) else None,
                span_meta=span_meta,
                support_level=item.get("support_level"),
            )
        )
    return out


def _get_domains(job_ir: Any) -> List[DomainRequirement]:
    doms = _get_field(job_ir, "domain_requirements", None)
    if isinstance(doms, list) and doms:
        # If already objects (or dicts that match structure), use them.
        # But if they are dicts, we might need to be careful if downstream expects objects.
        # For now, let's assume if it's a list, we trust it or partial re-parse.
        if all(isinstance(x, DomainRequirement) for x in doms):
            return cast(List[DomainRequirement], doms)
        # If list of dicts, let's pass to _extract_domains_from_raw-like logic,
        # or just wrap them? _extract_domains_from_raw is robust.
        # Let's treat the whole structure as "raw-like" if it's a list of dicts.
        raw = {"job": {"domain_requirements": doms}}
        return _extract_domains_from_raw(raw)

    # dict-style job_ir may nest domains under job.domain_requirements
    if isinstance(job_ir, dict):
        job = job_ir.get("job")
        if isinstance(job, dict):
            dr_list = job.get("domain_requirements")
            if isinstance(dr_list, list) and dr_list:
                return _extract_domains_from_raw(job_ir)

    raw = _get_raw_llm(job_ir)
    if isinstance(raw, dict):
        dr_top = raw.get("domain_requirements")
        if isinstance(dr_top, list) and dr_top:
            return _extract_domains_from_raw({"job": {"domain_requirements": dr_top}})
        if isinstance(raw.get("job"), dict) and isinstance(raw["job"].get("domain_requirements"), list):
            return _extract_domains_from_raw(raw)
    return _extract_domains_from_raw(raw)


def _normalize_skill_list(skills: List[Any]) -> List[str]:
    out: List[str] = []
    for s in skills:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if not t:
            continue
        parts = re.split(r"[;/,\|]+", t)
        for p in parts:
            p2 = p.strip()
            if p2:
                out.append(p2)
    return out


def _apply_skill_aliases(skills: List[str]) -> List[str]:
    out: List[str] = []
    for s in skills:
        out.append(s)
        key = s.strip().lower()
        for k, vals in _SKILL_ALIAS_MAP.items():
            if key == k:
                out.extend(vals)
    return out


def _canonicalize_skill_list(skills: List[str]) -> List[str]:
    out: List[str] = []
    for s in skills or []:
        t = canon_tool(s) or s
        if t:
            out.append(t)
    return out


def _get_candidate_skills(job_ir: Any, candidate_skills: Optional[List[str]] = None) -> List[str]:
    """
    Return candidate skills list.
    Priority:
      1) explicit candidate_skills passed into scorer
      2) fall back to job_ir["candidate_skills"] if present
    """
    merged: List[Any] = []

    # explicit list first
    if isinstance(candidate_skills, list):
        merged.extend(candidate_skills)

    # then job_ir candidate_skills from dict or object
    if isinstance(job_ir, dict):
        merged.extend(job_ir.get("candidate_skills") or [])
    else:
        merged.extend(getattr(job_ir, "candidate_skills", None) or [])

    # also merge raw_llm_json candidate skills (if available)
    raw = _get_raw_llm(job_ir) or {}
    cand = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {}
    for key in ("skills", "candidate_skills", "candidateSkills"):
        s = cand.get(key)
        if isinstance(s, list):
            merged.extend(s)

    merged = _apply_skill_aliases(_normalize_skill_list(merged))
    merged = _canonicalize_skill_list(merged)

    # normalize: trim + drop empties + de-dup (case-insensitive)
    out: List[str] = []
    seen = set()
    for s in merged:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _get_candidate_evidence_claims(job_ir: Any) -> List[CandidateEvidenceClaim]:
    raw_claims = _get_field(job_ir, "candidate_evidence_claims", None)
    if not isinstance(raw_claims, list):
        return []
    claims: List[CandidateEvidenceClaim] = []
    for item in raw_claims:
        try:
            claim = item if isinstance(item, CandidateEvidenceClaim) else CandidateEvidenceClaim.model_validate(item)
        except Exception:
            continue
        if claim.resume_quote.strip():
            claims.append(claim)
    return claims


def _get_explicit_jd_tools(job_ir: Any) -> List[str]:
    tools = _get_field(job_ir, "tools_in_jd", None)
    if not isinstance(tools, list):
        raw = _get_raw_llm(job_ir)
        tools = raw.get("tools_in_jd") if isinstance(raw.get("tools_in_jd"), list) else None
        if tools is None and isinstance(raw.get("job"), dict):
            tools = raw["job"].get("tools_in_jd")
    # Preserve literal alternatives/groups ("Relational / NoSQL database")
    # instead of splitting them into fragments that lose shared context.
    normalized = _canonicalize_skill_list(
        [str(item).strip() for item in tools if str(item).strip()]
        if isinstance(tools, list)
        else []
    )
    seen = set()
    out: List[str] = []
    for tool in normalized:
        key = _norm(tool)
        if key and key not in seen:
            seen.add(key)
            out.append(tool)
    return out


def _get_explicit_jd_keyword_meta(job_ir: Any) -> Dict[str, Dict[str, Any]]:
    raw_keywords = _get_field(job_ir, "jd_keywords", None)
    if not isinstance(raw_keywords, list):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for item in raw_keywords:
        name = str(_get_field(item, "name", "") or "").strip()
        evidence_quote = str(_get_field(item, "evidence_quote", "") or "").strip()
        evidence_ids = [
            str(value).strip()
            for value in (_get_field(item, "jd_evidence_ids", []) or [])
            if str(value).strip()
        ]
        if (
            not name
            or not evidence_quote
            or name.casefold() not in evidence_quote.casefold()
            or not evidence_ids
        ):
            continue
        result[_norm(name)] = {
            "name": name,
            "keyword_type": str(_get_field(item, "keyword_type", "other") or "other"),
            "importance": str(_get_field(item, "importance", "should") or "should"),
            "jd_evidence_ids": evidence_ids[:3],
            "jd_evidence": evidence_quote[:500],
        }
    return result


def _is_explicit_named_example(tool_name: str, evidence_quote: Optional[str]) -> bool:
    """Avoid turning LLM-normalized capability labels into fake keywords."""
    quote = str(evidence_quote or "")
    name = str(tool_name or "").strip()
    if not quote or not name:
        return False
    # Literal presence in the backend-owned parent passage is an
    # industry-neutral signal for software, credentials, standards and terms.
    return name.lower() in quote.lower()

def _get_job_level(job_ir: Any) -> SeniorityLabel:
    lvl = _get_field(job_ir, "job_seniority_signal", None)
    if isinstance(lvl, str) and lvl.strip():
        return cast(SeniorityLabel, lvl.strip().lower())

    raw = _get_raw_llm(job_ir)
    job = raw.get("job") if isinstance(raw.get("job"), dict) else {}
    jl = job.get("job_seniority")
    if isinstance(jl, str) and jl.strip():
        return cast(SeniorityLabel, jl.strip().lower())

    return "unknown"


def _get_candidate_title_signal(job_ir: Any) -> str:
    s = _get_field(job_ir, "candidate_seniority_signal", None)
    if isinstance(s, str) and s.strip():
        return s

    raw = _get_raw_llm(job_ir)
    cand = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {}
    se = cand.get("seniority_evidence")
    if isinstance(se, str) and se.strip():
        return se

    meta = raw.get("_debug_meta") if isinstance(raw.get("_debug_meta"), dict) else {}
    ct = meta.get("candidate_title_signal")
    if isinstance(ct, str) and ct.strip():
        return ct

    return "unknown"


# ============================================================
# Normalization helpers
# ============================================================
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = _WS.sub(" ", s)
    return s


def _canon_token(s: str) -> str:
    t = _norm(s)
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[“”\"']", "", t)
    t = re.sub(r"[^a-z0-9\+#/ \-\.]+", " ", t)  # keep + # / - .
    t = _WS.sub(" ", t).strip()
    return t


def _compact(s: str) -> str:
    t = _canon_token(s)
    if not t:
        return ""
    t = re.sub(r"[\s/\-\.]+", "", t)
    t = re.sub(r"[^a-z0-9\+#]+", "", t)
    return t


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("importance", "value", "name", "label"):
            if k in x and isinstance(x[k], (str, int, float, bool)):
                return str(x[k])
        return str(x)
    if hasattr(x, "value"):
        try:
            v = getattr(x, "value")
            if isinstance(v, (str, int, float, bool)):
                return str(v)
        except Exception:
            pass
    if hasattr(x, "name"):
        try:
            n = getattr(x, "name")
            if isinstance(n, (str, int, float, bool)):
                return str(n)
        except Exception:
            pass
    return str(x)


def _strip_wrapping_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1].strip()
    return s


def _infer_level_from_text_blob(text: str) -> Optional[Tuple[SeniorityLabel, float, str]]:
    """
    Heuristic: detect clear apprentice/junior clues from free text when title is unknown.
    Use cautiously; keep patterns narrow to avoid overfitting.
    """
    if not isinstance(text, str):
        return None
    low = text.lower()
    if any(k in low for k in ["apprentice", "trainee"]):
        return "apprentice", _LEVEL_NUM["apprentice"], "text_apprentice"
    if any(k in low for k in ["intern", "internship"]):
        return "intern", _LEVEL_NUM["intern"], "text_intern"
    if any(k in low for k in ["junior", "graduate", "entry-level", "entry level"]):
        return "junior", _LEVEL_NUM["junior"], "text_junior_like"
    if "student" in low:
        return "intern", _LEVEL_NUM["intern"], "text_student"
    return None


_MUST_DOMAIN_ALLOWLIST = {
    "software development",
    "software engineering principles",
    "software engineering",
    "backend development",
    "backend engineering",
    "platform engineering",
    "infrastructure",
    "infrastructure engineering",
    "architecture",
    "system design",
    "cloud platforms",
    "cloud infrastructure",
    "devops",
    "ci/cd integration",
    "ci/cd pipelines",
    "sre",
    "site reliability engineering",
    "observability",
    "security engineering",
    "application security",
    "data engineering",
    "data platforms",
    "ml engineering",
    "mlops",
    "ai engineering",
    "quality engineering",
    "testing",
    "qa",
}


def _downgrade_must_if_not_allowed(domains: List[DomainRequirement]) -> Tuple[List[DomainRequirement], List[str]]:
    """
    Preserve MUST from JD evidence; only downgrade when evidence is explicitly unverified.
    """
    out: List[DomainRequirement] = []
    downgraded: List[str] = []
    for d in domains or []:
        try:
            name_norm = _canon_token(getattr(d, "name", "") or "").lower()
            is_must = _is_must(getattr(d, "importance", None))
            facet = _norm(getattr(d, "facet", "") or "")
        except Exception:
            out.append(d)
            continue

        if is_must:
            ev_status = str(getattr(d, "evidence_status", "") or "").lower()
            if ev_status == "unverified":
                try:
                    setattr(d, "importance", "should")
                    downgraded.append(f"{getattr(d, 'name', '') or name_norm} (evidence_unverified)")
                except Exception:
                    pass
        out.append(d)
    return out, downgraded


def _importance_norm(x: Any) -> str:
    s_raw = _as_text(x)
    s = _norm(s_raw)
    if not s:
        return ""

    s = _strip_wrapping_quotes(s)

    if "." in s:
        tail = s.split(".")[-1].strip()
        tail = _strip_wrapping_quotes(tail)
        if tail in ("must", "should"):
            return tail

    for sep in [":", "=", "/"]:
        if sep in s:
            tail = s.split(sep)[-1].strip()
            tail = _strip_wrapping_quotes(tail)
            if tail in ("must", "should"):
                return tail

    if s in ("must", "should"):
        return s

    if _RE_MUST.search(s):
        return "must"
    if _RE_SHOULD.search(s):
        return "should"

    if "nice" in s:
        return "should"

    return ""


def _is_must(x: Any) -> bool:
    return _importance_norm(x) == "must"


def _is_should(x: Any) -> bool:
    return _importance_norm(x) == "should"


# ============================================================
# Tokenization
# ============================================================
def _singularize_token(tok: str) -> str:
    tok = _canon_token(tok)
    if not tok:
        return ""
    if not re.fullmatch(r"[a-z]+", tok):
        return tok
    if len(tok) < 3 or len(tok) > 20:
        return tok
    if tok.endswith("ss"):
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and len(tok) > 3:
        return tok[:-1]
    return tok


def _tokenize(s: str) -> List[str]:
    t = _canon_token(s)
    if not t:
        return []
    toks = re.split(r"[ \t/,\|\-]+", t)
    out: List[str] = []
    for x in toks:
        x = _canon_token(x)
        if not x:
            continue
        if x in _STOP_TOKENS:
            continue
        if len(x) < 3 and not any(ch.isdigit() for ch in x):
            continue
        x1 = _singularize_token(x)
        if x1 and x1 not in _STOP_TOKENS and (len(x1) >= 3 or any(ch.isdigit() for ch in x1)):
            out.append(x1)
    return out[:30]


def _canonical_domain_key(name: str, domain_id: Optional[str] = None) -> str:
    """
    Use the model's decision-level label as the identity. Catalog IDs are broad
    normalization metadata and must not collapse distinct JD requirements.
    """
    return _canon_token(name).lower()


def _dedupe_domains_by_canon(domains: List[DomainRequirement]) -> List[DomainRequirement]:
    """
    Merge obvious duplicates by canonical key.
    - keeps the first occurrence
    - upgrades to MUST if any duplicate is MUST
    - merges anchors/examples for broader coverage
    """
    out: List[DomainRequirement] = []
    seen: Dict[str, int] = {}

    for d in domains or []:
        d_name = str(getattr(d, "name", "") or "").strip()
        key = _canonical_domain_key(d_name, getattr(d, "domain_id", None))
        if not key or key not in seen:
            seen[key] = len(out)
            out.append(d)
            continue

        idx = seen[key]
        existing = out[idx]

        # Upgrade importance if the newer one is MUST
        try:
            if _is_must(getattr(d, "importance", None)) and not _is_must(getattr(existing, "importance", None)):
                setattr(existing, "importance", getattr(d, "importance", None))
        except Exception:
            pass

        # Merge anchors/examples where present
        try:
            existing_anchors = getattr(existing, "anchors", None) or []
            new_anchors = getattr(d, "anchors", None) or []
            merged = list(existing_anchors) + [a for a in new_anchors if a not in existing_anchors]
            setattr(existing, "anchors", merged[:50])
        except Exception:
            pass

        try:
            existing_examples = getattr(existing, "examples", None) or []
            new_examples = getattr(d, "examples", None) or []
            merged_examples = list(existing_examples) + [e for e in new_examples if e not in existing_examples]
            setattr(existing, "examples", merged_examples[:20])
        except Exception:
            pass

    return out


# ============================================================
# Candidate evidence text -> merged skills
# ============================================================
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[\-\*\u2022•]+|\d+[\.\)]|\([a-z0-9]+\))\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?;])\s+")


def _extract_claims_from_candidate_text(candidate_text: str, *, max_items: int = 80) -> List[str]:
    """
    Convert a free-form evidence blob into claim-like strings.
    Intended input: "Candidate evidence (selected lines):\n- ...\n- ..."
    """
    if not isinstance(candidate_text, str):
        return []
    txt = candidate_text.strip()
    if not txt:
        return []

    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [ln.strip() for ln in txt.split("\n")]

    claims: List[str] = []
    for ln in raw_lines:
        if not ln:
            continue
        if ln.lower().startswith("candidate evidence"):
            continue
        if ln.lower().startswith("candidate explicit skills (extracted tokens):"):
            continue
        if ln.lower().startswith("heuristic skills tokens"):
            continue

        ln2 = _BULLET_PREFIX_RE.sub("", ln).strip()
        ln2 = ln2.strip("-•* ").strip()

        # Further split long sentences into smaller claims to reduce run-ons
        parts = [ln2]
        if len(ln2) > 140:
            parts = []
            for seg in _SENT_SPLIT_RE.split(ln2):
                seg = seg.strip()
                if seg:
                    parts.append(seg)

        for part in parts:
            if len(part) < 18:
                continue
            if len(part) > 260:
                part = part[:260].rstrip()

            low = part.lower()
            if low in ("experience", "projects", "skills", "education", "summary", "certifications"):
                continue
            if re.fullmatch(r"[a-z0-9\s/&\-]{3,35}:?", low) and (low.endswith(":") or part.isupper()):
                continue

            claims.append(part)
            if len(claims) >= max_items:
                break
        if len(claims) >= max_items:
            break

    seen = set()
    out: List[str] = []
    for c in claims:
        k = _norm(c)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(c)

    return out


def _domain_evidence_tag(d: DomainRequirement) -> str:
    """Best-effort evidence tag extraction across schema variants.

    We have multiple upstream representations:
    - `evidence_level` (e.g., exact/strong/weak/none)
    - `evidence_status` (e.g., verified/unverified)
    - `span_meta` (object or dict) with fields like `method` and/or `found`

    This helper normalizes them into a small set used by scorer filtering.
    """
    # 1) direct fields
    try:
        ev_level = str(getattr(d, "evidence_level", "") or "").strip().lower()
    except Exception:
        ev_level = ""
    if ev_level:
        return ev_level

    try:
        ev_status = str(getattr(d, "evidence_status", "") or "").strip().lower()
    except Exception:
        ev_status = ""
    if ev_status:
        # map status into tag-like values
        if ev_status in ("unverified", "none", "missing"):
            return "none"
        if ev_status in ("verified", "exact", "strong", "weak"):
            return ev_status
        return ev_status

    # 2) span_meta (object or dict)
    span = None
    try:
        span = getattr(d, "span_meta", None)
    except Exception:
        span = None

    method = ""
    found: Optional[bool] = None
    if isinstance(span, dict):
        method = str(span.get("method") or "").strip().lower()
        f = span.get("found")
        if isinstance(f, bool):
            found = f
    elif span is not None:
        try:
            method = str(getattr(span, "method", "") or "").strip().lower()
        except Exception:
            method = ""
        try:
            f = getattr(span, "found", None)
            if isinstance(f, bool):
                found = f
        except Exception:
            pass

    if method:
        return method
    if found is True:
        return "found"
    if found is False:
        return "none"

    # 3) unknown
    return ""


def _is_scored_evidence(tag: str) -> bool:
    """Return True if the evidence tag should be treated as scorable."""
    t = (tag or "").strip().lower()
    if not t:
        return False
    return t not in ("none", "unverified", "missing")

# ============================================================
# Core vs Evidence filter
# ============================================================
_NON_CORE_SUFFIXES = {
    "interaction",
    "integration",
    "deployment",
    "implementation",
    "configuration",
    "maintenance",
    "support",
    "usage",
    "fundamentals",
    "basics",
    "reporting",
    "documentation",
    "monitoring",
    "operations",
    "ops",
}

_CORE_SIGNAL_TOKENS = {
    "architecture",
    "architecting",
    "design",
    "strategy",
    "analysis",
    "modeling",
    "development",
    "engineering",
    "build",
    "building",
    "system",
    "systems",
    "backend",
    "api",
    "security",
    "reliability",
    "scalability",
    "performance",
    "governance",
    "data",
    "pipeline",
    "pipelines",
    "rag",
}

_NON_CORE_TOKENS_NORM = {_singularize_token(x) for x in _NON_CORE_SUFFIXES if x}
_CORE_SIGNAL_TOKENS_NORM = {_singularize_token(x) for x in _CORE_SIGNAL_TOKENS if x}


def _is_core_domain(d: DomainRequirement) -> bool:
    try:
        if _is_must(getattr(d, "importance", None)):
            return True
    except Exception:
        pass

    facet = _norm(getattr(d, "facet", "") or "")
    if facet in ("people", "process"):
        return True

    name = _canon_token(getattr(d, "name", "") or "")
    if not name:
        return False

    eq = _canon_token(getattr(d, "evidence_quote", "") or "")
    src = f"{name} {eq}".strip()

    toks = _tokenize(src)
    tokset = set(toks)

    if tokset & _CORE_SIGNAL_TOKENS_NORM:
        return True

    parts = [p for p in name.split() if p]
    if parts:
        last = _singularize_token(parts[-1])
        if last in _NON_CORE_TOKENS_NORM:
            return False

    if tokset & _NON_CORE_TOKENS_NORM:
        return False

    return True


# ============================================================
# Evidence strength heuristics
# ============================================================
_VERB_HINTS = {
    "build",
    "built",
    "implement",
    "implemented",
    "design",
    "designed",
    "deploy",
    "deployed",
    "migrate",
    "migrated",
    "optimize",
    "optimized",
    "monitor",
    "monitored",
    "debug",
    "debugged",
    "integrate",
    "integrated",
    "create",
    "created",
    "deliver",
    "delivered",
    "own",
    "owned",
    "ship",
    "shipped",
    "improve",
    "improved",
    "reduce",
    "reduced",
    "increase",
    "increased",
    "configure",
    "configured",
    "automate",
    "automated",
    "write",
    "wrote",
    "author",
    "authored",
    "review",
    "reviewed",
    "secure",
    "secured",
}

_ARTIFACT_HINTS = {
    "policy",
    "policies",
    "role",
    "roles",
    "permission",
    "permissions",
    "assume-role",
    "assumerole",
    "sts",
    "terraform",
    "cloudformation",
    "pipeline",
    "api",
    "apis",
    "endpoint",
    "endpoints",
    "lambda",
    "cloudwatch",
    "alarm",
    "alarms",
    "dashboard",
    "dashboards",
    "etl",
    "rag",
    "vector",
    "index",
    "indexes",
    "query",
    "queries",
    "cache",
    "caching",
    "auth",
    "oauth",
    "jwt",
}

_CONTEXT_HINTS = {
    "production",
    "prod",
    "oncall",
    "incident",
    "sla",
    "p0",
    "p1",
    "least privilege",
    "cross account",
    "cross-account",
    "multi account",
    "multi-account",
}

_METRIC_RE = re.compile(
    r"(\b\d{1,4}\s*%|\b\d{1,6}\s*(ms|s|sec|secs|seconds|mins|minutes|hrs|hours)\b|\b(p50|p90|p95|p99)\b|\b\d{1,6}\s*(req/s|rps)\b|\b\d{1,6}\s*(\$|usd|aud)\b)"
)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _is_short_claim(s: str) -> bool:
    t = _canon_token(s)
    if not t:
        return True
    toks = t.split()
    return len(toks) <= 2 and len(t) <= 12


def _is_strong_short_claim(raw: str) -> bool:
    t = _canon_token(raw)
    if not t:
        return False
    tokc = _compact(t)
    if any(ch.isdigit() for ch in t):
        return True
    if tokc in _STRONG_SHORT_TOKENS:
        return True
    toks = t.split()
    if len(toks) == 1 and 2 <= len(toks[0]) <= 5 and toks[0] not in _VENDOR_PREFIXES:
        return True
    return False


def _claim_confidence(raw: str) -> float:
    t = _canon_token(raw)
    if not t:
        return 0.0

    toks = t.split()
    short = _is_short_claim(t)

    if short and _is_strong_short_claim(raw):
        conf = 0.62
    else:
        conf = 0.38 if short else 0.50

    if any(ch in raw for ch in [":", ";", "—", "–", "->", "=>"]):
        conf += 0.08

    tok_set = set(toks)

    if tok_set & _VERB_HINTS:
        conf += 0.25

    if tok_set & _ARTIFACT_HINTS:
        conf += 0.15

    low = _norm(raw)
    if any(h in low for h in _CONTEXT_HINTS):
        conf += 0.10

    if _METRIC_RE.search(raw):
        conf += 0.25

    if len(toks) >= 18 and not (tok_set & _VERB_HINTS) and not (tok_set & _ARTIFACT_HINTS) and not _METRIC_RE.search(raw):
        conf -= 0.15

    if short and len(toks) == 1 and toks[0] in _VENDOR_PREFIXES:
        conf -= 0.12

    low = t.lower()
    for w in _WEAK_PHRASES:
        if w in low:
            conf -= 0.08
            break

    return _clamp01(conf)


@dataclass
class _CandidateEntry:
    text: str
    compact: str
    conf: float


def _make_candidate_index(
    candidate_skills: List[str],
) -> Tuple[set, set, set, str, List[_CandidateEntry], float, float]:
    cand_phrases: set = set()
    cand_compacts: set = set()
    cand_tokens: set = set()
    parts: List[str] = []

    entries: List[_CandidateEntry] = []
    confs: List[float] = []
    short_count = 0
    total_count = 0

    for s in candidate_skills or []:
        t = _canon_token(s)
        if not t:
            continue
        total_count += 1
        if _is_short_claim(t):
            short_count += 1

        conf = _claim_confidence(s)
        confs.append(conf)

        c = _compact(t)
        entries.append(_CandidateEntry(text=t, compact=c, conf=conf))

        cand_phrases.add(t)
        if c:
            cand_compacts.add(c)

        parts.append(t)
        for tok in _tokenize(t):
            cand_tokens.add(tok)
            cand_phrases.add(tok)
            tc = _compact(tok)
            if tc:
                cand_compacts.add(tc)

    cand_text = " " + " ".join(parts) + " "
    avg_conf = (sum(confs) / float(len(confs))) if confs else 0.0
    keyword_ratio = (short_count / float(total_count)) if total_count > 0 else 0.0

    return cand_phrases, cand_compacts, cand_tokens, cand_text, entries, avg_conf, keyword_ratio


# ============================================================
# Seniority
# ============================================================
_LEVEL_NUM: Dict[str, float] = {
    "intern": 0.0,
    "apprentice": 0.5,
    "junior": 1.0,
    "mid": 2.0,
    "senior": 3.0,
    "lead": 4.0,
    "unknown": 2.0,
}

_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*(years?|yrs?)\b", re.IGNORECASE)


def _infer_candidate_level_and_numeric(title_signal: str) -> Tuple[SeniorityLabel, float, str]:
    t = _norm(title_signal)
    if not t or t == "unknown":
        return "unknown", _LEVEL_NUM["unknown"], "unknown"

    if any(band in t for band in ("junior_to_mid", "junior-to-mid", "junior to mid")):
        return "junior", 1.5, "experience_band_junior_to_mid"

    years_match = _YEARS_RE.search(t)
    years_val = int(years_match.group(1)) if years_match else None
    strong_title = any(
        k in t for k in ["senior", "sr ", "sr.", "lead", "principal", "staff", "architect", "manager", "director", "vp"]
    )
    strong_years = years_val is not None and years_val >= 3
    strong_override = strong_title or strong_years

    if "apprentice" in t and not strong_override:
        return "apprentice", _LEVEL_NUM["apprentice"], "title_apprentice_override"

    if any(k in t for k in ["intern", "internship", "trainee", "student"]) and not strong_override:
        return "intern", _LEVEL_NUM["intern"], "title_intern_override"

    if any(k in t for k in ["graduate", "entry level", "entry-level"]) and not strong_override:
        return "junior", _LEVEL_NUM["junior"], "title_graduate_override"

    if any(k in t for k in ["vp", "vice president", "director", "head of", "principal", "staff", "lead", "architect", "manager"]):
        return "lead", _LEVEL_NUM["lead"], "title_lead"

    if "senior" in t or "sr " in t or "sr." in t:
        return "senior", _LEVEL_NUM["senior"], "title_senior"

    # IMPORTANT: label is apprentice; numeric stays 0.5
    if "apprentice" in t:
        return "apprentice", _LEVEL_NUM["apprentice"], "title_apprentice_0_5"

    if any(k in t for k in ["intern", "internship", "trainee", "student"]):
        return "intern", _LEVEL_NUM["intern"], "title_intern"

    if any(k in t for k in ["junior", "graduate", "entry", "assistant"]):
        return "junior", _LEVEL_NUM["junior"], "title_junior"

    if any(k in t for k in ["mid", "intermediate"]):
        return "mid", _LEVEL_NUM["mid"], "title_mid"

    return "unknown", _LEVEL_NUM["unknown"], "unknown"


def _infer_job_numeric(job_level: SeniorityLabel) -> float:
    jl = (job_level or "unknown").strip().lower()
    if jl in _LEVEL_NUM:
        return _LEVEL_NUM[jl]
    return _LEVEL_NUM["unknown"]


def _seniority_gap_numeric(job_level: SeniorityLabel, cand_numeric: float) -> float:
    return float(_infer_job_numeric(job_level) - float(cand_numeric))


def _gap_bucket(gap: float) -> SeniorityBucket:
    """
    Buckets designed to align with _LEVEL_NUM discrete gaps:
    possible gaps are typically {.., 0.5, 1.0, 1.5, 2.0, 2.5, ..}.

    Semantics:
    - none: same level, a half-step transition, or one level above the role
    - small: normal promotion (<=1.0)  e.g., junior->mid, mid->senior
    - medium: big stretch but still sometimes interviewable (<=1.5)
              e.g., apprentice(0.5)->mid(2.0) gap=1.5
    - cliff: >=2.0 is usually unrealistic for typical hiring funnels
            e.g., junior->senior gap=2.0, intern->mid gap=2.0, apprentice->senior gap=2.5
    """
    # One level above a role (for example mid -> junior) is common and does
    # not justify the product label "overqualified". Reserve that label for
    # a clear two-level mismatch such as senior -> junior or lead -> mid.
    if gap < -1.0:
        return "overqualified"
    if gap <= 0.5:
        return "none"
    if gap <= 1.0:
        return "small"
    if gap <= 1.5:
        return "medium"
    return "cliff"


def _cap_for_gap(bucket: SeniorityBucket) -> int:
    """
    Cap is a realism / expectation-management ceiling, not a match score.

    - none: keep < 100 to avoid "too perfect" illusion
    - overqualified: slightly lower than none (retention/scope risk)
    - small: strong but not insane
    - medium: allow "Maybe" ceiling; never "Yes"
    - cliff: hard ceiling
    """
    if bucket == "none":
        return 92
    if bucket == "overqualified":
        return 85
    if bucket == "small":
        return 80
    if bucket == "medium":
        return 60
    return 35  # cliff


def _medium_cap_uplift(
    *,
    must_missing_count: int,
    must_hit_rate: float,
    core_hit_rate: float,
    evidence_conf: float,
    risk_level: RiskLevel,
) -> int:
    """
    Allow strong candidates to break the 60 cap for Medium gaps (up to 70).
    Strict criteria:
    - Low risk (Must missing <= 1, ideally 0)
    - High match rates
    - High evidence confidence (not just keywords)
    """
    # 1. Safety check: high risk or too many missing musts = no uplift
    if risk_level not in ("none", "low"):
        return 0
    if must_missing_count > 0:
        return 0

    # 2. Tier 1: Exceptional match (+10 -> Cap 70)
    # Must be nearly perfect on requirements and have strong evidence.
    if must_hit_rate >= 0.85 and core_hit_rate >= 0.70 and evidence_conf >= 0.70:
        return 10

    # 3. Tier 2: Strong match (+5 -> Cap 65)
    if must_hit_rate >= 0.75 and core_hit_rate >= 0.65 and evidence_conf >= 0.60:
        return 5

    return 0


def _decide_should_apply(
    *,
    bucket: SeniorityBucket,
    final_score: int,
    risk_level: RiskLevel,
    job_level: SeniorityLabel,
) -> ShouldApply:
    """
    Decision policy (cap-aware + more realistic overqualified handling):

    - cliff => No
    - risk high => No
    - overqualified:
        * if target is intern/junior/apprentice => avoid Yes (flight risk)
        * otherwise allow Yes only if very strong + low risk
    - medium: never Yes (even if cap is 60), but allow Maybe when strong enough
    - small/none: standard thresholds
    """
    if bucket == "cliff":
        return "No"
    if risk_level == "high":
        return "No"

    # overqualified: nuanced handling (Gemini-style)
    if bucket == "overqualified":
        jl = (job_level or "unknown").lower()
        if jl in ("intern", "apprentice", "junior"):
            return "Maybe" if (final_score >= 75 and risk_level in ("none", "low")) else "No"
        # for non-junior roles, allow Yes only when very strong
        if final_score >= 80 and risk_level in ("none", "low"):
            return "Yes"
        return "Maybe" if final_score >= 60 and risk_level in ("none", "low", "medium") else "No"

    # thresholds aligned to caps
    if bucket == "none":
        yes_thr, maybe_thr = 75, 45
    elif bucket == "small":
        yes_thr, maybe_thr = 72, 50
    else:  # medium (cap=60) - never Yes
        yes_thr, maybe_thr = 10**9, 50  # <-- 比 45 略严一点，避免“泛 Maybe”

    if final_score >= yes_thr and risk_level in ("none", "low"):
        return "Yes"
    if final_score >= maybe_thr and risk_level in ("none", "low", "medium"):
        return "Maybe"
    return "No"


# ============================================================
# Matching logic (confidence)
# ============================================================
def _domain_anchor_terms(domain: DomainRequirement) -> List[str]:
    anchors: List[str] = []

    name_norm = _canon_token(getattr(domain, "name", "") or "")
    quote_norm = _canon_token(getattr(domain, "evidence_quote", "") or "")
    src_norm = f"{name_norm} {quote_norm}".strip()
    is_version_control = "version control" in name_norm

    # Boost expansions (RAG/Observability)
    boost: List[str] = []
    if ("rag" in src_norm) or ("retrieval augmented" in src_norm) or ("retrieval-augmented" in src_norm):
        boost.extend(
            [
                "embeddings",
                "embedding",
                "vector",
                "vector search",
                "vector database",
                "semantic search",
                "similarity search",
                "top k",
                "top-k",
                "chunk",
                "chunking",
                "index",
                "indexing",
                "rerank",
                "reranker",
            ]
        )

    if ("monitor" in src_norm) or ("observability" in src_norm) or ("logging" in src_norm) or ("logs" in src_norm) or ("metrics" in src_norm):
        boost.extend(
            [
                "cloudwatch",
                "log",
                "logs",
                "metric",
                "metrics",
                "alarm",
                "alarms",
                "dashboard",
                "dashboards",
                "trace",
                "tracing",
                "apm",
                "sli",
                "slo",
            ]
        )

    # GenAI (concept-first, vendor-agnostic)
    if any(k in src_norm for k in ["generative ai", "gen ai", "genai", "llm", "prompt"]):
        boost.extend(
            [
                "llm",
                "llms",
                "prompt",
                "prompts",
                "prompting",
                "prompt engineering",
                "system prompt",
                "inference",
                "inference latency",
                "token",
                "tokens",
                "context window",
                "guardrail",
                "guardrails",
                "hallucination",
                "safety",
                "moderation",
                "tool calling",
                "function calling",
                "evaluation",
                "evals",
                "offline evals",
                "observability",
                "telemetry",
                "instrumentation",
            ]
        )

    # CI/CD duplicate folding (Integration vs Pipelines vs Release Engineering)
    if _canonical_domain_key(name_norm) == "ci_cd":
        boost.extend(
            [
                "ci",
                "cd",
                "cicd",
                "ci/cd",
                "pipeline",
                "pipelines",
                "build pipeline",
                "deploy pipeline",
                "release pipeline",
                "build and deploy",
                "release engineering",
                "continuous integration",
                "continuous delivery",
                "continuous deployment",
                "automation pipeline",
            ]
        )

    # Software engineering principles (proxy signals)
    if any(k in src_norm for k in ["engineering principles", "software engineering", "best practices", "sdlc"]):
        boost.extend(
            [
                "testing",
                "unit testing",
                "integration testing",
                "e2e testing",
                "test coverage",
                "code review",
                "lint",
                "linting",
                "type checking",
                "static analysis",
                "security",
                "owasp",
                "authn",
                "authz",
                "authorization",
                "authentication",
                "input validation",
                "ci",
                "cd",
                "release",
                "release process",
                "rollback",
                "deployment",
                "observability",
                "logging",
                "monitoring",
                "alerts",
                "alerting",
                "oncall",
                "incident",
                "postmortem",
                "slo",
                "sla",
            ]
        )

    anchors.extend(boost)

    # 1) raw anchors
    if getattr(domain, "anchors", None):
        anchors.extend([_canon_token(a) for a in (domain.anchors or []) if str(a).strip()])

    # For Version Control, infer only from examples/aliases (avoid name-based matching)
    if is_version_control:
        anchors.extend(["git", "github", "gitlab", "bitbucket", "source control"])
    else:
        # 2) evidence_quote
        q = _canon_token(getattr(domain, "evidence_quote", "") or "")
        if q:
            anchors.append(q)
            anchors.extend(_tokenize(q))

        # 3) domain.name
        anchors.append(_canon_token(domain.name))
        anchors.extend(_tokenize(domain.name))

    # 4) examples tool names
    for ex in (domain.examples or []):
        anchors.append(_canon_token(ex.name))
        anchors.extend(_tokenize(ex.name))

    max_terms = 80 if boost else 40

    uniq: List[str] = []
    seen = set()
    for a in anchors:
        a = _canon_token(a)
        if not a or len(a) < 2:
            continue
        if a in _STOP_TOKENS:
            continue
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)

    return uniq[:max_terms]


def _strip_vendor_prefix(tool_name: str) -> str:
    t = _canon_token(tool_name)
    if not t:
        return ""

    parts = t.split()
    if len(parts) < 2:
        return ""

    lowered = t.lower()
    for vp in sorted(_VENDOR_PREFIXES, key=lambda x: -len(x)):
        vp_norm = _canon_token(vp)
        if not vp_norm:
            continue
        if lowered.startswith(vp_norm + " "):
            core = lowered[len(vp_norm) + 1 :].strip()
            return core

    if parts[0] in _VENDOR_PREFIXES:
        return " ".join(parts[1:]).strip()

    return ""


def _extract_acronym_variants(tool_name: str) -> List[str]:
    t = _norm(tool_name)
    if not t:
        return []
    m = _PAREN_ACRONYM_RE.search(t)
    if not m:
        return []
    long_name = _canon_token(m.group(1))
    acr = _canon_token(m.group(2))
    out: List[str] = []
    if acr and len(acr) >= 2:
        out.append(acr)
    if long_name and len(long_name) >= 3:
        out.append(long_name)
    return out


def _is_strong_alias_token(tok: str) -> bool:
    tok = _canon_token(tok)
    if not tok:
        return False
    if tok in _STOP_TOKENS:
        return False
    if any(ch.isdigit() for ch in tok):
        return len(tok) >= 2
    if 2 <= len(tok) <= 5 and tok not in _VENDOR_PREFIXES:
        return True
    return len(tok) >= 4


def _clamp_topk(confs: List[float], k: int = 3) -> List[float]:
    confs2 = [max(0.0, min(1.0, float(c))) for c in confs if c and c > 0.0]
    confs2.sort(reverse=True)
    return confs2[: max(1, int(k))]


def _noisy_or(confs: List[float]) -> float:
    if not confs:
        return 0.0
    confs2 = _clamp_topk(confs, 3)
    if not confs2:
        return 0.0
    prod = 1.0
    for c in confs2:
        prod *= (1.0 - c)
    return _clamp01(1.0 - prod)


def _has_exact_hit(
    q: str,
    qc: str,
    *,
    cand_phrases: set,
    cand_compacts: set,
    entries: List[_CandidateEntry],
) -> bool:
    if not q:
        return False
    if q in cand_phrases:
        return True
    if qc and qc in cand_compacts:
        return True

    for e in entries:
        if not e.text:
            continue
        if q == e.text:
            return True
        if qc and e.compact and qc == e.compact:
            return True
    return False


def _match_confidence_string(s: str, cand_phrases: set, cand_compacts: set, entries: List[_CandidateEntry]) -> float:
    q = _canon_token(s)
    if not q:
        return 0.0
    qc = _compact(q)

    if q not in cand_phrases and (not qc or qc not in cand_compacts):
        if len(q) < 3:
            return 0.0

    matched: List[float] = []
    for e in entries:
        if not e.text:
            continue
        if q == e.text:
            matched.append(e.conf)
            continue
        if qc and e.compact and qc == e.compact:
            matched.append(e.conf)
            continue
        if len(q) >= 3 and f" {q} " in f" {e.text} ":
            matched.append(e.conf)

    return _noisy_or(matched)


def _matching_candidate_evidence(anchor_terms: List[str], entries: List[_CandidateEntry], limit: int = 3) -> List[str]:
    """Return the resume claims that actually supported a domain match."""
    out: List[str] = []
    seen = set()
    anchors = [(_canon_token(a), _compact(_canon_token(a))) for a in (anchor_terms or [])]
    anchors = [(a, c) for a, c in anchors if a]
    for entry in entries:
        for anchor, compact in anchors:
            matched = (
                anchor == entry.text
                or (compact and compact == entry.compact)
                or (len(anchor) >= 3 and f" {anchor} " in f" {entry.text} ")
            )
            if not matched:
                continue
            key = entry.text.lower()
            if key not in seen:
                seen.add(key)
                out.append(entry.text[:280])
            break
        if len(out) >= max(1, int(limit)):
            break
    return out


def _domain_match_confidence(
    domain: DomainRequirement,
    *,
    anchor_terms: List[str],
    cand_phrases: set,
    cand_compacts: set,
    cand_tokens: set,
    entries: List[_CandidateEntry],
    strong_threshold: float,
    weak_threshold: float,
    exact_floor: float,
) -> Tuple[float, str, bool]:
    confs: List[float] = []
    exact_hit = False

    for a in anchor_terms:
        q = _canon_token(a)
        if not q:
            continue
        qc = _compact(q)

        if not exact_hit and _has_exact_hit(q, qc, cand_phrases=cand_phrases, cand_compacts=cand_compacts, entries=entries):
            exact_hit = True

        c = _match_confidence_string(q, cand_phrases, cand_compacts, entries)
        if c > 0.0:
            confs.append(c)

    best = _noisy_or(confs)

    src = f"{getattr(domain, 'name', '')} {getattr(domain, 'evidence_quote', '')}"
    toks = _tokenize(src)
    tok_hits = [t for t in toks if t in cand_tokens]
    uniq_hits = list(dict.fromkeys(tok_hits))
    if len(uniq_hits) >= 2:
        best = max(best, 0.35)

    if exact_hit:
        best = max(best, float(exact_floor))

    best = _clamp01(best)

    if best >= strong_threshold:
        return best, "hit", exact_hit
    if best >= weak_threshold:
        return best, "soft_hit", exact_hit
    return best, "missing", exact_hit


def _tool_match_confidence(
    tool: ToolEvidence,
    *,
    cand_phrases: set,
    cand_compacts: set,
    entries: List[_CandidateEntry],
) -> float:
    t = _canon_token(canon_tool(tool.name))
    if not t:
        return 0.0

    parts = [part for part in t.split() if part]
    if len(parts) >= 2:
        # Multi-word named tools/standards require the complete phrase.
        # Shared suffixes such as "DevOps" cannot prove "Azure DevOps".
        candidate_blob = " ".join(entry.text for entry in entries if entry.text)
        return 1.0 if _tool_text_hit(t, candidate_blob) else 0.0

    confs: List[float] = []
    confs.append(_match_confidence_string(t, cand_phrases, cand_compacts, entries))

    for v in _extract_acronym_variants(tool.name):
        confs.append(_match_confidence_string(v, cand_phrases, cand_compacts, entries))

    core = _strip_vendor_prefix(t)
    if core:
        confs.append(_match_confidence_string(core, cand_phrases, cand_compacts, entries))
        core_parts = [p for p in _canon_token(core).split() if p]
        if core_parts:
            last = core_parts[-1]
            if _is_strong_alias_token(last):
                confs.append(_match_confidence_string(last, cand_phrases, cand_compacts, entries))

    return _clamp01(_noisy_or(confs))


def _expand_tool_names(name: str) -> List[str]:
    raw = (name or "").strip()
    if not raw:
        return []
    cleaned = re.sub(r"\([^)]*\)", "", raw).strip()
    parts = [p.strip() for p in _TOOL_SPLIT_RE.split(cleaned) if p and p.strip()]
    if not parts:
        parts = [cleaned]
    out: List[str] = []
    seen = set()
    for p in parts:
        c = canon_tool(p) or p
        key = _norm(c)
        if not key or key in seen:
            continue
        if key in _TOOL_IGNORE:
            continue
        seen.add(key)
        out.append(c)
    return out


def _tool_text_hit(tool_name: str, candidate_text: str) -> bool:
    t = (canon_tool(tool_name) or "").strip()
    if not t:
        return False
    low = (candidate_text or "").lower()
    if not low:
        return False

    t_low = t.lower()
    if t_low == "gpt-4":
        return bool(re.search(r"\bgpt[- ]?4\b", low) or "openai gpt-4" in low)
    if t_low == "gemini":
        return bool(re.search(r"\bgemini\b", low))
    if t_low == "aws api gateway":
        return "api gateway" in low

    if " " in t_low:
        return t_low in low
    return bool(re.search(rf"\b{re.escape(t_low)}\b", low))


# ============================================================
# Coverage-based penalty
# ============================================================
def _coverage_penalty(rate: float, *, ok_threshold: float = 0.80) -> Tuple[float, str]:
    r = float(rate)
    if r >= ok_threshold:
        return 1.0, "ok"
    if r >= 0.5:
        return 0.85, "partial"
    if r >= 0.3:
        return 0.65, "low"
    return 0.4, "critical"


def _core_penalty(
    *,
    must_total: int,
    must_hit: int,
    core_total: int,
    core_hit_effective: float,
) -> Tuple[float, str, Dict[str, Any]]:
    must_rate = (must_hit / float(must_total)) if must_total > 0 else 1.0
    core_rate = (core_hit_effective / float(core_total)) if core_total > 0 else 1.0

    if must_total < 2:
        effective_rate = core_rate
        rule = "core_only_tiny_must"
        penalty, bucket = _coverage_penalty(effective_rate, ok_threshold=0.75)
        return (
            penalty,
            f"core_{bucket}",
            {
                "must_total": must_total,
                "must_hit": must_hit,
                "must_rate": must_rate,
                "core_total": core_total,
                "core_hit_effective": core_hit_effective,
                "core_rate": core_rate,
                "effective_rate": effective_rate,
                "rule": rule,
                "ok_threshold": 0.75,
            },
        )

    effective_rate = min(must_rate, core_rate)
    rule = "min(must,core)"
    penalty, bucket = _coverage_penalty(effective_rate, ok_threshold=0.80)
    return (
        penalty,
        f"must_core_{bucket}",
        {
            "must_total": must_total,
            "must_hit": must_hit,
            "must_rate": must_rate,
            "core_total": core_total,
            "core_hit_effective": core_hit_effective,
            "core_rate": core_rate,
            "effective_rate": effective_rate,
            "rule": rule,
            "ok_threshold": 0.80,
        },
    )


def _keyword_stuffing_penalty(keyword_ratio: float) -> Tuple[float, str]:
    """
    v3.5.2 softened:
    - <=0.80: no penalty
    - 1.00: floor at 0.92
    """
    r = float(keyword_ratio)
    if r <= 0.80:
        return 1.0, "no_stuffing"
    span = 0.20  # 0.80 -> 1.00
    p = 1.0 - (min(1.0, r) - 0.80) * (0.08 / span)
    p = max(0.92, min(1.0, p))
    return p, f"keyword_ratio_{r:.2f}"


# ============================================================
# Output structures
# ============================================================
@dataclass
class GateRisk:
    risk_level: RiskLevel
    must_missing_names: List[str]
    notes: List[str]


@dataclass
class ScoreBreakdown:
    per_domain: Dict[str, str]
    per_tool_must: Dict[str, str]
    per_tool_should: Dict[str, str]
    constraints: Dict[str, Any]


@dataclass
class ScoreResultV3:
    should_apply: ShouldApply
    final_score: int
    distance_score: int
    cap: int
    seniority_gap: str
    summary: str

    required_domains: List[str]
    display_required_domains: List[str]
    missing_domains_must: List[str]
    missing_domains_all: List[str]

    required_tool_must: List[str]
    missing_tool_must: List[str]

    tool_should_examples: List[str]
    missing_tool_should_examples: List[str]

    gate_risk: GateRisk
    debug_breakdown: ScoreBreakdown


# ============================================================
# Tool redundancy heuristics (NEW)
# ============================================================
_TOOL_REDUNDANCY: Dict[str, set] = {
    # domain -> tools that are "implied" and should not be repeated as "hard tools"
    "version control": {"git", "github", "gitlab", "bitbucket"},
    "backend development": {"fastapi", "flask", "django", "express", "spring"},
    "api development": {"rest apis", "rest", "graphql"},
}


def _is_tool_redundant_with_domain(domain_name: str, tool_name: str) -> bool:
    dn = _canon_token(domain_name)
    tn = _canon_token(tool_name)
    if not dn or not tn:
        return False
    for dom_key, tools in _TOOL_REDUNDANCY.items():
        if dom_key in dn:
            return tn in tools or _compact(tn) in {_compact(x) for x in tools}
    return False


# ============================================================
# Scoring
# ============================================================
def score_ir_v3(
    job_ir: AnalyzeIRv3,
    *,
    candidate_skills: Optional[List[str]] = None,
    candidate_text: Optional[str] = None,
) -> ScoreResultV3:
    """
    candidate_skills:
      - explicit list of skills/claims (highest priority if provided)
    candidate_text:
      - free-form evidence text (bullets / achievements). We extract claims and merge them
        into candidate_skills to improve evidence confidence and reduce keyword stuffing penalties.
    """
    candidate_sources = {
        "explicit": bool(isinstance(candidate_skills, list) and candidate_skills),
        "job_ir": bool(_get_field(job_ir, "candidate_skills", None)),
        "raw_llm": False,
    }
    raw_llm = _get_raw_llm(job_ir) or {}
    raw_cand = raw_llm.get("candidate") if isinstance(raw_llm.get("candidate"), dict) else {}
    raw_skills = raw_cand.get("skills")
    if isinstance(raw_skills, list) and raw_skills:
        candidate_sources["raw_llm"] = True

    base_skills = _get_candidate_skills(job_ir, candidate_skills)
    structured_claims = _get_candidate_evidence_claims(job_ir)
    claims_by_evidence_id = {
        claim.evidence_id: claim
        for claim in structured_claims
        if str(claim.evidence_id or "").strip()
    }

    extracted_claims: List[str] = []
    if isinstance(candidate_text, str) and candidate_text.strip():
        extracted_claims = _extract_claims_from_candidate_text(candidate_text)

    # Named skills remain useful for exact keyword matching, but they no
    # longer prove broad requirements. Domain matching uses grounded claims.
    tool_inputs: List[str] = []
    seen = set()
    for s in (
        (base_skills or [])
        + [claim.resume_quote for claim in structured_claims]
        + (extracted_claims or [])
    ):
        st = str(s).strip()
        if not st:
            continue
        k = _norm(st)
        if not k or k in seen:
            continue
        seen.add(k)
        tool_inputs.append(st)

    tool_inputs.sort(key=lambda x: x.lower())
    cand_skills = tool_inputs
    candidate_text_blob = " ".join(tool_inputs)
    if isinstance(candidate_text, str) and candidate_text.strip():
        candidate_text_blob = (candidate_text_blob + " " + candidate_text).strip()
    domains_raw = _dedupe_domains_by_canon(_get_domains(job_ir))
    if not domains_raw:
        raw = _get_raw_llm(job_ir) or {}
        if raw:
            fallback_domains = _extract_domains_from_raw(raw)
            if fallback_domains:
                domains_raw = _dedupe_domains_by_canon(fallback_domains)
    required_domains_all = [
        str(getattr(d, "name", "") or "")
        for d in (domains_raw or [])
        if str(getattr(d, "name", "") or "").strip()
        and getattr(d, "domain_id", "") != "other_info"
    ]
    domains, must_downgraded = _downgrade_must_if_not_allowed(domains_raw)
    semantic_match_available = any(
        str(getattr(domain, "match_status", "unknown") or "unknown")
        in {"matched", "partial", "missing"}
        for domain in domains
    )
    semantic_invalid_requirements = 0

    job_level: SeniorityLabel = _get_job_level(job_ir)
    cand_title_signal = _get_candidate_title_signal(job_ir)

    cand_label, cand_numeric, cand_reason = _infer_candidate_level_and_numeric(cand_title_signal)
    cand_level: SeniorityLabel = cand_label

    # Secondary inference from free text (only when unknown and not set by title)
    if cand_level == "unknown" and candidate_text:
        inferred = _infer_level_from_text_blob(candidate_text)
        if inferred:
            cand_level, cand_numeric, cand_reason = inferred

    # Unknown must remain unknown. Use the job level only as a neutral numeric
    # comparison so we do not invent a candidate level or apply a false gap.
    if cand_level == "unknown":
        cand_numeric = _infer_job_numeric(job_level)
        cand_reason = "unknown_neutral_no_seniority_evidence"

    domain_inputs = (
        [claim.resume_quote for claim in structured_claims]
        if structured_claims
        else list(tool_inputs)
    )
    (
        cand_phrases,
        cand_compacts,
        cand_tokens,
        _cand_text,
        cand_entries,
        cand_avg_conf,
        cand_keyword_ratio,
    ) = _make_candidate_index(domain_inputs)
    (
        tool_cand_phrases,
        tool_cand_compacts,
        _tool_cand_tokens,
        _tool_cand_text,
        tool_cand_entries,
        _tool_cand_avg_conf,
        _tool_cand_keyword_ratio,
    ) = _make_candidate_index(tool_inputs)
    explicit_jd_tools = _get_explicit_jd_tools(job_ir)
    explicit_jd_keyword_meta = _get_explicit_jd_keyword_meta(job_ir)

    gap = _seniority_gap_numeric(job_level, cand_numeric)
    bucket = _gap_bucket(gap)
    seniority_cap_base = _cap_for_gap(bucket)
    seniority_cap_uplift = 0
    cap = seniority_cap_base

    MUST_DOMAIN_W = 10
    SHOULD_DOMAIN_W = 5

    # Tools are bonus-only (no hard penalty); keep 1 weight for the bonus pool
    SHOULD_TOOL_BONUS = 1
    SHOULD_TOOL_BONUS_CAP = 8

    SENIORITY_MAX = 10

    DOMAIN_STRONG_THRESHOLD = 0.60
    DOMAIN_WEAK_THRESHOLD = 0.40
    TOOL_HIT_THRESHOLD = 0.50

    DOMAIN_EXACT_FLOOR = 0.45
    SOFT_CORE_WEIGHT = 0.50

    per_domain: Dict[str, str] = {}
    per_tool_must: Dict[str, str] = {}  # kept for debug/contract compatibility
    per_tool_should: Dict[str, str] = {}

    required_domains: List[str] = []
    display_required_domains: List[str] = []
    missing_domains_all: List[str] = []
    missing_domains_must: List[str] = []

    # "Hard tools" are informational only (appear in required_skills),
    # but do NOT affect scoring/penalties.
    required_tool_must: List[str] = []
    missing_tool_must: List[str] = []  # always empty under bonus-only policy

    tool_should_examples: List[str] = []
    missing_tool_should_examples: List[str] = []

    anchors_used: Dict[str, List[str]] = {}
    domain_anchor_hits: Dict[str, List[str]] = {}
    domain_match_confidence: Dict[str, float] = {}
    domain_resume_evidence: Dict[str, List[str]] = {}
    domain_evidence_ids: Dict[str, List[str]] = {}
    domain_match_reasons: Dict[str, str] = {}

    raw_score = 0
    raw_max = 0
    domain_raw_score = 0
    domain_raw_max = 0
    tool_bonus_raw = 0

    must_total_raw = 0
    must_hit_raw = 0
    must_domain_names_raw: List[str] = []
    must_domain_names_hit_raw: List[str] = []
    must_domain_names_missing_raw: List[str] = []

    def _domain_is_other_info(x: DomainRequirement) -> bool:
        return getattr(x, "domain_id", "") == "other_info"

    def _domain_needs_confirmation(x: DomainRequirement) -> bool:
        return str(getattr(x, "requirement_type", "") or "") == "work_condition"

    def _domain_tag(x: DomainRequirement) -> str:
        return _domain_evidence_tag(x)

    display_domain_evidence = {
        str(getattr(d, "name", "") or ""): _domain_tag(d)
        for d in (domains or [])
        if (not _domain_is_other_info(d))
    }

    display_required_domains = [
        str(getattr(d, "name", "") or "")
        for d in (domains or [])
        if (not _domain_is_other_info(d))
    ]

    scored_core_domains = [
        d
        for d in (domains or [])
        if (not _domain_is_other_info(d))
        and (not _domain_needs_confirmation(d))
        and _is_scored_evidence(_domain_tag(d))
    ]

    other_domains = [
        str(getattr(d, "name", "") or "")
        for d in (domains or [])
        if _domain_is_other_info(d)
    ]

    unverified_domains = [
        str(getattr(d, "name", "") or "")
        for d in (domains or [])
        if (not _domain_is_other_info(d))
        and (not _is_scored_evidence(_domain_tag(d)))
    ]
    raw_domains_total = len(domains_raw or [])
    core_total_all = len(scored_core_domains)
    signal_insufficient = (
        core_total_all == 0
        or str(_get_field(job_ir, "analysis_status", "success") or "success") == "degraded"
    )
    core_domains_filtered = [d for d in scored_core_domains if _is_core_domain(d)]
    core_domains_demoted = [d for d in scored_core_domains if d not in core_domains_filtered]

    if not core_domains_filtered and scored_core_domains:
        core_domains_filtered = list(scored_core_domains)
        core_domains_demoted = []

    core_total = len(core_domains_filtered)

    core_hit_strong = 0
    core_hit_soft = 0
    core_exact_hit_count = 0
    # ---- Domain scoring
    for d in scored_core_domains:
        d_name = getattr(d, "name", "") or ""
        required_domains.append(d_name)

        is_must = _is_must(getattr(d, "importance", None))
        is_core_for_penalty = d in core_domains_filtered

        anchors = _domain_anchor_terms(d)
        anchors_used[d_name or "unknown"] = anchors[:60]

        evidence_decision: Optional[EvidenceMatchDecision] = None
        semantic_status = str(getattr(d, "match_status", "unknown") or "unknown")
        if semantic_match_available:
            evidence_ids = [
                str(value).strip()
                for value in (getattr(d, "resume_evidence_ids", None) or [])
                if str(value).strip() in claims_by_evidence_id
            ]
            if semantic_status in {"matched", "partial"} and not evidence_ids:
                semantic_invalid_requirements += 1
                semantic_status = "missing"
            elif semantic_status not in {"matched", "partial", "missing"}:
                semantic_invalid_requirements += 1
                semantic_status = "missing"
            selected_claims = [claims_by_evidence_id[value] for value in evidence_ids]
            evidence_decision = EvidenceMatchDecision(
                status=cast(Any, semantic_status),
                confidence={"matched": 0.92, "partial": 0.62, "missing": 0.0}[semantic_status],
                evidence_ids=evidence_ids,
                evidence_quotes=[claim.resume_quote[:280] for claim in selected_claims],
                reason=(
                    "semantic_model:"
                    + str(getattr(d, "match_reason", "") or "candidate_profile_comparison")[:400]
                ),
            )
            dom_conf = float(evidence_decision.confidence)
            _dom_status = {
                "matched": "hit",
                "partial": "soft_hit",
                "missing": "missing",
            }[evidence_decision.status]
            dom_exact = evidence_decision.status == "matched"
            domain_resume_evidence[d_name] = list(evidence_decision.evidence_quotes)
            domain_evidence_ids[d_name] = list(evidence_decision.evidence_ids)
            domain_match_reasons[d_name] = evidence_decision.reason
        elif structured_claims:
            evidence_decision = match_requirement_to_evidence(d, structured_claims)
            dom_conf = float(evidence_decision.confidence)
            _dom_status = {
                "matched": "hit",
                "partial": "soft_hit",
                "missing": "missing",
            }[evidence_decision.status]
            dom_exact = evidence_decision.reason == "exact_grounded_phrase"
            domain_resume_evidence[d_name] = list(evidence_decision.evidence_quotes)
            domain_evidence_ids[d_name] = list(evidence_decision.evidence_ids)
            domain_match_reasons[d_name] = evidence_decision.reason
        else:
            dom_conf, _dom_status, dom_exact = _domain_match_confidence(
                d,
                anchor_terms=anchors,
                cand_phrases=cand_phrases,
                cand_compacts=cand_compacts,
                cand_tokens=cand_tokens,
                entries=cand_entries,
                strong_threshold=DOMAIN_STRONG_THRESHOLD,
                weak_threshold=DOMAIN_WEAK_THRESHOLD,
                exact_floor=DOMAIN_EXACT_FLOOR,
            )
            domain_resume_evidence[d_name] = _matching_candidate_evidence(anchors, cand_entries)
            domain_match_reasons[d_name] = "legacy_flat_text_match"
        domain_match_confidence[d_name] = round(float(dom_conf), 3)

        # Capture anchors that actually matched (>0 confidence) for debug transparency
        matched_anchors = []
        for a in anchors:
            q = _canon_token(a)
            if not q:
                continue
            qc = _compact(q)
            c = _match_confidence_string(q, cand_phrases, cand_compacts, cand_entries)
            if c > 0.0:
                matched_anchors.append(q)
        if matched_anchors:
            domain_anchor_hits[d_name or "unknown"] = matched_anchors[:12]

        if evidence_decision is not None:
            strong_hit = evidence_decision.status == "matched"
            weak_hit = evidence_decision.status == "partial"
        else:
            strong_hit = dom_conf >= DOMAIN_STRONG_THRESHOLD
            weak_hit = (dom_conf >= DOMAIN_WEAK_THRESHOLD) and not strong_hit

        if is_core_for_penalty:
            if dom_exact:
                core_exact_hit_count += 1
            if strong_hit:
                core_hit_strong += 1
            elif weak_hit:
                core_hit_soft += 1

        if is_must:
            must_total_raw += 1
            must_domain_names_raw.append(d_name)

            raw_max += MUST_DOMAIN_W
            domain_raw_max += MUST_DOMAIN_W

            if strong_hit:
                raw_score += int(round(MUST_DOMAIN_W * dom_conf))
                domain_raw_score += int(round(MUST_DOMAIN_W * dom_conf))
                must_hit_raw += 1
                must_domain_names_hit_raw.append(d_name)
                per_domain[d_name] = "hit"
            elif weak_hit:
                per_domain[d_name] = "soft_hit"
                must_hit_raw += 1
                must_domain_names_hit_raw.append(d_name)

                partial = 0.50 * (dom_conf / DOMAIN_STRONG_THRESHOLD)
                partial = max(0.25, min(0.50, partial))
                raw_score += int(round(MUST_DOMAIN_W * partial))
                domain_raw_score += int(round(MUST_DOMAIN_W * partial))
            else:
                per_domain[d_name] = "missing"
                missing_domains_all.append(d_name)
                missing_domains_must.append(d_name)
                must_domain_names_missing_raw.append(d_name)
        else:
            raw_max += SHOULD_DOMAIN_W
            domain_raw_max += SHOULD_DOMAIN_W

            if dom_conf >= DOMAIN_WEAK_THRESHOLD:
                raw_score += int(round(SHOULD_DOMAIN_W * dom_conf))
                domain_raw_score += int(round(SHOULD_DOMAIN_W * dom_conf))
                per_domain[d_name] = "hit" if strong_hit else "soft_hit"
            elif dom_conf > 0.0:
                raw_score += int(round(SHOULD_DOMAIN_W * min(dom_conf, 0.20)))
                domain_raw_score += int(round(SHOULD_DOMAIN_W * min(dom_conf, 0.20)))
                per_domain[d_name] = "weak_hit"
            else:
                per_domain[d_name] = "missing"
                missing_domains_all.append(d_name)

    # ---- Tool scoring (bonus-only)
    should_tool_bonus_used = 0
    tool_should_seen: set = set()
    tool_should_total_unique = 0

    for d in scored_core_domains:
        d_name = getattr(d, "name", "") or ""
        d_is_must = _is_must(getattr(d, "importance", None))

        for ex in (d.examples or []):
            ex_name_raw = getattr(ex, "name", "") or ""
            if not ex_name_raw.strip():
                continue
            if not _is_explicit_named_example(ex_name_raw, getattr(ex, "evidence_quote", None)):
                continue
            ex_names = _expand_tool_names(ex_name_raw)
            if not ex_names:
                continue

            for ex_name in ex_names:
                ex_tool = ToolEvidence(
                    name=ex_name,
                    importance=getattr(ex, "importance", None),
                    evidence_quote=getattr(ex, "evidence_quote", None),
                )

                tool_conf = _tool_match_confidence(
                    ex_tool,
                    cand_phrases=tool_cand_phrases,
                    cand_compacts=tool_cand_compacts,
                    entries=tool_cand_entries,
                )

                tool_strong = tool_conf >= TOOL_HIT_THRESHOLD

                # (A) Record "hard tool" list for public required_skills only
                if (
                    d_is_must
                    and _is_must(getattr(ex, "importance", None))
                    and not _is_tool_redundant_with_domain(d_name, ex_name)
                ):
                    required_tool_must.append(ex_name)

                # (B) Everything is treated as "should tool" for scoring/bonus
                key = _norm(ex_name)
                if key not in tool_should_seen:
                    tool_should_seen.add(key)
                    tool_should_total_unique += 1
                    tool_should_examples.append(ex_name)

                if tool_conf <= 0.0 and candidate_text_blob:
                    if _tool_text_hit(ex_name, candidate_text_blob):
                        tool_conf = TOOL_HIT_THRESHOLD
                        tool_strong = True

                if tool_conf > 0.0:
                    per_tool_should[ex_name] = "hit" if tool_strong else "soft_hit"
                    if should_tool_bonus_used < SHOULD_TOOL_BONUS_CAP:
                        bonus_add = int(round(SHOULD_TOOL_BONUS * tool_conf))
                        raw_score += bonus_add
                        tool_bonus_raw += bonus_add
                        should_tool_bonus_used += 1
                else:
                    per_tool_should[ex_name] = "missing"
                    missing_tool_should_examples.append(ex_name)

    # Deterministically extracted JD tools are evaluated once at the top level.
    # They must not be copied into every domain, which previously produced a
    # repetitive and misleading requirement matrix.
    for tool_name in explicit_jd_tools:
        key = _norm(tool_name)
        if not key or key in tool_should_seen:
            continue
        tool_should_seen.add(key)
        tool_should_total_unique += 1
        tool_should_examples.append(tool_name)
        tool_conf = _tool_match_confidence(
            ToolEvidence(name=tool_name, importance="should", evidence_quote=None),
            cand_phrases=tool_cand_phrases,
            cand_compacts=tool_cand_compacts,
            entries=tool_cand_entries,
        )
        if tool_conf <= 0.0 and candidate_text_blob and _tool_text_hit(tool_name, candidate_text_blob):
            tool_conf = TOOL_HIT_THRESHOLD
        if tool_conf > 0.0:
            per_tool_should[tool_name] = "hit" if tool_conf >= TOOL_HIT_THRESHOLD else "soft_hit"
            if should_tool_bonus_used < SHOULD_TOOL_BONUS_CAP:
                bonus_add = int(round(SHOULD_TOOL_BONUS * tool_conf))
                raw_score += bonus_add
                tool_bonus_raw += bonus_add
                should_tool_bonus_used += 1
        else:
            per_tool_should[tool_name] = "missing"
            missing_tool_should_examples.append(tool_name)

    # Bonus max is based on unique tool mentions (capped)
    bonus_cap_effective = min(int(SHOULD_TOOL_BONUS_CAP), int(tool_should_total_unique))
    bonus_max = int(bonus_cap_effective) * int(SHOULD_TOOL_BONUS)
    raw_max += bonus_max

    # Seniority component
    raw_max += SENIORITY_MAX
    if bucket in ("none", "overqualified"):
        raw_score += SENIORITY_MAX
    elif bucket == "small":
        raw_score += int(SENIORITY_MAX * 0.6)
    elif bucket == "medium":
        raw_score += int(SENIORITY_MAX * 0.3)

    if raw_max > 0 and raw_score > raw_max:
        raw_score = raw_max
    if raw_score < 0:
        raw_score = 0

    raw_distance = int(round(100.0 * (raw_score / raw_max))) if raw_max > 0 else 0
    raw_distance = max(0, min(100, raw_distance))
    domain_raw_distance = int(round(100.0 * (domain_raw_score / domain_raw_max))) if domain_raw_max > 0 else 0
    domain_raw_distance = max(0, min(100, domain_raw_distance))

    def _dedupe(seq: List[str]) -> List[str]:
        seen2 = set()
        out2: List[str] = []
        for x in seq or []:
            k2 = _norm(x)
            if not k2 or k2 in seen2:
                continue
            seen2.add(k2)
            out2.append(x)
        return out2

    must_domain_names_all = _dedupe(must_domain_names_raw)
    must_domain_names_hit = _dedupe(must_domain_names_hit_raw)
    must_domain_names_missing = _dedupe(must_domain_names_missing_raw)

    must_total = len(must_domain_names_all)
    must_hit = len(must_domain_names_hit)

    core_hit_effective = float(core_hit_strong) + float(SOFT_CORE_WEIGHT) * float(core_hit_soft)

    penalty, penalty_bucket, coverage_debug = _core_penalty(
        must_total=must_total,
        must_hit=must_hit,
        core_total=core_total,
        core_hit_effective=core_hit_effective,
    )

    stuffing_penalty, stuffing_bucket = _keyword_stuffing_penalty(cand_keyword_ratio)
    if semantic_match_available:
        # Semantic statuses already encode direct, transferable, and missing
        # evidence. Applying the old lexical coverage penalty again would
        # double-count the same gaps.
        penalty = 1.0
        penalty_bucket = "semantic_statuses_direct"
        stuffing_penalty = 1.0
        stuffing_bucket = "semantic_profile"
    penalty2 = penalty * stuffing_penalty
    penalty2 = max(0.0, min(1.0, penalty2))
    penalty_bucket2 = f"{penalty_bucket}+{stuffing_bucket}"

    distance = int(round(raw_distance * penalty2))
    distance = max(0, min(100, distance))
    coverage_score = int(round(domain_raw_distance * penalty2))
    coverage_score = max(0, min(100, coverage_score))

    # Calculate Risk Level FIRST (needed for Uplift)
    must_missing_names = _dedupe(missing_domains_must)
    must_missing_total = len(must_missing_names)

    if must_missing_total == 0:
        risk_level: RiskLevel = "none"
    elif must_missing_total == 1:
        risk_level = "low"
    elif must_missing_total == 2:
        risk_level = "medium"
    else:
        risk_level = "high"

    if required_domains_all and not required_domains:
        required_domains = list(required_domains_all)

    if signal_insufficient:
        missing_domains_all = []
        missing_domains_must = []

    # Uplift logic for Medium Seniority
    uplift = 0
    if bucket == "medium":
        # Calculate rates for uplift
        must_rate_uplift = (must_hit / float(must_total)) if must_total > 0 else 1.0
        core_rate_uplift = (core_hit_effective / float(core_total)) if core_total > 0 else 1.0

        uplift = _medium_cap_uplift(
            must_missing_count=len(must_missing_names),
            must_hit_rate=must_rate_uplift,
            core_hit_rate=core_rate_uplift,
            evidence_conf=cand_avg_conf,
            risk_level=risk_level,
        )
        if uplift > 0:
            cap = min(70, cap + uplift)
            seniority_cap_uplift = uplift

    final_score = min(distance, cap)
    if signal_insufficient:
        risk_level = "medium"
        must_missing_names = []
        distance = 0
        raw_distance = 0
        penalty2 = 0.0
        penalty_bucket2 = "insufficient_signal"
        final_score = 0
        coverage_score = 0
        domain_raw_distance = 0

    notes: List[str] = []
    if bucket == "cliff":
        notes.append("Seniority gap > 2.5 levels (cap applied).")
    if bucket == "overqualified":
        notes.append("Candidate appears overqualified by seniority for this role (non-blocking).")
    if uplift > 0:
        notes.append(f"Cap Uplift Triggered: +{uplift} (Strong evidence override for medium gap).")
    if signal_insufficient:
        notes.append("Insufficient signal: no domain requirements available for scoring.")

    notes.append(
        f"Coverage: must {must_hit}/{must_total}, core(filtered) strong {core_hit_strong}/{core_total}, soft {core_hit_soft}/{core_total} "
        f"(soft_weight {SOFT_CORE_WEIGHT:.2f}) => core_effective {core_hit_effective:.2f}/{core_total} "
        f"=> penalty {penalty:.2f} ({penalty_bucket})."
    )
    notes.append(f"Core denominator: all_domains {raw_domains_total} -> filtered_core {core_total}.")
    notes.append(
        f"Evidence: avg_conf {cand_avg_conf:.2f}, keyword_ratio {cand_keyword_ratio:.2f} => stuffing_penalty {stuffing_penalty:.2f}."
    )
    notes.append(f"Tool-should bonus cap: {SHOULD_TOOL_BONUS_CAP} (used {should_tool_bonus_used}).")
    notes.append(f"Tool-should bonus max: {bonus_max} (unique tools: {tool_should_total_unique}).")
    notes.append(
        f"Hit thresholds: domain strong {DOMAIN_STRONG_THRESHOLD:.2f}, domain weak {DOMAIN_WEAK_THRESHOLD:.2f}, tool {TOOL_HIT_THRESHOLD:.2f}"
    )
    notes.append(f"Exact-hit floor: {DOMAIN_EXACT_FLOOR:.2f} (filtered-core exact matches: {core_exact_hit_count}).")

    if extracted_claims:
        notes.append(f"Candidate_text merged: {len(extracted_claims)} extracted evidence claims (merged into candidate index).")

    # Decide should_apply (FIXED: no early return of str)
    should_apply: ShouldApply = _decide_should_apply(
        bucket=bucket,
        final_score=int(final_score),
        risk_level=risk_level,
        job_level=job_level,
    )
    model_recommendation = _get_field(job_ir, "application_recommendation", None)
    model_should_apply = str(_get_field(model_recommendation, "should_apply", "") or "")
    model_rationale = str(_get_field(model_recommendation, "rationale", "") or "").strip()
    semantic_recommendation_valid = bool(
        model_should_apply in {"Yes", "Maybe", "No"} and model_rationale
    )
    if semantic_match_available and semantic_recommendation_valid:
        should_apply = cast(ShouldApply, model_should_apply)
    # Recommendation and score answer different product questions. The model
    # may say a stretch role is worth applying for, but it must never rewrite
    # requirement coverage or break the deterministic seniority cap.
    final_score = min(int(final_score), int(cap))

    risk_text = "" if risk_level in ("none", "low") else f" Risk: {risk_level} (missing {len(must_missing_names)} must)."
    extra_seniority_text = " (overqualified)" if bucket == "overqualified" else ""
    summary = (
        f"Score {int(final_score)}/100. Coverage {coverage_score}/100; Distance {distance}/100 "
        f"(raw {raw_distance}/100, penalty {penalty2:.2f}). Cap: {cap}. Seniority gap: {bucket}{extra_seniority_text}.{risk_text}"
    )
    if signal_insufficient:
        summary = "Insufficient signal: domain requirements missing or empty; score is not reliable."

    importance_samples = []
    for d in (scored_core_domains or [])[:6]:
        importance_samples.append({"name": getattr(d, "name", ""), "importance_repr": repr(getattr(d, "importance", None))})

    source_bits: List[str] = []
    if candidate_sources.get("explicit"):
        source_bits.append("param")
    if candidate_sources.get("job_ir"):
        source_bits.append("schema")
    if candidate_sources.get("raw_llm"):
        source_bits.append("raw_llm_json")
    if extracted_claims:
        source_bits.append("candidate_text")
    adapter_source = "+".join(source_bits) if source_bits else "unknown"

    domains_source = "unknown"
    if isinstance(_get_field(job_ir, "domain_requirements", None), list):
        domains_source = "schema"
    elif isinstance(job_ir, dict) and isinstance(job_ir.get("job"), dict) and isinstance(job_ir["job"].get("domain_requirements"), list):
        domains_source = "job_dict"
    elif _get_raw_llm(job_ir):
        domains_source = "raw_llm_json"

    requirement_matrix: List[Dict[str, Any]] = []
    for domain in (domains or []):
        name = str(getattr(domain, "name", "") or "").strip()
        if not name or getattr(domain, "domain_id", "") == "other_info":
            continue
        requirement_type = str(
            getattr(domain, "requirement_type", "capability") or "capability"
        )
        raw_status = per_domain.get(name, "unverified")
        if requirement_type == "work_condition":
            semantic_status = str(getattr(domain, "match_status", "unknown") or "unknown")
            has_direct_evidence = bool(getattr(domain, "resume_evidence_ids", None))
            status = (
                "confirmed"
                if semantic_status == "matched" and has_direct_evidence
                else "needs_confirmation"
            )
        else:
            status = {
                "hit": "matched",
                "soft_hit": "partial",
                "weak_hit": "partial",
                "missing": "missing",
            }.get(raw_status, "unverified")
        matrix_evidence_ids = domain_evidence_ids.get(name, [])
        matrix_resume_evidence = domain_resume_evidence.get(name, [])
        if requirement_type == "work_condition":
            matrix_evidence_ids = [
                str(value).strip()
                for value in (getattr(domain, "resume_evidence_ids", None) or [])
                if str(value).strip() in claims_by_evidence_id
            ]
            matrix_resume_evidence = [
                claims_by_evidence_id[value].resume_quote[:280]
                for value in matrix_evidence_ids
            ]
        tools: List[Dict[str, str]] = []
        for example in (getattr(domain, "examples", None) or []):
            tool_name = str(getattr(example, "name", "") or "").strip()
            if not tool_name:
                continue
            tool_status = per_tool_should.get(tool_name, "unverified")
            tools.append(
                {
                    "name": tool_name,
                    "status": {
                        "hit": "matched",
                        "soft_hit": "partial",
                        "missing": "missing",
                    }.get(tool_status, "unverified"),
                }
            )
        requirement_matrix.append(
            {
                "name": name,
                "importance": str(getattr(domain, "importance", "unknown") or "unknown"),
                "status": status,
                "match_confidence": domain_match_confidence.get(name, 0.0),
                "jd_evidence": str(getattr(domain, "evidence_quote", "") or "")[:500],
                "resume_evidence": matrix_resume_evidence,
                "resume_evidence_ids": matrix_evidence_ids,
                "match_reason": (
                    str(getattr(domain, "match_reason", "") or "needs_user_confirmation")
                    if requirement_type == "work_condition"
                    else domain_match_reasons.get(name, "unverified")
                ),
                "requirement_type": requirement_type,
                "alternatives": list(getattr(domain, "alternatives", None) or [])[:10],
                "tools": tools[:15],
            }
        )

    requirement_counts = {
        "total": len(requirement_matrix),
        "matched": sum(1 for x in requirement_matrix if x["status"] == "matched"),
        "partial": sum(1 for x in requirement_matrix if x["status"] == "partial"),
        "missing": sum(1 for x in requirement_matrix if x["status"] == "missing"),
        "needs_confirmation": sum(
            1 for x in requirement_matrix if x["status"] == "needs_confirmation"
        ),
        "confirmed": sum(1 for x in requirement_matrix if x["status"] == "confirmed"),
        "unverified": sum(1 for x in requirement_matrix if x["status"] == "unverified"),
        "must_total": sum(
            1
            for x in requirement_matrix
            if x["importance"] == "must" and x["requirement_type"] != "work_condition"
        ),
        "must_missing": sum(
            1
            for x in requirement_matrix
            if x["importance"] == "must"
            and x["requirement_type"] != "work_condition"
            and x["status"] == "missing"
        ),
    }
    unsupported_requirement_matches = (
        sum(
            1
            for item in requirement_matrix
            if item["status"] in ("matched", "partial") and not item.get("resume_evidence_ids")
        )
        if structured_claims
        else 0
    ) + semantic_invalid_requirements + int(
        semantic_match_available and not semantic_recommendation_valid
    )
    tool_matrix = [
        {
            "name": tool,
            "status": {
                "hit": "matched",
                "soft_hit": "partial",
                "missing": "missing",
            }.get(per_tool_should.get(tool, "unverified"), "unverified"),
            **explicit_jd_keyword_meta.get(_norm(tool), {}),
        }
        for tool in tool_should_examples
    ]
    tool_counts = {
        "total": len(tool_matrix),
        "matched": sum(1 for x in tool_matrix if x["status"] == "matched"),
        "partial": sum(1 for x in tool_matrix if x["status"] == "partial"),
        "missing": sum(1 for x in tool_matrix if x["status"] == "missing"),
    }

    debug = ScoreBreakdown(
        per_domain=per_domain,
        per_tool_must=per_tool_must,  # kept for schema stability; unused under bonus-only policy
        per_tool_should=per_tool_should,
        constraints={
            "job_level": job_level,
            "cand_level": cand_level,
            "candidate_display_level": (
                "junior_to_mid"
                if cand_reason == "experience_band_junior_to_mid"
                else cand_level
            ),
            "cand_numeric_level": cand_numeric,
            "cand_level_reason": cand_reason,
            "candidate_title_signal": cand_title_signal,
            "gap": gap,
            "seniority_gap_bucket": bucket,
            "cap": cap,
            "cap_applied": (cap < 100),
            "blocked_by_seniority": (bucket == "cliff"),
            "signal_insufficient": signal_insufficient,
            "matcher_mode": (
                "semantic_profile_comparison"
                if semantic_match_available
                else ("evidence_grounded_v2" if structured_claims else "legacy_flat_text")
            ),
            "semantic_invalid_requirements": semantic_invalid_requirements,
            "application_recommendation": (
                model_recommendation.model_dump()
                if hasattr(model_recommendation, "model_dump")
                else model_recommendation
                if isinstance(model_recommendation, dict)
                else {}
            ),
            "candidate_evidence_claim_count": len(structured_claims),
            "unsupported_requirement_matches": unsupported_requirement_matches,
            "seniority_cap_base": seniority_cap_base,
            "seniority_cap_uplift": seniority_cap_uplift,
            "seniority_cap_final": cap,
            "cap_explainer": "none=92, overqualified=85, small=80, medium=60(+uplift, max 70), cliff=35",
            "raw_score": raw_score,
            "raw_max": raw_max,
            "raw_distance": raw_distance,
            "domain_raw_score": domain_raw_score,
            "domain_raw_max": domain_raw_max,
            "coverage_score": coverage_score,
            "tool_bonus_points": tool_bonus_raw,
            "distance_penalty": penalty2,
            "distance_penalty_bucket": penalty_bucket2,
            "final_score_breakdown": {
                "raw_score": raw_score,
                "raw_max": raw_max,
                "raw_distance": raw_distance,
                "domain_raw_score": domain_raw_score,
                "domain_raw_max": domain_raw_max,
                "coverage_score": coverage_score,
                "tool_bonus_points": tool_bonus_raw,
                "penalty": penalty2,
                "distance_after_penalty": distance,
                "cap_applied": cap,
                "final_score": int(final_score),
                "signal_insufficient": signal_insufficient,
            },
            "hit_threshold_domain_strong": DOMAIN_STRONG_THRESHOLD,
            "hit_threshold_domain_weak": DOMAIN_WEAK_THRESHOLD,
            "hit_threshold_tool": TOOL_HIT_THRESHOLD,
            "domain_exact_floor": DOMAIN_EXACT_FLOOR,
            "soft_core_weight": SOFT_CORE_WEIGHT,
            "must_domain_count": must_total,
            "must_domain_hit": must_hit,
            "must_domain_hit_rate": (must_hit / float(must_total)) if must_total > 0 else 1.0,
            "must_domain_count_raw": must_total_raw,
            "must_domain_hit_raw": must_hit_raw,
            "must_domains_detected": must_domain_names_all,
            "must_domains_hit": must_domain_names_hit,
            "must_domains_missing": must_domain_names_missing,
            "core_domain_count_all": raw_domains_total,
            "core_domain_count_filtered": core_total,
            "core_domains_filtered": _dedupe([getattr(x, "name", "") or "" for x in core_domains_filtered]),
            "core_domains_demoted": _dedupe([getattr(x, "name", "") or "" for x in core_domains_demoted]),
            "core_domain_hit_strong": core_hit_strong,
            "core_domain_hit_soft": core_hit_soft,
            "core_domain_hit_effective": core_hit_effective,
            "core_domain_hit_rate_effective": (core_hit_effective / float(core_total)) if core_total > 0 else 1.0,
            "coverage_debug": coverage_debug,
            "should_tool_bonus_used": should_tool_bonus_used,
            "should_tool_bonus_cap": SHOULD_TOOL_BONUS_CAP,
            "should_tool_bonus_max": bonus_max,
            "should_tool_unique_total": tool_should_total_unique,
            "importance_samples": importance_samples,
            "must_downgraded_to_should": must_downgraded,
            "candidate_avg_conf": cand_avg_conf,
            "candidate_keyword_ratio": cand_keyword_ratio,
            "keyword_stuffing_penalty": stuffing_penalty,
            "adapter_domains_source": domains_source,
            "adapter_candidate_skills_source": adapter_source,
            "adapter_parse_stats": {
                "candidate_sources": candidate_sources,
                "candidate_skills_total": len(cand_skills or []),
                "candidate_text_claims_total": len(extracted_claims or []),
                "domains_total_raw": len(domains_raw or []),
                "domains_total_scored": len(scored_core_domains or []),
                "domains_total_other_info": len([d for d in (domains or []) if getattr(d, "domain_id", "") == "other_info"]),
                "domains_source": domains_source,
            },
            "candidate_text_claims_count": len(extracted_claims),
            "candidate_text_claims_preview": extracted_claims[:8],
            "other_domains": _dedupe(other_domains),
            "required_domains_unverified": _dedupe(unverified_domains),
            "display_domain_evidence_tags": display_domain_evidence,
            "requirement_matrix": requirement_matrix,
            "requirement_counts": requirement_counts,
            "tool_matrix": tool_matrix,
            "tool_counts": tool_counts,
            "diagnostics": {
                "scoring_domains_empty_reason": "no_scored_domains_after_evidence_filter" if signal_insufficient else "",
                "unverified_domains_count": len(_dedupe(unverified_domains)),
                "unverified_domains_sample": _dedupe(unverified_domains)[:5],
            },
            "anchors_used": anchors_used,
            "anchors_hit": domain_anchor_hits,
            "tool_policy": {
                "tools_bonus_only": True,
                "tool_importance_policy": "bonus_only",
                "hard_tools_for_required_skills_only": True,
                "hard_tools_detected": _dedupe(required_tool_must),
                "hard_tools_excluded_as_redundant": [
                    {"domain": getattr(d, "name", "") or "", "tool": getattr(ex, "name", "") or ""}
                    for d in (scored_core_domains or [])
                    for ex in (d.examples or [])
                    if _is_must(getattr(d, "importance", None))
                    and _is_must(getattr(ex, "importance", None))
                    and _is_tool_redundant_with_domain(getattr(d, "name", "") or "", getattr(ex, "name", "") or "")
                ][:25],
            },
        },
    )

    gate_risk = GateRisk(risk_level=risk_level, must_missing_names=must_missing_names, notes=notes)

    return ScoreResultV3(
        should_apply=should_apply,
        final_score=int(final_score),
        distance_score=int(distance),
        cap=int(cap),
        seniority_gap=str(bucket),
        summary=summary,
        required_domains=_dedupe(required_domains_all or required_domains),
        display_required_domains=_dedupe(display_required_domains),
        missing_domains_must=_dedupe(missing_domains_must),
        missing_domains_all=_dedupe(missing_domains_all),
        required_tool_must=_dedupe(required_tool_must),
        missing_tool_must=_dedupe(missing_tool_must),
        tool_should_examples=_dedupe(tool_should_examples),
        missing_tool_should_examples=_dedupe(missing_tool_should_examples),
        gate_risk=gate_risk,
        debug_breakdown=debug,
    )


# ============================================================
# Kairos v3 Official Public Output Contract (Rendering-friendly)
# ============================================================
def actions_to_sentences(actions):
    """
    Render actions as bullet-style multi-line strings.
    Output remains List[str] for Notion-friendly display.
    """
    if not actions:
        return []
    out: List[str] = []
    for act in actions:
        if not isinstance(act, dict):
            out.append(str(act))
            continue

        title = (act.get("title") or "").strip()
        items = act.get("items") or []
        items_str = ""
        if isinstance(items, list) and items:
            items_str = " [" + ", ".join(str(x) for x in items[:3]) + (", …" if len(items) > 3 else "") + "]"

        why = (act.get("why") or "").strip()
        steps = act.get("steps") or []

        lines: List[str] = []
        head = (title + items_str).strip() or "Action"
        if why:
            lines.append(f"{head} — {why}")
        else:
            lines.append(head)

        if isinstance(steps, str):
            s = steps.strip()
            if s:
                lines.append(f"- {s}")
        elif isinstance(steps, list):
            cleaned = []
            for s in steps:
                s2 = str(s).strip()
                if not s2:
                    continue
                cleaned.append(s2.rstrip("."))
            for s2 in cleaned[:10]:
                lines.append(f"- {s2}")
        else:
            pass

        out.append("\n".join(lines).strip())

    return out


def score_to_public_dict(result: ScoreResultV3) -> Dict[str, Any]:
    # (UNCHANGED from your current version, except actions rendering and tool policy implications)

    def _dedupe_keep_order(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in seq or []:
            k = _norm(x)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    def _sort_ci(seq: List[str]) -> List[str]:
        return sorted((seq or []), key=lambda x: x.lower())

    def _top_n(seq: List[str], n: int) -> List[str]:
        return (seq or [])[: max(0, int(n))]

    def _layer_label(bucket: str) -> str:
        if bucket in ("none", "overqualified"):
            return "Fit"
        if bucket == "small":
            return "Stretch"
        if bucket == "medium":
            return "Reach"
        return "Cliff"

    def _severity_label(must_missing_count: int, seniority_bucket: str) -> str:
        if seniority_bucket == "cliff":
            return "high"
        if must_missing_count >= 2:
            return "high"
        if must_missing_count == 1:
            return "medium"
        return "low"

    final_score = int(result.final_score)

    distance_score = int(result.distance_score)
    if distance_score > 100:
        distance_score = 100
    if distance_score < 0:
        distance_score = 0

    cap = int(result.cap)
    seniority_bucket = str(result.seniority_gap or "unknown")

    # tools are not blockers; missing_tool_must should be empty, but keep contract compatible
    scored_required_domains = _sort_ci(_dedupe_keep_order(result.required_domains))
    display_required_domains = _sort_ci(
        _dedupe_keep_order(result.display_required_domains or result.required_domains)
    )
    must_missing = _sort_ci(_dedupe_keep_order(result.missing_domains_must))
    rec_missing_all = _sort_ci(_dedupe_keep_order(result.missing_domains_all))
    tool_should_examples = _sort_ci(_dedupe_keep_order(result.tool_should_examples))

    per_domain = (result.debug_breakdown.per_domain or {}) if result.debug_breakdown else {}
    per_tool_should = (result.debug_breakdown.per_tool_should or {}) if result.debug_breakdown else {}

    domain_strengths = _sort_ci(
        _dedupe_keep_order([k for k, v in per_domain.items() if v in ("hit", "soft_hit")])
    )
    tool_strengths = _sort_ci(
        _dedupe_keep_order([canon_tool(k) or k for k, v in per_tool_should.items() if v in ("hit", "soft_hit")])
    )

    rec_missing = _sort_ci([d for d in rec_missing_all if d not in must_missing])

    HEADLINE_MAX = 6
    headline_must = _top_n(must_missing, HEADLINE_MAX)
    remaining_slots = max(0, HEADLINE_MAX - len(headline_must))
    headline_rec = _top_n(rec_missing, remaining_slots)
    headline_chips = headline_must + headline_rec

    missing_groups: List[Dict[str, Any]] = []
    if must_missing:
        missing_groups.append({"level": "must", "label": "must", "items": _top_n(must_missing, 20)})
    if rec_missing:
        missing_groups.append(
            {"level": "recommended", "label": "recommended · lower confidence", "items": _top_n(rec_missing, 20)}
        )

    missing_block = {
        "groups": missing_groups,
        "must": must_missing,
        "recommended": rec_missing,
        "all": _dedupe_keep_order(must_missing + rec_missing),
        "notes": [
            "Recommended items are lower-confidence inferences from the JD; treat as suggestions, not hard blockers."
        ]
        if rec_missing
        else [],
    }

    constraints = (result.debug_breakdown.constraints if result.debug_breakdown else {})
    signal_insufficient = bool(constraints.get("signal_insufficient")) if isinstance(constraints, dict) else False
    requirement_matrix = constraints.get("requirement_matrix", []) if isinstance(constraints, dict) else []
    requirement_counts = constraints.get("requirement_counts", {}) if isinstance(constraints, dict) else {}
    tool_matrix = constraints.get("tool_matrix", []) if isinstance(constraints, dict) else []
    tool_counts = constraints.get("tool_counts", {}) if isinstance(constraints, dict) else {}
    model_recommendation = (
        constraints.get("application_recommendation", {})
        if isinstance(constraints, dict)
        else {}
    )
    must_partial_count = sum(
        1
        for item in requirement_matrix
        if isinstance(item, dict)
        and item.get("importance") == "must"
        and item.get("requirement_type") != "work_condition"
        and item.get("status") == "partial"
    )
    scored_partial_or_missing = any(
        isinstance(item, dict)
        and item.get("requirement_type") != "work_condition"
        and item.get("status") in {"partial", "missing"}
        for item in requirement_matrix
    )

    def _calibrate_recommendation_language(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return text
        constrained = bool(
            scored_partial_or_missing or seniority_bucket in {"medium", "cliff"}
        )
        if constrained:
            text = re.sub(
                r"\ban exceptionally (?:strong|excellent) (?:fit|match)\b",
                "a strong match",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bexceptionally (?:strong|excellent)\b",
                "strong",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\bexceptionally\b", "clearly", text, flags=re.IGNORECASE)
            text = re.sub(r"\ban exceptional\b", "a strong", text, flags=re.IGNORECASE)
            text = re.sub(r"\bexceptional\b", "strong", text, flags=re.IGNORECASE)
            text = re.sub(r"\bideal\b", "credible", text, flags=re.IGNORECASE)
            text = re.sub(r"\bexcellent\b", "strong", text, flags=re.IGNORECASE)
        prefix = ""
        if must_missing:
            prefix = (
                f"Worth considering, but {len(must_missing)} MUST requirement(s) "
                "currently lack resume evidence. "
            )
        elif must_partial_count:
            prefix = (
                f"Promising, with {must_partial_count} MUST requirement(s) only "
                "partially evidenced. "
            )
        if seniority_bucket in {"medium", "cliff"}:
            prefix += (
                "This is a stretch because the role appears above the candidate's "
                "current seniority evidence. "
            )
        return (prefix + text).strip()

    if (
        isinstance(model_recommendation, dict)
        and str(model_recommendation.get("rationale") or "").strip()
        and constraints.get("matcher_mode") == "semantic_profile_comparison"
    ):
        decision_reason = "semantic_candidate_assessment"
        decision_explanation = _calibrate_recommendation_language(
            str(model_recommendation.get("rationale") or "").strip()
        )
    elif signal_insufficient:
        decision_reason = "insufficient_signal"
        decision_explanation = "Insufficient domain requirements to score reliably."
    elif seniority_bucket == "cliff":
        decision_reason = "seniority_cliff"
        decision_explanation = "Seniority gap is a cliff; score is capped regardless of skill match."
    elif must_missing:
        decision_reason = "must_missing"
        decision_explanation = f"Missing {len(must_missing)} MUST requirement(s)."
    elif seniority_bucket == "overqualified":
        decision_reason = "overqualified"
        decision_explanation = "Candidate seniority appears higher than the role level; not a blocker, but may reduce fit."
    elif final_score < 45:
        decision_reason = "low_match"
        decision_explanation = "Overall match score is below threshold."
    else:
        decision_reason = "ok"
        decision_explanation = "No hard blockers detected."

    decision = {"verdict": result.should_apply, "reason": decision_reason, "explanation": decision_explanation}

    actions: List[Dict[str, Any]] = []

    if seniority_bucket == "cliff":
        actions.append(
            {
                "title": "Target closer seniority roles",
                "why": "Seniority gap is a cliff; cap applied regardless of skill match.",
                "steps": [
                    "Search for Junior/Mid GenAI Engineer / Backend Engineer roles with LLM integration.",
                    "Or reframe this application as 'adjacent' with a strong portfolio link.",
                ],
                "tags": ["seniority", "strategy"],
                "priority": "high",
            }
        )

    if seniority_bucket == "overqualified":
        actions.append(
            {
                "title": "Handle potential overqualification (optional)",
                "why": "Some teams worry about retention or expectations mismatch.",
                "steps": [
                    "If applying anyway, tailor your summary to emphasize learning goals and role alignment.",
                    "Highlight why this scope is attractive (domain, mentorship, growth path).",
                ],
                "tags": ["seniority", "positioning"],
                "priority": "low",
            }
        )

    if must_missing and seniority_bucket != "cliff":
        top_must = _top_n(must_missing, 3)
        actions.append(
            {
                "title": "Close MUST gaps",
                "why": "Missing MUST requirements creates a large penalty and increases rejection risk.",
                "steps": [f"Add evidence for: {x} (project bullet + measurable outcome)." for x in top_must],
                "tags": ["must", "resume", "portfolio"],
                "priority": "high",
                "items": top_must,
            }
        )

    if rec_missing and seniority_bucket != "cliff":
        top_rec = _top_n(rec_missing, 3)
        actions.append(
            {
                "title": "Improve recommended domains (optional)",
                "why": "These are likely levers to raise match score; confidence is lower than MUST.",
                "steps": [f"Add proof for: {x} (short demo or 1 bullet evidence)." for x in top_rec],
                "tags": ["recommended", "upskill"],
                "priority": "medium",
                "items": top_rec,
            }
        )

    if seniority_bucket != "cliff":
        per_domain2 = (result.debug_breakdown.per_domain or {}) if result.debug_breakdown else {}
        soft_domains = _dedupe_keep_order([k for k, v in per_domain2.items() if v == "soft_hit"])

        if (not must_missing) and (not rec_missing) and soft_domains:
            top_soft = _top_n(soft_domains, 3)

            def _evidence_steps(domain_name: str) -> List[str]:
                return [
                    f"[{domain_name}] Add 1 bullet that proves you built/shipped something (not just used tools).",
                    "Use an action verb: Built / Implemented / Deployed / Optimized / Migrated / Owned.",
                    "Include scope: requests/day, dataset size, users, services, latency, cost, accuracy (pick 1–2).",
                    "Include environment: prod/staging, CI/CD, monitoring/alerts, rollback, on-call (pick what applies).",
                    'Use a tight template: “Built X with Y; handled Z; deployed to A; improved B by N%.”',
                ]

            steps_flat: List[str] = []
            for dd in top_soft:
                steps_flat.extend(_evidence_steps(dd))
                steps_flat.append("")

            steps_flat = [s for s in steps_flat if s.strip()]

            actions.insert(
                0,
                {
                    "title": "Turn soft matches into strong evidence",
                    "why": "Several domains are counted as soft hits (weak evidence). Strengthening 1–3 bullets often lifts the score more than adding new skills.",
                    "steps": steps_flat
                    + [
                        "Add 1 artifact proof (choose one): repo link / API spec / architecture diagram / dashboard screenshot.",
                        "If possible, pin the artifact in your resume (top projects) and link it in the application.",
                    ],
                    "tags": ["evidence", "resume", "portfolio"],
                    "priority": "high",
                    "items": top_soft,
                },
            )

    missing_tool_should = _sort_ci(_dedupe_keep_order(result.missing_tool_should_examples))
    tool_policy = constraints.get("tool_policy") if isinstance(constraints, dict) else {}
    allow_tool_actions = not bool((tool_policy or {}).get("tools_bonus_only", True))
    if missing_tool_should and seniority_bucket != "cliff" and allow_tool_actions:
        t1 = missing_tool_should[0]
        t2 = missing_tool_should[1] if len(missing_tool_should) > 1 else None

        tool_steps: List[str] = [
            f"Pick 1 tool: {t1}.",
            "Create a tiny proof repo (or gist) that runs in < 5 minutes.",
            "Include: 1–2 endpoints or scripts + README + runnable command.",
            "Show: example request/response (curl) + basic input validation.",
            "Add: 1 small test (pytest or equivalent).",
        ]
        if t2:
            tool_steps += [f"Optional: add a second proof for {t2} (same minimal standard)."]
        tool_steps += ["If applicable: add a Dockerfile + “How to run” section."]

        actions.append(
            {
                "title": "Add 1–2 tool proofs (optional)",
                "why": "Tool matches provide small bonuses; prioritize domains first.",
                "steps": tool_steps,
                "tags": ["tools", "bonus"],
                "priority": "low",
                "items": _top_n(missing_tool_should, 2),
            }
        )

    layers: List[Dict[str, Any]] = []

    seniority_status = "ok"
    if seniority_bucket == "cliff":
        seniority_status = "blocked"
    elif seniority_bucket == "overqualified":
        seniority_status = "info"
    elif seniority_bucket != "none":
        seniority_status = "warning"

    cap_applied = bool(constraints.get("cap_applied")) if isinstance(constraints, dict) else False
    blocked_by_seniority = bool(constraints.get("blocked_by_seniority")) if isinstance(constraints, dict) else False

    if seniority_bucket in ("none", "overqualified"):
        seniority_summary = "Gap: none (no seniority cap)."
    else:
        seniority_summary = f"Gap: {seniority_bucket} (cap {cap})."

    layers.append(
        {
            "id": "seniority",
            "title": "Seniority",
            "status": seniority_status,
            "summary": seniority_summary,
            "meta": {
                "bucket": seniority_bucket,
                "cap": cap,
                "cap_applied": cap_applied,
                "blocked_by_seniority": blocked_by_seniority,
                "job_level": (constraints.get("job_level") if result.debug_breakdown else None),
                "candidate_level": (
                    constraints.get("candidate_display_level", constraints.get("cand_level"))
                    if result.debug_breakdown
                    else None
                ),
                "candidate_numeric_level": (constraints.get("cand_numeric_level") if result.debug_breakdown else None),
                "gap": (constraints.get("gap") if result.debug_breakdown else None),
            },
        }
    )

    layers.append(
        {
            "id": "domains",
            "title": "Core domains",
            "status": (
                "info"
                if signal_insufficient
                else ("blocked" if must_missing else ("ok" if (not rec_missing) else "warning"))
            ),
            "summary": (
                "No requirements available for scoring."
                if signal_insufficient
                else f"{len(domain_strengths)}/{len(scored_required_domains)} hit. "
                + (f"{len(must_missing)} MUST missing." if must_missing else "No MUST missing.")
            ),
            "strengths": _top_n(domain_strengths, 10),
            "gaps": [],
            "meta": {
                "required": display_required_domains,
                "missing_must": must_missing,
                "missing_all": _dedupe_keep_order(must_missing + rec_missing),
                "missing_groups": missing_groups,
                "headline_gaps_must": headline_must,
                "headline_gaps_recommended": headline_rec,
            },
        }
    )

    layers.append(
        {
            "id": "tools",
            "title": "Tool examples",
            "status": "ok" if tool_strengths else "info",
            "summary": f"{len(tool_strengths)} tool matches (bonus).",
            "strengths": _top_n(tool_strengths, 10),
            "recommended": _top_n(missing_tool_should, 12),
            "meta": {"examples_in_jd": _top_n(tool_should_examples, 25)},
        }
    )

    strengths_section = [
        {"type": "domains", "items": _top_n(domain_strengths, 10)},
        {"type": "tools", "items": _top_n(tool_strengths, 10)},
    ]

    gaps_section = [
        {"type": "must", "items": must_missing},
        {"type": "recommended", "items": _top_n(rec_missing, 10)},
    ]

    must_missing_count = len(must_missing)
    severity = _severity_label(must_missing_count, seniority_bucket)
    if signal_insufficient:
        severity = "medium"

    risk_level = result.gate_risk.risk_level if result.gate_risk else "low"
    if signal_insufficient:
        risk_level = "medium"
    risk = {
        "level": risk_level,
        "severity": severity,
        "must_missing_count": must_missing_count,
        "must_missing": must_missing,
        "notes": (result.gate_risk.notes if result.gate_risk else []),
    }

    score_summary = result.summary
    if (
        isinstance(model_recommendation, dict)
        and constraints.get("matcher_mode") == "semantic_profile_comparison"
        and str(model_recommendation.get("rationale") or "").strip()
    ):
        score_summary = _calibrate_recommendation_language(
            str(model_recommendation.get("rationale") or "").strip()
        )
    if seniority_bucket == "cliff":
        score_summary = f"{result.summary} (Capped primarily by seniority.)"

    score = {
        "should_apply": result.should_apply,
        "final_score": final_score,
        "distance_score": distance_score,
        "coverage_score": (constraints.get("coverage_score") if isinstance(constraints, dict) else distance_score),
        "tool_bonus_points": (constraints.get("tool_bonus_points") if isinstance(constraints, dict) else 0),
        "tool_bonus_max": (constraints.get("should_tool_bonus_max") if isinstance(constraints, dict) else 0),
        "uncapped_score": (constraints.get("raw_distance") if isinstance(constraints, dict) else distance_score),
        "cap": cap,
        "seniority_cap": constraints.get("seniority_cap_final", cap) if isinstance(constraints, dict) else cap,
        "cap_explainer": constraints.get("cap_explainer") if isinstance(constraints, dict) else None,
        "seniority_gap": seniority_bucket,
        "label": _layer_label(seniority_bucket),
        "summary": score_summary,
    }

    debug = {
        "constraints": constraints,
        "per_domain": (result.debug_breakdown.per_domain if result.debug_breakdown else {}),
        "per_tool_should": (result.debug_breakdown.per_tool_should if result.debug_breakdown else {}),
        "per_tool_must": (result.debug_breakdown.per_tool_must if result.debug_breakdown else {}),
    }

    contract: Dict[str, Any] = {
        "engine_version": "v3",
        "contract": {"name": "kairos_v3_public", "version": "2.6"},
        "decision": decision,
        "score": score,
        "analysis_quality": {
            "status": (
                "degraded"
                if signal_insufficient or int(constraints.get("unsupported_requirement_matches", 0) or 0) > 0
                else "complete"
            ),
            "score_reliable": (
                not signal_insufficient
                and int(constraints.get("unsupported_requirement_matches", 0) or 0) == 0
            ),
            "requirements_extracted": len(requirement_matrix) if isinstance(requirement_matrix, list) else 0,
            "matcher_mode": constraints.get("matcher_mode", "legacy_flat_text"),
            "candidate_evidence_claims": int(
                constraints.get("candidate_evidence_claim_count", 0) or 0
            ),
            "unsupported_matches": int(
                constraints.get("unsupported_requirement_matches", 0) or 0
            ),
        },
        "requirements": {
            "counts": requirement_counts if isinstance(requirement_counts, dict) else {},
            "items": requirement_matrix if isinstance(requirement_matrix, list) else [],
            "status_meaning": {
                "matched": "The cited resume evidence directly or equivalently satisfies the requirement.",
                "partial": "The cited evidence is transferable or adjacent, but scope, depth, or exact tooling differs.",
                "missing": "The stored Candidate Profile contains no relevant resume evidence.",
                "confirmed": "The resume directly confirms this eligibility or work-condition item.",
                "needs_confirmation": "This eligibility or work-condition item is not scored and needs the user to confirm it.",
                "unverified": "The JD requirement could not be verified reliably.",
            },
        },
        "tools": {
            "counts": tool_counts if isinstance(tool_counts, dict) else {},
            "items": tool_matrix if isinstance(tool_matrix, list) else [],
            "note": "Explicit JD tools are ATS-relevant evidence, but are bonus-only in Kairos scoring.",
        },
        "missing": missing_block,
        "required_skills": display_required_domains,
        "required_domains": display_required_domains,
        "display_required_domains": display_required_domains,
        "display_domain_evidence": constraints.get("display_domain_evidence_tags", {}) if isinstance(constraints, dict) else {},
        "required_domains_labels": {
            **{x: "inferred_unverified" for x in _sort_ci(_dedupe_keep_order(constraints.get("required_domains_unverified", [])))}
        }
        if isinstance(constraints, dict)
        else {},
        "missing_domains": _dedupe_keep_order(must_missing + rec_missing),
        "other_domains": _sort_ci(_dedupe_keep_order(constraints.get("other_domains", []))) if isinstance(constraints, dict) else [],
        "mentioned_but_unverified_domains": _sort_ci(
            _dedupe_keep_order(constraints.get("required_domains_unverified", []))
        )
        if isinstance(constraints, dict)
        else [],
        "tools_in_jd": tool_should_examples,
        "recommended_tools_in_jd": _sort_ci(_dedupe_keep_order(result.missing_tool_should_examples)),
        "missing_required_tools": [],
        "missing_tools": _sort_ci(_dedupe_keep_order(result.missing_tool_should_examples)),
        "missing_tools_labels": {t: "nice_to_have" for t in _dedupe_keep_order(result.missing_tool_should_examples)},
        "missing_tools_mode": "nice_to_have",
        "missing_skills_mode": ("must" if must_missing else "recommended"),
        "missing_skills": headline_chips,
        "missing_skills_labels": {**{x: "must" for x in headline_must}, **{x: "recommended_low_conf" for x in headline_rec}},
        "actions": actions_to_sentences(actions),
        "strengths": strengths_section,
        "layers": layers,
        "gaps": gaps_section,
        "risk": risk,
        "debug": debug,
        "render_hints": {
            "primary_missing_field": "missing",
            "avoid_duplicate_missing_sources": ["layers.domains.gaps"],
        },
    }

    return contract


__all__ = ["score_ir_v3", "score_to_public_dict", "ScoreResultV3"]
