# backend/llm/analyze_v3.py
"""
Kairos Job Analyzer v3 (Domain-first IR) — importable

This revision makes the prompt industry-agnostic.

Goal:
- Keep your Domain-first IR structure (portable domains + tool examples).
- Ensure candidate extraction is NOT tech-only:
  - Candidate.skills must include: (1) technical skills/tools, (2) functions (e.g., accounting, sales ops),
    (3) industries/domains (e.g., finance, healthcare), pulled from experience/education too.
- Ensure job domain requirements are also industry-agnostic:
  - Domains should include non-tech capabilities (e.g., regulatory compliance, stakeholder mgmt, accounting),
    not only engineering domains.

Notes:
- Extraction-only change (prompt). Your scoring remains deterministic.

v3.1+ patch (Jan 2026):
- Responsibilities default to SHOULD unless explicitly required.
- Parse "specifically ..." under Requirements/Must-Have context => promote listed tools to MUST (and auto-fill tool quotes).

方案B（Soft MUST + Weak Hit buffer）在 analyze_v3 的落点：
- scoring 侧会把 domain 匹配分为 strong/weak/missing（>=0.60 / >=0.40 / <0.40），弱命中不算 MUST missing。
- 为了让 weak hit 更稳定：analyze_v3 需要更“保守但有效”的 anchors：
  1) 只保留能在 JD 文本中验证出现的 anchors（本文件已做）
  2) 增强 anchor 生成：包含 domain 名、证据句、examples 名、括号缩写等
  3) 避免过泛 anchor（STOP_ANCHOR_TOKENS）
- schema 无需变更；本文件主要是：prompt 行业通用 + anchors/importance/tool-must 提升稳定性。

实现点（你这份代码里我已落地）：
- 进一步强化“职责段默认 should”的严格化（避免 MUST 抖动）
- “specifically/in particular” 下的工具提升 MUST（仅在 requirements/强制语境）
- anchors 更稳（验证出现 + 去泛化 + 去重 + 上限）
"""

from __future__ import annotations

import json
import os
import re
import time
import pathlib
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions
from google.generativeai.types import GenerationConfig

from backend.ir.schema_v3 import (
    AnalyzeIRv3,
    DomainRequirement,
    ToolEvidence,
    Importance,
    SeniorityLabel,
    DomainFacet,
    ExtractedOwnership,
    ApplicationRecommendation,
)
from backend.ir.candidate_profile import CandidateProfile
from backend.ir.domain_catalog import load_domain_catalogs, adjudicate_domains
from backend.ir.canonicalize import canon_tool

# =============================
# Gemini setup
# =============================
JOB_ANALYSIS_MODEL = (os.getenv("GEMINI_JOB_MODEL") or "gemini-3.5-flash").strip()
JOB_ANALYSIS_TIMEOUT_SECONDS = 60
JOB_ANALYSIS_MAX_ATTEMPTS = 2
CATALOG_DIR = pathlib.Path(__file__).parent.parent / "config" / "catalogs"
_DOMAIN_CATALOG = None


def _ensure_gemini_configured() -> None:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment/.env")
    genai.configure(api_key=key)


def _detect_job_family(text: str) -> str:
    t = (text or "").lower()
    family_signals = {
        "tech_swe": [
            "python", "java", "javascript", "typescript", "fastapi", "flask", "django",
            "aws", "gcp", "azure", "docker", "kubernetes", "ci/cd", "git", "github",
            "api", "backend", "serverless", "microservice",
        ],
        "tech_data": [
            "sql", "etl", "pipeline", "warehouse", "data lake", "spark", "airflow",
            "rag", "llm", "prompt", "embeddings", "vector database", "ml", "model",
        ],
        "finance": ["accounting", "finance", "budget", "reconciliation", "audit", "ledger"],
        "sales": ["sales", "account management", "pipeline", "quota", "client", "customer"],
        "marketing": ["marketing", "campaign", "growth", "acquisition", "brand", "lead gen"],
        "ops": ["operations", "workflow", "coordination", "logistics", "supply", "process"],
        "hr": ["hr", "recruitment", "hiring", "people operations", "talent", "onboarding"],
        "legal": ["legal", "contract", "compliance", "regulation", "policy", "audit"],
        "healthcare": ["clinical", "patient", "nursing", "healthcare", "medical", "hospital"],
        "education": ["curriculum", "teaching", "teacher", "education", "student", "school"],
    }

    counts = {k: sum(1 for s in v if s in t) for k, v in family_signals.items()}
    tech_swe_hits = counts.get("tech_swe", 0)
    tech_data_hits = counts.get("tech_data", 0)
    nontech_best = max((v for k, v in counts.items() if k not in ("tech_swe", "tech_data")), default=0)

    tech_best = max(tech_swe_hits, tech_data_hits)
    if tech_best >= 2 and tech_best >= nontech_best:
        return "tech_data" if tech_data_hits >= tech_swe_hits else "tech_swe"
    for fam in ("finance", "sales", "marketing", "ops", "hr", "legal", "healthcare", "education"):
        if counts.get(fam, 0) == nontech_best and nontech_best > 0:
            return fam
    return "general"


_JD_TOOL_IGNORE = {
    "tool",
    "tools",
    "tooling",
    "technology",
    "technologies",
    "platform",
    "platforms",
    "solution",
    "solutions",
    "framework",
    "frameworks",
    "stack",
    "planning",
}

def _normalize_tool_token(tok: str) -> List[str]:
    t = (tok or "").strip()
    if not t:
        return []
    t = re.sub(r"[()\[\]{}]+", " ", t)
    t = re.sub(r"[;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(?:such as|including|for example)\s+", "", t, flags=re.IGNORECASE).strip()
    if not t or "?" in t:
        return []
    if re.match(
        r"^(?:and|or|but|are you|you will|you are|we are|the role|learn|work|build|develop)\b",
        t,
        re.IGNORECASE,
    ):
        return []
    if len(t) > 64 or len(t.split()) > 6:
        return []
    key = t.lower()
    if key in _JD_TOOL_IGNORE:
        return []
    return [t]


def extract_tools_from_jd(page_text: str) -> List[str]:
    """Extract explicit keywords without relying on an industry tool list.

    Complete requirements come from the grounded domain extraction. This
    helper only captures short, explicitly labelled/cued keywords. Illustrative
    lists introduced by "such as"/"including" stay attached to their parent
    requirement instead of becoming individual missing skills.
    """
    if not (page_text or "").strip():
        return []
    tools: List[str] = []

    acronym_aliases: Dict[str, str] = {}
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9&/\-]*(?:\s+[A-Za-z][A-Za-z0-9&/\-]*){1,7})"
        r"\s*\(\s*([A-Z][A-Z0-9&/\-]{1,11})\s*\)",
        page_text,
    ):
        long_form = _normalize_whitespace(match.group(1))
        long_form = re.sub(
            r"^(?:skilled in|proficient in|proficiency in|experience with|experience using|"
            r"hands-on experience with|familiar with|knowledge of|certified in|"
            r"certification in|licensed in)\s+",
            "",
            long_form,
            flags=re.IGNORECASE,
        ).strip()
        acronym = match.group(2).strip()
        if long_form and acronym:
            acronym_aliases[acronym.lower()] = long_form

    def _expand_tool_chunk(chunk: str) -> List[str]:
        c = (chunk or "").strip()
        if not c:
            return []
        c = re.sub(r"^(?:and|or)\s+", "", c, flags=re.IGNORECASE).strip()
        base = c
        inner = ""
        if "(" in c and ")" in c:
            base = c.split("(", 1)[0].strip()
            inner = c.split("(", 1)[1].split(")", 1)[0].strip()

        items: List[str] = []
        base_key = re.sub(r"\s+", " ", base).strip().lower()
        if inner and inner.lower() in acronym_aliases:
            items.extend(_normalize_tool_token(acronym_aliases[inner.lower()]))
        else:
            items.extend(_normalize_tool_token(base))
        # Parenthetical examples such as "IaC (Terraform)" are distinct
        # explicit terms; acronym expansions are already merged above.
        if inner and inner.lower() not in acronym_aliases and inner.lower() != base_key:
            items.extend(_normalize_tool_token(inner))
        return [x for x in items if x]

    example_marker = re.compile(r"\b(such as|including|e\.g\.|for example|like)\b", re.IGNORECASE)
    label_cues = re.compile(
        r"\b(skill|tool|software|system|platform|certification|licen[cs]e|language|"
        r"technology|method|methodology|application)\b",
        re.IGNORECASE,
    )

    # Structured lists work for any profession: "Software: Excel, SAP",
    # "Certifications: CPA, CFA", "Methods: Agile, Six Sigma".
    for ln in page_text.splitlines():
        if not ln.strip():
            continue
        match = re.match(r"^\s*([^:\n]{2,45})\s*:\s*(.+)$", ln)
        if not match:
            continue
        label, rest = match.group(1).strip(), match.group(2).strip()
        list_like = len(re.findall(r"[,;|]", rest)) >= 1
        if not label_cues.search(label) and not list_like:
            continue
        if example_marker.search(rest):
            continue
        parts = [p.strip() for p in re.split(r"[,;|]|\s+\band\b\s+", rest) if p.strip()]
        for p in parts:
            p2 = p.strip().strip(".:;")
            for norm in _expand_tool_chunk(p2):
                tools.append(norm)

    # Direct requirement cues are industry-neutral. Example lists are skipped:
    # "experience with Excel" is explicit; "tools such as Excel" is not.
    cue_re = re.compile(
        r"\b(skilled in|proficient in|proficiency in|experience with|experience using|"
        r"hands-on experience with|familiar with|knowledge of|certified in|"
        r"certification in|licensed in)\b",
        re.IGNORECASE,
    )
    for ln in page_text.splitlines():
        if not ln.strip():
            continue
        m = cue_re.search(ln)
        if not m:
            continue
        if example_marker.search(ln[m.end() :]):
            continue
        tail = ln[m.end():].strip()
        tail = re.sub(r"^[\s:\-–—]+", "", tail)
        tail = re.split(r"(?<=[.!?])\s+", tail, maxsplit=1)[0].strip()
        if not tail:
            continue
        parts = [p.strip() for p in re.split(r"[,;|]|\s+\band\b\s+", tail) if p.strip()]
        for p in parts:
            for norm in _expand_tool_chunk(p.strip().strip(".:;")):
                tools.append(norm)

    out: List[str] = []
    seen = set()
    for t in tools:
        k = re.sub(r"[^a-z0-9+#]+", " ", t.strip().lower()).strip()
        k = re.sub(r"\s+", " ", k)
        if k in acronym_aliases:
            canonical = acronym_aliases[k]
            k = re.sub(r"[^a-z0-9+#]+", " ", canonical.lower()).strip()
            t = canonical
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t.strip())
    return out


_EVIDENCE_REPAIR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
    "experience",
    "knowledge",
    "skills",
    "required",
    "preferred",
}


def _evidence_repair_tokens(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+#./-]*", str(value or "").lower())
        if len(token) > 1 and token not in _EVIDENCE_REPAIR_STOPWORDS
    ]


def _repair_jd_evidence_quote(
    *,
    page_text: str,
    requirement_name: str,
    proposed_quote: str,
    evidence_summary: str = "",
) -> str:
    """Map a paraphrased model citation back to one exact JD passage."""
    page_flat = _normalize_whitespace(page_text)
    if not page_flat:
        return ""

    segments = [
        _normalize_whitespace(part)
        for part in re.split(r"(?:\n+|(?<=[.!?;])\s+|\s+[•●▪]\s+)", str(page_text or ""))
    ]
    candidates = [
        segment
        for segment in segments
        if 3 <= len(segment.split()) <= 90
    ]

    name_tokens = set(_evidence_repair_tokens(requirement_name))
    target_text = " ".join(
        value for value in (requirement_name, proposed_quote, evidence_summary) if value
    )
    target_tokens = set(_evidence_repair_tokens(target_text))

    # Add bounded exact windows around distinctive requirement terms. This
    # handles LinkedIn text where bullets were flattened into one paragraph.
    page_lower = page_flat.lower()
    for token in sorted(name_tokens, key=len, reverse=True)[:8]:
        for match in list(re.finditer(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", page_lower))[:3]:
            start = max(0, match.start() - 180)
            end = min(len(page_flat), match.end() + 320)
            while start > 0 and page_flat[start - 1] not in ".!?;":
                start -= 1
                if match.start() - start >= 260:
                    break
            while end < len(page_flat) and page_flat[end - 1] not in ".!?;":
                end += 1
                if end - match.end() >= 420:
                    break
            candidate = page_flat[start:end].strip(" .")
            if candidate:
                candidates.append(candidate)

    best_quote = ""
    best_score = 0.0
    seen = set()
    normalized_target = _normalize_whitespace(target_text).lower()
    for candidate in candidates:
        key = candidate.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        candidate_tokens = set(_evidence_repair_tokens(candidate))
        if not candidate_tokens:
            continue
        name_hits = len(name_tokens & candidate_tokens)
        target_hits = len(target_tokens & candidate_tokens)
        # Semantic requirement labels need not appear verbatim in the JD. When
        # the label itself has no hit, require at least two supporting terms
        # from the model's proposed quote/summary before considering a passage.
        if name_tokens and name_hits == 0 and target_hits < 2:
            continue
        coverage = target_hits / float(max(1, len(target_tokens)))
        precision = target_hits / float(max(1, len(candidate_tokens)))
        sequence = SequenceMatcher(None, normalized_target, key).ratio()
        score = 0.55 * coverage + 0.20 * precision + 0.25 * sequence
        if score > best_score:
            best_score = score
            best_quote = candidate

    if best_score < 0.22:
        return ""
    if not _validate_quote(best_quote, page_flat.lower()):
        return ""
    return best_quote[:900]


def _build_jd_passages(page_text: str, *, max_passages: int = 120) -> List[Dict[str, str]]:
    """Split a JD into stable, source-grounded passages for model citations.

    Gemini cites the IDs; the backend owns the source text. Every emitted
    passage is a contiguous substring of the whitespace-normalized JD.
    """
    normalized = _normalize_whitespace(page_text)
    if not normalized:
        return []

    # Preserve bullet and sentence boundaries before whitespace normalization.
    rough_parts = re.split(
        r"(?:\r?\n+|[•●▪◦]\s*|(?<=[.!?;:])\s+(?=[A-Z0-9]))",
        str(page_text or ""),
    )
    parts = [_normalize_whitespace(value) for value in rough_parts]
    parts = [value for value in parts if value]

    passages: List[str] = []
    for part in parts:
        words = list(re.finditer(r"\S+", part))
        if len(words) <= 85:
            passages.append(part)
            continue

        # Long LinkedIn sections are sometimes flattened into one paragraph.
        # Split at word boundaries without rewriting any source wording.
        start_word = 0
        while start_word < len(words):
            end_word = min(start_word + 65, len(words))
            start_char = words[start_word].start()
            end_char = words[end_word - 1].end()
            passages.append(part[start_char:end_char])
            start_word = end_word

    # Some page extractors remove all useful boundaries. The long-part branch
    # above still covers that case. Deduplicate repeated LinkedIn UI text.
    unique: List[str] = []
    seen = set()
    normalized_lower = normalized.lower()
    for passage in passages:
        passage = _normalize_whitespace(passage)
        key = passage.lower()
        if len(passage) < 8 or key in seen:
            continue
        if key not in normalized_lower:
            continue
        seen.add(key)
        unique.append(passage)
        if len(unique) >= max_passages:
            break

    if not unique:
        unique = [normalized[:4000]]

    return [
        {"id": f"jd_{index:03d}", "text": passage}
        for index, passage in enumerate(unique, start=1)
    ]


def _get_domain_catalog(job_family: str):
    global _DOMAIN_CATALOG
    if _DOMAIN_CATALOG is None:
        _DOMAIN_CATALOG = {}
    if job_family in _DOMAIN_CATALOG:
        return _DOMAIN_CATALOG[job_family]

    base = CATALOG_DIR / "base.yaml"
    tech = CATALOG_DIR / "tech_overlay.yaml"
    tech_data = CATALOG_DIR / "tech_data_overlay.yaml"
    nontech = CATALOG_DIR / "nontech_overlay.yaml"

    if not base.exists():
        raise RuntimeError(f"Domain catalog not found at {base}")

    if job_family == "tech_swe":
        paths = [base, tech]
    elif job_family == "tech_data":
        paths = [base, tech_data]
    else:
        paths = [base, nontech]

    catalog = load_domain_catalogs(paths)
    _DOMAIN_CATALOG[job_family] = catalog
    return catalog


# =============================
# Regex (module-level)
# =============================
_WS = re.compile(r"\s+")
_OR_SPLIT_RE = re.compile(r"\s+(?:or|\|)\s+", re.IGNORECASE)

RE_STRICT_QUOTE = re.compile(r"\b(required|must|essential)\b", re.IGNORECASE)
RE_STRICT_STRONG = re.compile(
    r"\b(are required|is required|required experience|must have|mandatory|essential|need to have)\b",
    re.IGNORECASE,
)
RE_EXAMPLE = re.compile(
    r"\b(such as|including|e\.g\.|tools like|frameworks like|platforms like|libraries like)\b",
    re.IGNORECASE,
)
RE_PREFERENCE = re.compile(
    r"\b(nice to have|bonus|preferred|plus|desired|advantage|desirable)\b",
    re.IGNORECASE,
)

# --- section cues (best-effort, deterministic) ---
RE_SECTION_REQUIREMENTS = re.compile(
    r"\b(requirements|must have|must-haves|minimum qualifications|min(?:imum)? requirements|"
    r"essential qualifications|what we are looking for|what you'll bring|you have)\b",
    re.IGNORECASE,
)
RE_SECTION_RESPONSIBILITIES = re.compile(
    r"\b(responsibilities|key responsibilities|what you'll do|what you will do|role overview|the role|you will)\b",
    re.IGNORECASE,
)

# --- specifically parsing ---
RE_SPECIFICALLY = re.compile(r"\b(specifically|in particular)\b", re.IGNORECASE)

RE_SENIOR_RESPONSIBILITY = re.compile(
    r"\b("
    r"technical leadership|technical direction|set technical direction|"
    r"architecture discussion|architecture discussions|architectures?:|"
    r"system design|software system design|design and implement|"
    r"scalable|maintainable|reliable software systems|"
    r"deliver resilient|reusable components|"
    r"code review|code reviews|review designs and code|"
    r"stakeholder management|influence stakeholders|"
    r"cross[- ]functional|align roadmaps|manage dependencies|"
    r"mentor|mentorship|technical guidance|"
    r"uplifting engineering craft|role[- ]model engineering standards|"
    r"iterative experimentation|from idea to prototype to pilot to product|"
    r"success metrics"
    r")\b",
    re.IGNORECASE,
)

RE_JUNIOR_GROWTH = re.compile(
    r"\b("
    r"work(?:ing)?\s+(?:closely\s+)?with\s+senior\s+engineers|"
    r"learn(?:ing)?\s+(?:from\s+)?(?:senior\s+engineers|seniors)|"
    r"learn(?:ing)?\s+(?:system\s+design|architecture|distributed\s+systems)|"
    r"under\s+(?:the\s+)?guidance\s+of|"
    r"with\s+(?:the\s+)?guidance\s+from|"
    r"mentored\s+by|"
    r"training\s+(?:program|track|rotation)|"
    r"early[- ]career|"
    r"entry[- ]level|"
    r"graduate|grad\b|"
    r"growth\s+mindset|"
    r"willing(?:ness)?\s+to\s+learn"
    r")\b",
    re.IGNORECASE,
)

RE_STRONG_SENIOR_ONLY = re.compile(
    r"\b("
    r"staff\s+engineer|principal\s+engineer|lead\s+engineer|"
    r"own(?:ing)?\s+(?:the\s+)?architecture|architecture\s+ownership|"
    r"set\s+(?:the\s+)?technical\s+direction|"
    r"drive\s+(?:org|organization)[- ]wide|"
    r"organization[- ]wide|org[- ]wide|"
    r"lead\s+(?:a\s+)?team|manage\s+(?:a\s+)?team|"
    r"mentor(?:ing)?\s+engineers|"
    r"technical\s+strategy"
    r")\b",
    re.IGNORECASE,
)

RE_APPRENTICE = re.compile(r"\b(apprentice|apprenticeship)\b", re.IGNORECASE)

RE_PEOPLE_FACET = re.compile(
    r"\b(leadership|stakeholder|influence|mentor|mentorship|guidance|collaborate|cross[- ]functional|"
    r"roadmaps|dependencies|role[- ]model|standards|best practices)\b",
    re.IGNORECASE,
)
RE_PROCESS_FACET = re.compile(
    r"\b(experimentation|prototype|pilot|success metrics|evaluation|fit[- ]for[- ]purpose|recommendations)\b",
    re.IGNORECASE,
)

RE_TECH_GATE = re.compile(
    r"\b("
    r"system design|software system design|design and implement|scalable|maintainable|reliable|"
    r"programming|proficiency|languages?|python|typescript|c#|java|c\+\+|"
    r"microservices?|rest|grpc|event[- ]driven|"
    r"public cloud|aws|azure|gcp|lambda|functions|ecs|aks|s3|blob|api gateway|step functions|logic apps|"
    r"security engineering|devsecops|sast|dast|sca|iam|oauth|oidc|tls|encryption|threat modeling|nist|iso 27001"
    r")\b",
    re.IGNORECASE,
)

RE_PREFER_SHOULD_DOMAINS = re.compile(
    r"\b(technical leadership|stakeholder management|leadership|mentorship|experimentation|growth mindset)\b",
    re.IGNORECASE,
)

_ALLOWED_SENIORITY: Tuple[str, ...] = ("intern", "junior", "mid", "senior", "lead", "unknown")

_PAREN_ACRONYM_RE = re.compile(r"\b([A-Za-z][A-Za-z \-\/]{2,60})\s*\(\s*([A-Za-z0-9]{2,12})\s*\)")

# anchors: avoid generic tokens, to prevent over-matching across industries
_STOP_ANCHOR_TOKENS = {
    "ai",
    "ml",
    "api",
    "apis",
    "sdk",
    "saas",
    "cloud",
    "data",
    "tools",
    "tool",
    "platform",
    "platforms",
    "framework",
    "frameworks",
    "system",
    "systems",
    "development",
    "engineering",
    # common HR boilerplate
    "experience",
    "knowledge",
    "skills",
    "ability",
    "responsibilities",
    "requirements",
}

_IMPORTANCE_RANK: Dict[str, int] = {
    "unknown": 0,
    "nice": 1,
    "nice_to_have": 1,
    "should": 2,
    "must": 3,
}


# =============================
# JSON helpers
# =============================
def _extract_json_block(text: str) -> str:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _loads_json_object(text: str) -> Dict[str, Any]:
    t = text or ""
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    json_str = _extract_json_block(t)
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    json_str = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", json_str)

    try:
        out = json.loads(json_str)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


# =============================
# Text normalization & quote validation
# =============================
def _normalize_whitespace(s: str) -> str:
    return _WS.sub(" ", (s or "").strip()).strip()


def _normalize_preserve_newlines(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _word_count(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    s = re.sub(r"\s+", " ", s)
    return s.count(" ") + 1


def _anchor_slug(anchor: str) -> str:
    anchor_clean = re.sub(r"[^a-zA-Z0-9\+\.#]+", " ", anchor)
    toks = [t for t in anchor_clean.split() if t]
    if not toks:
        return ""
    slug = max(toks, key=len).lower()
    return slug if len(slug) >= 3 else ""


def _find_anchor_center(window_lower: str, anchor: str) -> Optional[int]:
    a_norm = _normalize_whitespace(anchor).lower()
    idx = window_lower.find(a_norm)
    if idx != -1:
        return idx + max(1, len(a_norm) // 2)
    slug = _anchor_slug(anchor)
    if slug:
        idx2 = window_lower.find(slug)
        if idx2 != -1:
            return idx2 + max(1, len(slug) // 2)
    return None


def _shrink_to_max_words_substring(s: str, center: int, max_words: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if _word_count(s) <= max_words:
        return s

    start, end = max(0, center - 160), min(len(s), center + 160)
    if start >= end:
        start, end = 0, len(s)

    guard = 0
    while start < end and _word_count(s[start:end]) > max_words and guard < 600:
        guard += 1
        left_span = max(0, center - start)
        right_span = max(0, end - center)
        if left_span >= right_span:
            nxt = s.find(" ", start + 1, end)
            start = nxt + 1 if nxt != -1 else start + 1
        else:
            prv = s.rfind(" ", start, end - 1)
            end = prv if prv != -1 else end - 1

    out = s[start:end].strip()
    guard2 = 0
    while _word_count(out) > max_words and len(out) > 1 and guard2 < 200:
        guard2 += 1
        out = out[1:-1].strip()
    return out


def _smart_trim_quote(s: str, anchor: str, max_words: int = 22) -> str:
    s_norm = _normalize_whitespace(s)
    if _word_count(s_norm) <= max_words:
        return s_norm
    center = _find_anchor_center(s_norm.lower(), anchor)
    if center is None:
        center = len(s_norm) // 2
    return _shrink_to_max_words_substring(s_norm, center, max_words)


def _validate_quote(quote: str, page_text_flat_lower: str) -> bool:
    if not quote:
        return False

    q_norm = _normalize_whitespace(quote)
    # JD passage IDs may point to a complete sentence or compact paragraph.
    # The backend owns that source text, so length is not a trust signal.
    if len(q_norm.split()) > 90:
        return False
    if len(q_norm) < 2:
        return False

    q_lower = q_norm.lower()

    if len(q_lower) < 5:
        if q_lower == ".net":
            pattern = re.escape(q_lower) + r"(?:$|[^a-z0-9\+#])"
        else:
            pattern = r"(?:^|[^a-z0-9\+#])" + re.escape(q_lower) + r"(?:$|[^a-z0-9\+#])"
        return re.search(pattern, page_text_flat_lower) is not None

    return q_lower in page_text_flat_lower


def _validate_soft_anchor(anchor: str, page_text_flat_lower: str) -> bool:
    a = _normalize_whitespace(anchor).lower()
    if len(a) < 2:
        return False
    if len(a) < 5:
        pattern = r"(?:^|[^a-z0-9\+#])" + re.escape(a) + r"(?:$|[^a-z0-9\+#])"
        return re.search(pattern, page_text_flat_lower) is not None
    return a in page_text_flat_lower


def _extract_context_window_orig(anchor: str, text_flat_lower: str, text_flat_orig: str) -> Optional[str]:
    a = _normalize_whitespace(anchor).lower()
    idx = text_flat_lower.find(a)
    if idx == -1:
        return None
    start = max(0, idx - 260)
    end = min(len(text_flat_orig), idx + len(a) + 260)
    return text_flat_orig[start:end].strip()


def _sanitize_field(value: Any, default: Optional[str] = "Unknown") -> Optional[str]:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    if s.lower() in ["string", "str", "unknown", "n/a", "none", "null", "undefined"]:
        return default
    return s


def _norm_seniority_label(x: Any) -> SeniorityLabel:
    s = str(x or "").strip().lower()
    if s in ["entry", "entry-level", "graduate"]:
        return "junior"
    if s in ["intermediate"]:
        return "mid"
    if s in ["staff", "principal", "manager", "head", "director", "vp", "architect"]:
        return "lead"
    return s if s in _ALLOWED_SENIORITY else "unknown"  # type: ignore


def _strip_wrapping_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1].strip()
    return s


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("importance", "value", "name", "label"):
            if k in x:
                return _as_text(x[k])
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


def _norm_importance(x: Any) -> Importance:
    s = _as_text(x).strip().lower()
    s = _strip_wrapping_quotes(s)

    if not s:
        return "unknown"

    if "." in s:
        s = s.split(".")[-1].strip()
        s = _strip_wrapping_quotes(s)

    for sep in (":", "=", "/"):
        if sep in s:
            tail = s.split(sep)[-1].strip()
            tail = _strip_wrapping_quotes(tail)
            if tail in ("must", "should", "nice", "nice_to_have", "unknown"):
                return tail  # type: ignore

    if s in ("must", "should", "nice", "nice_to_have", "unknown"):
        return s  # type: ignore

    if s in ("required", "essential", "core", "mandatory"):
        return "must"
    if s in ("preferred", "bonus", "nice to have", "nice-to-have", "plus", "desired", "advantage", "desirable"):
        return "nice_to_have"

    if "must" in s or "mandatory" in s or "required" in s or "essential" in s:
        return "must"
    if "preferred" in s or "bonus" in s or "desirable" in s or "nice to have" in s:
        return "nice_to_have"
    if "should" in s:
        return "should"

    return "unknown"


def _is_valid_title_signal(tag: str) -> bool:
    t = (tag or "").strip()
    if not t:
        return False
    if len(t.split()) > 12:
        return False
    s_lower = t.lower()
    sentence_markers = [" i ", " i'm ", " my ", " currently ", " working ", " responsible ", " duties "]
    if any(m in f" {s_lower} " for m in sentence_markers):
        return False
    return True


def _refine_job_title(explicit: Optional[str], standardized: Optional[str], user_title: Optional[str]) -> str:
    def is_garbage(t: str) -> bool:
        return _is_garbage_title(t)

    def looks_like_job_board_title(t: str) -> bool:
        tl = (t or "").lower().strip()
        if not tl:
            return False
        if " job in " in f" {tl} " or " jobs in " in f" {tl} ":
            return True
        if any(x in tl for x in ["seek", "indeed", "linkedin", "glassdoor", "workday", "greenhouse", "lever"]):
            if " - " in tl or " | " in tl:
                return True
        return False

    e = str(explicit or "").strip()
    if e and not is_garbage(e):
        s = str(standardized or "").strip()
        if len(e.split()) > 10 and s and not is_garbage(s):
            return s
        return e

    s = str(standardized or "").strip()
    if s and not is_garbage(s):
        return s

    if user_title and not is_garbage(user_title) and not looks_like_job_board_title(user_title):
        return user_title.strip()

    return "Unknown Role"


def _is_garbage_title(t: str) -> bool:
    t = (t or "").lower().strip()
    garbage_triggers = {
        "job description",
        "role description",
        "about us",
        "who we are",
        "what you'll do",
        "what you will do",
        "responsibilities",
        "requirements",
        "benefits",
        "full time",
        "part time",
        "location",
        "salary",
        "apply now",
        "join us",
    }
    if not t or len(t) < 3:
        return True
    if t in garbage_triggers:
        return True
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}", t):
        return True
    return False


def _extract_job_title_from_text(page_text: str) -> Optional[str]:
    if not page_text:
        return None
    patterns = [
        r"(?im)^\s*(job title|role|position|title)\s*[:\-]\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, page_text)
        if not m:
            continue
        title = (m.group(2) or "").strip()
        if title and not _is_garbage_title(title):
            return title
    return None


# =============================
# Anchors
# =============================
def _slugify_token(s: str) -> str:
    s = _normalize_whitespace(s).lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[“”\"']", "", s)
    s = re.sub(r"[^a-z0-9\+#/ \-\.]+", " ", s)
    s = _normalize_whitespace(s)
    return s


def _split_tokens_for_anchors(name: str) -> List[str]:
    n = _slugify_token(name)
    if not n:
        return []
    out: List[str] = [n]

    parts = re.split(r"[\/\-:|,;]+", n)
    for p in parts:
        p = _normalize_whitespace(p)
        if p and p not in out:
            out.append(p)

    for tok in re.split(r"\s+", n):
        if len(tok) >= 3 and tok not in out:
            out.append(tok)

    return out[:20]


def _derive_acronym_anchors(evidence_quote: str) -> List[str]:
    q = _normalize_whitespace(evidence_quote)
    if not q:
        return []
    anchors: List[str] = []
    for long_name, acr in _PAREN_ACRONYM_RE.findall(q):
        long_slug = _slugify_token(long_name)
        acr_slug = _slugify_token(acr)
        if long_slug and long_slug not in anchors:
            anchors.append(long_slug)
        if acr_slug and acr_slug not in anchors:
            anchors.append(acr_slug)
    return anchors[:6]


def _is_anchor_too_generic(anchor: str) -> bool:
    a = _slugify_token(anchor)
    if not a:
        return True
    if a in _STOP_ANCHOR_TOKENS:
        return True
    if len(a) < 3:
        return True
    if a.isdigit():
        return True
    return False


def _build_domain_anchors(
    domain_name: str,
    evidence_quote: str,
    examples: List[ToolEvidence],
    page_text_flat_lower: str,
    *,
    max_anchors: int = 10,
) -> List[str]:
    """
    方案B 关键：anchors 要“保守但覆盖”：
    - 只保留能在 JD 中验证出现的 token/phrase
    - 去泛化（STOP_ANCHOR_TOKENS）
    - 结合 domain_name / evidence_quote / tools / acronym
    """
    candidates: List[str] = []
    candidates.extend(_split_tokens_for_anchors(domain_name))
    candidates.extend(_derive_acronym_anchors(evidence_quote))
    for ex in examples or []:
        candidates.extend(_split_tokens_for_anchors(ex.name))

    seen = set()
    kept: List[str] = []
    for c in candidates:
        c = _normalize_whitespace(c)
        if not c:
            continue
        if _is_anchor_too_generic(c):
            continue

        k = c.lower()
        if k in seen:
            continue
        seen.add(k)

        if _validate_soft_anchor(c, page_text_flat_lower):
            kept.append(c)

        if len(kept) >= max_anchors:
            break

    return kept


# =============================
# LLM Extraction (v3 contract)
# =============================
def _extract_with_gemini_v3(
    page_text: str,
    *,
    title: Optional[str],
    output_language: str,
    candidate_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ensure_gemini_configured()
    model = genai.GenerativeModel(JOB_ANALYSIS_MODEL)

    # Pre-load catalog for prompt context
    job_family = _detect_job_family(page_text)
    catalog = _get_domain_catalog(job_family)
    catalog_json_summary = json.dumps(
        [{"id": d.id, "label": d.label, "facet": d.facet, "aliases": d.aliases_strong + d.aliases_weak}
         for d in catalog.domains.values()],
        indent=2
    )
    try:
        prompt_profile = CandidateProfile.model_validate(candidate_profile or {})
    except Exception as exc:
        raise ValueError(f"Invalid Candidate Profile: {exc}") from exc
    candidate_profile_summary = {
        "candidate_seniority_signal": prompt_profile.candidate_seniority_signal,
        "seniority_reason": prompt_profile.seniority_reason,
        "evidence_claims": [
            {
                "evidence_id": claim.evidence_id,
                "resume_quote": claim.resume_quote[:500],
                "skills": list(claim.skills or [])[:12],
                "domains": list(claim.domains or [])[:8],
                "role": claim.role,
            }
            for claim in prompt_profile.evidence_claims[:40]
            if claim.evidence_id and claim.resume_quote
        ],
        "roles": [
            {
                "title": role.title,
                "organization": role.organization,
                "date_range": role.date_range,
            }
            for role in prompt_profile.roles[:20]
        ],
    }
    candidate_profile_json = json.dumps(candidate_profile_summary, ensure_ascii=False)
    jd_passages = _build_jd_passages(page_text)
    jd_passage_map = {item["id"]: item["text"] for item in jd_passages}
    jd_passages_json = json.dumps(jd_passages, ensure_ascii=False)

    prompt = f"""You are a Job Analysis Engine.

Your task: decide whether this candidate should apply and return a compact,
scorable comparison for ANY professional job.

GLOBAL RULES:

1) DECISION-LEVEL REQUIREMENTS:
   - Extract the complete set of hiring-relevant requirements, but group terms
     that would be assessed together by a recruiter.
   - Do not turn every noun, example, acronym, or tool into a separate
     requirement. Most JDs should produce roughly 8-20 decision-level items.
   - Combine equivalent or overlapping items from the same clause.
   - Do not invent a requirement that lacks a supporting job passage ID.

2) IMPORTANCE HIERARCHY:
   - "must": only explicit required/essential/minimum/mandatory language.
   - "should": responsibilities and normal expected capabilities.
   - "nice_to_have": preferred/bonus/desirable/plus language.
   - Do not promote a responsibility to MUST merely because it sounds senior.

3) LOGICAL RULES:
   - One OR list is ONE requirement. Put its choices in `alternatives`.
     Never emit reversed duplicate requirements for each choice.
   - Terms introduced by "such as", "including", "e.g.", or "for example"
     belong in `examples[]` and are not independent requirements.
   - A named tool/system is its own requirement only when the JD explicitly
     requires proficiency, certification, or experience with that exact item.

4) SEMANTIC CANDIDATE MATCH:
   - Judge professional transferability, not literal phrase overlap.
   - `matched`: direct evidence or clearly equivalent experience satisfies the
     hiring requirement.
   - `partial`: adjacent/transferable evidence exists, but scope, depth, domain,
     or exact tool differs.
   - `missing`: the Candidate Profile contains no relevant evidence.
   - Every matched/partial item MUST cite one or more valid resume_evidence_ids.
   - Do not use broad labels alone. Read the cited resume quotes in context.
   - Never treat the same broad label across unrelated professional contexts
     as equivalent; regulated or compliance evidence must concern the same
     professional/regulatory domain.
   - For a requirement containing multiple material components joined by AND,
     `matched` requires evidence for every component. If only a subset is
     supported, use `partial`.
   - Do not infer a specialized environment, regulated context, scale, or
     responsibility merely from an adjacent industry or broad job title.

5) SHOULD APPLY:
   - Make the same holistic recommendation a strong career adviser would make.
   - Recommend Yes or Maybe when the candidate has strong transferable core
     experience and the gaps are learnable tools or narrower scope.
   - Recommend No only for a genuine non-negotiable gate or when the core job
     function is substantially unsupported.
   - A missing resume phrase does not prove the candidate cannot do something.
   - Responsibilities are not hard blockers merely because they appear in the
     job description.
   - "Yes" means the role is worth applying for; it does not mean a high
     interview probability or a near-perfect match.
   - Calibrate the language. Do not say "exceptional", "ideal", or equivalent
     when any MUST requirement is partial/missing or when the candidate
     seniority signal is materially below the job seniority signal. Describe
     such a role as transferable, stretch, or reach as appropriate.

6) COUNT & COVERAGE:
   - Prefer complete decision coverage over a fixed output count.
   - Include non-technical constraints such as licences, education, work rights,
     travel, schedule, language, physical conditions, or regulated experience.
   - Keep evidence_summary to at most 20 words.

7) EVIDENCE:
   - **evidence_summary**: Summarize the requirement context (1 sentence) for human understanding.
   - **jd_evidence_ids**: Cite 1-3 exact IDs from `job_passages` that support
     the requirement. Never copy, paraphrase, or generate JD evidence text.
   - Resume support is represented ONLY by `resume_evidence_ids`.
   - Never use a JD passage ID as a resume evidence ID or vice versa.

8) SENIORITY RUBRIC (Apply Universally):
   Use these BEHAVIORAL markers to determine `job_seniority_signal`, regardless of tech stack:
   - **MID-LEVEL**: Focus on **Execution** & **Implementation**.
     Keywords: "Implement", "Support", "Maintain", "Collaborate". Scope: Tasks / Features.
   - **SENIOR**: Focus on **Design** & **System Complexity**.
     Keywords: "Design", "Architect", "Scalability", "Reliability", "End-to-end ownership". Scope: Systems / Services.
   - **LEAD / PRINCIPAL**: Focus on **Strategy** & **Influence**.
     Keywords: "Set direction", "Mentor", "Cross-functional alignment", "Drive standards", "Stakeholder management". Scope: Team / Organization.
   - **NOTE**: Many JDs don't put "Lead" in the title but describe Lead-level influence. Trust the **responsibilities** over the title.

9) OWNERSHIP & SCOPE (Critical for Precision):
   Extract these signals explicitly (Integer Levels 0-3):
   - **ownership.level_val**: 0=None, 1=Task Owner (Can do work), 2=Result Owner (Accountable), 3=System/End-to-End Owner.
   - **scope.level_val**: 0=Unknown, 1=Self only, 2=Team Impact, 3=Multi-Team/Org Impact.
   - **leadership.level_val**: 0=None, 1=Mentor/Guide, 2=Team Lead (unblock others), 3=Org Lead (Manager/Principal).

<job_passages>
{jd_passages_json}
</job_passages>

<candidate_profile>
{candidate_profile_json}
</candidate_profile>

OUTPUT FORMAT (JSON ONLY):
Produce a single valid JSON object following this structure. Do NOT wrap in markdown code blocks.

{{
  "job_title": "string",
  "company": "string",
  "job_seniority_signal": "junior|mid|senior|lead|principal",
  "domain_requirements": [
    {{
      "domain_id": "string (optional, copy 'id' from catalog if exact match)",
      "domain": "string",
      "requirement_type": "capability|tool|credential|education|experience|work_condition|responsibility|other",
      "importance": "must|should|nice_to_have",
      "alternatives": ["literal OR alternative"],
      "match_status": "matched|partial|missing",
      "resume_evidence_ids": ["exact evidence_id from candidate_profile"],
      "match_reason": "one concise candidate-specific reason",
      "evidence_summary": "string",
      "jd_evidence_ids": ["exact id from job_passages"],
      "examples": [
        {{
          "tool": "string",
          "importance": "must|should|nice_to_have"
        }}
      ]
    }}
  ],
  "ownership_and_scope": {{
    "ownership": {{ "present": true, "level_val": 0, "evidence": ["string"] }},
    "scope": {{ "level_val": 0, "evidence": ["string"] }},
    "leadership": {{ "present": true, "level_val": 0, "evidence": ["string"] }}
  }},
  "application_recommendation": {{
    "should_apply": "Yes|Maybe|No",
    "confidence": "low|medium|high",
    "rationale": "concise holistic explanation",
    "hard_blockers": ["only genuine non-negotiable missing gates"],
    "strongest_matches": ["string"],
    "key_gaps": ["string"]
  }},
  "evidence_hints": {{
    "years_experience": "string",
    "education_degree": "string"
  }}
}}

You have access to an optional normalization catalog. Use domain_id only for an
exact conceptual match. The catalog is not an allowlist; preserve grounded new
requirements with domain_id omitted:
{catalog_json_summary}

OUTPUT LANGUAGE: {output_language}
"""

    generation_config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        # Gemini 3.5 Flash may use part of this budget for reasoning before
        # emitting the structured JSON. Keep enough headroom to avoid a
        # truncated or empty response on requirement-heavy JDs.
        max_output_tokens=16384,
        response_mime_type="application/json",
    )

    # Seniority Rules (Regex-based Fusion)
    rule_seniority = _extract_seniority_rules(page_text)

    try:
        transient_errors = (
            google_api_exceptions.DeadlineExceeded,
            google_api_exceptions.GatewayTimeout,
            google_api_exceptions.BadGateway,
            google_api_exceptions.ServiceUnavailable,
            google_api_exceptions.InternalServerError,
        )
        resp = None
        for attempt in range(JOB_ANALYSIS_MAX_ATTEMPTS):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": JOB_ANALYSIS_TIMEOUT_SECONDS},
                )
                break
            except transient_errors:
                if attempt + 1 >= JOB_ANALYSIS_MAX_ATTEMPTS:
                    raise
                time.sleep(0.75)
        if resp is None:
            raise RuntimeError("Gemini job analysis returned no response")
        # Verify JSON
        parsed_ir = AnalyzeIRv3.model_validate_json(resp.text)
        invalid_jd_evidence = []
        for requirement in parsed_ir.domain_requirements:
            cited_ids = _dedupe_keep_order(
                [
                    str(value).strip()
                    for value in (requirement.jd_evidence_ids or [])
                    if str(value).strip() in jd_passage_map
                ]
            )[:3]
            if not cited_ids:
                invalid_jd_evidence.append(requirement.name)
                continue
            requirement.jd_evidence_ids = cited_ids
            # Downstream deterministic logic continues to consume one exact
            # source quote. It is supplied by the backend, never by Gemini.
            requirement.evidence_quote = jd_passage_map[cited_ids[0]]
        if invalid_jd_evidence:
            raise ValueError(
                "Gemini omitted a valid JD passage ID for: "
                + ", ".join(invalid_jd_evidence[:5])
            )
        valid_candidate_ids = {
            item["evidence_id"]
            for item in candidate_profile_summary["evidence_claims"]
            if item.get("evidence_id")
        }
        if valid_candidate_ids and parsed_ir.domain_requirements:
            invalid_semantic_items = []
            for requirement in parsed_ir.domain_requirements:
                status = str(requirement.match_status or "unknown")
                cited = {
                    value
                    for value in (requirement.resume_evidence_ids or [])
                    if value in valid_candidate_ids
                }
                if status == "unknown" or (
                    status in {"matched", "partial"} and not cited
                ):
                    invalid_semantic_items.append(requirement.name)
            if invalid_semantic_items:
                raise ValueError(
                    "Gemini omitted valid candidate evidence for requirement comparison: "
                    + ", ".join(invalid_semantic_items[:5])
                )
            if not parsed_ir.application_recommendation.rationale.strip():
                raise ValueError("Gemini omitted the application recommendation rationale")

        # Attach raw JSON for debugging/transparency
        try:
            parsed_ir.raw_llm_json = json.loads(resp.text)
        except Exception:
            parsed_ir.raw_llm_json = {}

        # Preserve the deterministic JD seniority signal alongside the model output.
        parsed_ir.evidence_hints['rule_seniority_level'] = str(rule_seniority)

        # Adjudicate against catalog (THE Deterministic Step)
        job_family = _detect_job_family(page_text)
        catalog = _get_domain_catalog(job_family)
        if isinstance(parsed_ir.raw_llm_json, dict):
            parsed_ir.raw_llm_json.setdefault("_debug_meta", {})
            parsed_ir.raw_llm_json["_debug_meta"]["job_family"] = job_family
            parsed_ir.raw_llm_json["_debug_meta"]["jd_passage_count"] = len(jd_passages)
        # Note: adjudicate_domains now handles verification internally!
        # So we do NOT call _verify_spans here anymore.
        parsed_ir.domain_requirements = adjudicate_domains(
            parsed_ir.domain_requirements,
            jd_text=page_text,
            catalog=catalog
        )

        return parsed_ir.model_dump()

    except Exception as e:
        print(f"Gemini V3 Extraction Error/Degraded: {e}")
        # Return a "Degraded" IR instead of empty dict
        # This prevents silent failure (Run 0 domains)
        return AnalyzeIRv3(
            job_title="Error (Degraded)",
            company="Unknown",
            job_seniority_signal="unknown",
            analysis_status="degraded",
            evidence_hints={"error": str(e)}
        ).model_dump()


def _extract_seniority_rules(text: str) -> int:
    """
    Deterministic regex-based seniority level (0-3).
    """
    t = text.lower()

    # Lev 3: Org/Multi-Team Impact
    if RE_STRONG_SENIOR_ONLY.search(t):
        return 3

    # Lev 2: Team Lead / Senior
    if RE_SENIOR_RESPONSIBILITY.search(t):
        return 2

    # Lev 1: Mid/Execution (Default if standard keywords exist)
    # Lev 0: Junior/Intern
    if RE_JUNIOR_GROWTH.search(t) or RE_APPRENTICE.search(t):
        return 0

    return 1 # Default mid


def _verify_spans(ir: AnalyzeIRv3, page_text: str) -> None:
    """
    Check if extracted evidence quotes actually exist in the source text.
    Grading:
    - exact: Long quote (>35 chars) found verbatim (is_verified=True).
    - anchor: Short quote (<=35 chars) found verbatim (is_verified=True).
    - fuzzy: Quote NOT found, implies conceptual match only (is_verified=False).
    - none: No quote provided.
    """
    haystack = _normalize_whitespace(page_text).lower()

    for req in ir.domain_requirements:
        # verify domain evidence
        if req.evidence_quote:
            needle = _normalize_whitespace(req.evidence_quote).lower()
            if len(needle) < 3:
                req.is_verified = False
                req.verification_method = "none"
            elif needle in haystack:
                # Found! Distinguish Exact vs Anchor by length
                if len(needle) > 35:
                    req.is_verified = True
                    req.verification_method = "exact"
                else:
                    req.is_verified = True
                    req.verification_method = "anchor"
            else:
                # Not found -> Fuzzy (Hallucinated/Conceptual)
                req.is_verified = False
                req.verification_method = "fuzzy"
        else:
            req.is_verified = False
            req.verification_method = "none"

        # verify examples evidence (optional, but good for completeness)
        for ex in req.examples:
            if ex.evidence_quote:
                 n_ex = _normalize_whitespace(ex.evidence_quote).lower()
                 pass


# =============================
# Post-process helpers
# =============================
def _context_window_around_quote(
    page_text_lines_lower: str,
    page_text_lines_flat_lower: str,
    page_text_flat_lower: str,
    quote: str,
) -> str:
    q_norm = _normalize_whitespace(quote).lower()
    if not q_norm:
        return ""

    idx = page_text_lines_lower.find(q_norm)
    if idx != -1:
        start = max(0, idx - 260)
        end = min(len(page_text_lines_lower), idx + len(q_norm) + 260)
        return page_text_lines_lower[start:end]

    idx2 = page_text_lines_flat_lower.find(q_norm)
    if idx2 != -1:
        start = max(0, idx2 - 260)
        end = min(len(page_text_lines_flat_lower), idx2 + len(q_norm) + 260)
        return page_text_lines_flat_lower[start:end]

    idx3 = page_text_flat_lower.find(q_norm)
    if idx3 != -1:
        start = max(0, idx3 - 260)
        end = min(len(page_text_flat_lower), idx3 + len(q_norm) + 260)
        return page_text_flat_lower[start:end]

    return ""


def _tight_context_around_quote(page_text_flat_lower: str, quote: str, pad: int = 120) -> str:
    q = _normalize_whitespace(quote).lower()
    if not q:
        return ""
    idx = page_text_flat_lower.find(q)
    if idx == -1:
        return ""
    start = max(0, idx - pad)
    end = min(len(page_text_flat_lower), idx + len(q) + pad)
    return page_text_flat_lower[start:end]


def _guess_section_kind(ctx_lower: str) -> str:
    """
    Best-effort local section classification using nearby text.
    Returns: "requirements" | "responsibilities" | "unknown"
    """
    c = _normalize_whitespace(ctx_lower or "").lower()
    if not c:
        return "unknown"

    has_req = RE_SECTION_REQUIREMENTS.search(c) is not None
    has_resp = RE_SECTION_RESPONSIBILITIES.search(c) is not None

    if has_req and not has_resp:
        return "requirements"
    if has_resp and not has_req:
        return "responsibilities"
    if has_req and has_resp:
        # Prefer requirements if both appear (safer for Must Have bullets that include "you will...")
        return "requirements"
    return "unknown"


def _clean_candidate_skill_list(skills: Any) -> List[str]:
    if not isinstance(skills, list):
        return []
    out: List[str] = []
    for s in skills:
        t = _normalize_whitespace(str(s)).strip()
        if not t:
            continue
        parts = [p.strip() for p in t.split(",")] if "," in t else [t]
        for p in parts:
            if p and len(p) <= 80:
                out.append(p)

    seen = set()
    final: List[str] = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(x)
    return final


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _window_has_requirement_signals(window_lower: str) -> bool:
    if not window_lower:
        return False
    return bool(
        RE_STRICT_QUOTE.search(window_lower)
        or RE_STRICT_STRONG.search(window_lower)
        or RE_EXAMPLE.search(window_lower)
        or RE_PREFERENCE.search(window_lower)
        or RE_SENIOR_RESPONSIBILITY.search(window_lower)
    )


def _infer_facet(domain_name: str, quote: str, ctx_lower: str) -> DomainFacet:
    text = f"{domain_name} {quote} {ctx_lower}".lower()
    if RE_PEOPLE_FACET.search(text):
        return "people"
    if RE_TECH_GATE.search(text):
        return "technical"
    if RE_PROCESS_FACET.search(text):
        return "process"
    return "unknown"


def _derive_domain_importance_from_context(
    domain_name: str,
    evidence_quote: str,
    tight_context_lower: str,
    raw_imp: Any,
    facet: DomainFacet,
    section_kind: str,
) -> Importance:
    """Preserve the model's section-aware classification with safe overrides.

    Gemini sees the complete ordered JD passages. This function must not
    independently reclassify an evidenced requirement from a small local
    window; it only enforces unambiguous preference/mandatory wording.
    """
    base = _norm_importance(raw_imp)
    if base == "unknown":
        base = "should"

    q = _normalize_whitespace(evidence_quote or "").lower()
    quote_has_preference = bool(RE_PREFERENCE.search(q))
    quote_has_strict = bool(RE_STRICT_QUOTE.search(q) or RE_STRICT_STRONG.search(q))
    # Preference language cannot become a hard requirement.
    if quote_has_preference:
        return "nice_to_have"

    # Explicit mandatory language remains authoritative.
    if quote_has_strict:
        return "must"

    return base


def _is_explicit_tool_must(ex_quote: str, ex_ctx: str, tool_name: str, *, section_kind: str) -> bool:
    """
    Tool is MUST only if text explicitly requires that tool by name.
    Conservative to avoid false MUST (which causes MUST-missing jitter).
    """
    if not tool_name:
        return False

    ctx = _normalize_whitespace((ex_ctx or "").lower())
    if not ctx:
        return False

    # Example/preference framing => not must
    if RE_EXAMPLE.search(ctx) or RE_PREFERENCE.search(ctx):
        return False

    # Strong requirement markers in vicinity => must
    if RE_STRICT_QUOTE.search(ctx) or RE_STRICT_STRONG.search(ctx):
        return True

    # If in responsibilities, be conservative
    if section_kind == "responsibilities":
        return False

    t = _normalize_whitespace(tool_name).lower()
    if t:
        pattern1 = re.compile(rf"\b{re.escape(t)}\b.*\b(required|mandatory|must)\b", re.IGNORECASE)
        pattern2 = re.compile(rf"\b(required|mandatory|must)\b.*\b{re.escape(t)}\b", re.IGNORECASE)
        if pattern1.search(ctx) or pattern2.search(ctx):
            return True

    return False


def _merge_examples_keep_strongest(examples: List[ToolEvidence]) -> List[ToolEvidence]:
    by_key: Dict[str, ToolEvidence] = {}
    order: List[str] = []
    for e in examples or []:
        name = _normalize_whitespace(e.name)
        if not name:
            continue
        k = name.lower()
        if k not in by_key:
            by_key[k] = ToolEvidence(
                name=name,
                importance=e.importance if e.importance in ("must", "should", "nice", "unknown") else "should",
                evidence_quote=e.evidence_quote,
            )
            order.append(k)
            continue

        cur = by_key[k]
        if _IMPORTANCE_RANK.get(e.importance, 0) > _IMPORTANCE_RANK.get(cur.importance, 0):
            cur.importance = e.importance

        if (not cur.evidence_quote) and e.evidence_quote:
            cur.evidence_quote = e.evidence_quote
        elif cur.evidence_quote and e.evidence_quote and len(e.evidence_quote) > len(cur.evidence_quote):
            cur.evidence_quote = e.evidence_quote

        by_key[k] = cur

    return [by_key[k] for k in order]


def _tool_key_matches(name_a: str, name_b: str) -> bool:
    a = _normalize_whitespace(name_a).lower()
    b = _normalize_whitespace(name_b).lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith(b) and len(b) >= 3:
        return True
    if b.endswith(a) and len(a) >= 3:
        return True
    return False


def _parse_specifically_list(domain_quote: str) -> List[str]:
    q = _normalize_whitespace(domain_quote or "")
    if not q:
        return []

    q_lower = q.lower()
    m = RE_SPECIFICALLY.search(q_lower)
    if not m:
        return []

    tail = q[m.end() :].strip()
    tail = tail.lstrip(" :;,-").strip()
    tail = tail[:120]

    # alternatives => do not promote
    if re.search(r"\s+or\s+", tail, flags=re.IGNORECASE):
        return []

    tail = re.sub(r"\s*&\s*", " and ", tail)
    parts = re.split(r",|\band\b", tail, flags=re.IGNORECASE)

    out: List[str] = []
    for p in parts:
        t = _normalize_whitespace(p).strip(" .);:-")
        if not t:
            continue
        if t.lower() in ("and", "or", "with", "to", "in", "of", "for"):
            continue
        if len(t) > 40:
            continue
        out.append(t)

    seen = set()
    final: List[str] = []
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(x)
    return final


def _promote_tools_from_specifically(
    *,
    domain_importance: Importance,
    section_kind: str,
    domain_evidence_quote: str,
    page_text_flat_lower: str,
    page_text_flat_orig: str,
    examples: List[ToolEvidence],
) -> List[ToolEvidence]:
    """
    If domain is MUST under Requirements-like context and the domain quote contains "specifically ...",
    promote the listed tools to MUST (and auto-fill/validate tool quotes).
    """
    if domain_importance != "must":
        return examples
    if section_kind not in ("requirements", "unknown"):
        return examples

    tokens = _parse_specifically_list(domain_evidence_quote)
    if not tokens:
        return examples

    if section_kind == "unknown":
        q_lower = _normalize_whitespace(domain_evidence_quote).lower()
        if not (RE_STRICT_QUOTE.search(q_lower) or RE_STRICT_STRONG.search(q_lower) or RE_SECTION_REQUIREMENTS.search(q_lower)):
            return examples

    promoted: List[ToolEvidence] = []
    for ex in examples or []:
        ex_name = _normalize_whitespace(ex.name)
        if not ex_name:
            continue

        should_promote = any(_tool_key_matches(ex_name, t) for t in tokens)
        if not should_promote:
            promoted.append(ex)
            continue

        ex_quote = _normalize_whitespace(ex.evidence_quote or "")
        if not ex_quote:
            ex_quote_candidate = _smart_trim_quote(domain_evidence_quote, ex_name, 22)
            if _validate_quote(ex_quote_candidate, page_text_flat_lower):
                ex_quote = ex_quote_candidate
            else:
                ctx2 = _extract_context_window_orig(ex_name, page_text_flat_lower, page_text_flat_orig)
                if ctx2:
                    ex_quote_candidate2 = _smart_trim_quote(ctx2, ex_name, 22)
                    if _validate_quote(ex_quote_candidate2, page_text_flat_lower):
                        ex_quote = ex_quote_candidate2

        if ex_quote and _validate_quote(ex_quote, page_text_flat_lower):
            ex.importance = "must"
            ex.evidence_quote = ex_quote
        else:
            ex.importance = "should"
            ex.evidence_quote = ex.evidence_quote or None

        promoted.append(ex)

    return promoted


# =============================
# Seniority helpers (Job)
# =============================
def _extract_verbatim_window_from_match(page_text_orig: str, m: re.Match, pad: int = 240) -> str:
    if not page_text_orig or not m:
        return ""
    start = max(0, m.start() - pad)
    end = min(len(page_text_orig), m.end() + pad)
    return page_text_orig[start:end].strip()


def _find_first_match_quote(page_text_orig: str, pattern: re.Pattern, anchor: str = "") -> str:
    if not page_text_orig:
        return ""
    m = pattern.search(page_text_orig)
    if not m:
        return ""
    ctx = _extract_verbatim_window_from_match(page_text_orig, m, pad=240)
    anch = (anchor or (m.group(0) or "").strip()) or " "
    return _smart_trim_quote(ctx, anch, 22)


def _title_seniority_hint(final_job_title: str, page_text_orig: str) -> Tuple[SeniorityLabel, str, str]:
    title_l = (final_job_title or "").lower().strip()

    if re.search(r"\b(intern|internship)\b", title_l):
        q = _find_first_match_quote(page_text_orig, re.compile(r"\b(intern|internship)\b", re.IGNORECASE), "intern")
        return "intern", q, "title_hint_intern"
    if re.search(r"\b(junior|jr\.?|graduate|grad)\b", title_l):
        q = _find_first_match_quote(
            page_text_orig, re.compile(r"\b(junior|jr\.?|graduate|grad)\b", re.IGNORECASE), "junior"
        )
        return "junior", q, "title_hint_junior"
    if re.search(r"\b(senior|sr\.?)\b", title_l):
        q = _find_first_match_quote(page_text_orig, re.compile(r"\b(senior|sr\.?)\b", re.IGNORECASE), "senior")
        return "senior", q, "title_hint_senior"
    if re.search(r"\b(lead|staff|principal)\b", title_l):
        q = _find_first_match_quote(page_text_orig, re.compile(r"\b(lead|staff|principal)\b", re.IGNORECASE), "lead")
        return "lead", q, "title_hint_lead"

    return "unknown", "", "no_title_hint"


def _revised_seniority_decision(
    *,
    llm_label: SeniorityLabel,
    llm_evidence_ok: bool,
    page_text_flat_lower: str,
    page_text_orig: str,
    final_job_title: str,
) -> Tuple[SeniorityLabel, str, bool, str, Dict[str, bool]]:
    signals: Dict[str, bool] = {
        "has_junior_growth": RE_JUNIOR_GROWTH.search(page_text_flat_lower) is not None,
        "has_strong_senior_only": RE_STRONG_SENIOR_ONLY.search(page_text_flat_lower) is not None,
        "has_generic_resp": RE_SENIOR_RESPONSIBILITY.search(page_text_flat_lower) is not None,
    }

    title_hint_label, title_hint_quote, title_hint_reason = _title_seniority_hint(final_job_title, page_text_orig)
    signals["has_title_hint"] = title_hint_label != "unknown"

    if title_hint_label != "unknown":
        if title_hint_label != llm_label:
            return title_hint_label, title_hint_quote, True, title_hint_reason, signals
        return llm_label, "", False, "keep_llm_title_agrees", signals

    if llm_label in _ALLOWED_SENIORITY and llm_label != "unknown":
        if llm_label in ("senior", "lead") and signals["has_junior_growth"]:
            q = _find_first_match_quote(page_text_orig, RE_JUNIOR_GROWTH, "learn")
            return "junior", q, True, "downgrade_senior_due_to_growth_cues", signals

        if llm_label in ("intern", "junior", "mid") and signals["has_strong_senior_only"]:
            q = _find_first_match_quote(page_text_orig, RE_STRONG_SENIOR_ONLY, "technical direction")
            if re.search(r"\b(staff|principal|lead)\b", page_text_flat_lower):
                return "lead", q, True, "upgrade_due_to_strong_senior_only", signals
            return "senior", q, True, "upgrade_due_to_strong_senior_only", signals

        if llm_evidence_ok:
            return llm_label, "", False, "keep_llm_evidence_ok", signals
        return llm_label, "", False, "llm_no_evidence_fallback", signals

    if signals["has_strong_senior_only"]:
        q = _find_first_match_quote(page_text_orig, RE_STRONG_SENIOR_ONLY, "technical direction")
        if re.search(r"\b(staff|principal|lead)\b", page_text_flat_lower):
            return "lead", q, True, "heuristic_strong_senior_only", signals
        return "senior", q, True, "heuristic_strong_senior_only", signals

    if signals["has_junior_growth"]:
        q = _find_first_match_quote(page_text_orig, RE_JUNIOR_GROWTH, "learn")
        return "junior", q, True, "heuristic_junior_growth", signals

    if signals["has_generic_resp"]:
        q = _find_first_match_quote(page_text_orig, RE_SENIOR_RESPONSIBILITY, "architecture")
        return "mid", q, True, "heuristic_generic_responsibility", signals

    return "unknown", "", False, "unknown", signals


# =============================
# Candidate seniority helpers (apprentice=0.5)
# =============================
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DATE_RANGE_RE = re.compile(
    r"\b(?P<start_m>[A-Za-z]{3,9})\s*(?P<start_y>\d{4})\s*[-–—]\s*(?P<end_m>[A-Za-z]{3,9}|present|current)\s*(?P<end_y>\d{4})?",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(r"\b(?P<start_y>\d{4})\s*[-–—]\s*(?P<end_y>\d{4}|present|current)\b", re.IGNORECASE)
_TITLE_LEVEL_RE = re.compile(
    r"\b(apprentice|intern|junior|jr\.?|mid|intermediate|senior|sr\.?|lead|principal|staff|manager|director|vp)\b",
    re.IGNORECASE,
)


def _title_level_from_text(title: str) -> Optional[str]:
    t = _normalize_whitespace(title).lower()
    if not t:
        return None
    if re.search(r"\b(apprentice)\b", t):
        return "apprentice"
    if re.search(r"\b(intern|internship)\b", t):
        return "intern"
    if re.search(r"\b(junior|jr\.?|graduate|entry[- ]level|grad)\b", t):
        return "junior"
    if re.search(r"\b(mid|intermediate)\b", t):
        return "mid"
    if re.search(r"\b(senior|sr\.?)\b", t):
        return "senior"
    if re.search(r"\b(lead|principal|staff|manager|director|vp|architect)\b", t):
        return "lead"
    return None


def _level_num(label: str) -> float:
    return {
        "intern": 0.0,
        "apprentice": 0.5,
        "junior": 1.0,
        "junior_to_mid": 1.5,
        "mid": 2.0,
        "senior": 3.0,
        "lead": 4.0,
    }.get(label, 2.0)


def _months_ago_from_year(year: int) -> int:
    now = time.gmtime()
    return max(0, (now.tm_year - year) * 12 + max(0, now.tm_mon - 1))


_EXPERIENCE_FAMILY_TERMS: Dict[str, Tuple[str, ...]] = {
    "tech_swe": (
        "aws", "azure", "gcp", "cloud", "devops", "engineer", "engineering", "software",
        "python", "javascript", "api", "docker", "terraform", "kubernetes", "linux", "sre",
        "infrastructure", "network", "iam", "cloudformation",
    ),
    "tech_data": (
        "ai", "ml", "machine learning", "data", "analytics", "sql", "etl", "pipeline", "python",
        "sagemaker", "bedrock", "llm", "rag", "model", "cloud", "aws", "azure", "gcp",
    ),
    "finance": (
        "finance", "financial", "accountant", "accounting", "commercial analyst", "budget",
        "forecast", "reconciliation", "tax", "audit", "ledger", "reporting",
    ),
    "sales": ("sales", "account management", "quota", "pipeline", "customer", "client"),
    "marketing": ("marketing", "campaign", "brand", "growth", "acquisition", "content"),
    "ops": ("operations", "logistics", "supply", "workflow", "process", "coordination"),
    "hr": ("human resources", "recruitment", "talent", "people operations", "hiring", "onboarding"),
    "legal": ("legal", "contract", "compliance", "regulatory", "policy", "counsel"),
    "healthcare": ("clinical", "patient", "nursing", "medical", "healthcare", "hospital"),
    "education": ("teaching", "teacher", "curriculum", "student", "education", "school"),
}

_MID_RESPONSIBILITY_RE = re.compile(
    r"\b(troubleshoot|investigat|root[- ]cause|implement|develop|build|deliver|guide|"
    r"recommend|optimis|optimiz|automate|resolve|own|manage|coordinate)\w*\b",
    re.IGNORECASE,
)
_SENIOR_RESPONSIBILITY_RE = re.compile(
    r"\b(architect|end[- ]to[- ]end|technical leadership|set direction|drive standards|"
    r"mentor|led a team|lead a team|system design|design authority|strategy|roadmap|"
    r"cross[- ]functional alignment|multi[- ]team)\b",
    re.IGNORECASE,
)


def _experience_relevance_score(block_text: str, job_family: str, *, title: str = "") -> int:
    if not block_text:
        return 0
    if not job_family or job_family == "general":
        return 1
    low = block_text.lower()
    title_low = (title or "").lower()
    terms = _EXPERIENCE_FAMILY_TERMS.get(job_family, ())

    def contains(term: str) -> bool:
        if " " in term:
            return term in low
        return re.search(rf"\b{re.escape(term)}\b", low) is not None

    # Financial modelling/reporting should not become data/AI experience merely
    # because it contains generic words such as "model" or "analytics".
    if job_family == "tech_data":
        finance_markers = ("accountant", "accounting", "financial", "budget", "forecast", "tax", "reconciliation")
        tech_core = ("python", "sql", "etl", "pipeline", "sagemaker", "bedrock", "llm", "rag", "machine learning", "cloud", "aws", "azure", "gcp")
        if any(contains(term) for term in finance_markers) and not any(contains(term) for term in tech_core):
            return 0

    # A resume section can run into the next section when PDF line breaks are
    # imperfect.  Treat an explicitly finance/accounting role as non-technical
    # even when the captured body happens to contain words from a later section.
    if job_family in ("tech_swe", "tech_data"):
        finance_title_markers = (
            "finance", "financial", "accountant", "accounting", "commercial analyst",
            "budget", "tax", "audit",
        )
        if any(marker in title_low for marker in finance_title_markers):
            return 0

    return sum(1 for term in terms if contains(term))


def _experience_title(lines: List[str], idx: int, match_start: int) -> str:
    # Handles both "Role | Company - JUN 2023 – JUN 2024" and a date on the next line.
    inline = lines[idx][:match_start].strip().rstrip("-|–—").strip()
    if inline:
        return inline
    for back in range(1, 4):
        if idx - back >= 0:
            candidate = lines[idx - back].strip()
            if candidate and len(candidate.split()) <= 14:
                return candidate
    return ""


def _experience_block_text(lines: List[str], idx: int, title: str) -> str:
    body: List[str] = [title] if title else []
    for line in lines[idx : min(len(lines), idx + 14)]:
        stripped = line.strip()
        if not stripped:
            continue
        if body and (stripped != lines[idx].strip()) and (_DATE_RANGE_RE.search(stripped) or _YEAR_RANGE_RE.search(stripped)):
            break
        body.append(stripped)
    return " ".join(body)


def _parse_experience_blocks(text: str, *, job_family: str = "general") -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    entries: List[Dict[str, Any]] = []
    for idx, ln in enumerate(lines):
        if not ln:
            continue

        m = _DATE_RANGE_RE.search(ln)
        if m:
            start_m = _MONTHS.get(m.group("start_m").lower(), 1)
            start_y = int(m.group("start_y"))
            end_m_raw = (m.group("end_m") or "").lower()
            end_y_raw = m.group("end_y")
            if end_m_raw in ("present", "current"):
                end_m = time.gmtime().tm_mon
                end_y = time.gmtime().tm_year
            else:
                end_m = _MONTHS.get(end_m_raw, start_m)
                end_y = int(end_y_raw) if end_y_raw and end_y_raw.isdigit() else start_y
            duration_months = max(1, (end_y - start_y) * 12 + (end_m - start_m))
            months_ago = max(0, (time.gmtime().tm_year - end_y) * 12 + (time.gmtime().tm_mon - end_m))

            title_line = _experience_title(lines, idx, m.start())
            block_text = _experience_block_text(lines, idx, title_line)
            level = _title_level_from_text(title_line) or _title_level_from_text(ln)
            entries.append(
                {
                    "title": title_line or ln,
                    "level": level,
                    "level_num": _level_num(level) if level else None,
                    "duration_months": duration_months,
                    "months_ago": months_ago,
                    "relevance_score": _experience_relevance_score(
                        block_text, job_family, title=title_line or ln
                    ),
                    "mid_signal_count": len(_MID_RESPONSIBILITY_RE.findall(block_text)),
                    "senior_signal_count": len(_SENIOR_RESPONSIBILITY_RE.findall(block_text)),
                }
            )
            continue

        m2 = _YEAR_RANGE_RE.search(ln)
        if m2:
            start_y = int(m2.group("start_y"))
            end_raw = (m2.group("end_y") or "").lower()
            end_y = time.gmtime().tm_year if end_raw in ("present", "current") else int(end_raw)
            duration_months = max(1, (end_y - start_y) * 12)
            months_ago = _months_ago_from_year(end_y)
            title_line = _experience_title(lines, idx, m2.start())
            block_text = _experience_block_text(lines, idx, title_line)
            level = _title_level_from_text(title_line) or _title_level_from_text(ln)
            entries.append(
                {
                    "title": title_line or ln,
                    "level": level,
                    "level_num": _level_num(level) if level else None,
                    "duration_months": duration_months,
                    "months_ago": months_ago,
                    "relevance_score": _experience_relevance_score(
                        block_text, job_family, title=title_line or ln
                    ),
                    "mid_signal_count": len(_MID_RESPONSIBILITY_RE.findall(block_text)),
                    "senior_signal_count": len(_SENIOR_RESPONSIBILITY_RE.findall(block_text)),
                }
            )
    return entries


def _weighted_median_level(entries: List[Dict[str, Any]]) -> Optional[str]:
    if not entries:
        return None
    weights = []
    for e in entries:
        months_ago = int(e.get("months_ago", 0))
        w = pow(2.71828, -months_ago / 24.0)
        weights.append(w)
        e["weight"] = w
    total = sum(weights)
    if total <= 0:
        return None
    entries_sorted = sorted(entries, key=lambda x: x["level_num"])
    acc = 0.0
    for e in entries_sorted:
        acc += e.get("weight", 0.0)
        if acc / total >= 0.5:
            return e["level"]
    return entries_sorted[-1]["level"]


def _derive_candidate_level_from_experience(
    text: str, *, job_family: str = "general"
) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
    entries = _parse_experience_blocks(text, job_family=job_family)
    if not entries:
        return None, "no_experience_blocks", []

    relevant_entries = [e for e in entries if int(e.get("relevance_score", 0)) > 0]
    if not relevant_entries:
        return None, f"no_relevant_experience_for_{job_family}", []

    explicit_entries = [e for e in relevant_entries if e.get("level")]

    def _most_recent_level(max_months: int) -> Optional[str]:
        recent = [e for e in explicit_entries if int(e.get("months_ago", 999)) <= max_months]
        if not recent:
            return None
        recent.sort(key=lambda x: (int(x.get("months_ago", 999)), -int(x.get("duration_months", 0))))
        return recent[0]["level"]

    recent_level_12 = _most_recent_level(12)
    recent_level_24 = _most_recent_level(24)

    overall = recent_level_12 or recent_level_24 or _weighted_median_level(explicit_entries)
    total_months = sum(int(e.get("duration_months", 0)) for e in relevant_entries)
    total_years = total_months / 12.0

    recent_entries = [e for e in explicit_entries if int(e.get("months_ago", 999)) <= 12]
    recent_levels = {e["level"] for e in recent_entries if e.get("level")}
    has_recent_senior = any(l in ("senior", "lead") for l in recent_levels)
    has_recent_juniorish = any(l in ("apprentice", "intern", "junior") for l in recent_levels)

    if recent_level_12:
        reason = "recent_explicit_12m"
    elif recent_level_24:
        reason = "recent_explicit_24m"
    elif overall:
        reason = "weighted_median"
    else:
        mid_signals = sum(int(e.get("mid_signal_count", 0)) for e in relevant_entries)
        senior_signals = sum(int(e.get("senior_signal_count", 0)) for e in relevant_entries)
        if total_months < 12:
            overall = "junior"
        elif total_months < 36:
            overall = "junior_to_mid"
        elif senior_signals >= 2 and total_years >= 4:
            overall = "senior"
        else:
            overall = "mid"
        reason = (
            f"relevant_experience_{total_months}m_mid_signals_{mid_signals}"
            f"_senior_signals_{senior_signals}"
        )

    if has_recent_juniorish and not has_recent_senior:
        recent_juniorish_entries = [
            e for e in recent_entries if e.get("level") in ("apprentice", "intern", "junior")
        ]
        recent_juniorish_months = max(
            (int(e.get("duration_months", 0)) for e in recent_juniorish_entries), default=0
        )
        recent_mid_signals = sum(
            int(e.get("mid_signal_count", 0)) for e in recent_juniorish_entries
        )
        if overall in ("apprentice", "intern") and (
            recent_juniorish_months >= 18 or recent_mid_signals >= 2
        ):
            # The entry title still matters, but it should not permanently pin
            # someone to apprentice/intern after sustained hands-on experience.
            overall = "junior"
            reason = "apprenticeship_progressed_by_experience"
        elif overall in ("mid", "senior", "lead"):
            overall = "junior"
            reason = "recent_junior_title_cap"
        else:
            reason = "recent_junior_title"
    elif has_recent_senior:
        if overall in ("intern", "apprentice", "junior"):
            overall = "mid"
            reason = "recent_upgrade_senior"

    top_entries = sorted(
        relevant_entries,
        key=lambda x: (int(x.get("relevance_score", 0)), -int(x.get("months_ago", 999))),
        reverse=True,
    )[:2]
    return overall, reason, top_entries


def _infer_candidate_level_from_title(title: str) -> Tuple[str, Optional[float], str]:
    t = _normalize_whitespace(title).lower()
    if not t:
        return "unknown", None, "empty_title"

    if RE_APPRENTICE.search(t):
        return "apprentice", 0.5, "title_apprentice"

    if re.search(r"\b(intern|internship)\b", t):
        return "intern", 0.0, "title_intern"

    if re.search(r"\b(junior|jr\.?|entry[- ]level|graduate|grad)\b", t):
        return "junior", 1.0, "title_junior"

    if re.search(r"\b(mid|intermediate)\b", t):
        return "mid", 2.0, "title_mid"

    if re.search(r"\b(senior|sr\.?)\b", t):
        return "senior", 3.0, "title_senior"

    if re.search(r"\b(lead|staff|principal)\b", t):
        return "lead", 4.0, "title_lead"

    return "unknown", None, "no_match"


# =============================
# Main entry
# =============================
def analyze_v3(
    *,
    page_text: str,
    title: Optional[str] = None,
    candidate_profile: Optional[Dict[str, Any]] = None,
    candidate_resume_text: Optional[str] = None,
    output_language: str = "en",
) -> AnalyzeIRv3:
    if not (page_text or "").strip():
        raise ValueError("Empty page text")

    if title and str(title).strip().lower() in ["string", "undefined", "null", "none", ""]:
        title = None

    page_text_orig = (page_text or "").strip()
    page_text_flat_orig = _normalize_whitespace(page_text_orig)
    page_text_flat_lower = page_text_flat_orig.lower()
    page_text_lines_lower = _normalize_preserve_newlines(page_text_orig).lower()
    page_text_lines_flat_lower = _normalize_whitespace(page_text_lines_lower)

    raw = _extract_with_gemini_v3(
        page_text=page_text_orig,
        title=title,
        output_language=output_language,
        candidate_profile=candidate_profile,
    )

    job = raw.get("job") or {}

    explicit_title = _sanitize_field(
        job.get("explicit_title")
        or job.get("job_title")
        or job.get("title")
        or raw.get("explicit_title")
        or raw.get("job_title")
        or raw.get("title"),
        default=None,
    )
    std_title = _sanitize_field(job.get("standardized_title") or raw.get("standardized_title"), default=None)
    final_job_title = _refine_job_title(explicit_title, std_title, title)
    if final_job_title == "Unknown Role":
        fallback_title = _extract_job_title_from_text(page_text_orig)
        if fallback_title:
            final_job_title = fallback_title

    company = _sanitize_field(job.get("company") or raw.get("company"), default=None)
    if not company:
        company = _sanitize_field(raw.get("company"), default="Unknown") or "Unknown"
    location = _sanitize_field(job.get("location") or raw.get("location"), default="Unknown") or "Unknown"

    j_sen_raw = _norm_seniority_label(
        job.get("job_seniority") or raw.get("job_seniority") or raw.get("job_seniority_signal")
    )
    j_sen_evidence = _normalize_whitespace(
        str(job.get("job_seniority_evidence") or raw.get("job_seniority_evidence") or "")
    )

    llm_evidence_ok = _validate_quote(j_sen_evidence, page_text_flat_lower)
    if not llm_evidence_ok:
        j_sen_evidence = ""

    llm_label: SeniorityLabel = j_sen_raw if j_sen_raw in _ALLOWED_SENIORITY else "unknown"  # type: ignore

    final_label, inferred_quote, override_applied, override_reason, signals = _revised_seniority_decision(
        llm_label=llm_label,
        llm_evidence_ok=llm_evidence_ok,
        page_text_flat_lower=page_text_flat_lower,
        page_text_orig=page_text_orig,
        final_job_title=final_job_title,
    )

    job_seniority_signal: SeniorityLabel = final_label

    if llm_evidence_ok and (j_sen_evidence or ""):
        job_seniority_evidence_final = j_sen_evidence
    else:
        job_seniority_evidence_final = ""
        if inferred_quote and _validate_quote(inferred_quote, page_text_flat_lower):
            job_seniority_evidence_final = inferred_quote

    try:
        profile = CandidateProfile.model_validate(candidate_profile or {})
    except Exception as exc:
        raise ValueError(f"Invalid Candidate Profile: {exc}") from exc
    candidate_skills = [canon_tool(s) or s for s in profile.candidate_skills]
    candidate_evidence_by_id = {
        claim.evidence_id: claim
        for claim in profile.evidence_claims
        if claim.evidence_id and claim.resume_quote
    }
    candidate_seniority_signal = profile.candidate_seniority_signal or "unknown"
    candidate_seniority_normalized = candidate_seniority_signal
    candidate_seniority_numeric = _level_num(candidate_seniority_signal)
    candidate_seniority_reason = profile.seniority_reason or "candidate_profile"
    candidate_title_signal = profile.roles[0].title if profile.roles else "unknown"

    # Restore the previous job-family-aware seniority behavior without sending
    # the resume to the JD Gemini request. A single global career level is
    # misleading for career changers: finance experience, for example, should
    # not make someone overqualified for a junior engineering role.
    candidate_job_family = _detect_job_family(page_text)
    exp_level, exp_reason, top_entries = _derive_candidate_level_from_experience(
        (candidate_resume_text or "").strip(), job_family=candidate_job_family
    )
    if exp_level:
        candidate_seniority_normalized = exp_level
        candidate_seniority_signal = exp_level
        candidate_seniority_numeric = _level_num(exp_level)
        candidate_seniority_reason = f"job_family_experience:{exp_reason}"
        raw.setdefault("_debug_meta", {})
        raw["_debug_meta"]["candidate_seniority_job_family"] = candidate_job_family
        raw["_debug_meta"]["candidate_seniority_exp_reason"] = exp_reason
        raw["_debug_meta"]["candidate_seniority_top_experiences"] = [
            {
                "title": entry.get("title"),
                "level": entry.get("level"),
                "months_ago": entry.get("months_ago"),
                "duration_months": entry.get("duration_months"),
                "relevance_score": entry.get("relevance_score"),
            }
            for entry in top_entries
        ]

    domain_requirements: List[DomainRequirement] = []
    raw_domains = _safe_list(job.get("domain_requirements") or raw.get("domain_requirements"))

    for d in raw_domains:
        if not isinstance(d, dict):
            continue

        name = _normalize_whitespace(str(d.get("name") or ""))
        if not name or len(name) > 80:
            continue

        raw_imp = d.get("importance")
        evidence_quote = _normalize_whitespace(str(d.get("evidence_quote") or ""))

        quote_for_signals = evidence_quote
        if not _validate_quote(evidence_quote, page_text_flat_lower):
            if _validate_soft_anchor(name, page_text_flat_lower):
                ctx_orig = _extract_context_window_orig(name, page_text_flat_lower, page_text_flat_orig)
                if ctx_orig:
                    ctx_lower = _normalize_whitespace(ctx_orig).lower()
                    if _window_has_requirement_signals(ctx_lower):
                        evidence_quote = _smart_trim_quote(ctx_orig, name, 22)
                        quote_for_signals = ctx_orig

            if not _validate_quote(evidence_quote, page_text_flat_lower):
                continue

        broad_ctx_lower = _context_window_around_quote(
            page_text_lines_lower, page_text_lines_flat_lower, page_text_flat_lower, evidence_quote
        )
        tight_ctx_lower = _tight_context_around_quote(page_text_flat_lower, evidence_quote, pad=120)

        section_kind = _guess_section_kind(broad_ctx_lower or tight_ctx_lower)

        facet = _infer_facet(name, evidence_quote, broad_ctx_lower)

        imp = _derive_domain_importance_from_context(
            domain_name=name,
            evidence_quote=quote_for_signals or evidence_quote,
            tight_context_lower=tight_ctx_lower,
            raw_imp=raw_imp,
            facet=facet,
            section_kind=section_kind,
        )
        if imp not in ("must", "should", "nice", "nice_to_have", "unknown"):
            imp = "should"
        if imp == "unknown":
            imp = "should"

        requirement_type = _normalize_whitespace(str(d.get("requirement_type") or "capability")).lower()
        if requirement_type not in {
            "capability",
            "tool",
            "credential",
            "education",
            "experience",
            "work_condition",
            "responsibility",
            "other",
        }:
            requirement_type = "other"
        alternatives = []
        for value in _safe_list(d.get("alternatives")):
            alternative = _normalize_whitespace(str(value))
            if (
                alternative
                and len(alternative) <= 100
                and _validate_soft_anchor(alternative, page_text_flat_lower)
            ):
                alternatives.append(alternative)
        alternatives = _dedupe_keep_order(alternatives)[:10]
        match_status = _normalize_whitespace(str(d.get("match_status") or "unknown")).lower()
        if match_status not in {"matched", "partial", "missing", "unknown"}:
            match_status = "unknown"
        resume_evidence_ids = [
            str(value).strip()
            for value in _safe_list(d.get("resume_evidence_ids"))
            if str(value).strip() in candidate_evidence_by_id
        ]
        resume_evidence_ids = _dedupe_keep_order(resume_evidence_ids)[:5]
        if match_status in {"matched", "partial"} and not resume_evidence_ids:
            match_status = "unknown"
        if match_status in {"missing", "unknown"}:
            resume_evidence_ids = []
        jd_evidence_ids = _dedupe_keep_order(
            [
                str(value).strip()
                for value in _safe_list(d.get("jd_evidence_ids"))
                if str(value).strip()
            ]
        )[:3]
        match_reason = _normalize_whitespace(str(d.get("match_reason") or ""))[:500] or None

        examples: List[ToolEvidence] = []
        raw_examples = _safe_list(d.get("examples"))

        for ex in raw_examples:
            if not isinstance(ex, dict):
                continue
            tname = _normalize_whitespace(str(ex.get("name") or ""))
            tname = canon_tool(tname)
            if not tname or len(tname) > 80:
                continue

            ex_quote = _normalize_whitespace(str(ex.get("evidence_quote") or ""))
            ex_raw_imp = ex.get("importance")

            # Examples cite their parent requirement passage. A separate model-
            # generated quote is unnecessary when the literal name is present
            # in the backend-owned JD source.
            if not ex_quote and tname.lower() in evidence_quote.lower():
                ex_quote = evidence_quote

            ex_imp = _norm_importance(ex_raw_imp)
            if ex_imp in ("unknown", "nice"):
                ex_imp = "should"

            if ex_quote:
                if not _validate_quote(ex_quote, page_text_flat_lower):
                    if _validate_soft_anchor(tname, page_text_flat_lower):
                        ctx2 = _extract_context_window_orig(tname, page_text_flat_lower, page_text_flat_orig)
                        if ctx2:
                            ctx2_lower = _normalize_whitespace(ctx2).lower()
                            if _window_has_requirement_signals(ctx2_lower):
                                ex_quote = _smart_trim_quote(ctx2, tname, 22)
                    if ex_quote and not _validate_quote(ex_quote, page_text_flat_lower):
                        ex_quote = ""

            # Enforce strict tool-must
            if ex_imp == "must":
                ex_ctx = _tight_context_around_quote(page_text_flat_lower, ex_quote, pad=120) if ex_quote else ""
                if not _is_explicit_tool_must(ex_quote, ex_ctx, tname, section_kind=section_kind):
                    ex_imp = "should"

            if ex_quote:
                ex_ctx2 = _tight_context_around_quote(page_text_flat_lower, ex_quote, pad=120)
                if RE_EXAMPLE.search(ex_ctx2) or RE_PREFERENCE.search(ex_ctx2):
                    ex_imp = "should"

            examples.append(ToolEvidence(name=tname, importance=ex_imp, evidence_quote=ex_quote or None))

        ex_final = _merge_examples_keep_strongest(examples)
        if len(ex_final) > 14:
            ex_final = ex_final[:14]

        # Promote tools from "specifically ..." when domain is MUST and context is requirements-like
        ex_final = _promote_tools_from_specifically(
            domain_importance=imp,
            section_kind=section_kind,
            domain_evidence_quote=evidence_quote,
            page_text_flat_lower=page_text_flat_lower,
            page_text_flat_orig=page_text_flat_orig,
            examples=ex_final,
        )

        anchors = _build_domain_anchors(
            domain_name=name,
            evidence_quote=evidence_quote,
            examples=ex_final,
            page_text_flat_lower=page_text_flat_lower,
            max_anchors=10,
        )

        evidence_level_raw = str(d.get("evidence_level") or "").lower()
        if evidence_level_raw not in ("exact", "anchored", "weak", "none"):
            evidence_level_raw = ""
        domain_requirements.append(
            DomainRequirement(
                name=name,
                importance=imp,
                requirement_type=requirement_type,
                alternatives=alternatives,
                match_status=match_status,
                jd_evidence_ids=jd_evidence_ids,
                resume_evidence_ids=resume_evidence_ids,
                match_reason=match_reason,
                evidence_quote=evidence_quote,
                facet=facet,
                anchors=anchors,
                examples=ex_final,
                # Phase 3: Pass domain_id so Adjudicator can use it
                domain_id=d.get("domain_id"),
                evidence_level=evidence_level_raw or "none",
                evidence_status=d.get("evidence_status") if isinstance(d.get("evidence_status"), str) else None,
                span_meta=d.get("span_meta") if isinstance(d.get("span_meta"), dict) else None,
            )
        )

    # Collapse reversed duplicates of the same literal OR choice group. For
    # example, "CDK alternatives=[CloudFormation]" and the reversed item are
    # one hiring requirement, not two independent matches.
    choice_groups: Dict[Tuple[str, ...], DomainRequirement] = {}
    collapsed_requirements: List[DomainRequirement] = []
    normalization_merge_count = 0
    match_rank = {"unknown": 0, "missing": 1, "partial": 2, "matched": 3}
    for dr in domain_requirements:
        choice_labels = _dedupe_keep_order([dr.name] + list(dr.alternatives or []))
        choice_key = tuple(sorted(_normalize_whitespace(x).lower() for x in choice_labels if x))
        if len(choice_key) < 2:
            collapsed_requirements.append(dr)
            continue
        existing_choice = choice_groups.get(choice_key)
        if existing_choice is None:
            choice_groups[choice_key] = dr
            collapsed_requirements.append(dr)
            continue
        normalization_merge_count += 1
        display_choices = _dedupe_keep_order(
            [existing_choice.name]
            + list(existing_choice.alternatives or [])
            + [dr.name]
            + list(dr.alternatives or [])
        )
        existing_choice.name = " / ".join(display_choices)
        existing_choice.alternatives = display_choices
        existing_choice.resume_evidence_ids = _dedupe_keep_order(
            list(existing_choice.resume_evidence_ids or []) + list(dr.resume_evidence_ids or [])
        )[:5]
        existing_choice.jd_evidence_ids = _dedupe_keep_order(
            list(existing_choice.jd_evidence_ids or []) + list(dr.jd_evidence_ids or [])
        )[:3]
        if match_rank.get(dr.match_status, 0) > match_rank.get(existing_choice.match_status, 0):
            existing_choice.match_status = dr.match_status
            existing_choice.match_reason = dr.match_reason
        if _IMPORTANCE_RANK.get(dr.importance, 0) > _IMPORTANCE_RANK.get(existing_choice.importance, 0):
            existing_choice.importance = dr.importance
    domain_requirements = collapsed_requirements

    # Dedupe exact normalized names and merge supporting metadata.
    dom_map: Dict[str, DomainRequirement] = {}
    facet_rank = {"people": 3, "technical": 2, "process": 1, "unknown": 0}

    for dr in domain_requirements:
        k = dr.name.lower()
        if k not in dom_map:
            dom_map[k] = dr
            continue
        normalization_merge_count += 1

        exist = dom_map[k]

        if _IMPORTANCE_RANK.get(dr.importance, 0) > _IMPORTANCE_RANK.get(exist.importance, 0):
            exist.importance = dr.importance

        if facet_rank.get(dr.facet, 0) > facet_rank.get(exist.facet, 0):
            exist.facet = dr.facet

        if len(dr.evidence_quote or "") > len(exist.evidence_quote or ""):
            exist.evidence_quote = dr.evidence_quote

        merged_examples = _merge_examples_keep_strongest((exist.examples or []) + (dr.examples or []))
        exist.examples = merged_examples[:18]

        a_seen = {a.lower() for a in (exist.anchors or [])}
        for a in (dr.anchors or []):
            if a.lower() not in a_seen:
                exist.anchors.append(a)
                a_seen.add(a.lower())
        if len(exist.anchors) > 12:
            exist.anchors = exist.anchors[:12]

        # Phase 3: Merge domain_id if present in new item but missing in existing
        if not exist.domain_id and dr.domain_id:
            exist.domain_id = dr.domain_id
        exist.resume_evidence_ids = _dedupe_keep_order(
            list(exist.resume_evidence_ids or []) + list(dr.resume_evidence_ids or [])
        )[:5]
        exist.jd_evidence_ids = _dedupe_keep_order(
            list(exist.jd_evidence_ids or []) + list(dr.jd_evidence_ids or [])
        )[:3]
        if match_rank.get(dr.match_status, 0) > match_rank.get(exist.match_status, 0):
            exist.match_status = dr.match_status
            exist.match_reason = dr.match_reason

        dom_map[k] = exist

    domain_requirements = list(dom_map.values())

    # Adjudicate against catalog: canonicalize, evidence verify, must gate
    try:
        job_family = _detect_job_family(page_text)
        catalog = _get_domain_catalog(job_family)
        domain_requirements = adjudicate_domains(domain_requirements, jd_text=page_text, catalog=catalog)
    except Exception:
        pass

    domain_requirements.sort(
        key=lambda x: (
            -_IMPORTANCE_RANK.get(x.importance, 0),
            -len(x.evidence_quote or ""),
            x.name.lower(),
        )
    )
    raw_requirement_count = len(raw_domains)
    verified_requirement_count = len(domain_requirements)
    expected_requirement_count = max(
        0,
        raw_requirement_count - normalization_merge_count,
    )
    normalization_incomplete = bool(
        expected_requirement_count
        and verified_requirement_count < expected_requirement_count
    )
    evidence_hints = job.get("evidence_hints") or raw.get("evidence_hints") or {}
    if not isinstance(evidence_hints, dict):
        evidence_hints = {}

    ownership_raw = job.get("ownership_and_scope") or raw.get("ownership_and_scope") or {}
    try:
        ownership_and_scope = ExtractedOwnership.model_validate(ownership_raw)
    except Exception:
        ownership_and_scope = ExtractedOwnership()
    recommendation_raw = (
        job.get("application_recommendation")
        or raw.get("application_recommendation")
        or {}
    )
    try:
        application_recommendation = ApplicationRecommendation.model_validate(recommendation_raw)
    except Exception:
        application_recommendation = ApplicationRecommendation()

    try:
        raw.setdefault("_debug_meta", {})
        raw["_debug_meta"]["engine_version"] = "v3"
        raw["_debug_meta"]["job_seniority_signal"] = job_seniority_signal
        raw["_debug_meta"]["job_seniority_evidence_final"] = job_seniority_evidence_final
        raw["_debug_meta"]["seniority_override_applied"] = override_applied
        raw["_debug_meta"]["seniority_override_reason"] = override_reason
        raw["_debug_meta"]["llm_job_seniority"] = llm_label
        raw["_debug_meta"]["llm_job_seniority_evidence_ok"] = llm_evidence_ok
        raw["_debug_meta"]["seniority_signals"] = signals

        raw["_debug_meta"]["candidate_title_signal"] = candidate_title_signal
        raw["_debug_meta"]["candidate_seniority_signal"] = candidate_seniority_signal
        raw["_debug_meta"]["candidate_seniority_normalized"] = candidate_seniority_normalized
        raw["_debug_meta"]["candidate_seniority_numeric"] = candidate_seniority_numeric
        raw["_debug_meta"]["candidate_seniority_reason"] = candidate_seniority_reason

        # helpful: anchors stats to validate 方案B 稳定性
        raw["_debug_meta"]["anchors_per_domain"] = {d.name: (d.anchors or []) for d in domain_requirements}
        raw["_debug_meta"]["raw_requirement_count"] = raw_requirement_count
        raw["_debug_meta"]["verified_requirement_count"] = verified_requirement_count
        raw["_debug_meta"]["normalization_merge_count"] = normalization_merge_count
        raw["_debug_meta"]["expected_requirement_count"] = expected_requirement_count
        raw["_debug_meta"]["requirement_retention_ratio"] = round(
            verified_requirement_count / float(raw_requirement_count),
            3,
        ) if raw_requirement_count else 0.0
        raw["_debug_meta"]["normalization_incomplete"] = normalization_incomplete
    except Exception:
        pass

    jd_tools = extract_tools_from_jd(page_text)
    if jd_tools:
        raw.setdefault("job", {})
        if isinstance(raw.get("job"), dict):
            raw["job"]["tools_in_jd"] = jd_tools

    return AnalyzeIRv3(
        job_title=final_job_title,
        company=company,
        location=location,
        job_seniority_signal=job_seniority_signal,
        candidate_seniority_signal=candidate_seniority_signal,
        candidate_skills=candidate_skills,
        candidate_evidence_claims=profile.evidence_claims,
        domain_requirements=domain_requirements,
        tools_in_jd=jd_tools,
        ownership_and_scope=ownership_and_scope,
        application_recommendation=application_recommendation,
        evidence_hints=evidence_hints,
        raw_llm_json=raw,
        analysis_status=(
            "success"
            if domain_requirements and not normalization_incomplete
            else "degraded"
        ),
        model_used=JOB_ANALYSIS_MODEL,
    )


__all__ = ["JOB_ANALYSIS_MODEL", "analyze_v3"]
