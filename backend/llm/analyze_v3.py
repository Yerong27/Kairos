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
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from backend.ir.schema_v3 import (
    AnalyzeIRv3,
    DomainRequirement,
    ToolEvidence,
    Importance,
    SeniorityLabel,
    DomainFacet,
    ExtractedOwnership,
)
from backend.ir.domain_catalog import load_domain_catalogs, adjudicate_domains
from backend.ir.canonicalize import canon_domain, canon_tool

# =============================
# Gemini setup
# =============================
_DEFAULT_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
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


_JD_TOOL_PREFIXES = (
    "devsecops:",
    "cloud:",
    "architectures:",
    "desirable:",
)

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
    "llm",
    "llms",
    "agentic system",
    "agentic systems",
}


_JD_TOOL_ALIAS_MAP = {
    "actions": ["GitHub Actions"],
    "pipelines": ["CI/CD Pipelines"],
    "actions/pipelines": ["GitHub Actions", "CI/CD Pipelines"],
    "github": ["GitHub"],
    "gitlab": ["GitLab"],
    "azure devops": ["Azure DevOps"],
    "github/gitlab/azure devops": ["GitHub", "GitLab", "Azure DevOps"],
    "iac": ["IaC"],
    "terraform": ["Terraform"],
    "sast": ["SAST"],
    "dast": ["DAST"],
    "sca": ["SCA"],
    "sast/dast/sca": ["SAST", "DAST", "SCA"],
    "secrets management": ["Secrets Management"],
    "rest": ["REST"],
    "grpc": ["gRPC"],
    "event driven": ["Event-driven"],
    "event-driven": ["Event-driven"],
    "microservices": ["Microservices"],
    "fastapi/fastmcp": ["FastAPI", "FastMCP"],
    "react + typescript": ["React", "TypeScript"],
    "react+typescript": ["React", "TypeScript"],
}


def _normalize_tool_token(tok: str) -> List[str]:
    t = (tok or "").strip()
    if not t:
        return []
    t = re.sub(r"[()\[\]{}]+", " ", t)
    t = re.sub(r"[;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    key = t.lower()
    if key in _JD_TOOL_IGNORE:
        return []
    if key in _JD_TOOL_ALIAS_MAP:
        return _JD_TOOL_ALIAS_MAP[key]
    if "+" in t:
        out: List[str] = []
        for part in [p.strip() for p in t.split("+") if p.strip()]:
            out.extend(_normalize_tool_token(part))
        return out
    return [t]


def extract_tools_from_jd(page_text: str) -> List[str]:
    if not (page_text or "").strip():
        return []
    tools: List[str] = []
    def _expand_tool_chunk(chunk: str) -> List[str]:
        c = (chunk or "").strip()
        if not c:
            return []
        full_key = c.lower()
        full_key = re.sub(r"\s*/\s*", "/", full_key)
        full_key = re.sub(r"\s+", " ", full_key).strip()
        if full_key in _JD_TOOL_ALIAS_MAP:
            return _JD_TOOL_ALIAS_MAP[full_key]

        base = c
        inner = ""
        if "(" in c and ")" in c:
            base = c.split("(", 1)[0].strip()
            inner = c.split("(", 1)[1].split(")", 1)[0].strip()

        items: List[str] = []
        for p in re.split(r"[|/\\+]", base):
            items.extend(_normalize_tool_token(p))
        if inner:
            items.extend(_normalize_tool_token(inner))
        return [x for x in items if x]

    for ln in page_text.splitlines():
        if not ln.strip():
            continue
        low = ln.strip().lower()
        if not any(low.startswith(p) for p in _JD_TOOL_PREFIXES):
            continue
        _, _, rest = ln.partition(":")
        if not rest.strip():
            continue
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        for p in parts:
            p2 = p.strip().strip(".:;")
            for norm in _expand_tool_chunk(p2):
                tools.append(norm)

    for ln in page_text.splitlines():
        low = ln.lower()
        if "such as" in low:
            _, _, tail = ln.partition("such as")
            if tail:
                parts = [p.strip() for p in tail.split(",") if p.strip()]
                for p in parts:
                    for norm in _expand_tool_chunk(p.strip().strip(".:;")):
                        tools.append(norm)

    # Generic pattern-based extraction from JD lines
    tool_patterns = [
        r"\b(skilled in|proficient in|experience with|hands-on experience with|familiar with|knowledge of|tools like|frameworks like|platforms like)\b",
        r"\b(languages|tech stack|stack)\b",
    ]
    tool_re = re.compile("|".join(tool_patterns), re.IGNORECASE)
    for ln in page_text.splitlines():
        if not ln.strip():
            continue
        m = tool_re.search(ln)
        if not m:
            continue
        tail = ln[m.end():].strip()
        tail = re.sub(r"^[\s:\-–—]+", "", tail)
        if not tail:
            continue
        parts = [p.strip() for p in tail.split(",") if p.strip()]
        for p in parts:
            for norm in _expand_tool_chunk(p.strip().strip(".:;")):
                tools.append(norm)

    out: List[str] = []
    seen = set()
    for t in tools:
        k = t.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t.strip())
    return out


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

_IMPORTANCE_RANK: Dict[str, int] = {"unknown": 0, "nice": 1, "should": 2, "must": 3}


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
    if len(q_norm.split()) > 22:
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
            if tail in ("must", "should", "nice", "unknown"):
                return tail  # type: ignore

    if s in ("must", "should", "nice", "unknown"):
        return s  # type: ignore

    if s in ("required", "essential", "core", "mandatory"):
        return "must"
    if s in ("preferred", "bonus", "nice to have", "plus", "desired", "advantage", "desirable"):
        return "should"

    if "must" in s or "mandatory" in s or "required" in s or "essential" in s:
        return "must"
    if "should" in s or "preferred" in s or "bonus" in s or "desirable" in s:
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
    user_profile: Optional[str],
    output_language: str,
) -> Dict[str, Any]:
    _ensure_gemini_configured()
    model = genai.GenerativeModel(_DEFAULT_MODEL)

    # Pre-load catalog for prompt context
    job_family = _detect_job_family(page_text)
    catalog = _get_domain_catalog(job_family)
    catalog_json_summary = json.dumps(
        [{"id": d.id, "label": d.label, "facet": d.facet, "aliases": d.aliases_strong + d.aliases_weak}
         for d in catalog.domains.values()],
        indent=2
    )

    prompt = f"""You are a Job Analysis Engine.

Your task: extract a *scorable* IR for ANY professional job.

GLOBAL RULES (STRICT KAIROS LOGIC):

1) CORE PHILOSOPHY (Domain vs Tools):
   - **Domains** are "Capabilities" (e.g., "Cloud Computing", "CI/CD", "Project Management").
     They determine if a candidate CAN do the job.
   - **Tools** are "Implementations" or "Examples" (e.g., "AWS", "GitHub Actions", "Jira").
     They are bonus matches only.
   - NEVER extract a specific Tool as a Domain Name (e.g., Domain must be "Cloud Platforms", NOT "AWS").

2) IMPORTANCE HIERARCHY:
   - **"must"**: HARD THRESHOLD. Missing this means the candidate is unqualified.
     Criteria: explicit "Must Have", "Required", "Minimum Qualifications" section.
     **EXCEPTION For SENIOR+ Roles**: Key strategic/leadership responsibilities (e.g., "Define architecture", "Mentor team", "Drive strategy") are MUSTs, even if listed under "Responsibilities".
   - **"should"**: Core capability but not a showstopper.
     Criteria: Standard "Responsibilities", "What you'll do" sections (unless it is a key Senior requirement).
   - **"nice_to_have"**: Bonus only.
     Criteria: "Nice to have", "Preferred", "Bonus", "Plus", "Desirable".

3) LOGICAL RULES:
   - **Domain First**: Determine importance based on the *Capability* required.
   - **Tools are Examples**: List specific tools in `examples[]`. They typically inherit importance "should" or "nice_to_have".
   - **OR Logic**: If tools are connected by OR (e.g. "Python or Java"), the *Domain* ("Programming") is "must", but the specific *Tools* must NOT be marked "must" (because neither is strictly mandatory).

3.5) TOOLS EXTRACTION (MANDATORY):
   - If the JD explicitly names tools/technologies (e.g., "Python, FastAPI/FastMCP, React + TypeScript"),
     you MUST include them under `examples[]` of the most relevant domain(s).
   - Do NOT omit explicit tools. Include at least 5 tools if the JD lists them.
   - Each tool in `examples[]` must include an evidence_quote copied from the JD.

4) COUNT & COVERAGE:
   - Aim for comprehensive coverage without duplication (typically 6–14 Domain Requirements).
   - Do NOT squash distinct capabilities just to save space.
   - Do NOT omit technical/soft skills if they are core to the role.
   - Keep evidence_summary to at most 20 words and evidence_quote to the shortest
     verbatim clause that proves the requirement.

5) EVIDENCE:
   - **evidence_summary**: Summarize the requirement context (1 sentence) for human understanding.
   - **evidence_quote**: Verbatim quote from the text.

6) SENIORITY RUBRIC (Apply Universally):
   Use these BEHAVIORAL markers to determine `job_seniority_signal`, regardless of tech stack:
   - **MID-LEVEL**: Focus on **Execution** & **Implementation**.
     Keywords: "Implement", "Support", "Maintain", "Collaborate". Scope: Tasks / Features.
   - **SENIOR**: Focus on **Design** & **System Complexity**.
     Keywords: "Design", "Architect", "Scalability", "Reliability", "End-to-end ownership". Scope: Systems / Services.
   - **LEAD / PRINCIPAL**: Focus on **Strategy** & **Influence**.
     Keywords: "Set direction", "Mentor", "Cross-functional alignment", "Drive standards", "Stakeholder management". Scope: Team / Organization.
   - **NOTE**: Many JDs don't put "Lead" in the title but describe Lead-level influence. Trust the **responsibilities** over the title.

7) OWNERSHIP & SCOPE (Critical for Precision):
   Extract these signals explicitly (Integer Levels 0-3):
   - **ownership.level_val**: 0=None, 1=Task Owner (Can do work), 2=Result Owner (Accountable), 3=System/End-to-End Owner.
   - **scope.level_val**: 0=Unknown, 1=Self only, 2=Team Impact, 3=Multi-Team/Org Impact.
   - **leadership.level_val**: 0=None, 1=Mentor/Guide, 2=Team Lead (unblock others), 3=Org Lead (Manager/Principal).

<job_text>
{page_text[:25000]}
</job_text>

<candidate_profile>
{(user_profile or "N/A")[:15000]}
</candidate_profile>

OUTPUT FORMAT (JSON ONLY):
Produce a single valid JSON object following this structure. Do NOT wrap in markdown code blocks.

{{
  "job_title": "string",
  "company": "string",
  "job_seniority_signal": "junior|mid|senior|lead|principal",
  "candidate_seniority_signal": "string",
  "domain_requirements": [
    {{
      "domain_id": "string (optional, copy 'id' from catalog if exact match)",
      "domain": "string",
      "importance": "must|should|nice_to_have",
      "evidence_summary": "string",
      "evidence_quote": "string",
      "examples": [
        {{
          "tool": "string",
          "importance": "must|should|nice_to_have",
          "evidence_quote": "string"
        }}
      ]
    }}
  ],
  "candidate_skills": ["string", "string"],
  "ownership_and_scope": {{
    "ownership": {{ "present": true, "level_val": 0, "evidence": ["string"] }},
    "scope": {{ "level_val": 0, "evidence": ["string"] }},
    "leadership": {{ "present": true, "level_val": 0, "evidence": ["string"] }}
  }},
  "evidence_hints": {{
    "years_experience": "string",
    "education_degree": "string"
  }}
}}

You have access to a Domain Catalog. PREFER using 'domain_id' from this list if the concept matches:
{catalog_json_summary}

OUTPUT LANGUAGE: {output_language}
"""

    generation_config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        # The previous 8192-token ceiling encouraged very verbose responses and
        # increased latency. The schema remains exhaustive, but summaries and
        # quotes are intentionally concise.
        max_output_tokens=6144,
        response_mime_type="application/json",
    )

    # Seniority Rules (Regex-based Fusion)
    rule_seniority = _extract_seniority_rules(page_text)

    try:
        resp = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": 30},
        )
        # Verify JSON
        parsed_ir = AnalyzeIRv3.model_validate_json(resp.text)

        # Attach raw JSON for debugging/transparency
        try:
            parsed_ir.raw_llm_json = json.loads(resp.text)
        except Exception:
            parsed_ir.raw_llm_json = {}

        # Inject Rule-based Seniority
        # We store it in a temporary field or handle it in Scoring.
        # For now, let's allow Scoring Engine to re-calculate rule seniority from raw text if needed,
        # OR we can attach it to metadata?
        # Plan says: "Seniority Cap: `seniority_final = min(max(rule_level, llm_level - 1), 3)`"
        # scoring_engine_v3 has access to page_text? No, it takes AnalyzeIRv3.
        # So we should probably inject the rule signal into the IR.
        # Let's override/augment candidate_seniority_signal or add a hidden field?
        # Actually, scoring_engine does NOT have access to page_text.
        # Let's put the rule-based determination into "job_seniority_signal" if logic allows,
        # OR better: Add `rule_seniority_level` to AnalyzeIRv3?
        # Schema change required?
        # Wait, the plan said: "Implement `extract_seniority_rules` (Regex-based) in `analyze_v3.py`".
        # It implies analyze_v3 does it.
        # But where does it store it?
        # Let's repurpose "evidence_hints" to store 'rule_seniority' for scoring engine?
        parsed_ir.evidence_hints['rule_seniority_level'] = str(rule_seniority)

        # Adjudicate against catalog (THE Deterministic Step)
        job_family = _detect_job_family(page_text)
        catalog = _get_domain_catalog(job_family)
        if isinstance(parsed_ir.raw_llm_json, dict):
            parsed_ir.raw_llm_json.setdefault("_debug_meta", {})
            parsed_ir.raw_llm_json["_debug_meta"]["job_family"] = job_family
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
    """
    Enforce strict rules to reduce MUST jitter (方案B 的前置稳定性)：
    - Responsibilities 默认 SHOULD，除非显式 required/must
    - Preference 语言 => SHOULD
    - MUST 仅在 Requirements 段或 quote/context 含强制词
    """
    base = _norm_importance(raw_imp)
    if base in ("unknown", "nice"):
        base = "should"

    q = _normalize_whitespace(evidence_quote or "").lower()
    ctx = _normalize_whitespace(tight_context_lower or "").lower()
    name_l = _normalize_whitespace(domain_name or "").lower()

    quote_has_preference = bool(RE_PREFERENCE.search(q))
    quote_has_strict = bool(RE_STRICT_QUOTE.search(q) or RE_STRICT_STRONG.search(q))
    ctx_has_strict = bool(RE_STRICT_QUOTE.search(ctx) or RE_STRICT_STRONG.search(ctx))
    ctx_has_example = bool(RE_EXAMPLE.search(ctx))

    # Preference always forces SHOULD
    if quote_has_preference or bool(RE_PREFERENCE.search(ctx)):
        return "should"

    # Responsibilities default SHOULD unless explicitly required
    if section_kind == "responsibilities" and not (quote_has_strict or ctx_has_strict):
        return "should"

    # Explicit strict language forces MUST
    if quote_has_strict:
        return "must"

    # Requirements section:
    if section_kind == "requirements":
        # If it's clearly example framing and lacks strict markers, keep SHOULD.
        if ctx_has_example and not ctx_has_strict:
            return "should"
        # In requirements section, defaulting to MUST is acceptable for domains;
        # tools will still be constrained separately.
        return "must"

    # If LLM said must but we don't see requirement section nor strict language => do NOT allow must
    if base == "must":
        if RE_PREFER_SHOULD_DOMAINS.search(name_l) or RE_PREFER_SHOULD_DOMAINS.search(q):
            return "should"
        if not ctx_has_strict:
            return "should"
        return "must"

    # Context strict markers (outside responsibilities) can promote to MUST if not example/preference
    if ctx_has_strict and not ctx_has_example and not bool(RE_PREFERENCE.search(ctx)):
        return "must"

    if RE_PREFER_SHOULD_DOMAINS.search(name_l) or RE_PREFER_SHOULD_DOMAINS.search(q):
        return "should"

    return "should"


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


def _experience_relevance_score(block_text: str, job_family: str) -> int:
    if not block_text:
        return 0
    if not job_family or job_family == "general":
        return 1
    low = block_text.lower()
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
                    "relevance_score": _experience_relevance_score(block_text, job_family),
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
                    "relevance_score": _experience_relevance_score(block_text, job_family),
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
        if overall in ("mid", "senior", "lead"):
            overall = "junior"
        reason = "recent_floor_apprentice"
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
    user_profile: Optional[str] = None,
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

    safe_profile = (user_profile or "").strip()[:15000]
    safe_profile_flat_lower = _normalize_whitespace(safe_profile).lower()

    raw = _extract_with_gemini_v3(
        page_text=page_text_orig,
        title=title,
        user_profile=safe_profile,
        output_language=output_language,
    )

    job = raw.get("job") or {}
    cand = raw.get("candidate") or {}

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

    raw_c_title = _normalize_whitespace(str(cand.get("seniority_evidence") or ""))
    candidate_title_signal = raw_c_title if raw_c_title else "unknown"

    candidate_seniority_signal = "unknown"
    candidate_seniority_normalized = "unknown"
    candidate_seniority_numeric: Optional[float] = None
    candidate_seniority_reason = "unknown"

    if raw_c_title and _validate_quote(raw_c_title, safe_profile_flat_lower) and _is_valid_title_signal(raw_c_title):
        norm_label, numeric_level, reason = _infer_candidate_level_from_title(raw_c_title)
        candidate_seniority_normalized = norm_label
        candidate_seniority_numeric = numeric_level
        candidate_seniority_reason = reason
        candidate_seniority_signal = norm_label if norm_label != "unknown" else raw_c_title
    elif raw_c_title:
        candidate_seniority_signal = raw_c_title
        candidate_seniority_reason = "title_not_validated_or_invalid_signal"

    candidate_job_family = _detect_job_family(page_text)
    exp_level, exp_reason, top_entries = _derive_candidate_level_from_experience(
        safe_profile, job_family=candidate_job_family
    )
    exp_cap = "junior" if exp_reason == "recent_floor_apprentice" else None
    if exp_level:
        try:
            raw.setdefault("_debug_meta", {})
            raw["_debug_meta"]["candidate_seniority_top_experiences"] = [
                {
                    "title": e.get("title"),
                    "level": e.get("level"),
                    "months_ago": e.get("months_ago"),
                    "duration_months": e.get("duration_months"),
                    "weight": round(float(e.get("weight", 0.0)), 4),
                    "relevance_score": e.get("relevance_score"),
                    "mid_signal_count": e.get("mid_signal_count"),
                    "senior_signal_count": e.get("senior_signal_count"),
                }
                for e in top_entries
            ]
            raw["_debug_meta"]["candidate_seniority_exp_reason"] = exp_reason
            raw["_debug_meta"]["candidate_seniority_exp_cap"] = exp_cap
            raw["_debug_meta"]["candidate_seniority_job_family"] = candidate_job_family
        except Exception:
            pass

    # Deterministic, job-family-relevant experience takes precedence over an
    # unsupported LLM label. This also supports the 1.5 junior-to-mid band.
    if exp_level:
        candidate_seniority_normalized = exp_level
        candidate_seniority_signal = exp_level
        candidate_seniority_numeric = _level_num(exp_level)
        candidate_seniority_reason = f"experience_inference:{exp_reason}"

    # Only trust an LLM seniority label when it supplied a candidate title/evidence
    # that was validated against the resume. A bare `mid` prediction is not evidence.
    if candidate_seniority_normalized == "unknown" and raw_c_title:
        raw_signal = _norm_seniority_label(raw.get("candidate_seniority_signal") or cand.get("seniority_signal"))
        if raw_signal != "unknown":
            candidate_seniority_normalized = raw_signal
            candidate_seniority_signal = raw_signal
            candidate_seniority_numeric = _level_num(raw_signal)
            candidate_seniority_reason = "raw_signal"

    if exp_cap and candidate_seniority_normalized in ("mid", "senior", "lead"):
        candidate_seniority_normalized = exp_cap
        candidate_seniority_signal = exp_cap
        candidate_seniority_numeric = _level_num(exp_cap)
        candidate_seniority_reason = "exp_floor_cap_applied"

    # _extract_with_gemini_v3 returns the validated AnalyzeIRv3 schema at the
    # top level. Keep compatibility with older cached/nested response shapes.
    candidate_skills = _clean_candidate_skill_list(cand.get("skills") or raw.get("candidate_skills"))
    candidate_skills = [canon_tool(s) or s for s in (candidate_skills or [])]

    domain_requirements: List[DomainRequirement] = []
    raw_domains = _safe_list(job.get("domain_requirements") or raw.get("domain_requirements"))
    if len(raw_domains) > 22:
        raw_domains = raw_domains[:22]

    for d in raw_domains:
        if not isinstance(d, dict):
            continue

        name = _normalize_whitespace(str(d.get("name") or ""))
        name = canon_domain(name)
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
        if imp not in ("must", "should", "nice", "unknown"):
            imp = "should"
        if imp == "unknown":
            imp = "should"

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

    # dedupe domains by name, merge best evidence/examples/anchors
    dom_map: Dict[str, DomainRequirement] = {}
    facet_rank = {"people": 3, "technical": 2, "process": 1, "unknown": 0}

    for dr in domain_requirements:
        k = dr.name.lower()
        if k not in dom_map:
            dom_map[k] = dr
            continue

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
    if len(domain_requirements) > 14:
        domain_requirements = domain_requirements[:14]

    evidence_hints = job.get("evidence_hints") or raw.get("evidence_hints") or {}
    if not isinstance(evidence_hints, dict):
        evidence_hints = {}

    ownership_raw = job.get("ownership_and_scope") or raw.get("ownership_and_scope") or {}
    try:
        ownership_and_scope = ExtractedOwnership.model_validate(ownership_raw)
    except Exception:
        ownership_and_scope = ExtractedOwnership()

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
        domain_requirements=domain_requirements,
        tools_in_jd=jd_tools,
        ownership_and_scope=ownership_and_scope,
        evidence_hints=evidence_hints,
        raw_llm_json=raw,
        analysis_status="success" if domain_requirements else "degraded",
        model_used=_DEFAULT_MODEL,
    )


__all__ = ["analyze_v3"]
