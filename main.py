# main.py
from __future__ import annotations

import os
import json
import pathlib
import hashlib
import re
import inspect
import base64
import secrets
import sqlite3
import time
import traceback
import html
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
from backend.notion.writer import create_notion_page

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# -----------------------------
# 1) Config & Imports
# -----------------------------
load_dotenv(BASE_DIR / ".env")

# v3 (new)
_V3_AVAILABLE = True
try:
    from backend.llm.analyze_v3 import analyze_v3 as kairos_analyze_v3
    from backend.scoring.scoring_engine_v3 import score_ir_v3, score_to_public_dict
except Exception:
    _V3_AVAILABLE = False
    kairos_analyze_v3 = None  # type: ignore
    score_ir_v3 = None  # type: ignore
    score_to_public_dict = None  # type: ignore

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
NOTION_OAUTH_DB = DATA_DIR / "notion_oauth.db"

OAUTH_STATE_TTL_SECONDS = 10 * 60
OAUTH_SESSION_TTL_SECONDS = 20 * 60
AUTH_CODE_TTL_SECONDS = 5 * 60
MAX_RESUME_BYTES = 10 * 1024 * 1024
MAX_RESUME_TEXT_CHARS = 200_000

# v3 cache dir
V3_CACHE_DIR = DATA_DIR / "v3_cache"
V3_CACHE_DIR.mkdir(exist_ok=True)

NOTION_API_TOKEN = os.getenv("NOTION_API_TOKEN", "").strip()
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "").strip()
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28").strip()
NOTION_CLIENT_ID = os.getenv("NOTION_CLIENT_ID", "").strip()
NOTION_CLIENT_SECRET = os.getenv("NOTION_CLIENT_SECRET", "").strip()
NOTION_REDIRECT_URI = os.getenv("NOTION_REDIRECT_URI", "http://localhost:8000/notion/callback").strip()

# Optional: allow bypassing cache
KAIROS_DISABLE_CACHE = os.getenv("KAIROS_DISABLE_CACHE", "").strip().lower() in ("1", "true", "yes")

# Debug-dump switch (optional)
KAIROS_DEBUG_DUMP = os.getenv("KAIROS_DEBUG_DUMP", "").strip().lower() in ("1", "true", "yes")
KAIROS_RUN_ID = os.environ.get("KAIROS_RUN_ID", "").strip() or "run1"


# -----------------------------
# 2) Pydantic Models (API Layer)
# -----------------------------
class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    page_text: str = Field(..., min_length=50, max_length=50_000)
    extraction_meta: Optional[Dict[str, Any]] = None
    output_language: Optional[str] = "en"
    use_v3: bool = False

    # ✅ NEW: run_id can be set from web UI request body
    run_id: Optional[str] = None
    notion_session_token: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_title: str
    company: str
    location: str

    # legacy top-level fields (keep stable)
    should_apply: str
    final_score: int
    distance_score: int
    seniority_gap: str
    cap: int

    summary: str
    required_skills: List[str]
    missing_skills: List[str]

    raw_json: Dict[str, Any]
    notion_url: Optional[str] = None

    # Phase 2 Fields (v3)
    uncapped_score: Optional[int] = None
    blocking_reason: Optional[str] = None
    ownership: Optional[Dict[str, Any]] = None


class AuthExchangeRequest(BaseModel):
    code: str = Field(..., min_length=8)


# -----------------------------
# 3) Helpers
# -----------------------------
def _get_in(d: Any, path: List[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        return str(x)
    except Exception:
        return default


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in items:
        k = (s or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _clamp_text(s: str, n: int = 300) -> str:
    s = (s or "").strip()
    return s[:n] if s else ""


def _init_notion_oauth_db() -> None:
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_sessions (
                session_id TEXT PRIMARY KEY,
                access_token TEXT NOT NULL,
                created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_codes (
                code TEXT PRIMARY KEY,
                user_token TEXT NOT NULL,
                created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notion_users (
                user_token TEXT PRIMARY KEY,
                access_token TEXT NOT NULL,
                database_id TEXT,
                database_name TEXT,
                created_at INTEGER,
                resume_text TEXT,
                resume_filename TEXT,
                resume_uploaded_at INTEGER
            )
            """
        )
        now = int(time.time())
        conn.execute("DELETE FROM oauth_states WHERE created_at < ?", (now - OAUTH_STATE_TTL_SECONDS,))
        conn.execute("DELETE FROM oauth_sessions WHERE created_at < ?", (now - OAUTH_SESSION_TTL_SECONDS,))
        conn.execute("DELETE FROM auth_codes WHERE created_at < ?", (now - AUTH_CODE_TTL_SECONDS,))
        conn.commit()


def _store_oauth_state(state: str) -> None:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO oauth_states (state, created_at) VALUES (?, ?)",
            (state, int(time.time())),
        )
        conn.commit()


def _consume_oauth_state(state: str) -> bool:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        cur = conn.execute("SELECT created_at FROM oauth_states WHERE state = ?", (state,))
        row = cur.fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
        conn.commit()
        return int(row[0] or 0) >= int(time.time()) - OAUTH_STATE_TTL_SECONDS


def _store_oauth_session(session_id: str, access_token: str) -> None:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO oauth_sessions (session_id, access_token, created_at) VALUES (?, ?, ?)",
            (session_id, access_token, int(time.time())),
        )
        conn.commit()


def _lookup_oauth_session(session_id: str) -> Optional[str]:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        cur = conn.execute(
            "SELECT access_token, created_at FROM oauth_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if int(row[1] or 0) < int(time.time()) - OAUTH_SESSION_TTL_SECONDS:
            conn.execute("DELETE FROM oauth_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return None
        return row[0]


def _delete_oauth_session(session_id: str) -> None:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute("DELETE FROM oauth_sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def _store_notion_user(
    *, user_token: str, access_token: str, database_id: str, database_name: str
) -> None:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO notion_users
            (user_token, access_token, database_id, database_name, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_token, access_token, database_id, database_name, int(time.time())),
        )
        conn.commit()


def _get_notion_user(user_token: str) -> Optional[Dict[str, Any]]:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        cur = conn.execute(
            """
            SELECT user_token, access_token, database_id, database_name,
                   resume_text, resume_filename, resume_uploaded_at
            FROM notion_users WHERE user_token = ?
            """,
            (user_token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "user_token": row[0],
            "access_token": row[1],
            "database_id": row[2],
            "database_name": row[3],
            "resume_text": row[4],
            "resume_filename": row[5],
            "resume_uploaded_at": row[6],
        }


def _store_resume_for_user(user_token: str, resume_text: str, filename: str) -> None:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute(
            """
            UPDATE notion_users
            SET resume_text = ?, resume_filename = ?, resume_uploaded_at = ?
            WHERE user_token = ?
            """,
            (resume_text, filename, int(time.time()), user_token),
        )
        conn.commit()


def _store_auth_code(code: str, user_token: str) -> None:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO auth_codes (code, user_token, created_at) VALUES (?, ?, ?)",
            (code, user_token, int(time.time())),
        )
        conn.commit()


def _consume_auth_code(code: str) -> Optional[str]:
    _init_notion_oauth_db()
    with sqlite3.connect(NOTION_OAUTH_DB) as conn:
        cur = conn.execute("SELECT user_token, created_at FROM auth_codes WHERE code = ?", (code,))
        row = cur.fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM auth_codes WHERE code = ?", (code,))
        conn.commit()
        if int(row[1] or 0) < int(time.time()) - AUTH_CODE_TTL_SECONDS:
            return None
        return row[0]


def _get_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _sha256_text(*parts: str) -> str:
    """
    Stable cache key from effective inputs.
    Include resume_text + page_text + title + output_language + contract version string.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()


def _canonical_jd_for_cache(text: str) -> str:
    """Remove harmless LinkedIn UI churn without changing JD wording."""
    dynamic = re.compile(
        r"^(?:reposted|posted)\s+\d+\s+(?:minute|hour|day|week)s?\s+ago$|"
        r"^\d[\d,]*\s+(?:applicant|applicants)$|^over\s+\d[\d,]*\s+applicants$|"
        r"^\d[\d,]*\s+people\s+clicked\s+apply$",
        re.IGNORECASE,
    )
    lines = []
    for raw_line in (text or "").replace("\r", "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line and not dynamic.match(line):
            lines.append(line)
    return "\n".join(lines)


def _contract_is_reliable(obj: Any) -> bool:
    quality = obj.get("analysis_quality") if isinstance(obj, dict) else None
    return bool(isinstance(quality, dict) and quality.get("score_reliable"))


def _read_json_file(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_file(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        # best-effort; never crash main flow
        pass


def _cache_looks_like_contract(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    contract = obj.get("contract")
    return isinstance(contract, dict) and contract.get("name") == "kairos_v3_public"


def _job_key_from_req(req: AnalyzeRequest) -> str:
    """
    Stable job_key for grouping multiple runs of the same JD.
    Prefer url hash; else title hash; else page_text hash (short).
    """
    if req.url and req.url.strip():
        src = req.url.strip()
    elif req.title and req.title.strip():
        src = req.title.strip()
    else:
        src = (req.page_text or "")[:2000]  # avoid huge
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# 3.0) Debug dump for diff
# -----------------------------
def _extract_diff_slices(final_result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract exactly the 3 items you want to diff across runs.
      1) raw_llm_json.job.domain_requirements (names list)
      2) required_skills (final used domains list)
      3) missing.groups / missing_skills_labels (and low_conf flags)
    """
    raw_json = final_result_dict.get("raw_json") or {}
    score_contract = _get_in(raw_json, ["score"], {})  # v3 stores contract here

    # (1) raw_llm_json.job.domain_requirements names
    raw_llm_json = _get_in(raw_json, ["raw_llm_json"], None)
    domain_reqs = _as_list(_get_in(raw_llm_json, ["job", "domain_requirements"], []))
    domain_req_names: List[str] = []
    for dr in domain_reqs:
        if isinstance(dr, dict):
            nm = _safe_str(dr.get("name", "")).strip()
            if nm:
                domain_req_names.append(nm)
    domain_req_names = _dedupe_keep_order(domain_req_names)

    # (2) required_skills
    required_skills = final_result_dict.get("required_skills")
    if not isinstance(required_skills, list):
        required_skills = []
    required_skills = [_safe_str(x).strip() for x in required_skills if _safe_str(x).strip()]
    required_skills = _dedupe_keep_order(required_skills)

    # (3) missing.groups / missing_skills_labels
    missing_groups = _get_in(score_contract, ["missing", "groups"], [])
    if not isinstance(missing_groups, list):
        missing_groups = []
    missing_skills_labels = _get_in(score_contract, ["missing_skills_labels"], {})
    if not isinstance(missing_skills_labels, dict):
        missing_skills_labels = {}

    any_recommended_low_conf = any(
        (_safe_str(v).strip().lower() == "recommended_low_conf") for v in missing_skills_labels.values()
    )

    return {
        "slice_1_raw_llm_domain_requirements_names": domain_req_names,
        "slice_2_required_skills_final": required_skills,
        "slice_3_missing_groups": missing_groups,
        "slice_3_missing_skills_labels": missing_skills_labels,
        "slice_3_has_recommended_low_conf": any_recommended_low_conf,
    }


def dump_kairos_debug(
    *,
    out_dir: str,
    run_id: str,
    job_key: str,
    result: Dict[str, Any],
) -> str:
    """
    Dump:
      - full result JSON: {out_dir}/{job_key}/{run_id}.json
      - slices JSON:     {out_dir}/{job_key}/{run_id}.slices.json
    Returns the full-result path (string).
    """
    base = pathlib.Path(out_dir) / job_key
    base.mkdir(parents=True, exist_ok=True)

    full_path = base / f"{run_id}.json"
    slices_path = base / f"{run_id}.slices.json"

    _write_json_file(full_path, result)
    _write_json_file(slices_path, _extract_diff_slices(result))

    return str(full_path)


# -----------------------------
# 3.1) Evidence helpers
# -----------------------------
_ACTION_VERBS = (
    "built", "implemented", "developed", "deployed", "shipped", "owned", "optimized", "migrated",
    "designed", "created", "integrated", "automated", "reduced", "improved", "led",
    "maintained", "instrumented", "monitored", "debugged", "troubleshot"
)
_SKILLS_LINE_PREFIXES = (
    "skills", "tool", "tools", "tech", "tech stack", "stack", "technologies", "technology",
    "languages", "frameworks", "platforms"
)
_COMMON_TOOL_TOKENS = [
    "git", "github", "gitlab", "bitbucket",
    "docker", "kubernetes", "k8s",
    "aws", "lambda", "s3", "iam", "cloudwatch", "api gateway",
    "gcp", "azure",
    "fastapi", "flask", "django",
    "postgres", "mysql", "dynamodb",
    "pandas", "numpy",
]


def _looks_like_evidence_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) < 20:
        return False

    lower = s.lower()

    # ignore obvious section titles
    if re.fullmatch(r"[a-z\s/&\-]{3,40}:?", lower) and (":" in s or s.isupper()):
        return False

    starts_bullet = bool(re.match(r"^(\-|\•|\*|\u2022)\s+", s))
    starts_verb = any(lower.startswith(v + " ") for v in _ACTION_VERBS)

    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", s))
    has_unitish = bool(re.search(r"\b(ms|s|sec|secs|%|users?|req/s|rps|days?|weeks?|months?)\b", lower))

    has_ops = any(k in lower for k in ("lambda", "api gateway", "s3", "cloudwatch", "docker", "fastapi", "flask", "rag", "embeddings"))

    return (starts_bullet or starts_verb) and (has_number or has_unitish or has_ops)


def _looks_like_skills_line(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) < 8:
        return False
    lower = s.lower()

    if any(lower.startswith(p + ":") for p in _SKILLS_LINE_PREFIXES):
        return True

    if ("|" in s or "," in s) and any(tok in lower for tok in ("git", "github", "docker", "kubernetes", "aws", "gcp", "azure")):
        return len(re.findall(r"[,\|]", s)) >= 2

    return False


def _extract_explicit_skill_tokens(text: str) -> List[str]:
    t = (text or "").lower()
    found: List[str] = []
    for tok in _COMMON_TOOL_TOKENS:
        if " " in tok:
            if tok in t:
                found.append(tok)
        else:
            if re.search(rf"\b{re.escape(tok)}\b", t):
                found.append(tok)

    norm_map = {
        "k8s": "kubernetes",
        "api gateway": "aws api gateway",
    }
    out: List[str] = []
    seen = set()
    for x in found:
        y = norm_map.get(x, x)
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def _extract_candidate_evidence_text(resume_text: str, *, max_lines: int = 60) -> str:
    text = (resume_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    picked: List[str] = []

    for ln in lines:
        if _looks_like_evidence_line(ln) or _looks_like_skills_line(ln):
            cleaned = re.sub(r"^(\-|\•|\*|\u2022)\s+", "", ln).strip()
            cleaned = re.sub(
                r"^(skills|tools|tech stack|stack|technologies|languages|frameworks)\s*:\s*",
                "",
                cleaned,
                flags=re.I,
            ).strip()
            if cleaned:
                picked.append(cleaned)

    picked = _dedupe_keep_order(picked)[:max_lines]

    if len(picked) < 8:
        lower_text = text.lower()
        idx = lower_text.find("projects")
        if idx != -1:
            snippet = text[idx: idx + 2000]
            snippet_lines = [re.sub(r"^(\-|\•|\*|\u2022)\s+", "", l).strip() for l in snippet.split("\n")]
            snippet_lines = [l for l in snippet_lines if len(l) >= 20]
            picked.extend([l for l in snippet_lines if l and l.lower() not in {p.lower() for p in picked}][:20])

    picked = _dedupe_keep_order(picked)[:max_lines]

    if len(picked) < 5:
        return text.strip()

    return "Candidate evidence (selected lines):\n- " + "\n- ".join(picked)


def _call_score_ir_v3_with_optional_candidate_text(ir3: Any, candidate_text: str) -> Any:
    if score_ir_v3 is None:
        raise RuntimeError("score_ir_v3 not available")

    try:
        sig = inspect.signature(score_ir_v3)  # type: ignore[arg-type]
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    for key in ("candidate_text", "candidate_profile_text", "user_profile_text", "user_profile", "candidate_profile"):
        if key in params:
            try:
                return score_ir_v3(ir3, **{key: candidate_text})  # type: ignore[misc]
            except TypeError:
                pass

    return score_ir_v3(ir3)  # type: ignore[misc]


# -----------------------------
# 5) FastAPI App
# -----------------------------
app = FastAPI(title="Kairos Engine", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "operational", "backend": "connected", "v3_available": _V3_AVAILABLE}


@app.get("/notion/start")
def notion_oauth_start():
    if not NOTION_CLIENT_ID or not NOTION_CLIENT_SECRET:
        raise HTTPException(status_code=400, detail="Notion OAuth not configured.")

    state = secrets.token_urlsafe(16)
    _store_oauth_state(state)
    query = urlencode(
        {
            "client_id": NOTION_CLIENT_ID,
            "response_type": "code",
            "owner": "user",
            "redirect_uri": NOTION_REDIRECT_URI,
            "state": state,
        }
    )
    url = f"https://api.notion.com/v1/oauth/authorize?{query}"
    return RedirectResponse(url=url, status_code=302)


@app.get("/{locale}/notion/start")
def notion_oauth_start_locale(locale: str):
    return RedirectResponse(url="/notion/start", status_code=302)


@app.get("/notion/callback")
def notion_oauth_callback(code: str = "", state: str = ""):
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing OAuth code/state.")
    if not _consume_oauth_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")

    auth = base64.b64encode(f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}".encode("utf-8")).decode("utf-8")
    token_res = requests.post(
        "https://api.notion.com/v1/oauth/token",
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        json={"grant_type": "authorization_code", "code": code, "redirect_uri": NOTION_REDIRECT_URI},
        timeout=10,
    )
    if not token_res.ok:
        raise HTTPException(status_code=400, detail=f"Notion OAuth failed: {token_res.text}")

    payload = token_res.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Notion OAuth missing access_token.")

    workspace_id = payload.get("workspace_id")
    owner = payload.get("owner") or {}
    owner_id = owner.get("user", {}).get("id") if isinstance(owner, dict) else None
    bot_id = payload.get("bot_id")

    session_id = secrets.token_urlsafe(24)
    _store_oauth_session(session_id, access_token)

    url = f"/notion/select?session={session_id}"
    return RedirectResponse(url=url, status_code=302)


@app.get("/notion/select")
def notion_select_database(session: str = "", db: str = "", name: str = ""):
    if not session:
        raise HTTPException(status_code=400, detail="Missing session token.")
    token = _lookup_oauth_session(session)
    if not token:
        raise HTTPException(status_code=400, detail="Invalid session token.")

    if db:
        db_name = (name or "").strip() or "Selected Database"
        user_token = secrets.token_urlsafe(24)
        _store_notion_user(user_token=user_token, access_token=token, database_id=db, database_name=db_name)
        auth_code = secrets.token_urlsafe(16)
        _store_auth_code(auth_code, user_token)
        _delete_oauth_session(session)
        return RedirectResponse(url=f"/notion/done?code={auth_code}", status_code=302)

    res = requests.post(
        "https://api.notion.com/v1/search",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        json={"filter": {"property": "object", "value": "database"}, "page_size": 50},
        timeout=10,
    )
    if not res.ok:
        raise HTTPException(status_code=400, detail=f"Notion search failed: {res.text}")

    data = res.json()
    results = data.get("results") or []
    items: List[str] = []
    for d in results:
        title_parts = (d.get("title") or [])
        title = "".join([p.get("plain_text", "") for p in title_parts]) if isinstance(title_parts, list) else ""
        title = title.strip() or "Untitled Database"
        db_id = d.get("id") or ""
        if not db_id:
            continue
        name_q = urlencode({"name": title})
        link = f"/notion/select?session={session}&db={db_id}&{name_q}"
        items.append(f'<li><a href="{html.escape(link, quote=True)}">{html.escape(title)}</a></li>')

    page_html = "<html><body>"
    page_html += "<h3>Select a Notion Database</h3><ul>"
    page_html += "".join(items) if items else "<li>No databases found. Wait a few seconds, then refresh this page.</li>"
    page_html += "</ul><button onclick=\"location.reload()\">Refresh</button></body></html>"
    return HTMLResponse(content=page_html)

@app.get("/notion/done")
def notion_done(code: str = ""):
    if not code:
        raise HTTPException(status_code=400, detail="Missing auth code.")
    html = (
        "<html><body>"
        "<h3>Notion Connected</h3>"
        "<p>You can close this tab and return to the extension.</p>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.post("/auth/exchange")
def auth_exchange(req: AuthExchangeRequest):
    user_token = _consume_auth_code(req.code.strip())
    if not user_token:
        raise HTTPException(status_code=400, detail="Invalid or expired auth code.")
    user = _get_notion_user(user_token)
    if not user:
        raise HTTPException(status_code=400, detail="User not found for auth code.")
    return {
        "user_token": user_token,
        "database_id": user.get("database_id"),
        "database_name": user.get("database_name"),
    }


@app.get("/status")
def status(authorization: Optional[str] = Header(None)):
    user_token = _get_bearer_token(authorization)
    if not user_token:
        raise HTTPException(status_code=401, detail="Notion auth required.")
    user = _get_notion_user(user_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user token.")
    return {
        "notion_connected": True,
        "database_id": user.get("database_id"),
        "database_name": user.get("database_name"),
        "resume_present": bool(user.get("resume_text")),
        "resume_filename": user.get("resume_filename"),
        "resume_uploaded_at": user.get("resume_uploaded_at"),
    }


@app.post("/resume/upload")
def upload_resume(file: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    user_token = _get_bearer_token(authorization)
    if not user_token:
        raise HTTPException(status_code=401, detail="Notion auth required.")
    user = _get_notion_user(user_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user token.")

    content = file.file.read()
    if len(content) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="Resume file is too large. Maximum size is 10 MB.")
    filename = (file.filename or "resume").strip()
    text = ""
    content_type = (file.content_type or "").lower()
    if filename.lower().endswith(".txt") or content_type.startswith("text/"):
        text = content.decode("utf-8", errors="ignore").strip()
    elif filename.lower().endswith(".pdf") or content_type == "application/pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise HTTPException(status_code=400, detail="PDF parsing requires pypdf.") from exc
        import io

        reader = PdfReader(io.BytesIO(content))
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts).strip()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use TXT or PDF.")

    if len(text) < 30:
        raise HTTPException(status_code=400, detail="Resume text too short.")
    if len(text) > MAX_RESUME_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Resume text is too large.")

    _store_resume_for_user(user_token, text, filename)
    return {"status": "saved", "resume_filename": filename, "resume_uploaded_at": int(time.time())}


@app.post("/analyze_and_save", response_model=AnalyzeResponse)
def analyze_and_save(req: AnalyzeRequest, authorization: Optional[str] = Header(None)):
    user_token = _get_bearer_token(authorization)
    if not user_token:
        raise HTTPException(status_code=401, detail="Notion auth required.")
    user = _get_notion_user(user_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user token.")
    resume_text = (user.get("resume_text") or "").strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="Please upload resume first.")

    use_v3 = bool(req.use_v3)
    if use_v3 and not _V3_AVAILABLE:
        raise HTTPException(status_code=400, detail="v3 not available: ensure v3 modules are importable.")

    # ✅ run_id: request body overrides env
    run_id = (_safe_str(req.run_id, "").strip() or _safe_str(KAIROS_RUN_ID, "").strip() or "run1")

    try:
        # -----------------------------
        # v3 path
        # -----------------------------
        if use_v3:
            request_started = time.perf_counter()
            timings_ms: Dict[str, int] = {}
            candidate_evidence_text = _extract_candidate_evidence_text(resume_text)
            extra_tokens = _extract_explicit_skill_tokens(resume_text)
            if extra_tokens:
                candidate_evidence_text += "\n\nHeuristic skills tokens (tool bonus only): " + ", ".join(extra_tokens)

            cache_key = _sha256_text(
                "kairos_v3_public:1.4",
                _canonical_jd_for_cache(req.page_text or ""),
                resume_text or "",
                req.title or "",
                req.output_language or "en",
            )
            cache_path = V3_CACHE_DIR / f"{cache_key}.json"

            cache_hit = False
            api_data: Dict[str, Any] = {}

            cache_started = time.perf_counter()
            if (not KAIROS_DISABLE_CACHE) and cache_path.exists():
                cached = _read_json_file(cache_path)
                if _cache_looks_like_contract(cached) and _contract_is_reliable(cached):
                    api_data = cached  # type: ignore[assignment]
                    cache_hit = True
            timings_ms["cache_lookup"] = int(round((time.perf_counter() - cache_started) * 1000))

            ir3 = None
            score_result = None

            if not cache_hit:
                llm_started = time.perf_counter()
                ir3 = kairos_analyze_v3(  # type: ignore[misc]
                    page_text=req.page_text,
                    title=req.title,
                    user_profile=resume_text,
                    output_language=req.output_language or "en",
                )
                timings_ms["llm_and_normalization"] = int(round((time.perf_counter() - llm_started) * 1000))

                scoring_started = time.perf_counter()
                score_result = _call_score_ir_v3_with_optional_candidate_text(ir3, candidate_evidence_text)
                api_data = score_to_public_dict(score_result)  # type: ignore[misc]
                timings_ms["scoring"] = int(round((time.perf_counter() - scoring_started) * 1000))

                if not _contract_is_reliable(api_data):
                    error_detail = "The model did not return enough verified JD requirements."
                    hints = getattr(ir3, "evidence_hints", None)
                    if isinstance(hints, dict) and hints.get("error"):
                        error_detail = _clamp_text(_safe_str(hints.get("error")), 500)
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Analysis was incomplete and was not written to Notion. "
                            f"Reason: {error_detail}"
                        ),
                    )

                if isinstance(api_data, dict):
                    api_data["_job_meta"] = {
                        "job_title": getattr(ir3, "job_title", None),
                        "company": req.company or getattr(ir3, "company", None),
                        "location": req.location or getattr(ir3, "location", None),
                        "title_input": req.title,
                        "url_input": req.url,
                        "extraction": req.extraction_meta or {},
                    }
                    api_data["_candidate_meta"] = {
                        "evidence_preview": _clamp_text(candidate_evidence_text, 800),
                        "evidence_mode": "extracted_lines",
                    }
                    if _contract_is_reliable(api_data):
                        _write_json_file(cache_path, api_data)

            if cache_hit:
                meta = api_data.get("_job_meta") if isinstance(api_data, dict) else None
                meta_missing = not (
                    isinstance(meta, dict)
                    and any((meta.get("job_title"), meta.get("company"), meta.get("location")))
                )
                if meta_missing:
                    try:
                        ir_meta = kairos_analyze_v3(  # type: ignore[misc]
                            page_text=req.page_text,
                            title=req.title,
                            user_profile=resume_text,
                            output_language=req.output_language or "en",
                        )
                        if isinstance(api_data, dict):
                            api_data["_job_meta"] = {
                                "job_title": getattr(ir_meta, "job_title", None),
                                "company": getattr(ir_meta, "company", None),
                                "location": getattr(ir_meta, "location", None),
                                "title_input": req.title,
                                "url_input": req.url,
                                "meta_refreshed_from_old_cache": True,
                            }
                            _write_json_file(cache_path, api_data)
                    except Exception:
                        pass

            score_obj = api_data.get("score") if isinstance(api_data, dict) else None
            if not isinstance(score_obj, dict):
                score_obj = {}

            should_apply = score_obj.get("should_apply") or "No"
            final_score = score_obj.get("final_score", 0)
            distance_score = score_obj.get("distance_score", 0)
            seniority_gap = api_data.get("seniority_gap") or score_obj.get("seniority_gap") or "unknown"
            cap = api_data.get("cap")
            if cap is None:
                cap = score_obj.get("cap", 100)
            summary = api_data.get("summary") or score_obj.get("summary") or "No summary available"

            required_skills = api_data.get("required_skills")
            missing_skills = api_data.get("missing_skills")

            if required_skills is None:
                required_skills = _as_list(api_data.get("required_domains", [])) + _as_list(
                    api_data.get("required_tool_must", [])
                )

            if missing_skills is None:
                must_missing = _as_list(api_data.get("missing_domains_must", [])) + _as_list(
                    api_data.get("missing_tool_must", [])
                )
                missing_skills = must_missing if must_missing else _as_list(api_data.get("missing_domains_all", []))[:6]

            # ✅ v3 debug normally lives inside score_obj["debug"]
            contract_debug = score_obj.get("debug")
            if not isinstance(contract_debug, dict):
                contract_debug = {}

            job_title = "Unknown"
            company = "Unknown"
            location = "Unknown"

            meta = api_data.get("_job_meta") if isinstance(api_data, dict) else None
            if isinstance(meta, dict):
                job_title = _safe_str(meta.get("job_title") or job_title, job_title)
                company = _safe_str(meta.get("company") or company, company)
                location = _safe_str(meta.get("location") or location, location)

            if ir3 is not None:
                job_title = _safe_str(getattr(ir3, "job_title", None) or job_title, job_title)
                company = _safe_str(getattr(ir3, "company", None) or company, company)
                location = _safe_str(getattr(ir3, "location", None) or location, location)

            if (job_title or "").strip().lower() == "unknown" and (req.title or "").strip():
                job_title = req.title.strip()
            if (company or "").strip().lower() == "unknown" and (req.company or "").strip():
                company = req.company.strip()
            if (location or "").strip().lower() == "unknown" and (req.location or "").strip():
                location = req.location.strip()

            # Phase 2: Extract Ownership Signals from IR (if available)
            ownership_data = None
            # Phase 2: Extract Ownership Signals from IR (if available)
            ownership_data = None

            # Helper to get field from dict or object
            def _get(obj, key, default=None):
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)

            os_obj = _get(ir3, "ownership_and_scope")
            if os_obj:
                ownership_data = {
                    "ownership_present": _get(_get(os_obj, "ownership"), "present", False),
                    "scope_level": _get(_get(os_obj, "scope"), "level", "unknown"),
                    "leadership_present": _get(_get(os_obj, "leadership"), "present", False),
                }

            response_obj = AnalyzeResponse(
                job_title=job_title,
                company=company,
                location=location,
                should_apply=_safe_str(should_apply, "No"),
                final_score=int(final_score or 0),
                distance_score=int(distance_score or 0),
                seniority_gap=_safe_str(seniority_gap, "unknown"),
                cap=int(cap or 100),
                summary=_safe_str(summary, "No summary available"),
                required_skills=[_safe_str(x) for x in _as_list(required_skills)],
                missing_skills=[_safe_str(x) for x in _as_list(missing_skills)],

                # Phase 2 Fields
                uncapped_score=(
                    int(api_data.get("score", {}).get("uncapped_score") or api_data.get("score", {}).get("distance_score") or 0)
                    if use_v3
                    else None
                ),
                blocking_reason=api_data.get("score", {}).get("blocking_reason") if use_v3 else None,
                ownership=ownership_data,

                raw_json={
                    "engine_version": "v3",
                    "cache": {
                        "key": cache_key,
                        "hit": cache_hit,
                        "path": str(cache_path),
                        "disabled": bool(KAIROS_DISABLE_CACHE),
                    },
                    "timings_ms": timings_ms,
                    "input": {
                        "jd_chars": len(req.page_text or ""),
                        "resume_chars": len(resume_text or ""),
                        "extraction": req.extraction_meta or {},
                    },
                    "score": api_data,
                    "debug": contract_debug,
                    "raw_llm_json": getattr(ir3, "raw_llm_json", None) if ir3 is not None else None,
                    "candidate_evidence_preview": _clamp_text(candidate_evidence_text, 800),
                },
            )

            # -----------------------------
            # ✅ DEBUG DUMP: dump final result + diff slices
            # -----------------------------
            if KAIROS_DEBUG_DUMP:
                final_result_dict = response_obj.model_dump()
                job_key = _job_key_from_req(req)
                debug_path = dump_kairos_debug(
                    out_dir=str(BASE_DIR / "debug_runs"),
                    run_id=run_id,
                    job_key=job_key,
                    result=final_result_dict,
                )
                print(f"[debug] dumped to: {debug_path}")

            notion_token = user.get("access_token") or NOTION_API_TOKEN
            notion_database_id = user.get("database_id") or NOTION_DATABASE_ID
            if not notion_database_id:
                raise HTTPException(status_code=400, detail="No Notion database selected.")

            notion_started = time.perf_counter()
            notion_url = create_notion_page(
                req,
                response_obj,
                score_result,
                public_contract=api_data,
                notion_token=notion_token,
                notion_database_id=notion_database_id,
                notion_version=NOTION_VERSION,
            )
            timings_ms["notion_write"] = int(round((time.perf_counter() - notion_started) * 1000))
            timings_ms["total"] = int(round((time.perf_counter() - request_started) * 1000))
            response_obj.raw_json["timings_ms"] = timings_ms
            if not notion_url:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Analysis completed and was cached locally, but Notion did not confirm page creation. "
                        "Check Notion for an existing page before retrying."
                    ),
                )
            response_obj.notion_url = notion_url
            return response_obj

        raise HTTPException(status_code=400, detail="v2 is disabled; set use_v3=true")

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Engine Error traceback:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")
