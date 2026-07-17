from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


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


def _trim_blocks_for_notion(blocks: List[dict], max_blocks: int = 100) -> List[dict]:
    if not isinstance(blocks, list):
        return []
    if len(blocks) <= max_blocks:
        return blocks

    out = list(blocks)
    for i in range(len(out) - 1, -1, -1):
        blk = out[i] if isinstance(out[i], dict) else {}
        if blk.get("type") == "toggle":
            toggle = blk.get("toggle") if isinstance(blk.get("toggle"), dict) else {}
            rich_text = toggle.get("rich_text") or []
            text = ""
            if rich_text and isinstance(rich_text, list):
                text = ((rich_text[0] or {}).get("text") or {}).get("content") or ""
            if "Raw Debug Data" in text:
                out.pop(i)
                break

    if len(out) > max_blocks:
        out = out[:max_blocks]
    return out


def _best_title(req: Any, job_title: str) -> str:
    jt = (job_title or "").strip()
    if jt and jt.lower() != "unknown":
        return jt
    t = (getattr(req, "title", None) or "").strip()
    return t if t else "Unknown"


def _clean_tag(text: str) -> str:
    return (text or "").replace(",", " ").strip()[:50]


@dataclass
class _NotionPostResult:
    url: Optional[str] = None
    status_code: Optional[int] = None
    error_kind: Optional[str] = None
    detail: str = ""


_NOTION_RETRYABLE_STATUS = {409, 429, 500, 502, 503, 504}


def _retry_delay_seconds(res: Any, attempt: int) -> float:
    if getattr(res, "status_code", None) == 429:
        try:
            return max(1.0, min(float(res.headers.get("Retry-After", "1")), 30.0))
        except (TypeError, ValueError):
            pass
    return float(min(2 ** (attempt - 1), 8))


def _post_to_notion(
    *,
    notion_token: str,
    notion_database_id: str,
    notion_version: str,
    properties: dict,
    children: list,
    max_attempts: int = 3,
) -> _NotionPostResult:
    payload = {"parent": {"database_id": notion_database_id}, "properties": properties, "children": children}
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": notion_version,
    }
    attempts = max(1, int(max_attempts))

    for attempt in range(1, attempts + 1):
        try:
            res = requests.post(
                "https://api.notion.com/v1/pages",
                headers=headers,
                json=payload,
                timeout=(10, 45),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            # A POST that times out may still have created the page server-side.
            # Blindly retrying can create duplicates, so report an unknown outcome.
            print(
                "❌ Notion transport error; write outcome is unknown. "
                f"Check Notion before retrying: {exc}"
            )
            return _NotionPostResult(error_kind="transport_unknown", detail=str(exc))
        except requests.RequestException as exc:
            print(f"❌ Notion network error: {exc}")
            return _NotionPostResult(error_kind="transport", detail=str(exc))

        if res.ok:
            return _NotionPostResult(url=res.json().get("url"), status_code=res.status_code)

        detail = (res.text or "")[:1200]
        if res.status_code in _NOTION_RETRYABLE_STATUS and attempt < attempts:
            delay = _retry_delay_seconds(res, attempt)
            print(
                f"⚠️ Notion API temporary error ({res.status_code}); "
                f"retrying in {delay:g}s ({attempt}/{attempts})..."
            )
            time.sleep(delay)
            continue

        print(f"❌ Notion API Error ({res.status_code}): {detail}")
        return _NotionPostResult(
            status_code=res.status_code,
            error_kind="api",
            detail=detail,
        )

    return _NotionPostResult(error_kind="unknown", detail="Notion write exhausted retries")


def _notion_rich_text(text: str) -> List[dict]:
    return [{"type": "text", "text": {"content": (text or "")[:2000]}}]


def _notion_heading(title: str, level: int = 2) -> dict:
    if level <= 1:
        return {"object": "block", "type": "heading_1", "heading_1": {"rich_text": _notion_rich_text(title)}}
    if level == 3:
        return {"object": "block", "type": "heading_3", "heading_3": {"rich_text": _notion_rich_text(title)}}
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": _notion_rich_text(title)}}


def _notion_paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _notion_rich_text(text)}}


def _notion_divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _notion_bullets(
    title: str,
    items: List[str],
    *,
    max_items: int = 50,
    heading_level: int = 2,
) -> List[dict]:
    blocks: List[dict] = []
    blocks.append(_notion_heading(title, level=heading_level))
    if not items:
        blocks.append(_notion_paragraph("—"))
        return blocks

    for it in items[:max_items]:
        blocks.append(
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _notion_rich_text(it)},
            }
        )
    return blocks


def _status_emoji(status: str) -> str:
    s = (status or "").lower().strip()
    if s in ("ok", "pass", "good"):
        return "✅"
    if s in ("warning", "warn", "caution"):
        return "⚠️"
    if s in ("blocked", "fail", "error"):
        return "⛔"
    return "ℹ️"


def _extract_public_risk(public_contract: Dict[str, Any]) -> Tuple[str, str, List[str], List[str]]:
    risk_obj = _get_in(public_contract, ["risk"], None)
    if not isinstance(risk_obj, dict):
        risk_obj = _get_in(public_contract, ["score", "risk"], None)
    if not isinstance(risk_obj, dict):
        return "none", "none", [], []

    level = _safe_str(risk_obj.get("level", "none")).lower().strip() or "none"
    severity = _safe_str(risk_obj.get("severity", "none")).lower().strip() or "none"
    must_missing = [_safe_str(x) for x in _as_list(risk_obj.get("must_missing", [])) if _safe_str(x).strip()]
    notes = [_safe_str(x) for x in _as_list(risk_obj.get("notes", [])) if _safe_str(x).strip()]
    return level, severity, _dedupe_keep_order(must_missing), _dedupe_keep_order(notes)


def _norm_strengths_gaps_items(xs: List[Any]) -> List[str]:
    out: List[str] = []

    def push(s: str):
        s = _clamp_text(s, 300)
        if s:
            out.append(s)

    for x in xs:
        if isinstance(x, str):
            push(x)
            continue

        if isinstance(x, dict):
            if isinstance(x.get("items"), list):
                items = [_safe_str(i).strip() for i in x.get("items", []) if _safe_str(i).strip()]
                for it in items:
                    push(it)
                continue

            title = _safe_str(x.get("title") or x.get("name") or x.get("skill") or "").strip()
            why = _safe_str(x.get("why") or x.get("detail") or x.get("note") or "").strip()
            evidence = _safe_str(x.get("evidence") or x.get("evidence_quote") or "").strip()
            parts = [p for p in [title, why, evidence] if p]
            if parts:
                push(" — ".join(parts))
                continue

        push(_safe_str(x))

    return _dedupe_keep_order(out)


def _split_strengths(public_contract: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    strengths = _as_list(_get_in(public_contract, ["strengths"], []))
    domains: List[str] = []
    tools: List[str] = []
    others: List[str] = []

    for s in strengths:
        if isinstance(s, dict) and isinstance(s.get("items"), list):
            t = _safe_str(s.get("type", "")).strip().lower()
            items = [_safe_str(i).strip() for i in s.get("items", []) if _safe_str(i).strip()]
            if t == "domains":
                domains.extend(items)
            elif t == "tools":
                tools.extend(items)
            else:
                others.extend(items)
        else:
            others.extend(_norm_strengths_gaps_items([s]))

    return _dedupe_keep_order(domains), _dedupe_keep_order(tools), _dedupe_keep_order(others)


def _infer_layer_gaps(layer: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    meta = layer.get("meta")
    if not isinstance(meta, dict):
        return [], []

    must = _as_list(meta.get("headline_gaps_must") or meta.get("missing_must") or [])
    rec = _as_list(meta.get("headline_gaps_recommended") or [])
    if not rec:
        missing_all = _as_list(meta.get("missing_all") or [])
        missing_must = set(_safe_str(x).strip() for x in must if _safe_str(x).strip())
        rec = [x for x in missing_all if _safe_str(x).strip() and _safe_str(x).strip() not in missing_must]

    must_s = _dedupe_keep_order([_safe_str(x).strip() for x in must if _safe_str(x).strip()])
    rec_s = _dedupe_keep_order([_safe_str(x).strip() for x in rec if _safe_str(x).strip()])
    return must_s, rec_s


def _render_layers_to_notion(public_contract: Dict[str, Any]) -> List[dict]:
    blocks: List[dict] = []
    layers = _as_list(_get_in(public_contract, ["layers"], []))
    if not layers:
        return blocks

    blocks.append(_notion_heading("🧠 Layered Breakdown", level=2))

    for layer in layers[:10]:
        if not isinstance(layer, dict):
            continue

        lid = _safe_str(layer.get("id", "")).strip()
        title = _safe_str(layer.get("title", lid or "Layer")).strip()
        status = _safe_str(layer.get("status", "")).strip()
        summary = _safe_str(layer.get("summary", "")).strip()

        is_tool_layer = (lid.lower() == "tools") or (title.strip().lower() == "tool examples")
        layer_icon = "🧩" if is_tool_layer else _status_emoji(status)

        blocks.append(_notion_heading(f"{layer_icon} {title}", level=2))
        if summary:
            blocks.append(_notion_paragraph(summary))

        strengths = [_safe_str(x) for x in _as_list(layer.get("strengths", [])) if _safe_str(x).strip()]
        gaps = [_safe_str(x) for x in _as_list(layer.get("gaps", [])) if _safe_str(x).strip()]
        recommended = [_safe_str(x) for x in _as_list(layer.get("recommended", [])) if _safe_str(x).strip()]

        inferred_must, inferred_rec = _infer_layer_gaps(layer)

        if not gaps:
            inferred_all = [_safe_str(x).strip() for x in (inferred_must + inferred_rec) if _safe_str(x).strip()]
            gaps = inferred_all

        gaps = _dedupe_keep_order(gaps)

        if strengths:
            blocks.extend(_notion_bullets("✅ Matched", strengths, max_items=30, heading_level=3))
        if gaps:
            blocks.extend(_notion_bullets("⚠️ Gaps", gaps, max_items=30, heading_level=3))

        if recommended:
            rec_title = "✨ Nice-to-have" if is_tool_layer else "💡 Recommended"
            blocks.extend(_notion_bullets(rec_title, recommended, max_items=30, heading_level=3))

        meta = layer.get("meta")
        if isinstance(meta, dict) and meta:
            keep_keys = ["bucket", "cap", "job_level", "candidate_level", "gap", "missing_must", "missing_all"]
            meta_small = {k: meta.get(k) for k in keep_keys if k in meta}
            if meta_small:
                blocks.append(
                    {
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": _notion_rich_text("ℹ️ Layer meta"),
                            "children": [
                                {
                                    "object": "block",
                                    "type": "code",
                                    "code": {
                                        "language": "json",
                                        "rich_text": _notion_rich_text(
                                            json.dumps(meta_small, ensure_ascii=False, indent=2)[:2000]
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                )

        blocks.append(_notion_divider())

    return blocks


def _render_missing_to_notion(public_contract: Dict[str, Any]) -> List[dict]:
    blocks: List[dict] = []

    missing_obj = _get_in(public_contract, ["missing"], None)
    groups = []
    if isinstance(missing_obj, dict):
        groups = _as_list(missing_obj.get("groups", []))

    if not groups:
        legacy_gaps = _as_list(_get_in(public_contract, ["gaps"], []))
        tmp = []
        for g in legacy_gaps:
            if isinstance(g, dict) and isinstance(g.get("items"), list):
                tmp.append(
                    {
                        "level": _safe_str(g.get("type", "")).strip() or "recommended",
                        "label": _safe_str(g.get("type", "")).strip() or "recommended",
                        "items": g.get("items", []),
                    }
                )
        groups = tmp

    must_items: List[str] = []
    rec_items: List[str] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        level = _safe_str(g.get("level", "")).lower().strip()
        label = _safe_str(g.get("label", "")).strip()
        items = [_safe_str(x).strip() for x in _as_list(g.get("items", [])) if _safe_str(x).strip()]
        if not items:
            continue
        if level == "must" or label.lower().strip() == "must":
            must_items.extend(items)
        else:
            rec_items.extend(items)

    must_items = _dedupe_keep_order(must_items)
    rec_items = _dedupe_keep_order(rec_items)

    if not must_items and not rec_items:
        blocks.append(_notion_heading("❌ Missing", level=2))
        blocks.append(_notion_paragraph("—"))
        return blocks

    if must_items:
        blocks.append(_notion_heading("❌ Missing", level=2))
        blocks.append(_notion_heading("[must]", level=3))
        for it in must_items[:20]:
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": _notion_rich_text(it)},
                }
            )

    if rec_items:
        blocks.append(_notion_heading("⚠️ Gaps (recommended)", level=2))
        for it in rec_items[:20]:
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": _notion_rich_text(it)},
                }
            )
        notes = []
        if isinstance(missing_obj, dict):
            notes = [_safe_str(x).strip() for x in _as_list(missing_obj.get("notes", [])) if _safe_str(x).strip()]
        if notes:
            blocks.append(_notion_paragraph(_clamp_text(" ".join(notes), 400)))

    return blocks


def _render_decision_constraints_to_notion(public_contract: Dict[str, Any], resp: Any) -> List[dict]:
    blocks: List[dict] = []

    level, severity, must_missing_names, risk_notes = _extract_public_risk(public_contract)

    job_level = None
    cand_level = None
    gap_levels = None
    cap_val = None
    try:
        layers = _as_list(_get_in(public_contract, ["layers"], []))
        for ly in layers:
            if isinstance(ly, dict) and _safe_str(ly.get("id", "")).strip().lower() == "seniority":
                meta = ly.get("meta")
                if isinstance(meta, dict):
                    job_level = meta.get("job_level")
                    cand_level = meta.get("candidate_level")
                    gap_levels = meta.get("gap")
                    cap_val = meta.get("cap")
                break
    except Exception:
        pass

    has_any = bool(must_missing_names) or (cap_val is not None) or bool(getattr(resp, "cap", None)) or bool(
        getattr(resp, "seniority_gap", None)
    )
    if not has_any:
        return blocks

    blocks.append(_notion_heading("🔒 Decision Constraints", level=2))
    blocks.append(
        _notion_paragraph(
            "This recommendation is driven by a small number of strong, structural constraints — not by overall skill quality."
        )
    )

    blocks.append(_notion_heading("1️⃣ Seniority Constraint (Primary)", level=3))
    lines: List[str] = []
    jl = _safe_str(job_level, "").strip()
    cl = _safe_str(cand_level, "").strip()
    gl = _safe_str(gap_levels, "").strip()

    bucket = _safe_str(getattr(resp, "seniority_gap", ""), "").strip().lower()
    cv = cap_val if cap_val is not None else getattr(resp, "cap", None)

    blocking_reason = getattr(resp, "blocking_reason", None)
    if blocking_reason:
        lines.append(f"⛔ BLOCKED: {blocking_reason}")

    ownership = getattr(resp, "ownership", {}) or {}
    own_present = ownership.get("ownership_present")
    scope = ownership.get("scope_level", "unknown")
    lead_present = ownership.get("leadership_present")

    if own_present or lead_present or (scope and scope != "unknown"):
        details = []
        if own_present:
            details.append("Ownership ✅")
        if lead_present:
            details.append("Leadership ✅")
        if scope and scope != "unknown":
            details.append(f"Scope: {scope}")
        lines.append(f"Signals: {', '.join(details)}.")

    if jl or cl or gl:
        parts = []
        if jl:
            parts.append(f"Job level: {jl}")
        if cl:
            parts.append(f"Your level: {cl}")
        if gl:
            parts.append(f"Gap: {gl}")
        lines.append(" • ".join(parts) + ".")

    def _cap_phrase(v) -> str:
        if v is None:
            return ""
        try:
            return f" capped at {int(v)}"
        except Exception:
            sv = _safe_str(v, "").strip()
            return f" capped at {sv}" if sv else ""

    cap_phrase = _cap_phrase(cv)

    if bucket == "cliff":
        lines.append(
            "Seniority gap is a cliff for this role. "
            f"As a result, the final score is{cap_phrase}, regardless of skill overlap."
        )
    elif bucket in ("small", "medium"):
        lines.append(
            "Seniority gap may reduce fit and applies a cap to the final score. "
            f"The final score is{cap_phrase} after skill overlap is computed."
        )
    elif bucket in ("none", "overqualified"):
        if bucket == "overqualified":
            lines.append(
                "Candidate may be overqualified by seniority. This is not a hard blocker, "
                "but some teams may worry about role scope or retention."
            )
        else:
            lines.append("No seniority cap applied.")
    else:
        if cap_phrase:
            lines.append(f"Seniority assessment is uncertain; final score may be{cap_phrase} depending on role level.")
        else:
            lines.append("Seniority assessment is uncertain; no cap decision could be derived.")

    blocks.append(_notion_paragraph(" ".join([x for x in lines if x])))

    if must_missing_names:
        blocks.append(_notion_heading("2️⃣ Critical Must Skill Missing", level=3))
        blocks.append(
            _notion_paragraph(
                "One required (must-have) capability is missing:\n" + "– " + "\n– ".join(must_missing_names[:10])
            )
        )
        blocks.append(_notion_paragraph("This is treated as a hard requirement in the decision logic."))
    else:
        blocks.append(_notion_heading("✅ No critical must missing", level=3))
        blocks.append(_notion_paragraph("All must-have requirements are covered based on current evidence."))

    blocks.append(_notion_heading("3️⃣ Model Confidence", level=3))
    blocks.append(_notion_paragraph("Confidence: High"))
    blocks.append(_notion_paragraph("These constraints are rule-based and explicit, not inferred from weak signals."))

    lvl = (level or "none").upper()
    sev = (severity or "none").upper()
    if (lvl != "NONE") or (sev != "NONE"):
        blocks.append(_notion_paragraph(f"Likelihood (changes with more evidence): {lvl} • Impact (on verdict): {sev}"))

    if risk_notes:
        blocks.extend(_notion_bullets("Notes", risk_notes, max_items=10, heading_level=3))

    return blocks


def _render_requirement_matrix_to_notion(public_contract: Dict[str, Any], resp: Any) -> List[dict]:
    """Render every verified JD requirement once, with JD and resume evidence."""
    requirements = _get_in(public_contract, ["requirements"], {})
    if not isinstance(requirements, dict):
        return []
    items = [x for x in _as_list(requirements.get("items")) if isinstance(x, dict)]
    if not items:
        return []

    counts = requirements.get("counts") if isinstance(requirements.get("counts"), dict) else {}
    quality = _get_in(public_contract, ["analysis_quality"], {})
    reliable = bool(quality.get("score_reliable")) if isinstance(quality, dict) else True
    total = int(counts.get("total", len(items)) or len(items))
    matched = int(counts.get("matched", 0) or 0)
    partial = int(counts.get("partial", 0) or 0)
    missing = int(counts.get("missing", 0) or 0)
    must_total = int(counts.get("must_total", 0) or 0)
    must_missing = int(counts.get("must_missing", 0) or 0)

    blocks: List[dict] = [
        _notion_heading("🎯 JD ↔ Resume Match Coverage", level=2),
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "icon": {"emoji": "✅" if reliable and must_missing == 0 else "⚠️"},
                "rich_text": _notion_rich_text(
                    f"{matched} matched • {partial} partial • {missing} missing • {total} total"
                    + (f" • MUST missing {must_missing}/{must_total}" if must_total else "")
                    + (" • Score reliable" if reliable else " • Score NOT reliable")
                ),
            },
        },
        _notion_paragraph(
            "Matched means Kairos found resume evidence above its matching threshold. Partial means related evidence exists but should be made more explicit. Missing means Kairos found no resume evidence; do not add it unless it is true."
        ),
    ]

    status_icon = {"matched": "✅", "partial": "🟡", "missing": "❌", "unverified": "❓"}
    importance_order = {"must": 0, "should": 1, "nice_to_have": 2, "nice": 2, "unknown": 3}
    status_order = {"missing": 0, "partial": 1, "matched": 2, "unverified": 3}
    ordered = sorted(
        items,
        key=lambda x: (
            importance_order.get(_safe_str(x.get("importance")).lower(), 3),
            status_order.get(_safe_str(x.get("status")).lower(), 3),
            _safe_str(x.get("name")).lower(),
        ),
    )

    current_importance = None
    for item in ordered[:30]:
        importance = _safe_str(item.get("importance"), "unknown").lower()
        if importance != current_importance:
            current_importance = importance
            label = {"must": "MUST requirements", "should": "SHOULD requirements", "nice_to_have": "NICE-TO-HAVE", "nice": "NICE-TO-HAVE"}.get(importance, "Other requirements")
            blocks.append(_notion_heading(label, level=3))

        status = _safe_str(item.get("status"), "unverified").lower()
        name = _safe_str(item.get("name"), "Unnamed requirement")
        title = f"{status_icon.get(status, '❓')} {name} — {status.replace('_', ' ')}"

        detail_children: List[dict] = []
        jd_evidence = _safe_str(item.get("jd_evidence")).strip()
        if jd_evidence:
            detail_children.append(_notion_paragraph(_clamp_text(f"JD evidence: {jd_evidence}", 800)))
        resume_evidence = [_safe_str(x).strip() for x in _as_list(item.get("resume_evidence")) if _safe_str(x).strip()]
        if resume_evidence:
            detail_children.extend(_notion_bullets("Resume evidence", resume_evidence, heading_level=3, max_items=3))
        else:
            detail_children.append(_notion_paragraph("Resume evidence: none found."))

        tools = [x for x in _as_list(item.get("tools")) if isinstance(x, dict)]
        if tools:
            tool_text = ", ".join(
                f"{_safe_str(x.get('name'))} ({_safe_str(x.get('status'), 'unverified')})" for x in tools[:15]
            )
            detail_children.append(_notion_paragraph(_clamp_text(f"JD tools: {tool_text}", 900)))

        blocks.append(
            {
                "object": "block",
                "type": "toggle",
                "toggle": {"rich_text": _notion_rich_text(title), "children": detail_children[:20]},
            }
        )

    tool_items = [x for x in _as_list(_get_in(public_contract, ["tools", "items"], [])) if isinstance(x, dict)]
    if tool_items:
        blocks.append(_notion_divider())
        blocks.append(_notion_heading("🧩 Explicit JD Tools / ATS Keywords", level=2))
        blocks.append(
            _notion_paragraph(
                "These are explicit JD mentions. They help ATS keyword coverage, but Kairos treats tools as supporting evidence rather than hard blockers."
            )
        )
        for status, label in (("missing", "❌ Missing"), ("partial", "🟡 Partial"), ("matched", "✅ Matched")):
            names = [_safe_str(x.get("name")) for x in tool_items if _safe_str(x.get("status")) == status]
            if names:
                blocks.append(
                    {
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": _notion_rich_text(f"{label} ({len(names)})"),
                            "children": _notion_bullets("", names, heading_level=3, max_items=40),
                        },
                    }
                )

    actions = _norm_strengths_gaps_items(_as_list(public_contract.get("actions")))
    if actions:
        blocks.append(_notion_divider())
        blocks.extend(_notion_bullets("🛠 Resume / Skill Actions", actions, heading_level=2, max_items=10))

    blocks.append(_notion_divider())
    blocks.append(
        {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": _notion_rich_text("ℹ️ Score and seniority details"),
                "children": _notion_bullets(
                    "",
                    [
                        f"Skill match: {getattr(resp, 'distance_score', 0)}%",
                        f"Final score: {getattr(resp, 'final_score', 0)}/100",
                        f"Seniority gap: {getattr(resp, 'seniority_gap', 'unknown')}",
                        f"Score cap: {getattr(resp, 'cap', 100)}",
                    ],
                    heading_level=3,
                    max_items=10,
                ),
            },
        }
    )
    return blocks


def create_notion_page(
    req: Any,
    resp: Any,
    score_result: Any,
    *,
    public_contract: Optional[Dict[str, Any]] = None,
    notion_token: str,
    notion_database_id: str,
    notion_version: str,
) -> Optional[str]:
    if not notion_token or not notion_database_id:
        return None

    notion_job_title = _best_title(req, getattr(resp, "job_title", ""))

    full_properties = {
        "Job Title": {"title": [{"text": {"content": notion_job_title[:100]}}]},
        "Company": {"rich_text": [{"text": {"content": (_safe_str(getattr(resp, "company", None)) or "Unknown")[:100]}}]},
        "Match Score": {"number": int(getattr(resp, "final_score", 0))},
        "Should Apply?": {"select": {"name": _safe_str(getattr(resp, "should_apply", "No"))}},
        "Seniority Gap": {"select": {"name": _safe_str(getattr(resp, "seniority_gap", "unknown"))}},
        "Missing Skills": {
            "multi_select": [{"name": _clean_tag(s)} for s in (_as_list(getattr(resp, "missing_skills", [])))]
        },
        "Required Skills": {
            "multi_select": [{"name": _clean_tag(s)} for s in (_as_list(getattr(resp, "required_skills", [])))]
        },
    }
    if getattr(req, "url", None):
        full_properties["URL"] = {"url": getattr(req, "url")}

    safe_properties = {"Job Title": {"title": [{"text": {"content": notion_job_title[:100]}}]}}

    emoji_map = {"Yes": "✅", "Maybe": "🤔", "No": "❌"}
    action_emoji = emoji_map.get(_safe_str(getattr(resp, "should_apply", None)), "❓")

    cap_explainer = None
    if isinstance(public_contract, dict):
        cap_explainer = _safe_str(_get_in(public_contract, ["score", "cap_explainer"], None)).strip()
    if not cap_explainer:
        cap_val = getattr(resp, "cap", None)
        if isinstance(cap_val, int) and cap_val < 100:
            cap_explainer = f"Capped to {cap_val} due to seniority gap."

    blocks: List[dict] = [
        _notion_heading(f"{action_emoji} Recommendation: {getattr(resp, 'should_apply', 'No')}", level=1),
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "icon": {"emoji": "📊"},
                "rich_text": _notion_rich_text(
                    f"Final Score: {getattr(resp, 'final_score', 0)}/100"
                    + (f" • {cap_explainer}" if cap_explainer else "")
                ),
            },
        },
        _notion_paragraph(getattr(resp, "summary", "") or ""),
        _notion_divider(),
    ]

    # Keep the user-facing page focused on the decision, gaps, strengths, and
    # actions. The requirement evidence matrix remains available internally
    # for scoring/debugging but is intentionally not rendered into Notion.
    if isinstance(public_contract, dict):
        display_required = _as_list(public_contract.get("display_required_domains") or [])
        display_evidence = public_contract.get("display_domain_evidence") or {}
        tools_in_jd = _as_list(public_contract.get("tools_in_jd") or [])
        missing_tools = set(_as_list(public_contract.get("missing_tools") or []))
        tool_labels = public_contract.get("missing_tools_labels") or {}

        if display_required:
            blocks.append(_notion_heading("📌 Extracted Domains", level=2))
            for d in display_required[:25]:
                tag = str(display_evidence.get(d, "unknown")).upper()
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": _notion_rich_text(f"{d} — {tag}")},
                    }
                )
            blocks.append(_notion_divider())

        if tools_in_jd:
            stop_words = {
                "tool",
                "tools",
                "tooling",
                "platform",
                "platforms",
                "experience",
                "framework",
                "frameworks",
                "technology",
                "technologies",
                "solution",
                "solutions",
                "stack",
                "planning",
            }
            filtered_tools = [t for t in tools_in_jd if _safe_str(t).strip().lower() not in stop_words]
            must_tools: List[str] = []
            should_tools: List[str] = []
            nice_tools: List[str] = []
            for t in filtered_tools:
                label = _safe_str(tool_labels.get(t, "")).lower()
                if label == "must":
                    must_tools.append(t)
                elif label == "nice_to_have":
                    nice_tools.append(t)
                else:
                    should_tools.append(t)

            blocks.append(_notion_heading("🧰 Extracted Tools (JD)", level=2))
            blocks.append(
                _notion_paragraph(
                    "Scoring uses only verified & matched tools. This list is raw JD mentions."
                )
            )
            if must_tools:
                blocks.extend(_notion_bullets("[must]", must_tools, heading_level=3, max_items=30))
            if should_tools:
                blocks.extend(_notion_bullets("[should]", should_tools, heading_level=3, max_items=30))
            if nice_tools:
                blocks.append(
                    {
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": _notion_rich_text("[nice-to-have]"),
                            "children": _notion_bullets("", nice_tools, heading_level=3, max_items=30),
                        },
                    }
                )
            blocks.append(_notion_divider())

        blocks.extend(_render_missing_to_notion(public_contract))
        blocks.append(_notion_divider())

        strengths = _as_list(_get_in(public_contract, ["strengths"], []))
        missing_all = _as_list(_get_in(public_contract, ["missing", "all"], []))

        if (not strengths or all(not s.get("items") for s in strengths if isinstance(s, dict))) and not missing_all:
            if display_required:
                blocks.append(_notion_heading("📌 Requirements (unverified)", level=2))
                items = []
                for d in display_required[:20]:
                    tag = str(display_evidence.get(d, "unknown")).upper()
                    items.append(f"{d} — {tag}")
                for it in items:
                    blocks.append(
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": _notion_rich_text(it)},
                        }
                    )
                blocks.append(_notion_divider())

        actions = _as_list(_get_in(public_contract, ["actions"], []))
        blocks.extend(_notion_bullets("✅ Actions", _norm_strengths_gaps_items(actions), heading_level=2))
        blocks.append(_notion_divider())

        # Details toggle: Fit/Skill Match/Coverage/Distance
        details_lines: List[str] = []
        uncapped = getattr(resp, "uncapped_score", None)
        if uncapped is not None:
            details_lines.append(f"Fit (uncapped): {uncapped}")
        details_lines.append(f"Skill Match: {getattr(resp, 'distance_score', 0)}%")
        coverage_score = getattr(resp, "coverage_score", None)
        if coverage_score is None and isinstance(public_contract, dict):
            coverage_score = _get_in(public_contract, ["score", "coverage_score"], None)
        if coverage_score is not None:
            details_lines.append(f"Coverage: {coverage_score}%")
        details_lines.append(f"Seniority Gap: {getattr(resp, 'seniority_gap', 'unknown')}")
        if details_lines:
            blocks.append(
                {
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": _notion_rich_text("Details"),
                        "children": _notion_bullets("", details_lines, heading_level=3, max_items=10),
                    },
                }
            )
            blocks.append(_notion_divider())

        domains_s, tools_s, other_s = _split_strengths(public_contract)
        blocks.append(_notion_heading("💪 Strengths", level=2))
        if domains_s:
            blocks.extend(_notion_bullets("Domains", domains_s, heading_level=3, max_items=30))
        if tools_s:
            blocks.extend(_notion_bullets("Tools", tools_s, heading_level=3, max_items=30))
        if other_s:
            blocks.extend(_notion_bullets("Other", other_s, heading_level=3, max_items=30))
        blocks.append(_notion_divider())

        blocks.extend(_render_layers_to_notion(public_contract))

        dc_blocks = _render_decision_constraints_to_notion(public_contract, resp)
        if dc_blocks:
            blocks.append(_notion_divider())
            blocks.extend(dc_blocks)
            blocks.append(_notion_divider())

    blocks.append(
        {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": _notion_rich_text("🔧 Raw Debug Data"),
                "children": [
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "language": "json",
                            "rich_text": _notion_rich_text(
                                json.dumps(getattr(resp, "raw_json", {}), ensure_ascii=False, indent=2)[:2000]
                            ),
                        },
                    }
                ],
            },
        }
    )

    blocks = _trim_blocks_for_notion(blocks, 100)

    full_result = _post_to_notion(
        notion_token=notion_token,
        notion_database_id=notion_database_id,
        notion_version=notion_version,
        properties=full_properties,
        children=blocks,
    )
    if full_result.url:
        return full_result.url

    # Safe Mode is for validation/payload problems, not transport errors. A
    # timed-out POST has an unknown outcome and an immediate second create can
    # produce a duplicate page.
    if full_result.status_code == 400:
        safe_children = blocks[:24]
        print(
            "⚠️ Notion rejected the full payload. Retrying with Safe Mode "
            f"(title + {len(safe_children)} core blocks)..."
        )
        safe_result = _post_to_notion(
            notion_token=notion_token,
            notion_database_id=notion_database_id,
            notion_version=notion_version,
            properties=safe_properties,
            children=safe_children,
        )
        return safe_result.url

    if full_result.error_kind == "transport_unknown":
        print("⚠️ Notion write was not retried to avoid creating a duplicate page.")
    return None
