from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from backend.ir.schema_v3 import DomainRequirement, Importance, DomainFacet


@dataclass
class CatalogDomain:
    id: str
    label: str
    facet: DomainFacet
    allowed_must: bool
    aliases_strong: List[str]
    aliases_weak: List[str]
    domain_type: str  # "hard" | "soft"
    default_weight: float


@dataclass
class DomainCatalog:
    domains: Dict[str, CatalogDomain]
    aliases_global: Dict[str, str]
    allowed_must_domains: List[str]
    max_must_count: Optional[int]


def load_domain_catalog(path: pathlib.Path) -> DomainCatalog:
    data = json.loads(path.read_text(encoding="utf-8"))
    domains: Dict[str, CatalogDomain] = {}
    for item in data.get("domains", []):
        try:
            dom = CatalogDomain(
                id=str(item["id"]),
                label=str(item.get("label") or item["id"]),
                facet=str(item.get("facet") or "unknown"),
                allowed_must=bool(item.get("allowed_must", False)),
                aliases_strong=[str(x).lower() for x in item.get("aliases_strong", []) if str(x).strip()],
                aliases_weak=[str(x).lower() for x in item.get("aliases_weak", []) if str(x).strip()],
                domain_type=str(item.get("domain_type", "hard")),
                default_weight=float(item.get("default_weight", 1.0)),
            )
            domains[dom.id.lower()] = dom
        except Exception:
            continue

    aliases_global = {}
    for k, v in (data.get("aliases_global", {}).get("canonicalize", {}) or {}).items():
        aliases_global[str(k).lower()] = str(v).lower()

    policy = data.get("policy") if isinstance(data.get("policy"), dict) else {}
    allowed_must_domains = [str(x).lower() for x in policy.get("allowed_must_domains", []) if str(x).strip()]
    max_must_count = policy.get("max_must_count")
    if not isinstance(max_must_count, int):
        max_must_count = None

    return DomainCatalog(
        domains=domains,
        aliases_global=aliases_global,
        allowed_must_domains=allowed_must_domains,
        max_must_count=max_must_count,
    )


def load_domain_catalogs(paths: List[pathlib.Path]) -> DomainCatalog:
    merged_domains: Dict[str, CatalogDomain] = {}
    merged_aliases: Dict[str, str] = {}
    merged_allowed_must: List[str] = []
    max_must_count: Optional[int] = None
    for path in paths:
        data = load_domain_catalog(path)
        merged_domains.update(data.domains)
        merged_aliases.update(data.aliases_global)
        merged_allowed_must.extend(data.allowed_must_domains or [])
        if isinstance(data.max_must_count, int):
            max_must_count = max(data.max_must_count, max_must_count or 0)

    merged_allowed_must = sorted(set(merged_allowed_must))
    return DomainCatalog(
        domains=merged_domains,
        aliases_global=merged_aliases,
        allowed_must_domains=merged_allowed_must,
        max_must_count=max_must_count,
    )


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


_INVALID_TOOL_NAMES = {
    "tool",
    "tools",
    "technology",
    "technologies",
    "llm",
    "llms",
    "agentic system",
    "agentic systems",
}


def _is_valid_tool_name(name: str) -> bool:
    n = _normalize(name)
    if not n:
        return False
    if n in _INVALID_TOOL_NAMES:
        return False
    if len(n) < 2:
        return False
    return True


def _find_catalog_domain(catalog: DomainCatalog, name: str, hint_id: Optional[str] = None) -> Optional[CatalogDomain]:
    # 1. Try Hint First
    if hint_id:
        hint = _normalize(hint_id)
        if hint in catalog.domains:
            return catalog.domains[hint]

    n = _normalize(name)
    if not n:
        return None

    # 2. direct id/label match
    for dom in catalog.domains.values():
        if n == dom.id.lower() or n == _normalize(dom.label):
            return dom

    # 3. global alias map
    if n in catalog.aliases_global:
        mapped = catalog.aliases_global[n]
        return catalog.domains.get(mapped)

    # 4. alias match (strong/weak)
    for dom in catalog.domains.values():
        if n in dom.aliases_strong or n in dom.aliases_weak:
            return dom

    return None


def _span_exact_match(quote: str, text: str) -> Optional[Tuple[int, int, str]]:
    if not quote or not text:
        return None
    # Verbatim
    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote), text[idx : idx + len(quote)]

    # Normalized
    q_norm = _normalize(quote)
    t_norm = _normalize(text)
    idx2 = t_norm.find(q_norm)
    if idx2 >= 0 and len(q_norm) > 5: # min length check
        return idx2, idx2 + len(q_norm), q_norm
    return None


def _split_sentences(text: str) -> List[str]:
    # Simple heuristic splitting
    parts = re.split(r"(?<=[\.\!\?\n])\s+", text or "")
    return [p.strip() for p in parts if p.strip()][:250]


def _find_anchored_span(
    dom: CatalogDomain,
    text: str,
    window: int = 1
) -> Optional[Tuple[int, int, str]]:
    """
    Strong Alias + Window ±1 sentence check.
    """
    if not text or not dom.aliases_strong:
        return None

    sentences = _split_sentences(text)
    low_sents = [_normalize(s) for s in sentences]

    for i, s_low in enumerate(low_sents):
        for alias in dom.aliases_strong:
            # check exact word boundary or strong substring
            # for safety, we require boundary if alias is short (<4 chars) and not "aws"/"gcp" etc.
            if alias in s_low:
                # Construct window text
                start_i = max(0, i - window)
                end_i = min(len(sentences), i + window + 1)
                window_text = " ".join(sentences[start_i:end_i])
                return 0, 0, window_text # we don't return precise indices for anchor, just text

    return None


def _is_weak_hit(dom: CatalogDomain, name_extracted: str) -> bool:
    """
    Check if the extraction itself matches a weak alias, even if no span found.
    Soft Fallback logic.
    """
    n = _normalize(name_extracted)
    return n in dom.aliases_weak


def adjudicate_domains(
    domains: List[DomainRequirement],
    *,
    jd_text: str,
    catalog: DomainCatalog,
) -> List[DomainRequirement]:
    out: List[DomainRequirement] = []

    # Parameters (Logged)
    # anchored_window = 1
    # weak_window = 3 (Not used in code, simplified to alias check fallback)
    # fuzzy_threshold = 0.75 (Not used, rely on strict rules)

    for d in domains or []:
        try:
            if getattr(d, "examples", None):
                d.examples = [ex for ex in (d.examples or []) if _is_valid_tool_name(getattr(ex, "name", ""))]
        except Exception:
            pass
        try:
            if getattr(d, "examples", None):
                d.examples = [ex for ex in (d.examples or []) if _is_valid_tool_name(getattr(ex, "name", ""))]
        except Exception:
            pass
        # 1. Canonicalize
        dom = _find_catalog_domain(catalog, d.name, hint_id=getattr(d, "domain_id", None))

        if not dom:
            # The catalog is a normalization aid, not an allowlist. A new or
            # role-specific requirement with a verbatim JD quote is still a
            # valid requirement and must survive into matching/output.
            evidence_quote = getattr(d, "evidence_quote", "") or ""
            exact_span = _span_exact_match(evidence_quote, jd_text)
            try:
                if exact_span:
                    slug = re.sub(r"[^a-z0-9]+", "_", _normalize(d.name)).strip("_")
                    setattr(d, "domain_id", f"custom:{slug or 'requirement'}")
                    setattr(d, "evidence_level", "exact")
                    setattr(d, "evidence_status", "verified")
                    setattr(
                        d,
                        "span_meta",
                        {"method": "exact", "found": True, "text": exact_span[2]},
                    )
                else:
                    setattr(d, "domain_id", "other_info")
                    setattr(d, "evidence_level", "none")
                    setattr(d, "evidence_status", "unverified")
                    setattr(d, "importance", "should")
                    setattr(d, "span_meta", {"method": "none", "found": False, "text": ""})
            except Exception:
                pass
            out.append(d)
            continue

        evidence_quote = getattr(d, "evidence_quote", "") or ""
        # Remap ML -> RAG when the quote is clearly about agentic/RAG frameworks
        if dom.id == "machine_learning":
            ql = _normalize(evidence_quote)
            if any(x in ql for x in ["langchain", "llamaindex", "langgraph", "crewai", "vector database", "vector databases", "agentic"]):
                rag_dom = catalog.domains.get("rag_systems")
                if rag_dom:
                    dom = rag_dom

        # Apply Catalog Properties
        setattr(d, "domain_id", dom.id)
        # Preserve the concise JD-facing label produced by the comparison
        # model. The catalog is internal normalization metadata, not a UI
        # rewrite layer (for example, SQL must remain SQL).
        setattr(d, "facet", dom.facet)

        # 2. Adjudicate Evidence (Deterministic)
        evidence_quote = getattr(d, "evidence_quote", "") or ""

        # Level: Exact
        span_exact = _span_exact_match(evidence_quote, jd_text)
        if span_exact:
            setattr(d, "evidence_level", "exact")
            setattr(d, "span_meta", {"method": "exact", "found": True, "text": span_exact[2]})
        else:
            # Level: Anchored
            span_anchor = _find_anchored_span(dom, jd_text, window=1)
            if span_anchor:
                setattr(d, "evidence_level", "anchored")
                setattr(d, "span_meta", {"method": "anchored", "found": True, "text": span_anchor[2]})
            else:
                # Level: Weak (Fallback: Strong/Weak alias match in Name or Quote but no Span)
                # Or just alias match in extracted name
                if _is_weak_hit(dom, d.name) or _is_weak_hit(dom, evidence_quote):
                    setattr(d, "evidence_level", "weak")
                    setattr(d, "span_meta", {"method": "weak_alias", "found": False, "text": ""})
                else:
                    setattr(d, "evidence_level", "none")
                    setattr(d, "span_meta", {"method": "none", "found": False, "text": ""})

        span_meta = getattr(d, "span_meta", None)
        span_found = bool(span_meta.get("found")) if isinstance(span_meta, dict) else False
        if span_found:
            setattr(d, "evidence_status", "verified")
        else:
            setattr(d, "evidence_status", "unverified")

        # 3. MUST Gate (Strict)
        # The catalog normalizes names; it must not decide which professions
        # are allowed to have mandatory requirements. Only grounded evidence
        # controls whether MUST survives.
        if getattr(d, "importance", "unknown") == "must":
            downgrade = False
            reasons = []

            # Evidence strength
            ev_level = getattr(d, "evidence_level", "none")
            if ev_level not in ("exact", "anchored"):
                downgrade = True
                reasons.append(f"weak_evidence({ev_level})")

            if downgrade:
                setattr(d, "importance", "should")
                # Store reason in metadata (span_meta or new field if we had one)
                # For now appending to span_meta text or logging
                d.span_meta["downgrade_reason"] = ", ".join(reasons)
        out.append(d)

    return out
