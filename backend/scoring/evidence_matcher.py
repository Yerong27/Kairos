"""Industry-neutral matching between atomic JD requirements and resume evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

from backend.ir.candidate_profile import CandidateEvidenceClaim
from backend.ir.schema_v3 import DomainRequirement


MatchStatus = Literal["matched", "partial", "missing"]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "have",
    "in",
    "including",
    "is",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "to",
    "using",
    "with",
    "work",
    "working",
    "ability",
    "experience",
    "experienced",
    "knowledge",
    "skills",
    "skill",
    "strong",
    "proficiency",
    "proficient",
    "required",
    "preferred",
}
_STRICT_TYPES = {"tool", "credential", "education", "experience", "work_condition"}


def _normalize(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\s/_]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\-\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: str) -> List[str]:
    tokens: List[str] = []
    for token in _TOKEN_RE.findall(_normalize(value)):
        if token in _STOPWORDS or len(token) <= 1:
            continue
        stem = token
        if len(stem) > 7 and stem.endswith("ment"):
            stem = stem[:-4]
        elif len(stem) > 6 and stem.endswith("ion"):
            stem = stem[:-3]
        elif len(stem) > 6 and stem.endswith("ing"):
            stem = stem[:-3]
        elif len(stem) > 5 and stem.endswith("ed"):
            stem = stem[:-2]
        elif len(stem) > 5 and stem.endswith("ies"):
            stem = stem[:-3] + "y"
        elif len(stem) > 4 and stem.endswith("s") and not stem.endswith("ss"):
            stem = stem[:-1]
        if len(stem) > 4 and stem.endswith("e"):
            stem = stem[:-1]
        tokens.append(stem)
    return tokens


def _phrase_present(phrase: str, text: str) -> bool:
    phrase_norm = _normalize(phrase)
    text_norm = _normalize(text)
    if not phrase_norm or not text_norm:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9+#]){re.escape(phrase_norm)}(?=$|[^a-z0-9+#])",
            text_norm,
        )
        is not None
    )


def _coverage(required_tokens: Sequence[str], candidate_text: str) -> float:
    required = set(required_tokens)
    if not required:
        return 0.0
    candidate = set(_tokens(candidate_text))
    return len(required & candidate) / float(len(required))


def _illustrative_targets(evidence_quote: str) -> List[str]:
    quote = str(evidence_quote or "")
    marker = re.search(r"\b(such as|including|e\.g\.|for example)\b", quote, re.IGNORECASE)
    if not marker:
        return []
    tail = quote[marker.end() :].strip()
    if ")" in tail:
        tail = tail.split(")", 1)[0]
    tail = tail.strip(" .:;()[]")
    values: List[str] = []
    seen = set()
    for part in re.split(r"[,;]|\s+\bor\b\s+|\s+\band\b\s+", tail, flags=re.IGNORECASE):
        value = part.strip(" .:;()[]")
        key = _normalize(value)
        if value and key and key not in seen and len(value.split()) <= 6:
            seen.add(key)
            values.append(value)
    return values


@dataclass(frozen=True)
class EvidenceMatchDecision:
    status: MatchStatus
    confidence: float
    evidence_ids: List[str]
    evidence_quotes: List[str]
    reason: str


@dataclass(frozen=True)
class _ClaimScore:
    score: float
    status: MatchStatus
    claim: CandidateEvidenceClaim
    reason: str


def _score_claim(
    requirement: DomainRequirement,
    claim: CandidateEvidenceClaim,
) -> _ClaimScore:
    name = str(requirement.name or "").strip()
    alternatives = [str(item).strip() for item in (requirement.alternatives or []) if str(item).strip()]
    targets = [name] + alternatives
    examples = [
        str(item.name).strip()
        for item in (requirement.examples or [])
        if str(getattr(item, "name", "") or "").strip()
    ]
    examples.extend(
        item for item in _illustrative_targets(str(requirement.evidence_quote or ""))
        if _normalize(item) not in {_normalize(existing) for existing in examples}
    )
    quote = str(claim.resume_quote or "").strip()
    skills = [str(item).strip() for item in (claim.skills or []) if str(item).strip()]
    domains = [str(item).strip() for item in (claim.domains or []) if str(item).strip()]

    exact_quote = any(_phrase_present(target, quote) for target in targets)
    exact_grounded_skill = any(
        _normalize(target) == _normalize(skill) and _phrase_present(skill, quote)
        for target in targets
        for skill in skills
    )
    exact_example = any(
        _phrase_present(example, quote)
        or any(
            _normalize(example) == _normalize(skill) and _phrase_present(skill, quote)
            for skill in skills
        )
        for example in examples
    )

    req_tokens = _tokens(name)
    quote_coverage = _coverage(req_tokens, quote)
    skill_coverage = max((_coverage(req_tokens, skill) for skill in skills), default=0.0)
    domain_coverage = max((_coverage(req_tokens, domain) for domain in domains), default=0.0)
    token_hits = len(set(req_tokens) & set(_tokens(quote)))
    requirement_type = str(getattr(requirement, "requirement_type", "capability") or "capability")

    if exact_quote or exact_grounded_skill:
        return _ClaimScore(0.96, "matched", claim, "exact_grounded_phrase")
    if exact_example:
        return _ClaimScore(0.64, "partial", claim, "grounded_parent_example")

    if requirement_type in _STRICT_TYPES:
        if token_hits >= 2 and quote_coverage >= 0.75:
            return _ClaimScore(0.72, "partial", claim, "related_strict_evidence")
        if skill_coverage >= 0.50 or domain_coverage >= 0.50:
            return _ClaimScore(0.55, "partial", claim, "structured_but_not_explicit")
        return _ClaimScore(0.0, "missing", claim, "no_explicit_strict_evidence")

    # Capability/responsibility matching is deliberately conservative. Two
    # distinctive tokens are required unless the full phrase matched above.
    if token_hits >= 2 and quote_coverage >= 0.75:
        return _ClaimScore(0.82, "matched", claim, "strong_claim_overlap")
    if token_hits >= 2 and quote_coverage >= 0.40:
        return _ClaimScore(0.58, "partial", claim, "related_claim_overlap")
    if skill_coverage >= 0.50 or domain_coverage >= 0.50:
        return _ClaimScore(0.48, "partial", claim, "structured_related_evidence")
    return _ClaimScore(0.0, "missing", claim, "no_grounded_evidence")


def match_requirement_to_evidence(
    requirement: DomainRequirement,
    claims: Iterable[CandidateEvidenceClaim],
    *,
    max_evidence: int = 3,
) -> EvidenceMatchDecision:
    scored = [_score_claim(requirement, claim) for claim in claims if str(claim.resume_quote or "").strip()]
    supported = [item for item in scored if item.status != "missing"]
    if not supported:
        return EvidenceMatchDecision(
            status="missing",
            confidence=0.0,
            evidence_ids=[],
            evidence_quotes=[],
            reason="no_grounded_resume_evidence",
        )

    supported.sort(key=lambda item: item.score, reverse=True)
    best = supported[0]
    selected = [item for item in supported if item.status == best.status][: max(1, int(max_evidence))]
    return EvidenceMatchDecision(
        status=best.status,
        confidence=round(best.score, 3),
        evidence_ids=[
            item.claim.evidence_id
            for item in selected
            if str(item.claim.evidence_id or "").strip()
        ],
        evidence_quotes=[item.claim.resume_quote[:280] for item in selected],
        reason=best.reason,
    )


__all__ = [
    "EvidenceMatchDecision",
    "MatchStatus",
    "match_requirement_to_evidence",
]
