"""Small, industry-neutral helpers for recognizing common resume sections."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


_SECTION_ALIASES = {
    "experience": {
        "career history",
        "earlier experience",
        "employment",
        "employment history",
        "experience",
        "professional experience",
        "relevant experience",
        "work experience",
        "work history",
    },
    "education": {
        "academic background",
        "academic history",
        "education",
        "education and training",
    },
    "credentials": {
        "certificates",
        "certification",
        "certifications",
        "credentials",
        "licences",
        "licenses",
        "professional certifications",
        "professional credentials",
    },
    "skills": {
        "capabilities",
        "competencies",
        "core capabilities",
        "core competencies",
        "expertise",
        "key skills",
        "selected qualifications",
        "skills",
        "technical capabilities",
        "technical skills",
    },
    "projects": {
        "key projects",
        "personal projects",
        "projects",
        "selected projects",
        "technical projects",
    },
}


def _heading_key(line: str) -> str:
    value = re.sub(r"^[\s•●▪◦\-–—]+|[\s:|]+$", "", str(line or "")).strip()
    value = re.sub(r"[^A-Za-z&/ ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value.replace("&", "and")


def resume_heading_kind(line: str) -> Optional[str]:
    """Return a broad section kind, ``other`` for a heading, or ``None``."""
    raw = str(line or "").strip()
    key = _heading_key(raw)
    if not key or len(key.split()) > 8:
        return None

    for kind, aliases in _SECTION_ALIASES.items():
        if key in aliases:
            return kind

    # Unknown all-caps headings still end the preceding section. This avoids
    # allowing a date in Education to leak into Professional Experience.
    if re.search(r"\d", raw):
        return None
    letters = re.sub(r"[^A-Za-z]", "", raw)
    if letters and raw.upper() == raw and len(letters) >= 3:
        return "other"
    return None


def resume_section_map(lines: Iterable[str]) -> Tuple[List[Optional[str]], bool]:
    """Map each line to its current section and report whether headings exist."""
    values = list(lines)
    current: Optional[str] = None
    mapped: List[Optional[str]] = []
    recognized = False
    for line in values:
        heading = resume_heading_kind(line)
        if heading is not None:
            current = heading
            recognized = recognized or heading != "other"
        mapped.append(current)
    return mapped, recognized


def section_items(text: str, kinds: set[str], *, limit: int = 30) -> List[Tuple[str, str]]:
    """Return short, verbatim lines from selected structured sections."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    mapped, recognized = resume_section_map(lines)
    if not recognized:
        return []

    out: List[Tuple[str, str]] = []
    seen = set()
    for line, kind in zip(lines, mapped):
        if kind not in kinds or resume_heading_kind(line) is not None:
            continue
        value = re.sub(r"^[\s•●▪◦\-–—]+", "", line).strip()
        key = re.sub(r"\s+", " ", value).lower()
        if len(value) < 3 or len(value) > 400 or key in seen:
            continue
        seen.add(key)
        out.append((kind or "other", value))
        if len(out) >= limit:
            break
    return out


__all__ = ["resume_heading_kind", "resume_section_map", "section_items"]
