from __future__ import annotations

import re
from typing import Dict


# Keep this table deliberately small and industry-neutral. Domain-specific
# equivalence should come from grounded context (including "Long Form (ABC)")
# rather than an ever-growing list of IT product names.
_DOMAIN_ALIASES: Dict[str, str] = {}
_TOOL_ALIASES: Dict[str, str] = {}


def _canon_key(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[\s/_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9\.\+\# ]+", "", s)
    return s.strip()


def canon_domain(name: str) -> str:
    key = _canon_key(name)
    if not key:
        return ""
    return _DOMAIN_ALIASES.get(key, name.strip())


def canon_tool(name: str) -> str:
    key = _canon_key(name)
    if not key:
        return ""
    return _TOOL_ALIASES.get(key, name.strip())
