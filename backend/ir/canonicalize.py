from __future__ import annotations

import re
from typing import Dict


_DOMAIN_ALIASES: Dict[str, str] = {
    "finance domain knowledge": "Financial Domain Knowledge",
    "financial domain knowledge": "Financial Domain Knowledge",
}

_TOOL_ALIASES: Dict[str, str] = {
    "api gateway": "AWS API Gateway",
    "aws api gateway": "AWS API Gateway",
    "s3": "AWS S3",
    "aws s3": "AWS S3",
    "lambda": "AWS Lambda",
    "aws lambda": "AWS Lambda",
    "iam": "AWS IAM",
    "aws iam": "AWS IAM",
    "rag": "Retrieval-Augmented Generation",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "vector based search": "Vector Search",
    "vector-based search": "Vector Search",
    "vector search": "Vector Search",
    "openai gpt-4": "GPT-4",
    "gpt-4": "GPT-4",
    "google gemini": "Gemini",
    "gemini": "Gemini",
    "rdbms": "Relational Databases",
    "relational": "Relational Databases",
    "relational database": "Relational Databases",
    "relational databases": "Relational Databases",
    "relational database management system": "Relational Databases",
    "relational database management systems": "Relational Databases",
    "nosql database": "NoSQL Databases",
    "nosql databases": "NoSQL Databases",
}


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
