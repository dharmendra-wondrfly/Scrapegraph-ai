"""
Normalize extractor payloads and merge program lists with deduplication.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "fuse_programs",
    "normalize_programs_payload",
    "safe_parse_raw",
    "serialize_answer",
]


def safe_parse_raw(raw: Any) -> Any:
    """
    Recover valid structure from pipe-concatenated JSON strings.

    SmartScraperMultiGraph / MergeAnswers occasionally concatenates subgraph
    outputs with ``|``, producing invalid JSON like ``{"programs":[]}|{"programs":[...]}``.
    """
    if not isinstance(raw, str):
        return raw
    if "|" not in raw:
        return raw

    all_programs: list[Any] = []
    meta_sources: list[Any] = []

    for chunk in raw.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            progs = parsed.get("programs", [])
            if isinstance(progs, list):
                all_programs.extend(progs)
            src = parsed.get("sources") or parsed.get("considered_urls")
            if src:
                if isinstance(src, list):
                    meta_sources.extend(src)
                else:
                    meta_sources.append(src)

    out: dict[str, Any] = {"programs": all_programs}
    if meta_sources:
        out["sources"] = meta_sources
    return out


def serialize_answer(result: Any) -> dict[str, Any]:
    """Mirror extract_youth_programs helpers — coerce models/str/dict."""
    if hasattr(result, "model_dump"):
        return result.model_dump(by_alias=True)
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_text": result}
    return {"repr": repr(result)}


def normalize_programs_payload(raw: Any) -> dict[str, Any]:
    """
    Return {\"programs\": [...]} stripping MergeAnswers extras like \"sources\".
    """
    if isinstance(raw, str) and "|" in raw:
        raw = safe_parse_raw(raw)

    data = serialize_answer(raw)
    if not isinstance(data, dict):
        return {"programs": []}
    programs = data.get("programs")
    if not isinstance(programs, list):
        programs = []
    out = {"programs": programs}
    meta = data.get("sources") or data.get("considered_urls")
    if meta:
        out["_meta_sources"] = meta
    return out


def _prog_key(p: dict[str, Any]) -> tuple[str, str]:
    name = str(p.get("name") or "").strip().lower()
    link = str(p.get("joiningLink") or "").strip().lower()
    return (name, link)


def fuse_programs(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    prefer_primary: bool = True,
) -> list[dict[str, Any]]:
    """
    Merge two program lists. When ``prefer_primary``, rows from ``primary``
    overwrite ``secondary`` for the same (name, joiningLink) key.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for p in secondary:
        if not isinstance(p, dict):
            continue
        k = _prog_key(p)
        if not k[0] and not k[1]:
            continue
        merged[k] = p

    for p in primary:
        if not isinstance(p, dict):
            continue
        k = _prog_key(p)
        if not k[0] and not k[1]:
            continue
        if prefer_primary or k not in merged:
            merged[k] = p

    return list(merged.values())
