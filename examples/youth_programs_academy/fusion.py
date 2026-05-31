"""
Normalize extractor payloads and merge program lists with deduplication.

Two merge strategies are supported:
  - ``primary_wins``   — legacy behaviour. Primary row replaces secondary on key collision.
  - ``field_level``    — default. Merge field-by-field, keeping the more-complete value.
                         Arrays (schedules, prices) are unioned with structural dedup.
                         Lets the DepthSearch pass fill empty primary fields without
                         clobbering good primary fields.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "fuse_programs",
    "fuse_provider_profiles",
    "normalize_programs_payload",
    "safe_parse_raw",
    "serialize_answer",
]


_EMPTY_STRINGS = {"", "no data available", "not specified"}


def _is_empty_scalar(value: Any) -> bool:
    """Treat ``""``, ``None``, and ``"no data available"`` (any casing) as empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _EMPTY_STRINGS
    return False


def _better_scalar(a: Any, b: Any) -> Any:
    """Pick the more informative of two scalar values."""
    a_empty = _is_empty_scalar(a)
    b_empty = _is_empty_scalar(b)
    if a_empty and not b_empty:
        return b
    if b_empty and not a_empty:
        return a
    # both empty or both populated — prefer the longer string, else keep `a`.
    if isinstance(a, str) and isinstance(b, str):
        return a if len(a) >= len(b) else b
    return a


def _is_numeric_age(value: Any) -> bool:
    if not isinstance(value, (str, int, float)):
        return False
    try:
        float(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False


def _merge_age_group(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Prefer the AgeGroup with numeric min+max; fall back to per-field scalar merge."""
    a = a or {}
    b = b or {}
    a_ok = _is_numeric_age(a.get("minAge")) and _is_numeric_age(a.get("maxAge"))
    b_ok = _is_numeric_age(b.get("minAge")) and _is_numeric_age(b.get("maxAge"))
    if a_ok and not b_ok:
        return a
    if b_ok and not a_ok:
        return b
    return {
        "minAge": _better_scalar(a.get("minAge"), b.get("minAge")),
        "maxAge": _better_scalar(a.get("maxAge"), b.get("maxAge")),
    }


def _schedule_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (
        str(entry.get("day") or "").strip().lower(),
        str(entry.get("startTime") or "").strip().lower(),
    )


def _price_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("priceType") or "").strip().lower(),
        str(entry.get("priceUnit") or "").strip().lower(),
        str(entry.get("pricePerParticipant") or "").strip().lower(),
    )


def _union_dicts(
    primary: list[Any], secondary: list[Any], key_fn
) -> list[dict[str, Any]]:
    """Union two lists of dicts, dedup by ``key_fn``. Primary's entry wins on collision."""
    seen: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for source in (secondary, primary):
        for item in source or []:
            if not isinstance(item, dict):
                continue
            k = key_fn(item)
            if k not in seen:
                seen[k] = item
                order.append(k)
            else:
                # Merge scalar fields inside the existing entry.
                existing = seen[k]
                for field, value in item.items():
                    existing[field] = _better_scalar(existing.get(field), value)
    return [seen[k] for k in order]


def _merge_programs(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Field-level merge: keep more-complete value per field; union arrays."""
    merged: dict[str, Any] = {**secondary, **primary}

    merged["ageGroup"] = _merge_age_group(
        primary.get("ageGroup") if isinstance(primary.get("ageGroup"), dict) else None,
        secondary.get("ageGroup") if isinstance(secondary.get("ageGroup"), dict) else None,
    )

    merged["schedules"] = _union_dicts(
        primary.get("schedules") or [],
        secondary.get("schedules") or [],
        _schedule_key,
    )
    merged["prices"] = _union_dicts(
        primary.get("prices") or [],
        secondary.get("prices") or [],
        _price_key,
    )

    for scalar_field in (
        "name",
        "description",
        "joiningLink",
        "type",
        "pricingData",
        "indoorOroutdoor",
        "inpersonOrVirtual",
        "parentalSupervisionRequired",
        "maxNumberOfStudents",
        "offerDiscount",
    ):
        merged[scalar_field] = _better_scalar(primary.get(scalar_field), secondary.get(scalar_field))

    # Booleans: OR (either pass spotted it).
    for bool_field in ("isFreeTrial",):
        merged[bool_field] = bool(primary.get(bool_field)) or bool(secondary.get(bool_field))

    # activityRecurring: union days; OR the flag.
    a_ar = primary.get("activityRecurring") if isinstance(primary.get("activityRecurring"), dict) else {}
    b_ar = secondary.get("activityRecurring") if isinstance(secondary.get("activityRecurring"), dict) else {}
    days = list({d for d in (a_ar.get("days") or []) + (b_ar.get("days") or []) if d})
    merged["activityRecurring"] = {
        "days": days,
        "activityRecurring": bool(a_ar.get("activityRecurring")) or bool(b_ar.get("activityRecurring")),
    }

    return merged


_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_SUFFIX_TOKENS = (
    "program", "programs",
    "classes", "class",
    "camps", "camp",
    "lessons", "lesson",
    "care", "daycare",
    "school", "schools",
    "therapy", "therapies",
    "classroom", "classrooms",
    "service", "services",
)


def _fuzzy_name(name: Any) -> str:
    """Normalize a program name for fuzzy dedup ("Tot Tumblers Class" == "tot tumblers",
    "Infant" == "Infant Care"). Single-token programs that ARE a suffix word (e.g. a program
    literally named "Care") keep their name so they don't collapse to an empty key."""
    raw = _PUNCT_RE.sub(" ", str(name or "").lower()).strip()
    tokens = [t for t in raw.split() if t]
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        popped = tokens.pop()
        if not tokens:
            # Restore — never return an empty fuzzy key for a non-empty name.
            tokens.append(popped)
            break
    return " ".join(tokens)


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


def _fuzzy_prog_key(p: dict[str, Any]) -> tuple[str, str]:
    """Dedup key that ignores punctuation, casing, and trailing 'Class/Program' suffix."""
    return (_fuzzy_name(p.get("name")), str(p.get("joiningLink") or "").strip().lower())


def fuse_programs(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    prefer_primary: bool = True,
    strategy: str = "field_level",
) -> list[dict[str, Any]]:
    """Merge two program lists.

    ``strategy``:
        * ``"field_level"`` (default) — merge field-by-field via ``_merge_programs``.
          The depth-search pass fills empty primary fields without clobbering good ones.
        * ``"primary_wins"`` — legacy behaviour: primary row replaces secondary on
          key collision. Selected when caller explicitly passes
          ``strategy="primary_wins"`` or the legacy ``prefer_primary=True`` while
          using the explicit legacy strategy.

    ``prefer_primary`` is retained for backwards compatibility with existing
    callers, but the new default strategy already produces a "primary wins on
    non-empty fields" merge that is strictly better.
    """
    if strategy not in ("field_level", "primary_wins"):
        strategy = "field_level"

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []

    for p in secondary:
        if not isinstance(p, dict):
            continue
        k = _fuzzy_prog_key(p)
        if not k[0] and not k[1]:
            continue
        if k not in merged:
            order.append(k)
        merged[k] = p

    for p in primary:
        if not isinstance(p, dict):
            continue
        k = _fuzzy_prog_key(p)
        if not k[0] and not k[1]:
            continue
        if k not in merged:
            merged[k] = p
            order.append(k)
            continue

        if strategy == "primary_wins" or not prefer_primary:
            # Keep legacy semantics: primary overwrites OR secondary wins when
            # the legacy bool says so.
            if strategy == "primary_wins":
                merged[k] = p
            elif not prefer_primary and k not in merged:
                merged[k] = p
        else:
            merged[k] = _merge_programs(p, merged[k])

    # Second pass: collapse entries that share a fuzzy name. The primary pass
    # often returns generic names with empty joiningLink ("Infant") while the
    # depth pass returns the same program with a specific link ("Infant Care",
    # link=/infant-daycare). These have different keys above so they survive
    # the first pass. Here we merge them when:
    #   - they share a fuzzy_name AND
    #   - at least one side has an empty link (empty acts as a wildcard).
    # When BOTH sides have distinct non-empty links we leave them separate —
    # that's a real "two programs with the same name at different URLs" case.
    if strategy == "field_level":
        by_name: dict[str, list[tuple[str, str]]] = {}
        for k in order:
            by_name.setdefault(k[0], []).append(k)
        for name_key, keys in by_name.items():
            if len(keys) < 2:
                continue
            linked = [k for k in keys if k[1]]
            unlinked = [k for k in keys if not k[1]]
            if not unlinked or not linked:
                continue  # all linked or all unlinked — leave as-is
            target = linked[0]  # merge everything into the first linked entry
            for k in keys:
                if k == target:
                    continue
                if not k[1] or k == target:
                    merged[target] = _merge_programs(merged[target], merged[k])
                    del merged[k]
                    order.remove(k)

    return [merged[k] for k in order if k in merged]


def fuse_provider_profiles(
    primary: dict[str, Any] | None,
    secondary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge two ProviderProfile dicts using the same scalar/array rules.

    Scalar fields keep the more-informative non-empty value (primary wins ties).
    Array fields (``categories``, ``subjects``) are unioned with case-insensitive dedup.
    """
    p = primary or {}
    s = secondary or {}

    out: dict[str, Any] = {}
    for field in ("name", "address", "phone", "email", "description"):
        out[field] = _better_scalar(p.get(field), s.get(field))

    for field in ("categories", "subjects"):
        seen: dict[str, str] = {}
        for src in (p.get(field) or [], s.get(field) or []):
            for item in src:
                if not isinstance(item, str):
                    continue
                k = item.strip().lower()
                if k and k not in seen:
                    seen[k] = item.strip()
        out[field] = list(seen.values())

    return out
