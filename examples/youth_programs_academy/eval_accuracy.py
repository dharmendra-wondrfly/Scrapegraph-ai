"""
Offline accuracy harness for the ScrapeGraphAI youth-program extractor.

Reads a fixtures file (one provider per row), hits ``POST /extract`` on the
local FastAPI service, then scores per-field completeness on the returned
programs so we can diff "before" vs "after" prompt/schema/fusion changes.

Usage:
    # default fixtures (eval_fixtures.json next to this file) → CSV in this dir
    python eval_accuracy.py

    # custom fixtures + label the run (BASELINE / AFTER_PY / AFTER_NEST etc.)
    python eval_accuracy.py --fixtures my_fixtures.json --label AFTER_PY

    # different service URL
    python eval_accuracy.py --service http://localhost:8001

Fixture file shape (JSON list):
    [
      {
        "url": "https://example.com",
        "expected_programs_min": 3,
        "must_have_fields": ["ageGroup", "prices", "schedules", "joiningLink"]
      },
      ...
    ]

If the fixtures file is missing, this script writes a starter file with two
illustrative entries — replace with your validated providers from
Union/Passaic/Kings counties.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


_DEFAULT_MUST_HAVE = ["ageGroup", "prices", "schedules", "joiningLink", "description"]
_EMPTY_STRINGS = {"", "no data available", "not specified"}
_HERE = Path(__file__).resolve().parent
_DEFAULT_FIXTURES = _HERE / "eval_fixtures.json"


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _EMPTY_STRINGS
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _field_populated(program: dict[str, Any], field: str) -> bool:
    if field == "ageGroup":
        ag = program.get("ageGroup") or {}
        if not isinstance(ag, dict):
            return False
        return not _is_empty(ag.get("minAge")) and not _is_empty(ag.get("maxAge"))
    value = program.get(field)
    return not _is_empty(value)


def _score_program(program: dict[str, Any], must_have: list[str]) -> tuple[float, list[str]]:
    """Return (completeness 0..1, list of missing field names) for one program."""
    if not must_have:
        return 1.0, []
    missing = [f for f in must_have if not _field_populated(program, f)]
    score = 1.0 - (len(missing) / len(must_have))
    return score, missing


def _provider_score(provider: dict[str, Any]) -> tuple[float, list[str]]:
    """Completeness across the provider-profile object."""
    if not isinstance(provider, dict):
        return 0.0, ["address", "description", "phone", "email", "categories"]
    fields = ["address", "description", "phone", "email"]
    missing = [f for f in fields if _is_empty(provider.get(f))]
    cats = provider.get("categories") or []
    if not isinstance(cats, list) or len(cats) == 0:
        missing.append("categories")
    populated = (len(fields) + 1) - len(missing)
    return populated / (len(fields) + 1), missing


def _call_extract(service: str, url: str, timeout: int) -> dict[str, Any]:
    resp = requests.post(
        f"{service.rstrip('/')}/extract",
        json={"url": url},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _ensure_fixtures(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    starter = [
        {
            "url": "https://example-childcare.com",
            "expected_programs_min": 3,
            "must_have_fields": _DEFAULT_MUST_HAVE,
        },
        {
            "url": "https://example-dance-studio.com",
            "expected_programs_min": 3,
            "must_have_fields": _DEFAULT_MUST_HAVE,
        },
    ]
    path.write_text(json.dumps(starter, indent=2), encoding="utf-8")
    print(
        f"[eval] wrote starter fixtures to {path} — replace with real provider URLs "
        "from already-validated counties (Union/Passaic/Kings) and re-run.",
        flush=True,
    )
    return starter


def run_eval(
    service: str,
    fixtures_path: Path,
    label: str,
    out_dir: Path,
    timeout: int,
) -> Path:
    fixtures = _ensure_fixtures(fixtures_path)
    if not fixtures:
        raise RuntimeError(f"no fixtures found in {fixtures_path}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"eval_results_{ts}_{label}.csv"

    rows: list[dict[str, Any]] = []
    all_avg: list[float] = []
    all_provider: list[float] = []

    for fx in fixtures:
        url = fx.get("url")
        must_have = fx.get("must_have_fields") or _DEFAULT_MUST_HAVE
        expected_min = int(fx.get("expected_programs_min") or 0)

        if not url:
            continue

        print(f"[eval] {url} …", flush=True)
        start = time.time()
        try:
            payload = _call_extract(service, url, timeout)
            error = ""
        except Exception as exc:
            payload = {}
            error = str(exc)[:200]

        programs = payload.get("programs") or []
        provider = payload.get("provider") or {}
        elapsed = round(time.time() - start, 2)

        per_program_scores: list[float] = []
        missing_counter: dict[str, int] = {}
        for prog in programs:
            score, missing = _score_program(prog, must_have)
            per_program_scores.append(score)
            for m in missing:
                missing_counter[m] = missing_counter.get(m, 0) + 1

        avg = sum(per_program_scores) / len(per_program_scores) if per_program_scores else 0.0
        provider_score, provider_missing = _provider_score(provider)

        all_avg.append(avg)
        all_provider.append(provider_score)

        rows.append(
            {
                "url": url,
                "elapsed_sec": elapsed,
                "program_count": len(programs),
                "expected_min": expected_min,
                "meets_expected_min": len(programs) >= expected_min,
                "avg_completeness": round(avg, 3),
                "missing_field_freq": json.dumps(missing_counter, sort_keys=True),
                "provider_completeness": round(provider_score, 3),
                "provider_missing": ",".join(provider_missing),
                "error": error,
            }
        )

    overall_avg = sum(all_avg) / len(all_avg) if all_avg else 0.0
    overall_provider = sum(all_provider) / len(all_provider) if all_provider else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url",
                "elapsed_sec",
                "program_count",
                "expected_min",
                "meets_expected_min",
                "avg_completeness",
                "missing_field_freq",
                "provider_completeness",
                "provider_missing",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow(
            {
                "url": f"OVERALL ({label})",
                "program_count": sum(r["program_count"] for r in rows),
                "avg_completeness": round(overall_avg, 3),
                "provider_completeness": round(overall_provider, 3),
            }
        )

    print(
        f"[eval] {label}: avg_program_completeness={overall_avg:.3f}, "
        f"avg_provider_completeness={overall_provider:.3f}, csv={out_path}",
        flush=True,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="http://localhost:8001")
    parser.add_argument("--fixtures", type=Path, default=_DEFAULT_FIXTURES)
    parser.add_argument("--label", default="RUN")
    parser.add_argument("--out-dir", type=Path, default=_HERE)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    try:
        run_eval(args.service, args.fixtures, args.label, args.out_dir, args.timeout)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
