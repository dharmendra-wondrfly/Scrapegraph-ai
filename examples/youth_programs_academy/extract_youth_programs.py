#!/usr/bin/env python3
"""
Unified youth-program extraction: URL discovery, SmartScraper multi/lite routing,
optional DepthSearch fusion.

Requires Python >= 3.12, Playwright browsers, OPENAI_API_KEY or OPENAI_APIKEY.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scrapegraphai.graphs import DepthSearchGraph, SmartScraperMultiGraph, SmartScraperMultiLiteGraph

from discovery import discover_program_urls
from fusion import fuse_programs, normalize_programs_payload, safe_parse_raw
from prompt import build_prompt
from schema import ProgramsResponse

# ACADEMY_URLS = [
#     "https://www.buzzingbeesacademy.com/",
#     "https://www.ldvacademy.com/",
#     "http://www.westorangetennisclub.com/",
#     "http://www.maplewoodclub.com/",
#     "https://www.harambeefamilyacademy.org/",
# ]

ACADEMY_URLS = [
    "https://www.buzzingbeesacademy.com/"
]


def _slug(url: str) -> str:
    cleaned = re.sub(r"^https?://", "", url.lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_") or "site"


_OUTPUT_NUM_PATTERN = re.compile(r"^output(\d+)$")


def _next_numbered_output_dir(parent: Path) -> Path:
    """
    Pick ``output{N}`` where N is one greater than the highest existing ``output\\d+``
    sibling directory under ``parent`` (e.g. output7 exists → output8).
    If none exist, use ``output1``.
    """
    max_n = 0
    try:
        for p in parent.iterdir():
            if p.is_dir():
                m = _OUTPUT_NUM_PATTERN.match(p.name)
                if m:
                    max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    return parent / f"output{max_n + 1}"


def _run_primary_extractor(
    *,
    discovered: list[str],
    prompt: str,
    api_key: str,
    model: str,
    headless: bool,
    lite_threshold: int,
) -> Any:
    cfg = {
        "llm": {"api_key": api_key, "model": model},
        "verbose": True,
        "headless": headless,
    }
    if len(discovered) <= lite_threshold:
        graph = SmartScraperMultiLiteGraph(
            prompt=prompt,
            source=discovered,
            config=cfg,
            schema=ProgramsResponse,
        )
    else:
        graph = SmartScraperMultiGraph(
            prompt=prompt,
            source=discovered,
            config=cfg,
            schema=ProgramsResponse,
        )
    return graph.run()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover URLs then extract youth programs (multi/lite + optional DepthSearch fusion)."
    )
    parser.add_argument(
        "--lite-threshold",
        type=int,
        default=8,
        help="Use SmartScraperMultiLiteGraph when len(urls) <= this (default: 8)",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=40,
        help="Cap discovered URLs passed to multi/lite graphs (default: 40)",
    )
    parser.add_argument(
        "--seed-depth",
        type=int,
        default=2,
        help="Internal link expansion depth during discovery (default: 2)",
    )
    parser.add_argument(
        "--min-urls-for-depth",
        type=int,
        default=5,
        help="With --fusion-depth sparse: merge DepthSearch when discovery yields fewer URLs (default: 5)",
    )
    parser.add_argument(
        "--fusion-depth",
        choices=("never", "sparse", "always"),
        default="sparse",
        help="never=no DepthSearch; sparse=DepthSearch when few URLs; always=always merge DepthSearch",
    )
    parser.add_argument(
        "--depth-search-depth",
        type=int,
        default=3,
        help="DepthSearchGraph fetch depth (default: 3)",
    )
    parser.add_argument(
        "--rag-score-threshold",
        type=float,
        default=0.35,
        help="DepthSearch RAG chunk score cutoff (default: 0.35; library default was 0.5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N academy sites from the built-in list",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help='Primary LLM id for SmartScraper multi/lite (default: openai/gpt-4o-mini)',
    )
    parser.add_argument(
        "--model-depth",
        type=str,
        default="openai/gpt-4o",
        help="LLM for DepthSearch fusion pass (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-site JSON files. "
            "Default: next numbered folder output1, output2, … under this example directory "
            "(after existing output7 → output8)."
        ),
    )
    parser.add_argument(
        "--emit-discovery",
        action="store_true",
        help="Include discovered_urls in JSON output",
    )
    args = parser.parse_args()

    _here = Path(__file__).resolve().parent
    if args.output_dir is None:
        args.output_dir = _next_numbered_output_dir(_here)
    _repo_root = _here.parents[2]
    load_dotenv(_repo_root / ".env")
    load_dotenv(_here / ".env")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        print(
            "Missing API key: set OPENAI_API_KEY or OPENAI_APIKEY "
            "(see examples/youth_programs_academy/.env.example).",
            file=sys.stderr,
        )
        return 1

    sites = ACADEMY_URLS[: args.limit] if args.limit else ACADEMY_URLS

    print(f"Output directory: {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for site in sites:
        base = site.strip().rstrip("/")
        slug = _slug(site)
        out_path = args.output_dir / f"{slug}.json"
        prompt = build_prompt(site)

        print(f"\n--- Processing {site} -> {out_path.name} ---", flush=True)

        discovered = discover_program_urls(
            site,
            max_urls=args.max_urls,
            seed_depth=args.seed_depth,
            headless=True,
            loader_kwargs={},
        )
        if not discovered:
            scheme = "https" if site.startswith("https") else "http"
            parsed_rest = site.replace("https://", "").replace("http://", "").strip("/")
            discovered = [f"{scheme}://{parsed_rest}".rstrip("/")]

        print(f"Discovered {len(discovered)} URLs (cap {args.max_urls})", flush=True)

        primary_raw = _run_primary_extractor(
            discovered=discovered,
            prompt=prompt,
            api_key=api_key,
            model=args.model,
            headless=True,
            lite_threshold=args.lite_threshold,
        )
        print(primary_raw)
        primary_raw = safe_parse_raw(primary_raw)
        primary_norm = normalize_programs_payload(primary_raw)
        programs_out = list(primary_norm["programs"])

        run_depth = False
        if args.fusion_depth == "always":
            run_depth = True
        elif args.fusion_depth == "sparse":
            run_depth = len(discovered) < args.min_urls_for_depth

        if run_depth:
            print(
                f"Merging DepthSearchGraph (depth={args.depth_search_depth}, "
                f"rag_score_threshold={args.rag_score_threshold})",
                flush=True,
            )
            depth_cfg = {
                "llm": {"api_key": api_key, "model": args.model_depth},
                "verbose": True,
                "headless": True,
                "depth": args.depth_search_depth,
                "only_inside_links": True,
                "rag_score_threshold": args.rag_score_threshold,
            }
            dg = DepthSearchGraph(
                prompt=prompt,
                source=base,
                config=depth_cfg,
                schema=ProgramsResponse,
            )
            depth_raw = dg.run()
            depth_raw = safe_parse_raw(depth_raw)
            depth_norm = normalize_programs_payload(depth_raw)
            programs_out = fuse_programs(programs_out, depth_norm["programs"], prefer_primary=True)

        payload: dict[str, Any] = {"programs": programs_out}
        if args.emit_discovery:
            payload["discovered_urls"] = discovered

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(programs_out)} programs to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
