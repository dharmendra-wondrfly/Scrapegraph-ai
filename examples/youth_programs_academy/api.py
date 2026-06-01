"""
FastAPI microservice — ScrapeGraphAI youth-program extraction pipeline.

Start:
    cd Scrapegrapghaiforprograms
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Endpoints:
    POST /extract  – discover URLs + LLM-extract programs from a provider website
    GET  /health   – readiness probe
"""

from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load env vars — same search order as extract_youth_programs.py CLI
_here = Path(__file__).resolve().parent
_repo_root = _here.parent
load_dotenv(_repo_root / ".env")
load_dotenv(_here / ".env")
load_dotenv()

# Alias OPEN_AI → OPENAI_API_KEY so ScrapeGraphAI finds the key from our .env
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPEN_AI"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI"]

# Ensure local modules (discovery, fusion, prompt, schema) are importable
sys.path.insert(0, str(_here))

from discovery import discover_program_and_aux_urls, discover_program_urls  # noqa: E402
from fusion import (  # noqa: E402
    drop_category_and_stub_programs,
    fuse_programs,
    fuse_provider_profiles,
    normalize_programs_payload,
    safe_parse_raw,
)


def _sanitize_json(raw: str) -> str:
    """Fix common LLM JSON escaping errors before parsing."""
    import re
    # LLM sometimes outputs \' instead of ' inside double-quoted strings (invalid JSON)
    # Safe to replace because ' doesn't need escaping in JSON double-quoted strings
    cleaned = raw.replace("\\'", "'")

    # Fix broken escape sequences like \x (not valid in JSON)
    # But preserve valid escapes like \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    cleaned = re.sub(r'\\[^"\\\/bfnrtu]', '', cleaned)

    return cleaned
from prompt import build_prompt, build_provider_profile_prompt  # noqa: E402
from schema import ProgramsResponse, ProviderProfile  # noqa: E402
from scrapegraphai.graphs import (  # noqa: E402
    DepthSearchGraph,
    SmartScraperMultiGraph,
    SmartScraperMultiLiteGraph,
)

app = FastAPI(title="ScrapeGraphAI Youth Programs Service", version="1.0.0")


@app.on_event("startup")
def _prewarm_fastembed() -> None:
    """Download HuggingFace fastembed models at startup so first DepthSearch doesn't stall."""
    try:
        from fastembed import TextEmbedding
        _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("[startup] fastembed model ready")
    except Exception as exc:
        print(f"[startup] fastembed prewarm skipped: {exc}")


class ExtractRequest(BaseModel):
    url: str
    emit_discovery: bool = True
    # gpt-4o-mini fabricates schedules/times (proven in eval); gpt-5-mini respects "no data available".
    # Default gpt-5-mini for accuracy; override to gpt-4o-mini only for cheap/fast smoke tests.
    model: str = "openai/gpt-5-mini"
    model_depth: str = "openai/gpt-5-mini"
    max_urls: int = 40
    seed_depth: int = 2
    lite_threshold: int = 8
    fusion_depth: Literal["never", "sparse", "always"] = "always"
    min_urls_for_depth: int = 5
    depth_search_depth: int = 3
    rag_score_threshold: float = 0.35
    extract_provider_profile: bool = True
    max_aux_urls: int = 2


def _run_extraction(req: ExtractRequest) -> dict[str, Any]:
    """
    Synchronous extraction — FastAPI runs sync `def` routes in a thread-pool
    executor automatically, so Playwright / blocking IO won't block the event loop.
    """
    # Color codes for terminal: Cyan for init/final
    CYAN = "\033[96m"
    RESET = "\033[0m"
    print(f"\n{CYAN}[extract] START — url={req.url}{RESET}", flush=True)
    print(f"{CYAN}[extract] params: model={req.model}, model_depth={req.model_depth}, "
          f"fusion_depth={req.fusion_depth}, max_urls={req.max_urls}, "
          f"seed_depth={req.seed_depth}, lite_threshold={req.lite_threshold}, "
          f"extract_provider_profile={req.extract_provider_profile}{RESET}", flush=True)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY or OPENAI_APIKEY environment variable"
        )

    base = req.url.strip().rstrip("/")

    # STAGE 1: URL discovery (sitemap + shallow BFS) — split into program vs aux pages.
    BLUE = "\033[94m"
    print(f"\n{BLUE}[stage-1-discovery] discovering URLs with seed_depth={req.seed_depth}, "
          f"max_urls={req.max_urls}, max_aux={req.max_aux_urls}...{RESET}", flush=True)
    program_urls, aux_urls = discover_program_and_aux_urls(
        base,
        max_urls=req.max_urls,
        seed_depth=req.seed_depth,
        headless=True,
        loader_kwargs={},
        max_aux=req.max_aux_urls,
    )
    if not program_urls:
        program_urls = [base]
    discovered = program_urls + aux_urls
    print(f"{BLUE}[stage-1-discovery] found {len(program_urls)} program URLs, "
          f"{len(aux_urls)} aux URLs (about/contact){RESET}", flush=True)
    print(f"{BLUE}[stage-1-discovery] program_urls: {program_urls}{RESET}", flush=True)

    # joiningLink must come from the discovered program URLs — inject them into the prompt.
    prompt = build_prompt(req.url, candidate_urls=program_urls)

    # STAGE 2: Primary extraction (SmartScraperMultiGraph or Lite variant)
    # Route based on lite_threshold: if few URLs, use lite (faster); else full graph (more capable)
    GREEN = "\033[92m"
    # Accuracy-neutral speedups applied to every fetch: skip image/css/font/media
    # downloads (we only read text) and fail a dead page fast instead of hanging.
    FAST_LOADER = {"block_resources": True}
    FETCH_TIMEOUT = 45
    cfg = {
        "llm": {"api_key": api_key, "model": req.model},
        "verbose": True,
        "headless": True,
        "loader_kwargs": FAST_LOADER,
        "timeout": FETCH_TIMEOUT,
    }
    graph_type = "SmartScraperMultiLiteGraph" if len(program_urls) <= req.lite_threshold else "SmartScraperMultiGraph"
    print(f"\n{GREEN}[stage-2-primary] running {graph_type} with model={req.model} "
          f"on {len(program_urls)} URLs (lite_threshold={req.lite_threshold})...{RESET}", flush=True)

    if len(program_urls) <= req.lite_threshold:
        graph = SmartScraperMultiLiteGraph(
            prompt=prompt,
            source=program_urls,
            config=cfg,
            schema=ProgramsResponse,
        )
    else:
        graph = SmartScraperMultiGraph(
            prompt=prompt,
            source=program_urls,
            config=cfg,
            schema=ProgramsResponse,
        )

    try:
        primary_raw = graph.run()
    except Exception as exc:
        # A single un-renderable URL (file download, timeout) must not 500 the
        # whole request. Log and continue — depth + provider stages still run.
        print(f"{GREEN}[stage-2-primary] graph.run() FAILED (continuing with empty primary): {exc}{RESET}", flush=True)
        primary_raw = {"programs": []}
    print(f"{GREEN}[stage-2-primary] LLM RAW RESPONSE:\n{primary_raw}{RESET}", flush=True)
    if isinstance(primary_raw, str):
        primary_raw = primary_raw.replace("```json", "").replace("```", "")
        primary_raw = _sanitize_json(primary_raw)
        print(f"{GREEN}[stage-2-primary] after sanitize:\n{primary_raw}{RESET}", flush=True)

    try:
        primary_raw = safe_parse_raw(primary_raw)
        print(f"{GREEN}[stage-2-primary] after safe_parse_raw:\n{primary_raw}{RESET}", flush=True)
    except Exception as exc:
        print(f"{GREEN}[stage-2-primary] ERROR parsing JSON: {exc}{RESET}", flush=True)
        print(f"{GREEN}[stage-2-primary] raw string sample (first 500 chars):\n{str(primary_raw)[:500]}{RESET}", flush=True)
        raise

    primary_norm = normalize_programs_payload(primary_raw)
    programs_out = list(primary_norm["programs"])
    print(f"{GREEN}[stage-2-primary] extracted {len(programs_out)} programs (after dedup/normalize){RESET}", flush=True)

    # Log per-program field details
    for idx, prog in enumerate(programs_out):
        populated_fields = [k for k, v in prog.items() if v and v not in ("", "no data available", "not specified")]
        missing_fields = [k for k, v in prog.items() if not v or v in ("", "no data available", "not specified")]
        print(f"{GREEN}[stage-2-primary] program[{idx}] — populated: {populated_fields} | missing: {missing_fields}{RESET}", flush=True)
        print(f"{GREEN}[stage-2-primary] program[{idx}] details:\n{prog}{RESET}", flush=True)

    primary_provider: dict[str, Any] = {}
    if isinstance(primary_raw, dict) and isinstance(primary_raw.get("provider"), dict):
        primary_provider = primary_raw["provider"]

    # STAGE 4 (LAUNCH EARLY): the provider-profile graph only needs base + aux URLs,
    # so its slow fetch+LLM runs in a background thread CONCURRENT with stage-3 depth.
    # We collect + fuse its result after depth completes. Accuracy-neutral: same graph,
    # same inputs — only the wall-clock overlaps.
    MAGENTA = "\033[95m"
    _executor: ThreadPoolExecutor | None = None
    profile_future = None
    if req.extract_provider_profile:
        profile_sources = [base, *aux_urls]
        print(f"\n{MAGENTA}[stage-4-provider] LAUNCHED in background on {len(profile_sources)} sources "
              f"(base + {len(aux_urls)} aux_urls): {profile_sources}{RESET}", flush=True)

        def _run_provider_profile() -> Any:
            profile_cfg = {
                "llm": {"api_key": api_key, "model": req.model},
                "verbose": True,
                "headless": True,
                "loader_kwargs": FAST_LOADER,
                "timeout": FETCH_TIMEOUT,
            }
            pg = SmartScraperMultiLiteGraph(
                prompt=build_provider_profile_prompt(req.url),
                source=profile_sources,
                config=profile_cfg,
                schema=ProviderProfile,
            )
            return pg.run()

        _executor = ThreadPoolExecutor(max_workers=1)
        profile_future = _executor.submit(_run_provider_profile)

    # STAGE 3: Optional DepthSearch fusion (field-level merge to refine accuracy)
    # Controls: fusion_depth ("never"/"sparse"/"always") + min_urls_for_depth threshold
    # Decision logic: skip if fusion_depth="never"; run if "always"; run if "sparse" AND few URLs
    YELLOW = "\033[93m"
    run_depth = req.fusion_depth == "always" or (
        req.fusion_depth == "sparse" and len(program_urls) < req.min_urls_for_depth
    )
    print(f"\n{YELLOW}[stage-3-fusion] decision: fusion_depth={req.fusion_depth}, "
          f"min_urls_for_depth={req.min_urls_for_depth}, "
          f"discovered_urls={len(program_urls)}, run_depth={run_depth}{RESET}", flush=True)

    depth_provider: dict[str, Any] = {}
    if run_depth:
        print(f"{YELLOW}[stage-3-fusion] running DepthSearchGraph with model_depth={req.model_depth}, "
              f"depth={req.depth_search_depth}, rag_score_threshold={req.rag_score_threshold}...{RESET}", flush=True)
        depth_cfg = {
            "llm": {"api_key": api_key, "model": req.model_depth},
            "verbose": True,
            "headless": True,
            "depth": req.depth_search_depth,
            "only_inside_links": True,
            "rag_score_threshold": req.rag_score_threshold,
            "loader_kwargs": FAST_LOADER,
            "timeout": FETCH_TIMEOUT,
        }
        dg = DepthSearchGraph(
            prompt=prompt,
            source=base,
            config=depth_cfg,
            schema=ProgramsResponse,
        )
        try:
            depth_raw = dg.run()
        except Exception as exc:
            # DepthSearch crash (bad link, timeout) must not 500; keep primary results.
            print(f"{YELLOW}[stage-3-fusion] DepthSearch FAILED (skipping fusion): {exc}{RESET}", flush=True)
            depth_raw = {"programs": []}
        print(f"{YELLOW}[stage-3-fusion] DepthSearch LLM RAW RESPONSE:\n{depth_raw}{RESET}", flush=True)
        if isinstance(depth_raw, str):
            depth_raw = depth_raw.replace("```json", "").replace("```", "")
            depth_raw = _sanitize_json(depth_raw)

        depth_raw = safe_parse_raw(depth_raw)
        print(f"{YELLOW}[stage-3-fusion] after safe_parse_raw:\n{depth_raw}{RESET}", flush=True)

        depth_norm = normalize_programs_payload(depth_raw)
        print(f"{YELLOW}[stage-3-fusion] BEFORE fusion: {len(programs_out)} programs{RESET}", flush=True)
        for idx, prog in enumerate(programs_out):
            print(f"{YELLOW}[stage-3-fusion] before[{idx}]:\n{prog}{RESET}", flush=True)

        print(f"{YELLOW}[stage-3-fusion] DepthSearch extracted {len(depth_norm['programs'])} programs to merge{RESET}", flush=True)
        for idx, prog in enumerate(depth_norm["programs"]):
            print(f"{YELLOW}[stage-3-fusion] depth_extracted[{idx}]:\n{prog}{RESET}", flush=True)

        programs_out = fuse_programs(
            programs_out, depth_norm["programs"], strategy="field_level"
        )
        print(f"{YELLOW}[stage-3-fusion] AFTER fusion: {len(programs_out)} programs{RESET}", flush=True)
        for idx, prog in enumerate(programs_out):
            populated_fields = [k for k, v in prog.items() if v and v not in ("", "no data available", "not specified")]
            missing_fields = [k for k, v in prog.items() if not v or v in ("", "no data available", "not specified")]
            print(f"{YELLOW}[stage-3-fusion] after[{idx}] — populated: {populated_fields} | missing: {missing_fields}{RESET}", flush=True)
            print(f"{YELLOW}[stage-3-fusion] after[{idx}] details:\n{prog}{RESET}", flush=True)

        if isinstance(depth_raw, dict) and isinstance(depth_raw.get("provider"), dict):
            depth_provider = depth_raw["provider"]
    else:
        print(f"{YELLOW}[stage-3-fusion] skipped (fusion_depth={req.fusion_depth}){RESET}", flush=True)

    provider_profile = fuse_provider_profiles(primary_provider, depth_provider)

    # STAGE 4 (COLLECT): join the background provider-profile graph and fuse its
    # result into the stage-2/3 provider object.
    if profile_future is not None:
        print(f"\n{MAGENTA}[stage-4-provider] collecting background result...{RESET}", flush=True)
        try:
            profile_raw = profile_future.result()
            print(f"{MAGENTA}[stage-4-provider] LLM RAW RESPONSE:\n{profile_raw}{RESET}", flush=True)
            if isinstance(profile_raw, str):
                profile_raw = profile_raw.replace("```json", "").replace("```", "")
                profile_raw = _sanitize_json(profile_raw)
            profile_raw = safe_parse_raw(profile_raw)
            print(f"{MAGENTA}[stage-4-provider] after safe_parse_raw:\n{profile_raw}{RESET}", flush=True)
            if isinstance(profile_raw, dict):
                # When the graph returns the full ProviderProfile dict directly.
                print(f"{MAGENTA}[stage-4-provider] BEFORE merge (from stage-2/3): {provider_profile}{RESET}", flush=True)
                provider_profile = fuse_provider_profiles(profile_raw, provider_profile)
                print(f"{MAGENTA}[stage-4-provider] AFTER merge: {provider_profile}{RESET}", flush=True)
                populated_provider_fields = [k for k, v in provider_profile.items() if v and v not in ("", "no data available", "not specified")]
                missing_provider_fields = [k for k, v in provider_profile.items() if not v or v in ("", "no data available", "not specified")]
                print(f"{MAGENTA}[stage-4-provider] merged provider — populated: {populated_provider_fields} | missing: {missing_provider_fields}{RESET}", flush=True)
        except Exception as exc:  # pragma: no cover — non-fatal; just log and continue
            print(f"{MAGENTA}[stage-4-provider] error (non-fatal): {exc}{RESET}", flush=True)
        finally:
            if _executor is not None:
                _executor.shutdown(wait=False)
    else:
        print(f"\n{MAGENTA}[stage-4-provider] skipped (extract_provider_profile=False){RESET}", flush=True)

    # STAGE 5: Drop fake programs — category/umbrella cards ("Camps", "Gym Classes")
    # and empty daycare/care stubs ("Care", "Outdoor Play") that carry no concrete
    # data (no numeric age, no schedule, no price). Keeps real detail pages.
    RED = "\033[91m"
    before_n = len(programs_out)
    programs_out, dropped = drop_category_and_stub_programs(programs_out)
    print(f"\n{RED}[stage-5-filter] dropped {len(dropped)} fake/category programs "
          f"({before_n} → {len(programs_out)}){RESET}", flush=True)
    for d in dropped:
        print(f"{RED}[stage-5-filter] DROPPED: name={d.get('name')!r} "
              f"link={d.get('joiningLink')!r} (no age/schedule/price){RESET}", flush=True)

    payload: dict[str, Any] = {
        "programs": programs_out,
        "provider": provider_profile,
    }
    if req.emit_discovery:
        payload["discovered_urls"] = discovered
        payload["aux_urls"] = aux_urls

    # FINAL: Return complete extraction payload
    print(f"\n{CYAN}[extract] DONE — returning {len(programs_out)} programs + provider profile{RESET}", flush=True)
    print(f"{CYAN}[extract] final_payload keys: {list(payload.keys())}{RESET}", flush=True)
    print(f"{CYAN}[extract] programs sample: {programs_out[:1] if programs_out else []}{RESET}", flush=True)
    return payload


@app.post("/extract")
def extract(req: ExtractRequest) -> dict[str, Any]:
    """Discover URLs and extract youth programs from a provider website."""
    try:
        return _run_extraction(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
