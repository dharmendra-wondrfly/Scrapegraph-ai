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
    fuse_programs,
    fuse_provider_profiles,
    normalize_programs_payload,
    safe_parse_raw,
)
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
    model: str = "openai/gpt-4o-mini"
    model_depth: str = "openai/gpt-4o"
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
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY or OPENAI_APIKEY environment variable"
        )

    base = req.url.strip().rstrip("/")

    # Step 1: URL discovery (sitemap + shallow BFS) — split into program vs aux pages.
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

    # joiningLink must come from the discovered program URLs — inject them into the prompt.
    prompt = build_prompt(req.url, candidate_urls=program_urls)

    # Step 2: Primary extraction (SmartScraperMultiGraph or Lite variant)
    cfg = {
        "llm": {"api_key": api_key, "model": req.model},
        "verbose": True,
        "headless": True,
    }
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

    primary_raw = graph.run()
    if isinstance(primary_raw, str):
        primary_raw = primary_raw.replace("```json", "").replace("```", "")

    primary_raw = safe_parse_raw(primary_raw)
    primary_norm = normalize_programs_payload(primary_raw)
    programs_out = list(primary_norm["programs"])

    primary_provider: dict[str, Any] = {}
    if isinstance(primary_raw, dict) and isinstance(primary_raw.get("provider"), dict):
        primary_provider = primary_raw["provider"]

    # Step 3: Optional DepthSearch fusion (field-level merge by default).
    run_depth = req.fusion_depth == "always" or (
        req.fusion_depth == "sparse" and len(program_urls) < req.min_urls_for_depth
    )

    depth_provider: dict[str, Any] = {}
    if run_depth:
        depth_cfg = {
            "llm": {"api_key": api_key, "model": req.model_depth},
            "verbose": True,
            "headless": True,
            "depth": req.depth_search_depth,
            "only_inside_links": True,
            "rag_score_threshold": req.rag_score_threshold,
        }
        dg = DepthSearchGraph(
            prompt=prompt,
            source=base,
            config=depth_cfg,
            schema=ProgramsResponse,
        )
        depth_raw = dg.run()
        if isinstance(depth_raw, str):
            depth_raw = depth_raw.replace("```json", "").replace("```", "")

        depth_raw = safe_parse_raw(depth_raw)
        depth_norm = normalize_programs_payload(depth_raw)
        programs_out = fuse_programs(
            programs_out, depth_norm["programs"], strategy="field_level"
        )
        if isinstance(depth_raw, dict) and isinstance(depth_raw.get("provider"), dict):
            depth_provider = depth_raw["provider"]

    provider_profile = fuse_provider_profiles(primary_provider, depth_provider)

    # Step 4: Provider-profile pass — aux pages (about/contact/location) feed a small
    # lite-graph run that focuses only on provider-level fields. Merged into the final
    # provider object so home + about + contact data agree.
    if req.extract_provider_profile:
        profile_sources = [base, *aux_urls]
        try:
            profile_cfg = {
                "llm": {"api_key": api_key, "model": req.model},
                "verbose": True,
                "headless": True,
            }
            pg = SmartScraperMultiLiteGraph(
                prompt=build_provider_profile_prompt(req.url),
                source=profile_sources,
                config=profile_cfg,
                schema=ProviderProfile,
            )
            profile_raw = pg.run()
            if isinstance(profile_raw, str):
                profile_raw = profile_raw.replace("```json", "").replace("```", "")
            profile_raw = safe_parse_raw(profile_raw)
            if isinstance(profile_raw, dict):
                # When the graph returns the full ProviderProfile dict directly.
                provider_profile = fuse_provider_profiles(profile_raw, provider_profile)
        except Exception as exc:  # pragma: no cover — non-fatal; just log and continue
            print(f"[provider_profile] skipped: {exc}", flush=True)

    payload: dict[str, Any] = {
        "programs": programs_out,
        "provider": provider_profile,
    }
    if req.emit_discovery:
        payload["discovered_urls"] = discovered
        payload["aux_urls"] = aux_urls
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
