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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

# ── GLM (Z.ai) LLM override ─────────────────────────────────────────────────
# Z.ai's GLM API is OpenAI-compatible. When GLM_API_KEY is set we route the
# ScrapeGraphAI "openai" provider at Z.ai's endpoint so EVERY LLM call uses GLM
# instead of OpenAI. We override the OpenAI env vars (so langchain_openai picks
# up the Z.ai base_url) AND pass api_key/model/base_url explicitly in each llm
# config below — belt-and-suspenders so it works regardless of how the library
# resolves the endpoint. Unset GLM_API_KEY to fall back to OpenAI.
GLM_API_KEY = os.environ.get("GLM_API_KEY")
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/paas/v4/")
GLM_MODEL = os.environ.get("GLM_MODEL", "openai/glm-4.5-flashx")
USE_GLM = bool(GLM_API_KEY)
if USE_GLM:
    os.environ["OPENAI_API_KEY"] = GLM_API_KEY
    os.environ["OPENAI_BASE_URL"] = GLM_BASE_URL
    os.environ["OPENAI_API_BASE"] = GLM_BASE_URL
    print(f"[startup] GLM mode ON — model={GLM_MODEL} via {GLM_BASE_URL}")

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

# Resource types that text extraction never reads — aborting them speeds fetch 2-5x.
_BLOCKED_RESOURCE_TYPES = ("image", "media", "font", "stylesheet")


def _install_block_resources_support() -> None:
    """Make ``loader_kwargs={"block_resources": True}`` work on ANY scrapegraphai build.

    Our patched ChromiumLoader handles ``block_resources`` natively (pops the kwarg,
    adds a context.route that aborts image/media/font/css). A fresh ``pip install
    scrapegraphai`` from PyPI does NOT — there the kwarg would fall through into
    ``browser.launch()`` and raise. This monkeypatch reproduces the native behaviour
    so the speedup is self-contained and can't be silently lost on a clean deploy.

    No-op when native support is detected (i.e. running against the patched repo), so
    we never double-install the route.
    """
    import contextvars
    import inspect

    from scrapegraphai.docloaders.chromium import ChromiumLoader

    try:
        if "block_resources" in inspect.getsource(ChromiumLoader):
            return  # patched build — native support present, nothing to do
    except (OSError, TypeError):
        pass  # source unavailable (compiled/zipped) — fall through and patch defensively

    # Flag scoped to the current async task; read by the wrapped new_context below.
    _block_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "sgi_block_resources", default=False
    )

    from playwright.async_api import Browser as _PWBrowser

    _orig_new_context = _PWBrowser.new_context

    async def _new_context(self, *args, **kwargs):
        ctx = await _orig_new_context(self, *args, **kwargs)
        if _block_flag.get():
            await ctx.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES
                    else route.continue_()
                ),
            )
        return ctx

    _PWBrowser.new_context = _new_context

    # 1) Pop block_resources in __init__ so it is never forwarded to browser.launch().
    _orig_init = ChromiumLoader.__init__

    def _patched_init(self, *args, **kwargs):
        self.block_resources = bool(kwargs.pop("block_resources", False))
        _orig_init(self, *args, **kwargs)

    ChromiumLoader.__init__ = _patched_init

    # 2) Set the contextvar around each scrape so _new_context installs the route.
    for _method in ("ascrape_playwright", "ascrape_with_js_support", "ascrape_undetected_chromium"):
        _orig = getattr(ChromiumLoader, _method, None)
        if _orig is None:
            continue

        def _make_wrapper(orig):
            async def _wrapped(self, *args, **kwargs):
                token = _block_flag.set(bool(getattr(self, "block_resources", False)))
                try:
                    return await orig(self, *args, **kwargs)
                finally:
                    _block_flag.reset(token)

            return _wrapped

        setattr(ChromiumLoader, _method, _make_wrapper(_orig))


# Hosts that explode the depth crawl off-target: social networks, media, link
# shorteners, maps/review/app aggregators. A single provider-footer link to
# instagram.com drags the DepthSearch crawl into Instagram's /popular/<tag>/ graph
# (each page ~80 more tag links) and burns the whole timeout on garbage. We drop
# these from the crawl's next-level link set. Other external links (e.g. booking
# platforms) are still allowed — only these known link-explosive hosts are cut.
_BLOCKED_LINK_HOSTS = frozenset({
    "instagram.com", "facebook.com", "fb.com", "fb.me",
    "twitter.com", "x.com", "t.co",
    "tiktok.com", "youtube.com", "youtu.be",
    "linkedin.com", "pinterest.com", "pin.it",
    "snapchat.com", "reddit.com", "tumblr.com", "threads.net",
    "whatsapp.com", "wa.me", "telegram.org", "t.me",
    "yelp.com", "tripadvisor.com",
    "maps.google.com", "google.com", "goo.gl", "g.page",
    "apple.com", "apps.apple.com", "play.google.com",
})


def _link_host_blocked(netloc: str) -> bool:
    host = (netloc or "").lower().split(":")[0].lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if host in _BLOCKED_LINK_HOSTS:
        return True
    # match subdomains too (e.g. l.instagram.com, business.facebook.com)
    parts = host.split(".")
    if len(parts) >= 2 and ".".join(parts[-2:]) in _BLOCKED_LINK_HOSTS:
        return True
    return False


def _install_link_host_blocklist() -> None:
    """Stop the DepthSearch depth crawl from following social/media/aggregator links.

    scrapegraphai's ``FetchNodeLevelK.get_full_links()`` returns every absolute link
    it finds on a page as a next-level crawl target (``only_inside_links`` defaults
    False). A provider's footer Instagram/Facebook link therefore drags the crawl
    off-domain into link-explosive social graphs, wasting the whole timeout on
    irrelevant pages. We wrap ``get_full_links`` to drop blocked hosts from its
    result. Kept in OUR api.py as a runtime patch so the scrapegraphai library is
    never edited. Idempotent; fails open if the library layout changes.
    """
    try:
        from scrapegraphai.nodes.fetch_node_level_k import FetchNodeLevelK
    except Exception as exc:  # library moved — don't crash startup, just skip
        print(f"[startup] link-host blocklist skipped: {exc}")
        return

    if getattr(FetchNodeLevelK, "_link_host_blocklist_installed", False):
        return

    from urllib.parse import urlparse

    _orig_get_full_links = FetchNodeLevelK.get_full_links

    def _patched_get_full_links(self, base_url, links):
        full = _orig_get_full_links(self, base_url, links)
        kept = [u for u in full if not _link_host_blocked(urlparse(u).netloc)]
        dropped = len(full) - len(kept)
        if dropped:
            try:
                self.logger.info(
                    f"[host-blocklist] dropped {dropped}/{len(full)} off-target "
                    f"links (social/media/aggregator)"
                )
            except Exception:
                pass
        return kept

    FetchNodeLevelK.get_full_links = _patched_get_full_links
    FetchNodeLevelK._link_host_blocklist_installed = True
    print("[startup] link-host blocklist installed (depth crawl stays on-target)")


_install_block_resources_support()
_install_link_host_blocklist()

app = FastAPI(title="ScrapeGraphAI Youth Programs Service", version="1.0.0")


@app.on_event("startup")
def _prewarm_fastembed() -> None:
    """Fully download the fastembed models at startup so the FIRST DepthSearch doesn't
    stall — and, critically, so concurrent providers don't race to download the same
    model and corrupt the snapshot (the ONNX 'File doesn't exist' error). DepthSearch's
    RAG uses 'BAAI/bge-small-en' (→ Qdrant/bge-small-en), which is a DIFFERENT model from
    the '-v1.5' one; we warm both, and actually run an embed() so the ONNX is materialized.
    """
    from fastembed import TextEmbedding
    for name in ("BAAI/bge-small-en", "BAAI/bge-small-en-v1.5"):
        try:
            emb = TextEmbedding(model_name=name)
            list(emb.embed(["warmup"]))  # force the ONNX model to download + load fully
            print(f"[startup] fastembed ready: {name}")
        except Exception as exc:
            print(f"[startup] fastembed prewarm skipped for {name}: {exc}")


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

    # Build the per-graph llm config. In GLM mode every graph uses the GLM model
    # + Z.ai endpoint; otherwise the OpenAI model requested by the caller.
    def _llm(requested_model: str) -> dict[str, Any]:
        if USE_GLM:
            # NOTE: this scrapegraphai build forwards unknown llm keys (e.g.
            # model_tokens) straight into the OpenAI client call, which errors —
            # so we only pass the keys it expects. The "default token size 8192"
            # warning it prints is non-fatal.
            return {"api_key": GLM_API_KEY, "model": GLM_MODEL, "base_url": GLM_BASE_URL}
        return {"api_key": api_key, "model": requested_model}

    # GLM doesn't support OpenAI's structured-output `.parse()` — passing a rigid
    # pydantic schema makes scrapegraphai call `.parse()`, and GLM's markdown-fenced
    # JSON then fails validation (0 programs). In GLM mode we omit the schema so the
    # library uses a lenient JSON parser (strips ```json fences); api.py's
    # _sanitize_json + safe_parse_raw + normalize_programs_payload enforce the shape.
    programs_schema = None if USE_GLM else ProgramsResponse
    profile_schema = None if USE_GLM else ProviderProfile

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
    # GLM-4.7 is a reasoning model — it emits long internal reasoning before answering,
    # so the LLM call needs much more than the 45s used for fast (non-reasoning) models.
    # Env-overridable via SGI_FETCH_TIMEOUT.
    FETCH_TIMEOUT = int(os.environ.get("SGI_FETCH_TIMEOUT", "240" if USE_GLM else "45"))
    cfg = {
        "llm": _llm(req.model),
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
            schema=programs_schema,
        )
    else:
        graph = SmartScraperMultiGraph(
            prompt=prompt,
            source=program_urls,
            config=cfg,
            schema=programs_schema,
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
                "llm": _llm(req.model),
                "verbose": True,
                "headless": True,
                "loader_kwargs": FAST_LOADER,
                "timeout": FETCH_TIMEOUT,
            }
            pg = SmartScraperMultiLiteGraph(
                prompt=build_provider_profile_prompt(req.url),
                source=profile_sources,
                config=profile_cfg,
                schema=profile_schema,
            )
            return pg.run()

        _executor = ThreadPoolExecutor(max_workers=1)
        profile_future = _executor.submit(_run_provider_profile)

    # STAGE 3: Optional DepthSearch fusion (field-level merge to refine accuracy)
    # Controls: fusion_depth ("never"/"sparse"/"always") + min_urls_for_depth threshold
    # Decision logic: skip if fusion_depth="never"; run if "always"; run if "sparse" AND few URLs
    YELLOW = "\033[93m"
    # NOTE: DepthSearch holds a Playwright browser open across its LLM calls. Fast models
    # (OpenAI, glm-4.7-flashx) keep the page alive; SLOW GLM models (air/4.5/4.6/4.7)
    # time out mid-crawl and crash the browser. flashx is fast enough, so depth stays on.
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
            "llm": _llm(req.model_depth),
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
            schema=programs_schema,
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


# Server-side wall-clock cap for one /extract call. Matched to the NestJS receive-side
# axios timeout (30 min) so neither end orphans the other: when extraction exceeds this,
# we return 504 cleanly instead of running on after the client has already given up.
EXTRACTION_DEADLINE_SECONDS = 1800  # 30 min


@app.post("/extract")
def extract(req: ExtractRequest) -> dict[str, Any]:
    """Discover URLs and extract youth programs from a provider website."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run_extraction, req)
    try:
        return future.result(timeout=EXTRACTION_DEADLINE_SECONDS)
    except FuturesTimeoutError:
        # Deadline hit. The worker thread can't be force-killed, so we abandon it and
        # return 504; the NestJS side will retry with lighter settings on a fresh request.
        print(
            f"[extract] ABORTED — exceeded {EXTRACTION_DEADLINE_SECONDS}s deadline for url={req.url}",
            flush=True,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Extraction exceeded {EXTRACTION_DEADLINE_SECONDS}s deadline",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")
    finally:
        executor.shutdown(wait=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
