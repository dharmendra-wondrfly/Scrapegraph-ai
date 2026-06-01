# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ScrapeGraphAI is a web-scraping library that builds LLM-powered extraction pipelines as directed graphs of nodes. The user states a prompt + source (URL or local file); a graph fetches, parses, RAG-filters, and LLM-extracts structured output (optionally validated against a Pydantic schema).

Python `>=3.12,<4.0`. Package manager is **uv**. Playwright browsers are required for fetching live sites (`playwright install`).

## Commands

```bash
make install        # uv sync + install pre-commit hooks
make lint           # ruff check + black --check + isort --check-only
make type-check     # mypy (strict) on scrapegraphai + tests
make test           # pytest with coverage (term + html + xml)
make pre-commit     # run all pre-commit hooks on all files
make all            # lint + type-check + test

# Single test / file / marker (pytest.ini sets testpaths=tests, asyncio_mode=auto)
uv run pytest tests/test_<name>.py
uv run pytest tests/test_<name>.py::TestClass::test_func
uv run pytest -m unit          # markers: unit, integration, slow, e2e, llm_provider, requires_api_key, benchmark
uv run pytest -m "not integration and not requires_api_key"   # offline-only
```

Lint config: line length 88, ruff selects F/E/W/C (ignores E203/E501/C901), black + isort (black profile), mypy strict with `ignore_missing_imports`. Commits follow semantic-commit conventions (see `SEMANTIC_COMMITS.md`); releases are automated via `.releaserc.yml` — do not hand-edit `CHANGELOG.md` or the version in `pyproject.toml`.

## Architecture

The core abstraction is a **graph of nodes**. Three layers:

1. **`scrapegraphai/graphs/`** — pipeline definitions. `AbstractGraph` (`abstract_graph.py`) is the base: its `__init__` builds the LLM from `config["llm"]` (`_create_llm`), then calls the subclass `_create_graph()` and broadcasts common params (headless, verbose, llm_model, timeout, loader_kwargs) to every node via `set_common_params`. `BaseGraph` (`base_graph.py`) holds `nodes` + `edges` and runs them in `execute()`, threading a shared **state dict** between nodes. Each concrete graph (e.g. `smart_scraper_graph.py`) wires specific nodes into edges; many select an edge/node variation at build time from a config-keyed table (e.g. `(html_mode, reasoning, reattempt)` tuple → nodes+edges).

2. **`scrapegraphai/nodes/`** — units of work, all subclass `BaseNode`. A node declares `input`/`output` keys (string expressions over the state dict, e.g. `"user_prompt & (parsed_doc | doc)"`) and implements `execute(state)`. Key nodes: `FetchNode` (Playwright/loader fetch), `ParseNode` / `ParseNodeDepthK` (chunking), `RAGNode` (embeddings + vector filter), `GenerateAnswerNode` (the LLM extraction call), `MergeAnswersNode` (combine multi-source results), `ConditionalNode` (branch). The `*_level_k` / `*_depth_k` variants power recursive crawling.

3. **Supporting modules**: `models/` (custom LLM clients beyond LangChain's — XAI, DeepSeek, Nvidia, OneApi, CLoD, MiniMax; OpenAI/etc. come via `init_chat_model`), `docloaders/` (Chromium/Playwright, BrowserBase, scrape.do fetchers), `helpers/` (`models_tokens.py` token limits per model, `schemas.py`, robots), `prompts/`, `utils/`, `telemetry/`, `integrations/` (Burr bridge for observability via `burr_kwargs`).

**Multi-source graphs** (`*_multi_graph.py`, `*_multi_lite_graph.py`) run a per-URL sub-graph inside `GraphIteratorNode` (batched, async) then merge. **Lite** variants skip heavier nodes for speed/cost. `DepthSearchGraph` crawls links recursively to a configured depth and RAG-filters before extraction.

Graph output: `graph.run()` returns the extracted answer (dict if schema/JSON, else string). LLM JSON output is parsed by LangChain's output parser, which raises `OUTPUT_PARSING_FAILURE` on malformed JSON (e.g. invalid escape sequences in LLM output).

## Example app: `examples/youth_programs_academy/`

A self-contained FastAPI extraction service (its own pipeline on top of the library) used for accuracy work — **not** part of the published package. Key files:

- `api.py` — FastAPI service (`uvicorn api:app --port 8001 --reload`). `POST /extract` runs a 4-stage pipeline: (1) URL discovery → (2) primary extraction via `SmartScraperMultiGraph` or the Lite variant (routed by `lite_threshold`), (3) optional `DepthSearchGraph` fusion, (4) provider-profile pass over aux pages. `ExtractRequest` exposes all tuning params (model, model_depth, max_urls, seed_depth, lite_threshold, fusion_depth, rag_score_threshold, etc.). Loads `OPENAI_API_KEY` (aliases `OPEN_AI`) from `.env`.
- `discovery.py` — sitemap + shallow-BFS URL discovery, splitting program vs auxiliary (about/contact) URLs.
- `fusion.py` — field-level merge of program lists across stages (`fuse_programs`, `fuse_provider_profiles`), plus `safe_parse_raw` / `normalize_programs_payload` for tolerant JSON handling.
- `schema.py` — Pydantic `Program` / `ProviderProfile` / `ProgramsResponse` output schemas.
- `prompt.py` — `build_prompt` (injects candidate URLs for joiningLink) and `build_provider_profile_prompt`.
- `eval_accuracy.py` — offline harness: hits `POST /extract` for each fixture in `eval_fixtures.json`, scores per-field completeness, writes `eval_results_*.csv`. Run `python eval_accuracy.py [--fixtures f.json] [--label TAG] [--service URL]`.
- `extract_youth_programs.py` — CLI equivalent of the same pipeline (no server).

Accuracy note: for JS-rendered booking widgets (embedded iframes), real schedule/price data is not in the fetched static HTML — a stronger model reduces hallucinated filler but cannot recover data that the fetch never retrieved.
