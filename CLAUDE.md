# Agent Instructions

> This file is mirrored across CLAUDE.md, AGENTS.md, and GEMINI.md so the same instructions load in any AI environment.

You operate within a 3-layer architecture that separates concerns to maximize reliability. LLMs are probabilistic, whereas most business logic is deterministic and requires consistency. This system fixes that mismatch.

## The 3-Layer Architecture

**Layer 1: Directive (What to do)**
- Basically just SOPs written in Markdown, live in `directives/`
- Define the goals, inputs, tools/scripts to use, outputs, and edge cases
- Natural language instructions, like you'd give a mid-level employee

**Layer 2: Orchestration (Decision making)**
- This is you. Your job: intelligent routing.
- Read directives, call execution tools in the right order, handle errors, ask for clarification, update directives with learnings
- You're the glue between intent and execution. E.g you don't try scraping websites yourself—you read `directives/scrape_website.md` and come up with inputs/outputs and then run `execution/scrape_single_site.py`

**Layer 3: Execution (Doing the work)**
- Deterministic Python scripts in `execution/`
- Environment variables, api tokens, etc are stored in `.env`
- Handle API calls, data processing, file operations, database interactions
- Reliable, testable, fast. Use scripts instead of manual work.

**Why this works:** if you do everything yourself, errors compound. 90% accuracy per step = 59% success over 5 steps. The solution is push complexity into deterministic code. That way you just focus on decision-making.

## Operating Principles

**1. Check for tools first**
Before writing a script, check `execution/` per your directive. Only create new scripts if none exist.


**2. Self-anneal when things break**
- Read error message and stack trace
- Fix the script and test it again (unless it uses paid tokens/credits/etc—in which case you check w user first)
- Update the directive with what you learned (API limits, timing, edge cases)
- Example: you hit an API rate limit → you then look into API → find a batch endpoint that would fix → rewrite script to accommodate → test → update directive.

**3. Update directives as you learn**
Directives are living documents. When you discover API constraints, better approaches, common errors, or timing expectations—update the directive. But don't create or overwrite directives without asking unless explicitly told to. Directives are your instruction set and must be preserved (and improved upon over time, not extemporaneously used and then discarded).

## Self-annealing loop

Errors are learning opportunities. When something breaks:
1. Fix it
2. Update the tool
3. Test tool, make sure it works
4. Update directive to include new flow
5. System is now stronger

## File Organization

**Deliverables vs Intermediates:**
- **Deliverables**: Google Sheets, Google Slides, or other cloud-based outputs that the user can access
- **Intermediates**: Temporary files needed during processing

**Directory structure:**
- `.tmp/` - All intermediate files (dossiers, scraped data, temp exports). Never commit, always regenerated.
- `execution/` - Python scripts (the deterministic tools)
- `directives/` - SOPs in Markdown (the instruction set)
- `.env` - Environment variables and API keys
- `credentials.json`, `token.json` - Google OAuth credentials (required files, in `.gitignore`)

**Key principle:** Local files are only for processing. Deliverables live in cloud services (Google Sheets, Slides, etc.) where the user can access them. Everything in `.tmp/` can be deleted and regenerated.

## Cloud Webhooks (Modal)

The system supports event-driven execution via Modal webhooks. Each webhook maps to exactly one directive with scoped tool access.

**When user says "add a webhook that...":**
1. Read `directives/add_webhook.md` for complete instructions
2. Create the directive file in `directives/`
3. Add entry to `execution/webhooks.json`
4. Deploy: `modal deploy execution/modal_webhook.py`
5. Test the endpoint

**Key files:**
- `execution/webhooks.json` - Webhook slug → directive mapping
- `execution/modal_webhook.py` - Modal app (do not modify unless necessary)
- `directives/add_webhook.md` - Complete setup guide

**Endpoints:**
- `https://nick-90891--claude-orchestrator-list-webhooks.modal.run` - List webhooks
- `https://nick-90891--claude-orchestrator-directive.modal.run?slug={slug}` - Execute directive
- `https://nick-90891--claude-orchestrator-test-email.modal.run` - Test email

**Available tools for webhooks:** `send_email`, `read_sheet`, `update_sheet`

**All webhook activity streams to Slack in real-time.**

## Known Issues & Learnings

Hard-won discoveries from development and debugging sessions. Follow these to avoid repeating past mistakes.

### Security

**Never hardcode credentials in source files.** Always load database connection strings, API keys, and passwords from environment variables via `.env`. Ensure `.env` is in `.gitignore` and never staged in git. This was discovered via hardcoded Neon DB connection strings with plaintext passwords in `cloud_migrate.py` and `test_neon.py` -- those must be refactored to use `os.getenv()`.

### Embedding Service Architecture

**Three embedding providers exist: Cohere, Voyage AI (`voyage-law-2`), and local BGE-M3.** Voyage AI's `voyage-law-2` is the primary choice for legal documents (6-10% better retrieval accuracy on legal benchmarks). When modifying embedding code, always use the base class pattern to avoid duplicating cache/batch logic across providers. Do not add a new provider without extending the base class.

### Vector Store & Database Connections

**Always use `_ensure_connection()` or the `get_connection()` context manager when accessing the database.** Never access `self._conn` directly in VectorStore methods -- the connection may be stale or the pool may have replaced it. The VectorStore class supports both single-connection and pooled modes; the context manager handles both correctly.

**Multi-tenant isolation uses PostgreSQL Row-Level Security (RLS).** All queries and inserts must include `client_id`. Tenant context is set via `SET app.current_tenant = client_id`. The demo mode uses client ID `00000000-0000-0000-0000-000000000001`. Omitting `client_id` will cause silent data leakage between tenants.

### Cloud Deployment (Streamlit Cloud)

**Streamlit Cloud has a 1GB RAM limit.** Docling is too heavy for cloud deployment -- use PyMuPDF4LLM exclusively (`use_docling` is hardcoded to `False` in the cloud config). Do not re-enable Docling without switching to a higher-memory deployment target.

**All AI/model cache paths must point to `/tmp/` directories** on Streamlit Cloud. Local paths like `~/.cache/` are not writable in that environment.

### Retrieval Pipeline

**Query classification drives pipeline cost.** The retriever classifies queries into four types: `simple`, `factual`, `analytical`, and `standard`. Each type has different pipeline configurations:
- `analytical` -- full enhancement (query expansion + HyDE + multi-query = 3 LLM calls). Most expensive.
- `simple` -- skips all enhancement for speed. Cheapest. Triggered for queries ≤4 words.
- Always test query classification when modifying retrieval logic to avoid unexpected cost spikes.

**Smart reranking saves 30-50% on Cohere reranking costs.** The retriever skips expensive Cohere reranking API calls when the top result has high confidence (>0.85 score) and a clear lead (>0.15 gap to the second result). Do not remove this optimization without understanding the cost impact.

### LLM Provider

**Uses NVIDIA NIM API (OpenAI-compatible), not Anthropic Claude.** Three models are in use:
- **Answer generation:** Qwen 3 235B (`qwen/qwen3-235b-a22b`) -- 60s timeout, max_tokens=3500
- **Query enhancement** (expansion, HyDE, multi-query): Llama 3.3 70B (`meta/llama-3.3-70b-instruct`) -- 30s timeout. Uses a separate cached client (`_get_enhancement_llm_client()`) for lower latency.
- **Contextual chunking:** Llama 3.2 3B (`meta/llama-3.2-3b-instruct`)

All are accessed via the NVIDIA NIM endpoint. Do not assume Anthropic API conventions when modifying LLM integration code. The enhancement and answer LLM clients are intentionally separate -- do not merge them.

### Latency & Caching

**pgvector HNSW ef_search is tuned to 25** (default is 40). This trades marginal recall for faster vector search. Do not increase without benchmarking latency impact on production query volumes.

**Full-pipeline answer caching stores both retrieval results and generated answers.** `QueryResultCache` returns a `(results, answer_data)` tuple where `answer_data` is `Optional[dict]` with `'answer'` and `'sources'` keys. TTL is 24 hours. Cache is invalidated on document upload or delete. When modifying the cache, always preserve this tuple structure.

**Simple query classification threshold is ≤4 words** (widened from ≤3). This means more queries skip the expensive enhancement pipeline. Test classification changes against representative query sets before deploying.

### SSE Streaming

**The `/api/v1/query/stream` endpoint uses Server-Sent Events** for progressive answer rendering. Event protocol:
- `sources` -- sent once after retrieval with serialized SourceInfo list
- `token` -- sent per chunk during LLM streaming
- `done` -- sent once with `{"latency_ms": float}`
- `error` -- sent on failure

Frontend creates an empty assistant message immediately and updates it progressively via `setMessages` map updates. The skeleton loader only shows when the last message content is an empty string.

### Testing

**The project currently uses script-style tests, not pytest.** When adding new tests, create proper pytest tests in a `tests/` directory with mocks for all external services (Cohere, Voyage AI, NVIDIA NIM, PostgreSQL) so tests can run without API keys or live database connections.

### API Security

**CORS origins are configurable via `CORS_ORIGINS` env var** (comma-separated). Defaults to `http://localhost:5173,http://localhost:3000`. Never use `*` in production. Rate limiting is enforced at 60 requests/minute per API key (configurable via `RATE_LIMIT_RPM` env var).

**Document deletion is tenant-isolated.** The `delete_document()` method accepts an optional `client_id` parameter. The API endpoint always passes the authenticated client's ID to prevent cross-tenant deletion.

### Dependencies

**All dependencies are in `requirements.txt`** (cloud-optimized, no Docling). When adding new dependencies, add them to `requirements.txt`. Unused packages (anthropic, llama-index, asyncpg, httpx, aiofiles, tenacity, pypdf, streamlit-chat) have been removed.

## Summary

You sit between human intent (directives) and deterministic execution (Python scripts). Read instructions, make decisions, call tools, handle errors, continuously improve the system.

Be pragmatic. Be reliable. Self-anneal.

