# Plan: Live End-to-End Test (#31)

**Ticket**: https://github.com/Rememora/prememora/issues/31
**Status**: Plan mode — awaiting approval

## Summary

Build a single-command E2E test runner (`e2e/run_live.py`) that exercises the full PreMemora pipeline against real Polymarket markets. This is NOT a pytest unit test — it is a live integration script that validates every stage works together with real APIs. The script will also produce a comprehensive report of what happened so we can diagnose any stage that fails.

The pipeline under test:

```
1. Ingestion (a few sources → Graphiti/Neo4j)
2. Market discovery (Gamma API → active markets)
3. Agent interviews (MiroFish simulation → probability estimates)
4. Edge calculation (Kelly sizing + risk limits)
5. Paper trade execution (SQLite)
6. Auto-resolution + strategy report (wait for market resolution or use already-resolved markets)
```

## Key Design Decision: Two Modes

The ticket says "wait for markets to resolve" but real markets can take days/weeks. We need two modes:

1. **`--mode live`** (default): Full pipeline against active markets. Opens real paper positions. Resolution happens later when you re-run with `--resolve-only`.
2. **`--mode smoke`**: Uses recently-resolved markets from Gamma API. We already know the outcomes so we can complete the full cycle (ingest → interview → trade → resolve → report) in a single run without waiting. This is the primary validation mode.

The smoke mode works by:
- Fetching recently resolved markets from Gamma API
- Using their historical YES price as the "current" market price
- Running the full pipeline (context → interview → edge calc → paper trade)
- Immediately resolving positions using the known outcomes
- Generating the strategy report

## Implementation Steps

### Step 1: Create `e2e/` directory structure

```
e2e/
├── __init__.py
├── run_live.py        # Main CLI entry point
└── helpers.py         # Shared utilities (health checks, logging, timing)
```

### Step 2: `e2e/helpers.py` — Infrastructure health checks + utilities

Functions:
- `check_neo4j(uri, user, password) -> bool` — ping Neo4j with a simple MATCH query
- `check_mirofish(url) -> bool` — GET `/api/health` or `/api/simulation/list`
- `check_minimax_api(api_key) -> bool` — minimal LLM completion test
- `ensure_graph_exists(client, graph_id) -> bool` — check if graph has any data
- `StageTimer` — context manager that tracks duration + success/failure per stage
- `E2EReport` — dataclass collecting results from each stage, with `summary()` method that prints a human-readable report

### Step 3: `e2e/run_live.py` — Main orchestrator

The script runs stages sequentially. Each stage is wrapped in error handling so we get a clear report of exactly which stage failed and why.

```python
async def run_e2e(mode: str, config: E2EConfig) -> E2EReport:
    report = E2EReport()

    # Stage 0: Health checks
    with report.stage("health_checks"):
        assert check_neo4j(...)
        assert check_mirofish(...)  # only if simulation_id provided
        assert check_minimax_api(...)

    # Stage 1: Ingest some data (lightweight — just RSS + crypto news for ~2 min)
    # Skip if graph already has data (allow re-runs without re-ingesting)
    with report.stage("ingestion"):
        if not ensure_graph_has_data(graph_id):
            run_mini_ingestion(graph_id, duration=120)  # 2 minutes
        else:
            report.note("Graph already has data, skipping ingestion")

    # Stage 2: Discover markets
    with report.stage("market_discovery"):
        if mode == "smoke":
            markets = fetch_recently_resolved_markets(limit=5)
        else:
            markets = fetch_active_markets(limit=5)

    # Stage 3: Context enrichment (search graph for each market)
    with report.stage("context_enrichment"):
        for market in markets:
            context = context_builder.build_context(market.question)
            # Log how many facts found

    # Stage 4: Agent interviews
    with report.stage("agent_interviews"):
        for market in markets:
            # Call MiroFish interview API
            # Parse probability from responses
            # If no MiroFish simulation available, use a synthetic fallback
            # (so we can still validate stages 5-6)

    # Stage 5: Edge calculation + paper trading
    with report.stage("edge_calculation_and_trading"):
        for market in markets:
            signal = edge_calc.evaluate(estimate, market_price)
            if signal.action != "SKIP" and paper_engine:
                paper_engine.open_position(...)

    # Stage 6: Resolution + strategy report (smoke mode only)
    with report.stage("resolution"):
        if mode == "smoke":
            for market in resolved_markets:
                paper_engine.resolve_market(market.id, market.outcome)
            review = StrategyReview(paper_engine, signal_log)
            strategy_report = review.generate_report()

    # Print final report
    print(report.summary())
    return report
```

**CLI interface:**
```
python -m e2e.run_live --mode smoke --max-markets 3
python -m e2e.run_live --mode live --simulation-id sim_xxx --graph-id mirofish_xxx
python -m e2e.run_live --resolve-only  # just check for resolved markets and score
```

### Step 4: Handle "no MiroFish simulation" gracefully

The ticket requires MiroFish interviews, but having a running simulation requires manual setup (prepare + start via MiroFish API, takes ~15 minutes). For the smoke test to work standalone:

- If `--simulation-id` is provided, use real MiroFish interviews
- If not provided, generate **synthetic probability estimates** with a note that this is simulated. Use a simple heuristic: `0.5 + random.uniform(-0.2, 0.2)` seeded by market question hash. This still exercises the edge calculator and paper engine.
- The E2E report clearly marks which mode was used

### Step 5: Add lightweight ingestion for E2E (RSS + crypto news only)

We don't want to run all 6 connectors for 2 hours during a test. Create a helper that:
- Runs only RSS and crypto_news connectors (no API keys required for RSS, crypto_news uses free API)
- Ingests for a configurable duration (default 60s for smoke, longer for live)
- Uses the existing `IngestionOrchestrator` with most connectors disabled
- If graph already has data from a previous run, skip ingestion entirely

### Step 6: Fetch recently-resolved markets for smoke mode

Add a function that queries Gamma API for recently resolved markets:
```python
async def fetch_recently_resolved(limit=5) -> list[ResolvedMarket]:
    """Fetch markets that resolved in the last 7 days."""
    # GET /markets?closed=true&order=volume&ascending=false&limit=N
    # Filter to those with resolution + resolvedAt set
    # Return market + outcome + last YES price before resolution
```

This reuses the existing Gamma API code from `pipeline/trigger.py` and `backtesting/polymarket_history.py`.

### Step 7: Tests (pytest)

Add `tests/test_e2e_helpers.py`:
- Test `StageTimer` and `E2EReport` formatting
- Test `fetch_recently_resolved` response parsing (mocked HTTP)
- Test synthetic probability generation (deterministic given same market)
- Test the full smoke flow with mocked APIs (no real network calls)

This is a **unit test for the E2E helpers**, not the E2E test itself. The E2E test is meant to be run manually.

### Step 8: Update data/.gitkeep handling

The E2E test creates files in `data/` (SQLite DBs, signal logs). Add `data/e2e/` subdirectory with its own `.gitkeep` so E2E outputs don't mix with production paper trading data. Use `--db-prefix e2e_` or `--data-dir data/e2e/` flag.

## Testing Strategy

1. **Unit tests** (`tests/test_e2e_helpers.py`): Test helpers with mocked APIs. Must pass in CI.
2. **Manual smoke test**: `python -m e2e.run_live --mode smoke --max-markets 3` — requires Neo4j running + MiniMax API key. This is the primary acceptance criteria validation.
3. **Manual live test**: Full pipeline with real active markets and MiroFish simulation.

## Acceptance Criteria Mapping

| Criteria | How validated |
|----------|--------------|
| Pipeline completes at least one full cycle without errors | Smoke mode runs all 6 stages, E2E report shows all stages green |
| At least 1 paper trade opened based on real signal | Edge calc with resolved-market prices (market_price != our_estimate) should produce at least 1 trade. If all SKIPs, we lower `min_edge` temporarily for the test. |
| Strategy review produces meaningful report after resolution | Smoke mode resolves positions and generates report with Brier score |
| Brier score computed and calibration curve generated | `strategy_review.generate_report()` called on resolved positions |

## Files to Create/Modify

**Create:**
- `e2e/__init__.py`
- `e2e/helpers.py`
- `e2e/run_live.py`
- `tests/test_e2e_helpers.py`
- `data/e2e/.gitkeep`

**Modify:**
- None — this is purely additive. All existing code is reused as-is.

## Risk / Open Questions

1. **Gamma API rate limiting**: The API is public and unauthed. We limit to 5 markets to stay polite.
2. **MiniMax API cost**: Each graph search involves LLM calls (reranking). With 5 markets and ~5 search terms each, that's ~25 LLM calls for context enrichment. Cost is minimal (~$0.01 total).
3. **Neo4j must be running**: The test requires Docker Neo4j. The health check will fail fast with a clear message if it's not running.
4. **No MiroFish simulation by default**: The smoke test uses synthetic probabilities unless `--simulation-id` is explicitly provided. The E2E report clearly shows which mode was used.
