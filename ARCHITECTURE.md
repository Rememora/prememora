# Architecture

## System Overview

PreMemora is a prediction pipeline with five stages:

```
1. INGEST           2. ANALYZE           3. SIMULATE          4. EVALUATE        5. TRADE
Event Sources  -->  Graphiti+Neo4j  -->  MiroFish Agents  -->  Edge Calc  -->  Paper Trading
(6 connectors)      (knowledge graph)    (22 agent swarm)     (Kelly sizing)   (SQLite positions)
                         ^                                         |
                         |                                         v
                    Data Collector                           Calibration Gate
                    (market prices)                          (oracle-based)
```

All stages are implemented. The system runs in data-collection mode (stages 1-2 + market tracking), with on-demand prediction and paper trading (stages 3-5). Live trading (#13) is deferred until the system proves profitable in paper mode.

## Components

### Graphiti Adapter (`adapter/`)

Drop-in replacement for the Zep Cloud SDK. MiroFish was built against `zep_cloud.client.Zep`; this adapter provides the same interface backed by local Graphiti + Neo4j.

```
adapter/
├── client.py              # GraphitiZepClient — main entry point
│                          #   .graph.search()     — semantic edge search
│                          #   .graph.node.*        — node CRUD + listing
│                          #   .graph.edge.*        — edge CRUD + listing
│                          #   .graph.episode.*     — episode ingestion
├── zep_types.py           # Type stubs (NodeResult, EdgeResult, SearchResults, etc.)
├── ontology_stubs.py      # EntityModel, EdgeModel, EntityText stubs
├── local_embedder.py      # fastembed ONNX embedder (BAAI/bge-small-en-v1.5)
├── minimax_llm_client.py  # Graphiti OpenAIClient subclass for MiniMax
├── minimax_embedder.py    # MiniMax embedding API (rate-limited, unused)
├── minimax_reranker.py    # LLM-based reranker (MiniMax has no logprobs)
└── patch_mirofish.py      # Rewrites MiroFish import statements (7 files)
```

**Key design choice**: The adapter translates at the API boundary. MiroFish services call `client.graph.search()` thinking they're talking to Zep Cloud; the adapter translates to Graphiti's `graphiti.search()` and Neo4j Cypher queries, then wraps results in Zep-compatible types.

### MiroFish (`vendor/mirofish/`)

Cloned upstream MiroFish with patched imports (all `zep_cloud` → `adapter`). Runs as a Flask backend on port 5001.

```
vendor/mirofish/backend/
├── app/
│   ├── api/
│   │   ├── graph.py           # /api/graph/*      — graph build endpoints
│   │   ├── simulation.py      # /api/simulation/*  — simulation lifecycle
│   │   └── report.py          # /api/report/*      — report generation + chat
│   ├── services/
│   │   ├── graph_builder.py           # Ontology → episode ingestion → Neo4j
│   │   ├── ontology_generator.py      # LLM-generated entity/edge ontology
│   │   ├── report_agent.py            # ReACT loop: plan → per-section tool calls
│   │   ├── zep_tools.py               # 4 graph tools for report agent (~1500 lines)
│   │   ├── zep_entity_reader.py       # Entity filtering for simulation setup
│   │   ├── zep_graph_memory_updater.py # Stream agent activities to graph
│   │   ├── simulation_manager.py      # Simulation lifecycle (prepare/start/stop)
│   │   ├── simulation_runner.py       # Subprocess management for OASIS
│   │   ├── oasis_profile_generator.py # LLM-generated agent personas
│   │   └── simulation_config_generator.py # LLM-generated simulation config
│   ├── utils/
│   │   ├── llm_client.py     # OpenAI-format LLM client (MiniMax-adapted)
│   │   └── zep_paging.py     # Cursor-based pagination helper
│   └── config.py             # Env-based config loader
├── scripts/
│   ├── run_twitter_simulation.py   # OASIS Twitter simulation entry point
│   └── run_reddit_simulation.py    # OASIS Reddit simulation entry point
└── frontend/                       # Vue.js UI (port 3000)
```

### Ingestion Orchestrator (`ingestors/orchestrator.py`)

Wires 6 event source connectors to Graphiti for knowledge-graph ingestion. Runs continuously, normalizing events into episode text, deduplicating via LRU set (10K entries), and ingesting via `asyncio.to_thread()` to bridge event loops.

```
ingestors/
├── orchestrator.py        # Main loop: callbacks → dedup → normalize → graph.add()
├── polymarket_ws.py       # Polymarket WebSocket (price changes, trades)
├── crypto_news.py         # Free Crypto News API (streaming + polling fallback)
├── rss_feeds.py           # 7 RSS feeds (AP, Reuters, White House, BBC, etc.)
├── whale_tracker.py       # Whale Alert API (large crypto transfers)
├── reddit_sentiment.py    # Reddit API (requires credentials)
└── fred_macro.py          # FRED macro data (GDP, rates, unemployment)
```

Currently running: RSS + crypto news (no API keys needed). Whale/Reddit/FRED require API keys.

### Pipeline Trigger (`pipeline/trigger.py`)

Orchestrates the prediction cycle: fetch active Polymarket markets, filter by agent expertise, interview MiroFish agents, parse probability estimates, calculate edge, and optionally execute paper trades.

```
pipeline/
├── trigger.py             # PipelineTrigger — main evaluation loop
│                          #   fetch_active_markets() — Gamma API
│                          #   interview_agents() — MiroFish /api/simulation/interview/all
│                          #   parse_probability() — regex extraction from agent text
│                          #   aggregate_probabilities() — trimmed mean
│                          #   relevance filter — keyword matching for agent expertise
├── context.py             # ContextBuilder — searches graph for market-relevant facts
│                          #   extract_search_terms() — entity/noun extraction
│                          #   build_enriched_prompt() — formats facts into interview prompt
└── __init__.py
```

**Relevance filter**: The pipeline skips markets outside agent expertise (sports, weather) using word-boundary keyword matching. Default keywords cover crypto, finance, geopolitics, and tech — matching the 11 MiroFish agent personas (BlackRock, Fed, MicroStrategy, Jerome Powell, Ethereum, etc.).

### Trading Stack (`trading/`)

```
trading/
├── paper_engine.py        # PaperTradingEngine — SQLite position lifecycle
│                          #   open_position(), close_position(), resolve_market()
│                          #   Tracks: entry_confidence, market_deadline, P&L
├── edge_calculator.py     # EdgeCalculator — Kelly sizing with risk limits
│                          #   kelly_fraction() = edge / (1 - market_price)
│                          #   Quarter-Kelly default (0.25×), 10% min edge
├── exit_monitor.py        # ExitMonitor — 4 trigger types for early exit
│                          #   confidence_drop, contradictory_evidence, time_decay, stop_loss
├── strategy_review.py     # StrategyReview — Brier scoring, calibration, recommendations
│                          #   auto_resolve() — polls Gamma API for settled markets
│                          #   brier_score(), calibration_buckets(), source attribution
├── calibration_gate.py    # CalibrationGate — oracle-based pre-trade circuit breaker
│                          #   Blocks trading when Brier > 0.25 (worse than random)
│                          #   Caches results in SQLite, per-category tracking
└── __init__.py
```

### Backtesting (`backtesting/`)

```
backtesting/
├── polymarket_history.py  # Market discovery + CLOB price history fetching
│                          #   discover_markets() — Gamma API with resolution parsing
│                          #   fetch_price_history() — CLOB API hourly prices
├── event_replay.py        # Converts price history into narrative events for graph
├── hindsight.py           # HindsightOracle — backtests against resolved markets
│                          #   LLM probability estimator (MiniMax fallback)
│                          #   Graph vs no-graph comparison mode
└── __init__.py
```

**Key finding**: CLOB API doesn't serve price history for resolved markets. The data collector (`e2e/data_collector.py`) solves this by capturing hourly snapshots while markets are active.

### E2E and Data Collection (`e2e/`)

```
e2e/
├── run_live.py            # E2E test runner (smoke + live modes)
├── soak_test.py           # Continuous paper trading runner (72h)
├── data_collector.py      # Market price + resolution tracker (15min polls)
│                          #   Stores to data/collector/market_history.db
├── helpers.py             # Gamma API helpers, synthetic probs, timing
└── __init__.py
```

### Neo4j (Docker)

Graph database storing all entities and relationships extracted by Graphiti.

- Image: `neo4j:5.26` with APOC plugin
- Ports: 7474 (browser), 7687 (bolt)
- Auth: `neo4j/prememora_local`
- Volumes: `neo4j-data/`, `neo4j-logs/`

**Storage model**: Graphiti stores all relationships as `RELATES_TO` type with the actual name in `r.name` property. Entity labels are stored as a `labels` property on nodes, not as Neo4j node labels.

### OASIS Simulation

Multi-agent social media simulation using camel-ai + camel-oasis. Runs in a separate Python 3.11 venv (`.venv-oasis/`) because camel-oasis requires Python < 3.12.

`simulation_runner.py` spawns the OASIS script as a subprocess using `.venv-oasis/bin/python` with PYTHONPATH set to include both the project root and `vendor/mirofish/backend`.

Output goes to SQLite (`twitter_simulation.db`) in the simulation directory.

## Data Flow

### Continuous Data Collection (always running)
```
RSS feeds + Crypto News API
  → Ingestion Orchestrator (dedup, normalize)
    → asyncio.to_thread(adapter.graph.add())
      → Graphiti extracts entities + edges
        → Neo4j knowledge graph (world events accumulate over time)

Polymarket Gamma API (every 15 min)
  → Data Collector (snapshot_markets)
    → SQLite: market_id, yes_price, volume, timestamp
  → Data Collector (check_resolutions)
    → SQLite: market_id, outcome, last_yes_price, resolved_at
```

### On-Demand Prediction Cycle
```
Trigger: python -m pipeline.trigger --once --simulation-id SIM_ID

  1. Fetch active markets (Gamma API, top 20 by volume)
  2. Relevance filter (keyword match → skip sports/weather)
  3. Build context (search graph for market-relevant facts)
  4. Interview 22 MiroFish agents (enriched prompt with graph context)
  5. Parse probabilities (regex: "70%", "probability: 0.7", "65-75%")
  6. Aggregate (trimmed mean: drop highest + lowest, average rest)
  7. Edge calculation (our_prob - market_price, Kelly sizing)
  8. Calibration gate check (Brier < 0.25 on recent resolved markets)
  9. Execute paper trade if edge > 10% and gate passes
 10. Log signal to JSONL (for strategy review)
```

### Backtesting (on-demand)
```
python -m backtesting.hindsight run --max-markets 10 --use-graph

  1. Discover resolved markets (Gamma API)
  2. Fetch price history (data collector SQLite or CLOB API)
  3. Build graph context from accumulated news events
  4. LLM probability estimate (MiniMax, or synthetic fallback)
  5. Edge calc → paper trade → resolve (outcome known)
  6. Score: Brier, calibration, P&L
  7. Optional: compare mode (graph vs no-graph A/B test)
```

### MiroFish Simulation Lifecycle
```
POST /api/simulation/start
  → simulation_runner.py
    → subprocess: .venv-oasis/bin/python run_parallel_simulation.py
    → OASIS runs 72 rounds (Twitter + Reddit platforms)
    → 22 agents with crypto/finance/geopolitics personas
    → Agents post, like, quote, follow on simulated social media
    → After rounds complete: enters "alive" command-waiting mode
    → Accepts /api/simulation/interview/all for probability questions
```

### Graph Build (manual, via MiroFish UI)
```
User uploads text
  → ontology_generator.py (LLM extracts entity/edge types)
  → graph_builder.py
    → adapter.graph.episode.add() (ingest episodes)
    → Graphiti extracts entities + edges
    → Neo4j stores graph
```

### Report Generation (manual, via MiroFish UI)
```
POST /api/report/generate
  → report_agent.py
    → LLM plans outline (sections)
    → Per section: ReACT loop with tools:
        - InsightForge    → adapter.graph.search()
        - PanoramaSearch  → adapter.graph.node/edge.get_by_graph_id()
        - QuickSearch     → adapter.graph.search()
        - InterviewAgents → /api/simulation/interview/batch (if sim running)
    → Assembled report stored as markdown
```

## Dependencies

| Component | Key Dependencies |
|-----------|-----------------|
| Adapter | graphiti-core, neo4j, fastembed, openai |
| MiroFish Backend | flask, openai, python-dotenv |
| OASIS Simulation | camel-ai==0.2.78, camel-oasis==0.2.5, torch, sentence-transformers |
| LLM Provider | MiniMax M2.5 via OpenAI-compatible API (`api.minimaxi.chat`) |

## Constraints

- MiniMax allows only ONE system message per request (error 2013)
- MiniMax does not support `response_format` or logprobs
- MiniMax responses include `<think>...</think>` blocks that must be stripped
- OASIS requires Python < 3.12 (separate venv: `.venv-oasis/`)
- Agent active hours start at hour 9 — simulations need >= 15 rounds to reach active period
- Graphiti stores relationship type as `RELATES_TO` — actual names are in properties
- CLOB API does not serve price history for resolved markets — use data collector snapshots
- Gamma API has no `resolution` field — derive from `outcomePrices` + `resolvedBy`
- MiroFish simulation subprocess must stay "alive" for interviews — Flask debug mode kills it on file changes
- Interview timeout needs 300s+ for 22 agents (each makes an LLM call)
- MiroFish agents have crypto/finance/geopolitics expertise only — sports/weather markets return 50%

## Operational Notes

### Running Services (for continuous data collection)

```bash
# 1. Neo4j
docker start neo4j

# 2. MiroFish backend (no debug mode to prevent subprocess kills)
FLASK_DEBUG=0 PYTHONPATH="$PWD:$PWD/vendor/mirofish/backend" python vendor/mirofish/backend/run.py

# 3. Ingestion orchestrator (RSS + crypto news → graph)
PREMEMORA_GRAPH_ID=mirofish_4fe1013db29c49e0 python -m ingestors.orchestrator

# 4. Data collector (market prices + resolutions, every 15 min)
python -m e2e.data_collector start --interval 900

# 5. (Optional) Start OASIS simulation for agent interviews
curl -s http://localhost:5001/api/simulation/start -H "Content-Type: application/json" \
  -d '{"simulation_id": "sim_38bc8d05388f"}'
# Wait for env_status.json to show "alive" before interviewing
```

### Monitoring

```bash
python -m e2e.data_collector stats          # market snapshots + resolutions
grep -c Ingested /tmp/orchestrator.log      # events ingested into graph
python -m trading.calibration_gate last     # last calibration result
python -m e2e.soak_test status              # soak test progress (if running)
```
