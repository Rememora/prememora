# PreMemora

Automated prediction engine: ingest world events, build knowledge graphs, run agent swarm simulations, generate analysis reports, and trade on Polymarket.

```
Event Sources → Graphiti+Neo4j → MiroFish agent swarms → Edge Calc → Paper Trading
(RSS, crypto)   (knowledge graph)  (22 agent interviews)  (Kelly sizing) (SQLite)
       ↑                                                        ↓
Data Collector                                          Calibration Gate
(market prices)                                         (oracle-based)
```

Everything runs locally except LLM API calls (MiniMax M2.5) and Polymarket.

## Quick Start

### Prerequisites

- Python 3.11+ (3.11 required for OASIS simulation)
- Docker (for Neo4j)
- MiniMax API key

### Setup

```bash
# Clone and configure
cp .env.sample .env
# Edit .env with your MiniMax API key

# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/prememora_local \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v $(pwd)/neo4j-data:/data \
  -v $(pwd)/neo4j-logs:/logs \
  neo4j:5.26

# Install main dependencies
pip install -e ".[dev]"

# Install MiroFish backend deps
pip install -r vendor/mirofish/backend/requirements.txt
```

### OASIS Simulation (separate venv)

OASIS requires Python < 3.12 due to camel-ai constraints.

```bash
python3.11 -m venv .venv-oasis
source .venv-oasis/bin/activate
pip install camel-ai==0.2.78 camel-oasis==0.2.5
pip install graphiti-core flask fastembed python-dotenv neo4j>=5.26
```

### Run MiroFish Backend

```bash
cd vendor/mirofish/backend
python run.py  # Starts Flask on port 5001
```

## Project Structure

```
prememora/
├── adapter/           # Graphiti adapter (Zep Cloud drop-in replacement)
├── ingestors/         # 6 event source connectors + unified orchestrator
├── pipeline/          # Prediction pipeline: trigger, context builder
├── trading/           # Paper engine, edge calc, exit monitor, strategy review, calibration gate
├── backtesting/       # Hindsight oracle, price history, event replay
├── e2e/               # E2E tests, soak test, data collector
├── vendor/
│   └── mirofish/      # MiroFish backend + frontend (patched imports)
├── data/              # SQLite DBs, signal logs, market history (gitignored)
├── docs/              # Architecture insights, API quirks, decision records
└── tests/             # 280+ unit tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, component relationships, data flow |
| [docs/architecture-insights.md](docs/architecture-insights.md) | Design decisions and the *why* behind them |
| [AGENTS.md](AGENTS.md) | Agent/LLM coding assistant instructions |
| [docs/decisions/](docs/decisions/) | Architecture Decision Records |
| [docs/minimax-quirks.md](docs/minimax-quirks.md) | MiniMax M2.5 API compatibility notes |
| [docs/zep-graphiti-mapping.md](docs/zep-graphiti-mapping.md) | Zep Cloud to Graphiti API mapping |

## Current Status

### Core Pipeline
- [x] Graphiti adapter (Zep Cloud replacement)
- [x] MiroFish graph build pipeline
- [x] Graph read path (search, statistics, entities)
- [x] Report agent (multi-section reports + chat)
- [x] OASIS simulation (22-agent Twitter/Reddit simulation)
- [x] Event ingestion — 6 connectors (Polymarket WS, crypto news, RSS, whale alerts, Reddit, FRED)
- [x] Unified ingestion orchestrator — all sources → Graphiti with dedup

### Trading Stack
- [x] Paper trading engine — SQLite, position lifecycle, fee calc, P&L
- [x] Edge calculator — Kelly sizing, risk limits, drawdown protection
- [x] Pipeline trigger — MiroFish interviews → edge calc → paper trading
- [x] Graph context enrichment — relevant facts from knowledge graph fed to agent interviews
- [x] Strategy review — Brier score, calibration, source attribution, recommendations
- [x] Exit monitor — 4 trigger types (confidence drop, contradictory evidence, time decay, stop-loss)
- [x] Calibration gate — oracle-based pre-trade circuit breaker (blocks when Brier > 0.25)
- [x] Market relevance filter — skips sports/weather markets agents can't reason about

### Validation
- [x] E2E smoke test — full pipeline against resolved markets in one shot
- [x] Hindsight oracle — backtest predictions against known outcomes
- [x] Soak test runner — continuous paper trading mode
- [x] Data collector — market price snapshots + resolution tracking (running continuously)

### Currently Running
- Ingestion orchestrator (RSS + crypto news → knowledge graph)
- Data collector (Polymarket prices + resolutions, every 15 min)

### Next
- [ ] Accumulate data for 2+ weeks (prices, graph events, resolutions)
- [ ] Run hindsight oracle with real historical data
- [ ] Paper trade when calibration proves out
- [ ] Live execution on Polymarket (#13)

## License

MIT
