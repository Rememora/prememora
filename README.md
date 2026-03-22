# PreMemora

Automated prediction engine: ingest world events, build knowledge graphs, run agent swarm simulations, generate analysis reports, and trade on Polymarket.

```
World Events --> OpenViking --> Graphiti+Neo4j --> MiroFish agent swarms --> Predictions --> Polymarket
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
├── vendor/
│   └── mirofish/      # MiroFish backend + frontend (patched imports)
├── .env.sample        # Environment template
├── pyproject.toml     # Project metadata and dependencies
└── test_adapter.py    # Adapter integration tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, component relationships, data flow |
| [AGENTS.md](AGENTS.md) | Agent/LLM coding assistant instructions |
| [docs/decisions/](docs/decisions/) | Architecture Decision Records |
| [docs/minimax-quirks.md](docs/minimax-quirks.md) | MiniMax M2.5 API compatibility notes |
| [docs/zep-graphiti-mapping.md](docs/zep-graphiti-mapping.md) | Zep Cloud to Graphiti API mapping |

## Current Status

- [x] Graphiti adapter (Zep Cloud replacement)
- [x] MiroFish graph build pipeline
- [x] Graph read path (search, statistics, entities)
- [x] Report agent (multi-section reports + chat)
- [x] OASIS simulation (multi-agent Twitter simulation)
- [ ] OpenViking event ingestion
- [ ] Real-world data pipeline
- [ ] Polymarket trading integration

## License

MIT
