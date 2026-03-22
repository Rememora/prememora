# Architecture

## System Overview

PreMemora is a prediction pipeline with four stages:

```
1. INGEST        2. ANALYZE           3. SIMULATE          4. TRADE
OpenViking  -->  Graphiti+Neo4j  -->  MiroFish Agents  -->  Polymarket
(events)         (knowledge graph)    (swarm simulation)    (execution)
```

Only stages 2 and 3 are implemented. Stages 1 and 4 are planned.

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

### Graph Build
```
User uploads text
  → ontology_generator.py (LLM extracts entity/edge types)
  → graph_builder.py
    → adapter.graph.episode.add() (ingest episodes)
    → Graphiti extracts entities + edges
    → Neo4j stores graph
```

### Report Generation
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

### OASIS Simulation
```
POST /api/simulation/prepare
  → oasis_profile_generator.py (LLM creates agent personas from graph entities)
  → simulation_config_generator.py (LLM creates simulation config)

POST /api/simulation/start
  → simulation_runner.py
    → subprocess: .venv-oasis/bin/python run_twitter_simulation.py
    → OASIS runs N rounds (60 min/round simulated time)
    → Agents post, like, quote, follow on simulated Twitter
    → Results in SQLite
```

## Dependencies

| Component | Key Dependencies |
|-----------|-----------------|
| Adapter | graphiti-core, neo4j, fastembed, openai |
| MiroFish Backend | flask, openai, python-dotenv |
| OASIS Simulation | camel-ai==0.2.78, camel-oasis==0.2.5, torch, sentence-transformers |
| LLM Provider | MiniMax M2.5 via OpenAI-compatible API (`api.minimaxi.chat`) |

## Constraints

- MiniMax allows only ONE system message per request
- MiniMax does not support `response_format` or logprobs
- OASIS requires Python < 3.12 (separate venv)
- Agent active hours start at hour 9 — simulations need >= 15 rounds to reach active period
- Graphiti stores relationship type as `RELATES_TO` — actual names are in properties
