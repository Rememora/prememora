# AGENTS.md

Instructions for AI coding agents working in this repository.

## Build & Run

```bash
# Start Neo4j (must be running for anything graph-related)
docker start neo4j

# Run MiroFish backend (port 5001)
cd vendor/mirofish/backend && python run.py

# Run adapter integration tests
python -m pytest test_adapter.py -v

# OASIS simulation (requires .venv-oasis)
.venv-oasis/bin/python vendor/mirofish/backend/scripts/run_twitter_simulation.py \
  --config <path-to-config.json> --max-rounds 15
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

- `adapter/` — Graphiti adapter mimicking Zep Cloud SDK. Entry point: `client.py`
- `vendor/mirofish/` — Patched MiroFish. Do not update imports manually; use `adapter/patch_mirofish.py`
- `.venv-oasis/` — Python 3.11 venv for OASIS simulation (not committed)

## Key Conventions

- **LLM provider**: MiniMax M2.5 only. See [docs/minimax-quirks.md](docs/minimax-quirks.md)
  - ONE system message per request (error 2013 if multiple)
  - No `response_format` parameter
  - Strip `<think>...</think>` from responses
  - Endpoint: `https://api.minimaxi.chat/v1` (note: minimax**i**)
- **Embeddings**: Local fastembed (`BAAI/bge-small-en-v1.5`), never MiniMax embedding API
- **Config**: `.env` file at project root, symlinked to `vendor/mirofish/.env`
- **Python**: Main venv is 3.11+. OASIS venv must be Python 3.11 (< 3.12)

## Code Style

- Adapter types in `adapter/zep_types.py` must match Zep Cloud SDK interface
- MiroFish vendor code: minimize changes, document all patches in commit messages
- When fixing MiroFish compatibility: prefer adapter-side fixes over vendor-side changes

## Git

- Do not commit `.env`, `.venv-oasis/`, `neo4j-data/`, `neo4j-logs/`, or `vendor/mirofish/backend/uploads/`
- Commit messages: describe what changed and why
- Vendor patches: note which files were modified and the reason

## Testing

```bash
# Full adapter test (needs Neo4j running)
python -m pytest test_adapter.py -v

# Test MiroFish endpoints manually
curl http://localhost:5001/api/graph/projects
curl -X POST http://localhost:5001/api/report/tools/search \
  -H "Content-Type: application/json" \
  -d '{"graph_id": "<id>", "query": "test"}'
```

## Boundaries

- Do not add new LLM providers without explicit approval
- Do not modify Graphiti internals — fix issues in the adapter layer
- Do not install packages in the main venv that require Python < 3.12 (use .venv-oasis)
- Do not write to Neo4j outside of the adapter layer
