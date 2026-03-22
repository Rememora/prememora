# ADR-004: Separate Python 3.11 Venv for OASIS Simulation

## Status

Accepted

## Context

camel-oasis (the OASIS multi-agent simulation framework) requires Python < 3.12 due to dependency constraints. Our main project uses Python 3.13.

## Decision

Create a separate virtualenv (`.venv-oasis/`) with Python 3.11 specifically for OASIS simulation. The `simulation_runner.py` subprocess launcher resolves `.venv-oasis/bin/python` and uses it to spawn the simulation script.

## Consequences

- Main project can use any Python 3.11+ version
- OASIS dependencies (torch, sentence-transformers, camel-ai) stay isolated
- `.venv-oasis/` is ~1.2GB (mostly PyTorch) — excluded from git
- `simulation_runner.py` sets PYTHONPATH to include project root + vendor/mirofish/backend so the subprocess can import the adapter
- neo4j version mismatch warning (camel-oasis pins 5.23.0, graphiti needs >=5.26.0) — accepted, works in practice with 5.28.3
