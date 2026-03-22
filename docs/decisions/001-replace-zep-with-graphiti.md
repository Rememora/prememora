# ADR-001: Replace Zep Cloud with Local Graphiti + Neo4j

## Status

Accepted

## Context

MiroFish uses Zep Cloud for graph memory (entity extraction, relationship storage, semantic search). Zep deprecated its open-source offering in April 2025, making the hosted service the only option. We need local-first infrastructure with no external dependencies beyond LLM APIs.

## Decision

Build a drop-in adapter (`adapter/`) that implements the Zep Cloud SDK interface using Graphiti (open-source graph memory framework) backed by a local Neo4j instance.

## Consequences

- All MiroFish graph operations work without code changes to service files (only import paths change)
- Adapter must handle differences in storage model (Graphiti uses `RELATES_TO` for all edges, stores names as properties)
- Entity type inference requires heuristic matching since Graphiti doesn't store typed entity labels
- No ongoing Zep Cloud costs or vendor lock-in
- We own the full data pipeline locally
