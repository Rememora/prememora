# ADR-003: Use Local fastembed for Embeddings

## Status

Accepted

## Context

Graphiti requires embeddings for semantic search. MiniMax offers an embedding model (`embo-01`) but enforces aggressive RPM limits on the free tier, making it impractical for batch operations during graph building (dozens of episodes, each triggering multiple embedding calls).

## Decision

Use fastembed with `BAAI/bge-small-en-v1.5` (ONNX runtime, CPU-only) for all embeddings. The MiniMax embedding client (`adapter/minimax_embedder.py`) exists but is not used in production.

## Consequences

- No API rate limits — embeddings are instant and free
- Model runs locally on CPU, no GPU required
- 384-dimension vectors (smaller than OpenAI's 1536 but sufficient for graph search)
- ~50MB model download on first run (cached afterwards)
- Embedding quality is adequate for entity/relationship search in knowledge graphs
