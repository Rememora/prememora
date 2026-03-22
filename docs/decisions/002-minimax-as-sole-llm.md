# ADR-002: Use MiniMax M2.5 as Sole LLM Provider

## Status

Accepted

## Context

MiroFish needs an LLM for ontology generation, report writing, agent profile creation, simulation config, and OASIS agent behavior. We need an OpenAI-compatible API that's cost-effective for high-volume usage.

## Decision

Use MiniMax M2.5 via their OpenAI-compatible endpoint (`api.minimaxi.chat/v1`) for all LLM calls. No fallback providers.

## Consequences

- Uniform pricing: $0.30/M input, $1.20/M output
- Must work around MiniMax limitations: single system message, no `response_format`, no logprobs, `<think>` tag stripping
- Embedding API is rate-limited — use local fastembed instead
- All compatibility fixes documented in `docs/minimax-quirks.md`
- Stale connection issue with long-running processes (no clean fix)
