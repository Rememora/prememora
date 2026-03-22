# MiniMax M2.5 API Quirks

MiniMax M2.5 is used as the sole LLM provider via an OpenAI-compatible API. This document tracks compatibility issues and workarounds.

## Endpoint

```
Base URL: https://api.minimaxi.chat/v1
```

Note the extra "i" — `minimaxi` not `minimax`. Using the wrong URL gives silent failures or DNS errors.

## Pricing

All standard models cost the same: $0.30/M input tokens, $1.20/M output tokens. No cheaper model exists.

## Known Issues

### Single System Message Only

MiniMax returns error 2013 ("invalid chat setting") if a request contains more than one `system` message.

**Workaround**: Merge all system-level instructions into a single system message. When adding JSON formatting instructions, append to the existing system message rather than adding a new one.

### No `response_format` Support

Passing `response_format: {"type": "json_object"}` causes error 2013.

**Workaround**: Append "You MUST respond with valid JSON only. No markdown fences, no explanation." to the system message. Parse the response and strip markdown code fences if present.

### `<think>` Tags in Responses

MiniMax M2.5 sometimes includes `<think>...</think>` blocks containing chain-of-thought reasoning in the response content.

**Workaround**: Strip with regex after receiving response:
```python
import re
content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
```

### No Logprobs

MiniMax does not support the `logprobs` parameter. Any reranking or confidence scoring must use alternative approaches.

**Workaround**: `adapter/minimax_reranker.py` uses an LLM-based reranking prompt instead of logprob-based scoring.

### Embedding Rate Limits

The `embo-01` embedding model has very aggressive RPM limits on the free tier, making it impractical for batch operations like graph building.

**Workaround**: Use local fastembed with `BAAI/bge-small-en-v1.5` (ONNX, runs on CPU). See `adapter/local_embedder.py`.

### Stale Connections

Long-running processes (like OASIS simulation) can experience hung HTTP connections to the MiniMax API. The server closes the connection but the client doesn't detect it, causing the process to block indefinitely.

**Workaround**: No clean fix yet. Force-kill stuck processes after timeout. The simulation data in SQLite is intact even when the process hangs.

## Files Modified for MiniMax Compatibility

| File | Change |
|------|--------|
| `vendor/mirofish/backend/app/utils/llm_client.py` | Removed `response_format`, added `<think>` stripping, JSON instruction merging |
| `vendor/mirofish/backend/app/services/simulation_config_generator.py` | Removed `response_format`, added `<think>` stripping |
| `vendor/mirofish/backend/app/services/oasis_profile_generator.py` | Removed `response_format`, added `<think>` stripping |
| `adapter/minimax_llm_client.py` | Custom OpenAIClient subclass for Graphiti |
| `adapter/minimax_reranker.py` | LLM-based reranker (no logprobs) |
