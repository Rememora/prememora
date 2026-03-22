"""
Simple reranker for MiniMax that uses the LLM to score relevance.

MiniMax doesn't support logprobs, so we ask the model to score directly.
"""

import json
import logging
import re

from openai import AsyncOpenAI

from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.llm_client import LLMConfig

logger = logging.getLogger("prememora.reranker")


class MiniMaxReranker(CrossEncoderClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        # For small lists, just return them with equal scores (skip LLM call)
        if len(passages) <= 2:
            return [(p, 1.0) for p in passages]

        # Ask LLM to rank passages by relevance
        numbered = "\n".join(f"[{i}] {p[:200]}" for i, p in enumerate(passages))
        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Rank these passages by relevance (most relevant first). "
            f"Return a JSON array of indices, e.g. [2, 0, 1].\n\n"
            f"{numbered}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model or "MiniMax-M2.5",
                messages=[
                    {"role": "system", "content": "You rank passages by relevance. Return only a JSON array of indices. No markdown, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
            )

            content = response.choices[0].message.content or "[]"
            # Strip <think>...</think> blocks from MiniMax responses
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # Parse ranking — might be {"ranking": [2,0,1]} or just [2,0,1]
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                ranking = parsed.get("ranking", parsed.get("indices", list(range(len(passages)))))
            else:
                ranking = parsed

            # Build scored results
            results = []
            for rank_pos, idx in enumerate(ranking):
                if isinstance(idx, int) and 0 <= idx < len(passages):
                    score = 1.0 - (rank_pos / len(ranking))
                    results.append((passages[idx], score))

            # Add any passages that weren't ranked
            ranked_indices = set(ranking) if isinstance(ranking, list) else set()
            for i, p in enumerate(passages):
                if i not in ranked_indices:
                    results.append((p, 0.0))

            return results

        except Exception as e:
            logger.warning(f"Reranker failed, returning unranked: {e}")
            return [(p, 1.0 / (i + 1)) for i, p in enumerate(passages)]
