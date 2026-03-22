"""
MiniMax embedder for Graphiti.

Uses MiniMax's native embedding API (embo-01) since their OpenAI-compatible
endpoint doesn't support /embeddings.

Requires MINIMAX_API_KEY and MINIMAX_GROUP_ID env vars.
"""

import asyncio
import logging
import os
from collections.abc import Iterable
from typing import Union

import httpx

from graphiti_core.embedder import EmbedderClient

import random

logger = logging.getLogger("prememora.embedder")

MINIMAX_EMBED_MODEL = "embo-01"
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 3.0
# Serialize all embedding requests to avoid burst rate limits
_embed_semaphore = asyncio.Semaphore(1)


class MiniMaxEmbedder(EmbedderClient):
    def __init__(self, api_key: str | None = None, group_id: str | None = None):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.environ.get("MINIMAX_GROUP_ID", "")
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY not set")
        if not self.group_id:
            raise ValueError("MINIMAX_GROUP_ID not set — find it at https://platform.minimax.io")
        self._url = f"https://api.minimaxi.chat/v1/embeddings?GroupId={self.group_id}"
        self._client = httpx.AsyncClient(timeout=30)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with _embed_semaphore:
            return await self._embed_inner(texts)

    async def _embed_inner(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            resp = await self._client.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MINIMAX_EMBED_MODEL,
                    "texts": texts,
                    "type": "db",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            status_code = data.get("base_resp", {}).get("status_code", 0)
            if status_code == 0:
                return [item["embedding"] for item in data["vectors"]]

            # Rate limit — retry with exponential backoff + jitter
            if status_code == 1002:
                delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
                logger.warning(f"Embedding rate limited, retrying in {delay:.1f}s (attempt {attempt+1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue

            raise RuntimeError(f"MiniMax embedding error: {data['base_resp']}")

        raise RuntimeError("MiniMax embedding rate limit exceeded after max retries")

    async def create(
        self, input_data: Union[str, list[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            texts = input_data
        else:
            texts = [str(input_data)]

        results = await self._embed(texts)
        return results[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._embed(input_data_list)
