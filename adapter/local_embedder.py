"""
Local embedder using fastembed (ONNX-based, no API rate limits).

Replaces MiniMax embedding API which has very restrictive RPM limits.
"""

from collections.abc import Iterable
from typing import Union

from fastembed import TextEmbedding

from graphiti_core.embedder import EmbedderClient

# Small but effective model, downloads on first use (~130MB)
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbedder(EmbedderClient):
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model = TextEmbedding(model_name=model_name)

    async def create(
        self, input_data: Union[str, list[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            texts = input_data
        else:
            texts = [str(input_data)]

        embeddings = list(self._model.embed(texts))
        return embeddings[0].tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        embeddings = list(self._model.embed(input_data_list))
        return [e.tolist() for e in embeddings]
