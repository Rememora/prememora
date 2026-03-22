"""
MiniMax LLM client for Graphiti.

Subclasses Graphiti's OpenAIClient but overrides structured completion
and regular completion to avoid features MiniMax doesn't support
(Responses API, response_format json_object).
"""

import json
import logging
import re

from pydantic import BaseModel

from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client import LLMConfig

logger = logging.getLogger("prememora.minimax_llm")


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown fences and think tags."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def _merge_system_instruction(messages: list, suffix: str) -> list:
    """Append suffix to the existing system message (MiniMax only allows one)."""
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] += suffix
            return messages
    # No system message found — create one
    return [{"role": "system", "content": suffix.strip()}] + messages


class MiniMaxLLMClient(OpenAIClient):
    """OpenAI-compatible client that avoids unsupported MiniMax features."""

    async def _create_structured_completion(
        self,
        model,
        messages,
        temperature,
        max_tokens,
        response_model,
        reasoning=None,
        verbosity=None,
    ):
        """Use chat completions with schema in prompt instead of responses.parse()."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_suffix = (
            f"\n\nYou must respond with valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Return ONLY the JSON object, no markdown, no explanation."
        )

        augmented_messages = _merge_system_instruction(list(messages), schema_suffix)

        response = await self.client.chat.completions.create(
            model=model,
            messages=augmented_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return _ChatAsStructuredResponse(response)

    async def _create_completion(
        self,
        model,
        messages,
        temperature,
        max_tokens,
        response_model=None,
        reasoning=None,
        verbosity=None,
    ):
        """Regular completion without response_format (MiniMax doesn't support it)."""
        json_suffix = "\n\nAlways respond with valid JSON only. No markdown, no explanation."
        augmented_messages = _merge_system_instruction(list(messages), json_suffix)

        return await self.client.chat.completions.create(
            model=model,
            messages=augmented_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class _ChatAsStructuredResponse:
    """Wraps a chat completion response to look like a structured response."""

    def __init__(self, chat_response):
        self._chat = chat_response
        content = chat_response.choices[0].message.content or "{}"
        self.output_text = _extract_json(content)
        self.usage = chat_response.usage
        self.refusal = None
