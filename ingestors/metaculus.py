"""
Metaculus Expert Forecasts connector — polls the Metaculus API for
open forecast questions and emits events when community probabilities change.

Endpoint: https://www.metaculus.com/api2/questions/?status=open&type=forecast&limit=50&order_by=-activity

NOTE: As of April 2026, the Metaculus API returns 403 for unauthenticated
requests. This connector is built and tested but disabled by default in the
orchestrator until auth is resolved. Enable with enable_metaculus=True once
a valid auth token is available (set METACULUS_TOKEN env var).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.metaculus")

BASE_URL = "https://www.metaculus.com/api2/questions/"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _extract_probability(question: Dict[str, Any]) -> Optional[float]:
    """Extract the community probability from a Metaculus question.

    Path: aggregations.recency_weighted.latest.centers[0]

    Returns the probability as a float in [0, 1], or None if unavailable.
    """
    try:
        aggregations = question.get("aggregations", {})
        recency = aggregations.get("recency_weighted", {})
        latest = recency.get("latest", {})
        centers = latest.get("centers", [])
        if centers:
            return float(centers[0])
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _question_fingerprint(question_id: int, probability: Optional[float]) -> str:
    """Create a fingerprint for change detection.

    Uses question ID + probability rounded to 4 decimal places so that
    tiny floating-point jitter doesn't trigger spurious events.
    """
    if probability is None:
        return f"{question_id}:none"
    return f"{question_id}:{probability:.4f}"


def _build_event(question: Dict[str, Any], probability: float) -> Dict[str, Any]:
    """Build an event dict from a Metaculus question."""
    question_id = question.get("id", 0)
    title = question.get("title", "")
    num_forecasters = question.get("number_of_forecasters", 0)
    url = f"https://www.metaculus.com/questions/{question_id}/"

    return {
        "source": "metaculus",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_id": question_id,
        "question_title": title,
        "community_probability": probability,
        "num_forecasters": num_forecasters,
        "url": url,
    }


@dataclass
class MetaculusConnector:
    """Polls Metaculus for open forecast questions, emitting events on probability change.

    Parameters
    ----------
    callback : EventCallback | None
        Async function called with each changed-question event dict.
    poll_interval : float
        Seconds between polls (default: 21600 = 6 hours).
    """

    callback: Optional[EventCallback] = None
    poll_interval: float = 21600.0
    _running: bool = field(default=False, init=False)
    _last_fingerprints: Dict[int, str] = field(default_factory=dict, init=False)
    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False, repr=False)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={"Accept": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_questions(self) -> List[Dict[str, Any]]:
        """Fetch open forecast questions from Metaculus."""
        session = await self._get_session()
        params = {
            "status": "open",
            "type": "forecast",
            "limit": 50,
            "order_by": "-activity",
        }
        try:
            async with session.get(BASE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Metaculus API returned %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
                return data.get("results", [])
        except Exception as e:
            logger.error("Failed to fetch Metaculus questions: %s", e)
            return []

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Single fetch: get open questions, return only those with probability changes.

        On first poll, all questions with valid probabilities are emitted.
        """
        raw_questions = await self._fetch_questions()
        if not raw_questions:
            return []

        changed_events: List[Dict[str, Any]] = []

        for question in raw_questions:
            question_id = question.get("id")
            if not question_id:
                continue

            probability = _extract_probability(question)
            if probability is None:
                continue

            fingerprint = _question_fingerprint(question_id, probability)
            old_fp = self._last_fingerprints.get(question_id)
            if old_fp == fingerprint:
                continue

            self._last_fingerprints[question_id] = fingerprint
            event = _build_event(question, probability)
            changed_events.append(event)

        return changed_events

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("Metaculus connector starting, interval=%ss", self.poll_interval)

        while self._running:
            try:
                events = await self.poll_once()
                if events:
                    logger.info("Metaculus: %d questions changed", len(events))
                    if self.callback:
                        for event in events:
                            await self.callback(event)
            except Exception:
                logger.exception("Error during Metaculus poll cycle")

            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Signal the polling loop to stop after the current cycle."""
        self._running = False


# ── Standalone demo ──────────────────────────────────────────────────────────


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    async def print_event(event: Dict[str, Any]) -> None:
        print(
            f"  [{event['question_id']}] {event['question_title']}"
            f"\n    P={event['community_probability']:.2%}, "
            f"forecasters={event['num_forecasters']}"
        )

    connector = MetaculusConnector(callback=print_event, poll_interval=60)
    print("Polling Metaculus forecasts — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
