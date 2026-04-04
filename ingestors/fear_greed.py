"""
Crypto Fear & Greed Index connector — polls the Alternative.me API for
current sentiment values and computes short-term trend direction.

Endpoint: https://api.alternative.me/fng/
Free, no API key required, ~60 req/min.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.fear_greed")

BASE_URL = "https://api.alternative.me/fng/"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _compute_trend(values: List[int]) -> str:
    """Compute trend direction from a list of chronological FNG values.

    The API returns newest-first, so callers must reverse before passing
    here if needed.  Expects oldest-first order.

    Returns 'rising', 'falling', or 'stable'.
    """
    if len(values) < 2:
        return "stable"

    # Compare recent half average to older half average.
    mid = len(values) // 2
    old_avg = sum(values[:mid]) / mid
    new_avg = sum(values[mid:]) / (len(values) - mid)
    diff = new_avg - old_avg

    if diff > 3:
        return "rising"
    elif diff < -3:
        return "falling"
    return "stable"


def _parse_fng_entry(entry: Dict[str, Any]) -> tuple[int, str, str]:
    """Parse a single FNG API entry into (value, classification, timestamp_iso)."""
    value = int(entry.get("value", 0))
    classification = entry.get("value_classification", "Unknown")
    ts_unix = int(entry.get("timestamp", 0))
    ts_iso = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat() if ts_unix else datetime.now(timezone.utc).isoformat()
    return value, classification, ts_iso


@dataclass
class FearGreedConnector:
    """Polls the Crypto Fear & Greed Index and emits events via callback.

    Parameters
    ----------
    callback : EventCallback | None
        Async function called with each new FNG event dict.
    poll_interval : float
        Seconds between polls (default: 3600 = 1 hour).
    """

    callback: Optional[EventCallback] = None
    poll_interval: float = 3600.0
    _running: bool = field(default=False, init=False)
    _last_value: Optional[int] = field(default=None, init=False)
    _session: Optional[aiohttp.ClientSession] = field(default=None, init=False, repr=False)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Accept": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_fng(self, limit: int = 1) -> List[Dict[str, Any]]:
        """Fetch FNG data from the API. Returns raw entries list."""
        session = await self._get_session()
        try:
            async with session.get(BASE_URL, params={"limit": limit}) as resp:
                if resp.status != 200:
                    logger.warning("FNG API returned %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
                return data.get("data", [])
        except Exception as e:
            logger.error("Failed to fetch FNG data: %s", e)
            return []

    async def poll_once(self) -> Optional[Dict[str, Any]]:
        """Single fetch: get current value + 7-day history for trend.

        Returns an event dict or None on failure.
        """
        # Fetch 7 days of history (includes current).
        entries = await self._fetch_fng(limit=7)
        if not entries:
            return None

        # Current value is the first entry (newest).
        current_value, classification, timestamp = _parse_fng_entry(entries[0])

        # Previous value (yesterday) if available.
        previous_value: Optional[int] = None
        if len(entries) > 1:
            previous_value = int(entries[1].get("value", 0))

        # Compute trend from history (reverse to oldest-first).
        history_values = [int(e.get("value", 0)) for e in reversed(entries)]
        trend = _compute_trend(history_values)

        event: Dict[str, Any] = {
            "source": "fear_greed",
            "timestamp": timestamp,
            "value": current_value,
            "classification": classification,
            "previous_value": previous_value,
            "trend": trend,
        }

        self._last_value = current_value
        return event

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("Fear & Greed connector starting, interval=%ss", self.poll_interval)

        while self._running:
            try:
                event = await self.poll_once()
                if event and self.callback:
                    await self.callback(event)
                if event:
                    logger.info(
                        "FNG: %d (%s), trend=%s",
                        event["value"],
                        event["classification"],
                        event["trend"],
                    )
            except Exception:
                logger.exception("Error during FNG poll cycle")

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
            f"[{event['timestamp']}] Fear & Greed: {event['value']} "
            f"({event['classification']}), prev={event['previous_value']}, "
            f"trend={event['trend']}"
        )

    connector = FearGreedConnector(callback=print_event, poll_interval=60)
    print("Polling Fear & Greed Index — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
