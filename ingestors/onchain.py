"""
Bitcoin On-Chain Metrics connector — polls the BGeometrics API for
key on-chain indicators (MVRV Z-Score, SOPR, NUPL) and emits events
with current values, previous values, and market-phase interpretations.

Endpoints:
  - https://charts.bgeometrics.com/apiservice?chart=mvrv_zscore&format=json
  - https://charts.bgeometrics.com/apiservice?chart=sopr&format=json
  - https://charts.bgeometrics.com/apiservice?chart=nupl&format=json

Free, no API key required.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger("prememora.ingestors.onchain")

BASE_URL = "https://charts.bgeometrics.com/apiservice"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

# Metrics to fetch, with their chart parameter names.
METRICS = ["mvrv_zscore", "sopr", "nupl"]


def _interpret_mvrv(value: float) -> str:
    """Interpret MVRV Z-Score.

    <0 = undervalued (historically good buy zone)
    0-3 = fair value range
    >3 = overvalued / potential market top signal
    """
    if value < 0:
        return "undervalued"
    elif value <= 3:
        return "fair"
    else:
        return "overvalued"


def _interpret_sopr(value: float) -> str:
    """Interpret Spent Output Profit Ratio.

    <1 = selling at loss (capitulation / accumulation)
    =1 = breakeven
    >1 = selling at profit (distribution)
    """
    if value < 1:
        return "selling_at_loss"
    elif value == 1.0:
        return "breakeven"
    else:
        return "selling_at_profit"


def _interpret_nupl(value: float) -> str:
    """Interpret Net Unrealized Profit/Loss.

    <0 = capitulation (holders underwater)
    0-0.25 = hope / fear
    0.25-0.5 = optimism / anxiety
    0.5-0.75 = belief / denial
    >0.75 = euphoria / greed
    """
    if value < 0:
        return "capitulation"
    elif value < 0.25:
        return "hope_fear"
    elif value < 0.5:
        return "optimism"
    elif value < 0.75:
        return "belief"
    else:
        return "euphoria"


_INTERPRETERS = {
    "mvrv_zscore": _interpret_mvrv,
    "sopr": _interpret_sopr,
    "nupl": _interpret_nupl,
}


def _parse_metric_data(data: Any) -> Optional[Tuple[float, float]]:
    """Parse BGeometrics JSON response into (latest_value, previous_value).

    The API returns a list of [date, value] pairs sorted by date ascending.
    We take the last two entries for current and previous values.

    Returns None if the data is not parseable.
    """
    if not isinstance(data, list) or len(data) < 1:
        return None

    try:
        # Try list-of-lists format: [[date, value], ...]
        if isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
            latest = float(data[-1][1])
            previous = float(data[-2][1]) if len(data) >= 2 else latest
            return latest, previous

        # Try list-of-dicts format: [{"date": ..., "value": ...}, ...]
        if isinstance(data[0], dict):
            val_key = None
            for key in ("value", "v", "y"):
                if key in data[0]:
                    val_key = key
                    break
            if val_key:
                latest = float(data[-1][val_key])
                previous = float(data[-2][val_key]) if len(data) >= 2 else latest
                return latest, previous

        # Try plain list of numbers
        if isinstance(data[0], (int, float)):
            latest = float(data[-1])
            previous = float(data[-2]) if len(data) >= 2 else latest
            return latest, previous

    except (TypeError, ValueError, IndexError, KeyError):
        pass

    return None


@dataclass
class OnChainConnector:
    """Polls BGeometrics for Bitcoin on-chain metrics, emitting events per metric.

    Parameters
    ----------
    callback : EventCallback | None
        Async function called with each metric event dict.
    poll_interval : float
        Seconds between polls (default: 21600 = 6 hours).
    """

    callback: Optional[EventCallback] = None
    poll_interval: float = 21600.0
    _running: bool = field(default=False, init=False)
    _last_values: Dict[str, float] = field(default_factory=dict, init=False)
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

    async def _fetch_metric(self, chart: str) -> Any:
        """Fetch a single metric from BGeometrics."""
        session = await self._get_session()
        params = {"chart": chart, "format": "json"}
        try:
            async with session.get(BASE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("BGeometrics API returned %d for %s", resp.status, chart)
                    return None
                return await resp.json(content_type=None)
        except Exception as e:
            logger.error("Failed to fetch on-chain metric %s: %s", chart, e)
            return None

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Fetch all metrics and return events for each.

        Each metric produces one event with the latest value, previous
        value, and a human-readable interpretation.
        """
        events: List[Dict[str, Any]] = []

        for metric in METRICS:
            raw_data = await self._fetch_metric(metric)
            if raw_data is None:
                continue

            parsed = _parse_metric_data(raw_data)
            if parsed is None:
                logger.warning("Could not parse data for %s", metric)
                continue

            value, previous_value = parsed
            interpreter = _INTERPRETERS.get(metric)
            interpretation = interpreter(value) if interpreter else "unknown"

            event: Dict[str, Any] = {
                "source": "onchain",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric": metric,
                "value": value,
                "previous_value": previous_value,
                "interpretation": interpretation,
            }

            self._last_values[metric] = value
            events.append(event)

        return events

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("On-chain connector starting, interval=%ss", self.poll_interval)

        while self._running:
            try:
                events = await self.poll_once()
                if events:
                    logger.info("On-chain: %d metrics fetched", len(events))
                    if self.callback:
                        for event in events:
                            await self.callback(event)
            except Exception:
                logger.exception("Error during on-chain poll cycle")

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
            f"  {event['metric']}: {event['value']:.4f} "
            f"(prev={event['previous_value']:.4f}) "
            f"— {event['interpretation']}"
        )

    connector = OnChainConnector(callback=print_event, poll_interval=60)
    print("Polling Bitcoin on-chain metrics — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
