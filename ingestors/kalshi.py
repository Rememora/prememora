"""
Kalshi Prediction Market Prices connector — polls the Kalshi API for
open market data and emits events for markets with price changes.

Endpoint: https://api.elections.kalshi.com/trade-api/v2/markets?limit=50&status=open
Free, no API key required for market data read access.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.kalshi")

API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _extract_yes_price(market: Dict[str, Any]) -> Optional[float]:
    """Extract the yes price from a Kalshi market.

    Kalshi uses cents (0-100), so we normalize to a 0-1 float.
    The field may be 'yes_bid', 'last_price', or 'yes_price' depending
    on API version. We try several paths.
    """
    # Try direct price fields (normalized 0-1)
    for key in ("yes_price", "last_price"):
        val = market.get(key)
        if val is not None:
            v = float(val)
            # If > 1, assume cents (0-100) and normalize
            return v / 100 if v > 1 else v

    # Try cent-based fields
    for key in ("yes_bid", "yes_ask"):
        val = market.get(key)
        if val is not None:
            return float(val) / 100

    return None


def _market_fingerprint(ticker: str, yes_price: Optional[float]) -> str:
    """Create a fingerprint for change detection.

    Uses ticker + yes price rounded to 2 decimal places so that
    tiny fluctuations still register as changes when meaningful.
    """
    if yes_price is None:
        return f"{ticker}:none"
    return f"{ticker}:{yes_price:.2f}"


def _build_event(market: Dict[str, Any], yes_price: float) -> Dict[str, Any]:
    """Build an event dict from a Kalshi market."""
    ticker = market.get("ticker", "")
    title = market.get("title", "") or market.get("event_title", "")
    volume = market.get("volume", 0) or market.get("volume_24h", 0)
    category = market.get("category", "") or market.get("series_ticker", "")

    return {
        "source": "kalshi",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": ticker,
        "market_title": title,
        "yes_price": yes_price,
        "volume": volume,
        "category": category,
    }


@dataclass
class KalshiConnector:
    """Polls Kalshi for open market data, emitting events on price change.

    Parameters
    ----------
    callback : EventCallback | None
        Async function called with each changed-market event dict.
    poll_interval : float
        Seconds between polls (default: 1800 = 30 minutes).
    """

    callback: Optional[EventCallback] = None
    poll_interval: float = 1800.0
    _running: bool = field(default=False, init=False)
    _last_fingerprints: Dict[str, str] = field(default_factory=dict, init=False)
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

    async def _fetch_markets(self) -> List[Dict[str, Any]]:
        """Fetch open markets from Kalshi."""
        session = await self._get_session()
        params = {
            "limit": 50,
            "status": "open",
        }
        try:
            async with session.get(API_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Kalshi API returned %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
                return data.get("markets", [])
        except Exception as e:
            logger.error("Failed to fetch Kalshi markets: %s", e)
            return []

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Single fetch: get open markets, return only those with price changes.

        On first poll, all markets with valid prices are emitted.
        """
        raw_markets = await self._fetch_markets()
        if not raw_markets:
            return []

        changed_events: List[Dict[str, Any]] = []

        for market in raw_markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue

            yes_price = _extract_yes_price(market)
            if yes_price is None:
                continue

            fingerprint = _market_fingerprint(ticker, yes_price)
            old_fp = self._last_fingerprints.get(ticker)
            if old_fp == fingerprint:
                continue

            self._last_fingerprints[ticker] = fingerprint
            event = _build_event(market, yes_price)
            changed_events.append(event)

        return changed_events

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("Kalshi connector starting, interval=%ss", self.poll_interval)

        while self._running:
            try:
                events = await self.poll_once()
                if events:
                    logger.info("Kalshi: %d markets changed", len(events))
                    if self.callback:
                        for event in events:
                            await self.callback(event)
            except Exception:
                logger.exception("Error during Kalshi poll cycle")

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
            f"  [{event['market_id']}] {event['market_title']}"
            f"\n    YES={event['yes_price']:.2f}, vol={event['volume']}, "
            f"cat={event['category']}"
        )

    connector = KalshiConnector(callback=print_event, poll_interval=60)
    print("Polling Kalshi markets — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
