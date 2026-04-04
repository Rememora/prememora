"""
PredictIt Political Market Prices connector — polls the PredictIt API for
active prediction market data and emits events for markets with price changes.

Endpoint: https://www.predictit.org/api/marketdata/all/
Free, no API key required.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.predictit")

API_URL = "https://www.predictit.org/api/marketdata/all/"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _extract_contracts(market: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract contract data from a PredictIt market response."""
    contracts = []
    for c in market.get("contracts", []):
        contracts.append({
            "name": c.get("name", ""),
            "price": c.get("lastTradePrice"),
            "volume": c.get("totalSharesTraded", 0),
        })
    return contracts


def _market_fingerprint(contracts: List[Dict[str, Any]]) -> str:
    """Create a fingerprint of contract prices for change detection.

    Returns a string like "name1:0.62|name2:0.38" sorted by contract name
    so the fingerprint is stable across API call ordering changes.
    """
    parts = sorted(f"{c['name']}:{c['price']}" for c in contracts)
    return "|".join(parts)


def _build_market_event(market: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
    """Build an event dict from a PredictIt market."""
    contracts = _extract_contracts(market)
    return {
        "source": "predictit",
        "timestamp": timestamp,
        "market_id": str(market.get("id", "")),
        "market_name": market.get("name", ""),
        "contracts": contracts,
    }


@dataclass
class PredictItConnector:
    """Polls PredictIt for active market data, emitting events on price change.

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
        """Fetch all active markets from PredictIt."""
        session = await self._get_session()
        try:
            async with session.get(API_URL) as resp:
                if resp.status != 200:
                    logger.warning("PredictIt API returned %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
                return data.get("markets", [])
        except Exception as e:
            logger.error("Failed to fetch PredictIt markets: %s", e)
            return []

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Single fetch: get all markets, return only those with price changes.

        On first poll, all markets are considered 'changed' (initial snapshot).
        """
        raw_markets = await self._fetch_markets()
        if not raw_markets:
            return []

        timestamp = datetime.now(timezone.utc).isoformat()
        changed_events: List[Dict[str, Any]] = []

        for market in raw_markets:
            market_id = str(market.get("id", ""))
            if not market_id:
                continue

            contracts = _extract_contracts(market)
            fingerprint = _market_fingerprint(contracts)

            old_fp = self._last_fingerprints.get(market_id)
            if old_fp == fingerprint:
                # No price change — skip.
                continue

            self._last_fingerprints[market_id] = fingerprint
            event = _build_market_event(market, timestamp)
            changed_events.append(event)

        return changed_events

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("PredictIt connector starting, interval=%ss", self.poll_interval)

        while self._running:
            try:
                events = await self.poll_once()
                if events:
                    logger.info("PredictIt: %d markets changed", len(events))
                    if self.callback:
                        for event in events:
                            await self.callback(event)
            except Exception:
                logger.exception("Error during PredictIt poll cycle")

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
        contracts_str = ", ".join(
            f"{c['name']}: ${c['price']}" for c in event["contracts"][:3]
        )
        suffix = "..." if len(event["contracts"]) > 3 else ""
        print(f"  [{event['market_id']}] {event['market_name']}")
        print(f"    {contracts_str}{suffix}")

    connector = PredictItConnector(callback=print_event, poll_interval=60)
    print("Polling PredictIt markets — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
