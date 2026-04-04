"""
CoinGecko Trending & Market Data connector — polls the free CoinGecko API
for trending coins and key cryptocurrency prices.

Endpoints:
  - GET /api/v3/search/trending     (no auth, updates ~30min)
  - GET /api/v3/simple/price        (no auth, real-time)

Free tier: 5-15 calls/min rate limit, no API key needed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.coingecko")

BASE_URL = "https://api.coingecko.com/api/v3"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

DEFAULT_PRICE_IDS: List[str] = ["bitcoin", "ethereum", "solana"]


def _parse_trending(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the trending API response into an event dict."""
    coins: List[Dict[str, Any]] = []
    for item in data.get("coins", []):
        coin_info = item.get("item", {})
        coins.append({
            "name": coin_info.get("name", ""),
            "symbol": coin_info.get("symbol", ""),
            "market_cap_rank": coin_info.get("market_cap_rank"),
        })

    return {
        "source": "coingecko",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "trending",
        "coins": coins,
    }


def _parse_prices(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the simple/price API response into an event dict."""
    prices: Dict[str, Any] = {}
    for coin_id, price_data in data.items():
        prices[coin_id] = {
            "usd": price_data.get("usd"),
            "usd_24h_change": price_data.get("usd_24h_change"),
        }

    return {
        "source": "coingecko",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "prices",
        "prices": prices,
    }


@dataclass
class CoinGeckoConnector:
    """Polls CoinGecko for trending coins and key crypto prices.

    Parameters
    ----------
    callback : EventCallback | None
        Async function called with each event dict (trending or price).
    poll_interval : float
        Seconds between polls (default: 3600 = 1 hour).
    price_ids : list[str]
        Coin IDs to fetch prices for.
    """

    callback: Optional[EventCallback] = None
    poll_interval: float = 3600.0
    price_ids: List[str] = field(default_factory=lambda: list(DEFAULT_PRICE_IDS))
    _running: bool = field(default=False, init=False)
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

    async def _fetch_trending(self) -> Optional[Dict[str, Any]]:
        """Fetch trending coins from CoinGecko."""
        session = await self._get_session()
        try:
            async with session.get(f"{BASE_URL}/search/trending") as resp:
                if resp.status != 200:
                    logger.warning("CoinGecko trending API returned %d", resp.status)
                    return None
                data = await resp.json(content_type=None)
                return _parse_trending(data)
        except Exception as e:
            logger.error("Failed to fetch CoinGecko trending: %s", e)
            return None

    async def _fetch_prices(self) -> Optional[Dict[str, Any]]:
        """Fetch prices for tracked coins from CoinGecko."""
        if not self.price_ids:
            return None
        session = await self._get_session()
        params = {
            "ids": ",".join(self.price_ids),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
        }
        try:
            async with session.get(f"{BASE_URL}/simple/price", params=params) as resp:
                if resp.status != 200:
                    logger.warning("CoinGecko price API returned %d", resp.status)
                    return None
                data = await resp.json(content_type=None)
                return _parse_prices(data)
        except Exception as e:
            logger.error("Failed to fetch CoinGecko prices: %s", e)
            return None

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Fetch trending + prices, return list of events."""
        events: List[Dict[str, Any]] = []

        trending = await self._fetch_trending()
        if trending:
            events.append(trending)

        prices = await self._fetch_prices()
        if prices:
            events.append(prices)

        return events

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("CoinGecko connector starting, interval=%ss", self.poll_interval)

        while self._running:
            try:
                events = await self.poll_once()
                if events and self.callback:
                    for event in events:
                        await self.callback(event)
                if events:
                    logger.info("CoinGecko: fetched %d events", len(events))
            except Exception:
                logger.exception("Error during CoinGecko poll cycle")

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
        etype = event.get("event_type", "unknown")
        if etype == "trending":
            coins = event.get("coins", [])
            names = ", ".join(c["name"] for c in coins[:5])
            print(f"[{event['timestamp'][:19]}] Trending: {names}")
        elif etype == "prices":
            for cid, p in event.get("prices", {}).items():
                change = p.get("usd_24h_change")
                change_str = f"{change:+.1f}%" if change is not None else "N/A"
                print(f"[{event['timestamp'][:19]}] {cid}: ${p.get('usd', '?')} ({change_str})")

    connector = CoinGeckoConnector(callback=print_event, poll_interval=60)
    print("Polling CoinGecko — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
