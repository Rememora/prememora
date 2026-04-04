"""
Bitcoin on-chain and market data connector — polls Blockchain.com and
CoinGecko for key metrics and emits events with values and interpretations.

Sources:
  - Blockchain.com: hashrate, difficulty, market price, fees, tx count
    https://api.blockchain.info/stats
    https://api.blockchain.info/charts/{metric}?timespan=7days&format=json
  - CoinGecko: BTC dominance, total market cap, trending coins
    https://api.coingecko.com/api/v3/global

Free, no API keys required.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.onchain")

BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"
BLOCKCHAIN_CHARTS_URL = "https://api.blockchain.info/charts"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def _interpret_dominance(value: float) -> str:
    """Interpret BTC dominance percentage."""
    if value > 60:
        return "btc_dominant"
    elif value > 50:
        return "btc_leading"
    elif value > 40:
        return "balanced"
    else:
        return "altcoin_season"


def _interpret_hashrate_change(pct: float) -> str:
    """Interpret hashrate 7-day change."""
    if pct > 5:
        return "growing_fast"
    elif pct > 0:
        return "growing"
    elif pct > -5:
        return "declining"
    else:
        return "declining_fast"


@dataclass
class OnChainConnector:
    """Polls Blockchain.com and CoinGecko for on-chain/market metrics.

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
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Accept": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_blockchain_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch Bitcoin network stats from Blockchain.com."""
        session = await self._get_session()
        try:
            async with session.get(BLOCKCHAIN_STATS_URL) as resp:
                if resp.status != 200:
                    logger.warning("Blockchain.com stats API returned %d", resp.status)
                    return None
                return await resp.json(content_type=None)
        except Exception as e:
            logger.error("Failed to fetch blockchain stats: %s", e)
            return None

    async def _fetch_blockchain_chart(self, metric: str, timespan: str = "7days") -> Optional[List]:
        """Fetch a chart metric from Blockchain.com."""
        session = await self._get_session()
        url = f"{BLOCKCHAIN_CHARTS_URL}/{metric}"
        try:
            async with session.get(url, params={"timespan": timespan, "format": "json"}) as resp:
                if resp.status != 200:
                    logger.warning("Blockchain.com chart API returned %d for %s", resp.status, metric)
                    return None
                data = await resp.json(content_type=None)
                return data.get("values", [])
        except Exception as e:
            logger.error("Failed to fetch blockchain chart %s: %s", metric, e)
            return None

    async def _fetch_coingecko_global(self) -> Optional[Dict[str, Any]]:
        """Fetch global crypto market data from CoinGecko."""
        session = await self._get_session()
        try:
            async with session.get(COINGECKO_GLOBAL_URL) as resp:
                if resp.status != 200:
                    logger.warning("CoinGecko global API returned %d", resp.status)
                    return None
                data = await resp.json(content_type=None)
                return data.get("data", {})
        except Exception as e:
            logger.error("Failed to fetch CoinGecko global: %s", e)
            return None

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Fetch all metrics and return events."""
        events: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()

        # 1. Blockchain.com stats
        stats = await self._fetch_blockchain_stats()
        if stats:
            price = stats.get("market_price_usd", 0)
            prev_price = self._last_values.get("btc_price", price)
            events.append({
                "source": "onchain",
                "timestamp": now,
                "metric": "btc_price",
                "value": price,
                "previous_value": prev_price,
                "interpretation": f"${price:,.0f}",
            })
            self._last_values["btc_price"] = price

            hashrate = stats.get("hash_rate", 0)
            prev_hashrate = self._last_values.get("hashrate", hashrate)
            pct_change = ((hashrate - prev_hashrate) / prev_hashrate * 100) if prev_hashrate else 0
            events.append({
                "source": "onchain",
                "timestamp": now,
                "metric": "hashrate",
                "value": hashrate,
                "previous_value": prev_hashrate,
                "interpretation": _interpret_hashrate_change(pct_change),
            })
            self._last_values["hashrate"] = hashrate

            n_tx = stats.get("n_tx", 0)
            events.append({
                "source": "onchain",
                "timestamp": now,
                "metric": "daily_transactions",
                "value": n_tx,
                "previous_value": self._last_values.get("daily_transactions", n_tx),
                "interpretation": f"{n_tx:,} txs",
            })
            self._last_values["daily_transactions"] = n_tx

        # 2. CoinGecko global
        global_data = await self._fetch_coingecko_global()
        if global_data:
            mcap_pct = global_data.get("market_cap_percentage", {})
            btc_dom = mcap_pct.get("btc", 0)
            prev_dom = self._last_values.get("btc_dominance", btc_dom)
            events.append({
                "source": "onchain",
                "timestamp": now,
                "metric": "btc_dominance",
                "value": round(btc_dom, 2),
                "previous_value": round(prev_dom, 2),
                "interpretation": _interpret_dominance(btc_dom),
            })
            self._last_values["btc_dominance"] = btc_dom

            total_mcap = global_data.get("total_market_cap", {}).get("usd", 0)
            events.append({
                "source": "onchain",
                "timestamp": now,
                "metric": "total_market_cap",
                "value": total_mcap,
                "previous_value": self._last_values.get("total_market_cap", total_mcap),
                "interpretation": f"${total_mcap/1e12:.2f}T",
            })
            self._last_values["total_market_cap"] = total_mcap

            mcap_change = global_data.get("market_cap_change_percentage_24h_usd", 0)
            events.append({
                "source": "onchain",
                "timestamp": now,
                "metric": "market_cap_change_24h",
                "value": round(mcap_change, 2),
                "previous_value": self._last_values.get("market_cap_change_24h", 0),
                "interpretation": "bullish" if mcap_change > 0 else "bearish",
            })
            self._last_values["market_cap_change_24h"] = mcap_change

        return events

    async def start(self) -> None:
        """Start the polling loop."""
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
        self._running = False
