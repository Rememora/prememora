"""
Polymarket WebSocket connector — streams real-time market odds changes.

Connects to the Polymarket CLOB WebSocket and emits normalized events
for price changes, trades, and order book updates.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("prememora.ingestors.polymarket_ws")

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
HEARTBEAT_INTERVAL = 10  # seconds
RECONNECT_BASE_DELAY = 1  # seconds
RECONNECT_MAX_DELAY = 60  # seconds

EventCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class PolymarketWSConnector:
    """Streams real-time Polymarket market data via WebSocket."""

    asset_ids: List[str] = field(default_factory=list)
    callback: Optional[EventCallback] = None
    _ws: Any = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _subscribed: Set[str] = field(default_factory=set, init=False)

    async def start(self):
        """Connect and stream events with automatic reconnection."""
        self._running = True
        delay = RECONNECT_BASE_DELAY

        while self._running:
            try:
                async with websockets.connect(WS_URL, ping_interval=None) as ws:
                    self._ws = ws
                    delay = RECONNECT_BASE_DELAY
                    logger.info("Connected to Polymarket WebSocket")

                    await self._subscribe(ws)
                    await asyncio.gather(
                        self._heartbeat_loop(ws),
                        self._receive_loop(ws),
                    )
            except ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e.code} {e.reason}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if not self._running:
                break

            logger.info(f"Reconnecting in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def stop(self):
        """Disconnect gracefully."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def subscribe(self, asset_ids: List[str]):
        """Subscribe to additional markets at runtime."""
        self.asset_ids.extend(asset_ids)
        if self._ws:
            await self._subscribe(self._ws, asset_ids)

    async def _subscribe(self, ws, asset_ids: Optional[List[str]] = None):
        """Send subscription messages for asset IDs."""
        ids = asset_ids or self.asset_ids
        if not ids:
            logger.warning("No asset IDs to subscribe to")
            return

        for asset_id in ids:
            if asset_id in self._subscribed:
                continue
            msg = json.dumps({
                "type": "market",
                "assets_ids": [asset_id],
            })
            await ws.send(msg)
            self._subscribed.add(asset_id)
            logger.debug(f"Subscribed to {asset_id}")

    async def _heartbeat_loop(self, ws):
        """Send PING every HEARTBEAT_INTERVAL seconds."""
        while self._running:
            try:
                await ws.send("PING")
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except ConnectionClosed:
                return

    async def _receive_loop(self, ws):
        """Receive and process messages."""
        async for raw in ws:
            if raw == "PONG":
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug(f"Non-JSON message: {raw[:100]}")
                continue

            events = self._parse_message(data)
            for event in events:
                if self.callback:
                    await self.callback(event)
                else:
                    logger.debug(f"Event (no callback): {event['event_type']}")

    def _parse_message(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a WebSocket message into normalized event dicts."""
        events = []
        now = datetime.now(timezone.utc).isoformat()

        # The WS sends arrays of market updates
        items = data if isinstance(data, list) else [data]

        for item in items:
            event_type = item.get("event_type") or item.get("type", "unknown")
            asset_id = item.get("asset_id", "")
            market_id = item.get("market") or item.get("condition_id", "")

            event = {
                "source": "polymarket_ws",
                "timestamp": item.get("timestamp", now),
                "market_id": market_id,
                "asset_id": asset_id,
                "event_type": event_type,
                "data": {},
            }

            if event_type == "price_change":
                event["data"] = {
                    "price": item.get("price"),
                    "old_price": item.get("old_price"),
                    "side": item.get("side"),
                }
            elif event_type == "last_trade_price":
                event["data"] = {
                    "price": item.get("price"),
                    "size": item.get("size"),
                    "side": item.get("side"),
                }
            elif event_type == "book":
                event["data"] = {
                    "bids": item.get("bids", []),
                    "asks": item.get("asks", []),
                    "best_bid": item.get("best_bid"),
                    "best_ask": item.get("best_ask"),
                    "spread": item.get("spread"),
                }
            elif event_type == "tick_size_change":
                event["data"] = {
                    "tick_size": item.get("tick_size"),
                    "old_tick_size": item.get("old_tick_size"),
                }
            else:
                event["data"] = {
                    k: v for k, v in item.items()
                    if k not in ("event_type", "type", "asset_id", "market", "condition_id", "timestamp")
                }

            events.append(event)

        return events


async def _print_callback(event: Dict[str, Any]):
    """Default callback for standalone testing — prints events."""
    ts = event.get("timestamp", "")
    etype = event["event_type"]
    market = event.get("market_id", "")[:16]
    data = event.get("data", {})
    print(f"[{ts}] {etype:20s} market={market}... {json.dumps(data, default=str)[:120]}")


async def main():
    """Standalone mode: connect and print all events for demonstration."""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Try to discover some active markets via Gamma API
    asset_ids = os.environ.get("POLYMARKET_ASSET_IDS", "").split(",")
    asset_ids = [a.strip() for a in asset_ids if a.strip()]

    if not asset_ids:
        print("Discovering active markets from Gamma API...")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"limit": 5, "active": "true", "order": "volume", "ascending": "false"},
                ) as resp:
                    if resp.status == 200:
                        markets = await resp.json()
                        for m in markets:
                            for token in m.get("tokens", []):
                                token_id = token.get("token_id")
                                if token_id:
                                    asset_ids.append(token_id)
                        print(f"Discovered {len(asset_ids)} tokens from {len(markets)} markets")
                    else:
                        print(f"Gamma API returned {resp.status}")
        except ImportError:
            print("Install aiohttp for market discovery: pip install aiohttp")
        except Exception as e:
            print(f"Market discovery failed: {e}")

    if not asset_ids:
        print("No asset IDs. Set POLYMARKET_ASSET_IDS env var or install aiohttp for auto-discovery.")
        return

    connector = PolymarketWSConnector(
        asset_ids=asset_ids,
        callback=_print_callback,
    )

    print(f"Connecting to Polymarket WebSocket with {len(asset_ids)} assets...")
    try:
        await connector.start()
    except KeyboardInterrupt:
        await connector.stop()


if __name__ == "__main__":
    asyncio.run(main())
