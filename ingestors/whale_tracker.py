"""
On-chain whale tracking connector.

Monitors large-value cryptocurrency transactions via the Whale Alert API
(https://whale-alert.io) and normalizes them into prediction-engine events.

Modes:
  - Whale Alert API (requires WHALE_ALERT_API_KEY, paid tier)
  - Free fallback: logs a warning and skips gracefully when no key is set

Events are delivered to an async callback with a normalized dict schema:
    source, timestamp, chain, from_entity, to_entity, amount_usd,
    token, tx_hash, classification
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("prememora.ingestors.whale_tracker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHALE_ALERT_BASE_URL = "https://api.whale-alert.io/v1"

SUPPORTED_CHAINS = {"bitcoin", "ethereum", "solana"}

# Well-known exchange addresses (lowercase).
# This is a representative starter set — a production deployment would use a
# much larger list maintained separately or fetched from an on-chain labeling
# service.
KNOWN_EXCHANGE_ADDRESSES: Dict[str, Set[str]] = {
    "bitcoin": {
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3",  # Binance
        "3m9sphqxzwlajz1xmjru1nmnymezyrljeu",          # Bitfinex
        "1kryen1begieyecbopyb6gqiytavktkuzc",           # Coinbase
        "3cbetxrboynkjcjbkghxojynvrgyvcckes",           # Kraken
    },
    "ethereum": {
        "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance 14
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance 15
        "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 16
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance (old)
        "0x71660c4005ba85c37ccec55d0c4493e66fe775d3",  # Coinbase 1
        "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase 2
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",  # Kraken
        "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0",  # Kraken 4
    },
    "solana": {
        "5tzfn6qwefq7ykfcebrqrcpshszzdrxdrbeqfi9oledm",  # Binance
        "9wfppneyagfq3cgymwq7em3wdrslup7zzacsawesvd1f",  # FTX (historical)
    },
}


class TxClassification(str, Enum):
    """Classification labels for whale transactions."""
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    WHALE_TRANSFER = "whale_transfer"
    EXCHANGE_TO_EXCHANGE = "exchange_to_exchange"
    UNKNOWN = "unknown"


# Type alias for the user-supplied async callback that receives events.
EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_chain(raw: str) -> str:
    """Map Whale Alert chain names to our canonical set."""
    mapping = {
        "bitcoin": "bitcoin",
        "btc": "bitcoin",
        "ethereum": "ethereum",
        "eth": "ethereum",
        "solana": "solana",
        "sol": "solana",
    }
    return mapping.get(raw.lower(), raw.lower())


def _is_exchange(address: str, chain: str) -> bool:
    """Return True if the address belongs to a known exchange."""
    if not address:
        return False
    addrs = KNOWN_EXCHANGE_ADDRESSES.get(chain, set())
    return address.lower() in addrs


def classify_transaction(
    from_address: str,
    to_address: str,
    chain: str,
    from_owner: str = "",
    to_owner: str = "",
) -> TxClassification:
    """Determine a human-readable classification for a whale tx.

    Uses both address lookups and owner-type hints from the API.
    """
    from_is_exchange = (
        _is_exchange(from_address, chain)
        or (from_owner.lower() in ("exchange", "unknown")
            and _is_exchange(from_address, chain))
    )
    to_is_exchange = (
        _is_exchange(to_address, chain)
        or (to_owner.lower() in ("exchange", "unknown")
            and _is_exchange(to_address, chain))
    )

    # Also classify based on the Whale Alert owner_type field when present.
    if from_owner.lower() == "exchange" and to_owner.lower() == "exchange":
        return TxClassification.EXCHANGE_TO_EXCHANGE
    if from_owner.lower() == "exchange":
        from_is_exchange = True
    if to_owner.lower() == "exchange":
        to_is_exchange = True

    if from_is_exchange and to_is_exchange:
        return TxClassification.EXCHANGE_TO_EXCHANGE
    if to_is_exchange:
        return TxClassification.EXCHANGE_INFLOW
    if from_is_exchange:
        return TxClassification.EXCHANGE_OUTFLOW

    return TxClassification.WHALE_TRANSFER


def _build_event(tx: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single Whale Alert transaction into our normalized schema."""
    chain = _normalize_chain(tx.get("blockchain", "unknown"))
    from_addr = tx.get("from", {}).get("address", "") if isinstance(tx.get("from"), dict) else ""
    to_addr = tx.get("to", {}).get("address", "") if isinstance(tx.get("to"), dict) else ""
    from_owner = tx.get("from", {}).get("owner", "") if isinstance(tx.get("from"), dict) else ""
    to_owner = tx.get("to", {}).get("owner", "") if isinstance(tx.get("to"), dict) else ""
    from_owner_type = tx.get("from", {}).get("owner_type", "") if isinstance(tx.get("from"), dict) else ""
    to_owner_type = tx.get("to", {}).get("owner_type", "") if isinstance(tx.get("to"), dict) else ""

    classification = classify_transaction(
        from_addr, to_addr, chain,
        from_owner=from_owner_type,
        to_owner=to_owner_type,
    )

    return {
        "source": "whale_alert",
        "timestamp": datetime.fromtimestamp(
            tx.get("timestamp", time.time()), tz=timezone.utc
        ).isoformat(),
        "chain": chain,
        "from_entity": from_owner or from_addr or "unknown",
        "to_entity": to_owner or to_addr or "unknown",
        "amount_usd": tx.get("amount_usd", 0.0),
        "token": tx.get("symbol", "").upper(),
        "tx_hash": tx.get("hash", ""),
        "classification": classification.value,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WhaleTrackerConfig:
    """Runtime configuration for the whale tracker."""

    api_key: str = ""
    min_value_usd: int = 1_000_000
    chains: List[str] = field(default_factory=lambda: ["bitcoin", "ethereum", "solana"])
    poll_interval_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("WHALE_ALERT_API_KEY", "")
        # Normalise chain names.
        self.chains = [_normalize_chain(c) for c in self.chains]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class WhaleTracker:
    """Async whale-transaction poller.

    Usage::

        tracker = WhaleTracker(config, callback=my_handler)
        await tracker.start()   # runs until cancelled
        # or
        events = await tracker.poll_once()
    """

    def __init__(
        self,
        config: Optional[WhaleTrackerConfig] = None,
        callback: Optional[EventCallback] = None,
    ) -> None:
        self.config = config or WhaleTrackerConfig()
        self.callback = callback
        self._cursor: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        # Track the last timestamp we fetched so the first request gets
        # recent transactions rather than the full history.
        self._last_timestamp: int = int(time.time()) - self.config.poll_interval_seconds

    # -- HTTP helpers -------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        """Cleanly close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- Core API -----------------------------------------------------------

    async def _fetch_whale_alert(self) -> List[Dict[str, Any]]:
        """Call the Whale Alert REST API and return raw transaction dicts."""
        if not self.config.api_key:
            logger.warning(
                "WHALE_ALERT_API_KEY not set — skipping Whale Alert fetch. "
                "Set the key in .env or pass it in WhaleTrackerConfig."
            )
            return []

        session = await self._ensure_session()
        params: Dict[str, Any] = {
            "api_key": self.config.api_key,
            "min_value": self.config.min_value_usd,
            "start": self._last_timestamp,
        }
        if self._cursor:
            params["cursor"] = self._cursor

        url = f"{WHALE_ALERT_BASE_URL}/transactions"
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 401:
                    logger.error("Whale Alert API returned 401 — check your API key")
                    return []
                if resp.status == 429:
                    logger.warning("Whale Alert rate-limited (429) — backing off")
                    return []
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Whale Alert API error %d: %s", resp.status, body[:500])
                    return []

                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Whale Alert request failed: %s", exc)
            return []

        self._cursor = data.get("cursor")
        txs: List[Dict[str, Any]] = data.get("transactions", [])

        # Filter to requested chains.
        allowed = set(self.config.chains)
        filtered = [
            tx for tx in txs
            if _normalize_chain(tx.get("blockchain", "")) in allowed
        ]

        # Advance last_timestamp so the next poll doesn't re-fetch.
        if filtered:
            max_ts = max(tx.get("timestamp", 0) for tx in filtered)
            if max_ts > self._last_timestamp:
                self._last_timestamp = max_ts

        return filtered

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Execute a single poll cycle and return normalized events.

        If a callback is registered it is invoked for each event.
        """
        raw_txs = await self._fetch_whale_alert()
        events: List[Dict[str, Any]] = []
        for tx in raw_txs:
            event = _build_event(tx)
            events.append(event)
            if self.callback is not None:
                try:
                    await self.callback(event)
                except Exception:
                    logger.exception("Callback error for tx %s", event.get("tx_hash"))
        if events:
            logger.info("Polled %d whale events", len(events))
        else:
            logger.debug("No new whale events this cycle")
        return events

    async def start(self) -> None:
        """Start the polling loop. Runs until ``stop()`` is called or cancelled."""
        self._running = True
        logger.info(
            "WhaleTracker started — polling every %ds, min_value=$%s, chains=%s, api=%s",
            self.config.poll_interval_seconds,
            f"{self.config.min_value_usd:,}",
            self.config.chains,
            "whale_alert" if self.config.api_key else "none (dry-run)",
        )
        try:
            while self._running:
                await self.poll_once()
                await asyncio.sleep(self.config.poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("WhaleTracker polling cancelled")
        finally:
            await self.close()

    def stop(self) -> None:
        """Signal the polling loop to exit after the current cycle."""
        self._running = False
        logger.info("WhaleTracker stop requested")


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------

async def _demo_callback(event: Dict[str, Any]) -> None:
    chain = event["chain"]
    amt = event["amount_usd"]
    cls = event["classification"]
    token = event["token"]
    print(f"  [{chain}] {token} ${amt:,.0f} — {cls}")


async def _main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = WhaleTrackerConfig()
    tracker = WhaleTracker(config=config, callback=_demo_callback)

    if not config.api_key:
        print(
            "No WHALE_ALERT_API_KEY found. Running a single dry-run poll "
            "(will return no data).\n"
            "Set WHALE_ALERT_API_KEY in .env to fetch live transactions."
        )
        events = await tracker.poll_once()
        print(f"Events returned: {len(events)}")
        await tracker.close()
        return

    print(
        f"Whale tracker running — min ${config.min_value_usd:,}, "
        f"chains {config.chains}, interval {config.poll_interval_seconds}s\n"
        "Press Ctrl-C to stop.\n"
    )
    try:
        await tracker.start()
    except KeyboardInterrupt:
        tracker.stop()
        await tracker.close()


if __name__ == "__main__":
    asyncio.run(_main())
