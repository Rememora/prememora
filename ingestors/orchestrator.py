"""
Unified ingestion orchestrator — wires all event source connectors to
Graphiti for knowledge-graph episode creation.

Receives events from all connectors via async callbacks, normalizes them
into episode text, deduplicates across sources, and ingests into the
Graphiti knowledge graph via the adapter layer.

Usage:
    orchestrator = IngestionOrchestrator(graph_id="my_graph")
    await orchestrator.start()   # blocks until stop() or Ctrl-C
    await orchestrator.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("prememora.ingestors.orchestrator")

# Max dedup entries before LRU eviction.
DEDUP_MAX_SIZE = 10_000

# ── Normalizers ───────────────────────────────────────────────────────────────
# Each source emits different event dicts. These functions turn them into a
# plain text string suitable for Graphiti episode ingestion (Graphiti handles
# entity and relationship extraction from the text itself).


def _normalize_polymarket(event: dict[str, Any]) -> str:
    etype = event.get("event_type", "update")
    market = event.get("market_id", "unknown")
    data = event.get("data", {})
    ts = event.get("timestamp", "")

    if etype == "price_change":
        old = data.get("old_price", "?")
        new = data.get("price", "?")
        side = data.get("side", "")
        return (
            f"Polymarket price change on market {market}: "
            f"{side} moved from {old} to {new} at {ts}."
        )
    if etype == "last_trade_price":
        price = data.get("price", "?")
        size = data.get("size", "?")
        return f"Polymarket trade on market {market}: price {price}, size {size} at {ts}."
    return f"Polymarket {etype} on market {market} at {ts}: {data}"


def _normalize_crypto_news(event: dict[str, Any]) -> str:
    title = event.get("title", "")
    content = event.get("content", "")
    source_name = event.get("source_name", "")
    entities = event.get("entities", [])
    sentiment = event.get("sentiment", "")

    parts = []
    if title:
        parts.append(title)
    if content:
        parts.append(content[:1000])
    if source_name:
        parts.append(f"Source: {source_name}")
    if entities:
        parts.append(f"Entities: {', '.join(str(e) for e in entities)}")
    if sentiment:
        parts.append(f"Sentiment: {sentiment}")
    return " | ".join(parts) if parts else "Crypto news event (no content)"


def _normalize_rss(event: dict[str, Any]) -> str:
    title = event.get("title", "")
    content = event.get("content", "")
    feed = event.get("feed_name", "")
    category = event.get("category", "")

    parts = []
    if feed:
        parts.append(f"[{feed}]")
    if category:
        parts.append(f"({category})")
    if title:
        parts.append(title)
    if content:
        parts.append(content[:1000])
    return " ".join(parts) if parts else "RSS article (no content)"


def _normalize_whale(event: dict[str, Any]) -> str:
    chain = event.get("chain", "unknown")
    token = event.get("token", "?")
    amount = event.get("amount_usd", 0)
    from_e = event.get("from_entity", "unknown")
    to_e = event.get("to_entity", "unknown")
    cls = event.get("classification", "unknown")
    tx = event.get("tx_hash", "")

    return (
        f"Whale alert on {chain}: {token} ${amount:,.0f} transfer from {from_e} "
        f"to {to_e} ({cls}). TX: {tx}"
    )


def _normalize_reddit(event: dict[str, Any]) -> str:
    sub = event.get("subreddit", "")
    title = event.get("title", "")
    content = event.get("content", "")
    score = event.get("score", 0)
    comments = event.get("num_comments", 0)

    parts = [f"Reddit r/{sub}: {title}"]
    if content:
        parts.append(content[:500])
    parts.append(f"Score: {score}, Comments: {comments}")
    return " | ".join(parts)


def _normalize_fred(event: dict[str, Any]) -> str:
    name = event.get("series_name", event.get("series_id", "unknown"))
    value = event.get("value", "?")
    change = event.get("change")
    units = event.get("units", "")

    text = f"FRED macro update: {name} = {value} {units}"
    if change is not None:
        sign = "+" if change >= 0 else ""
        text += f" (change: {sign}{change})"
    return text


_NORMALIZERS: dict[str, Any] = {
    "polymarket_ws": _normalize_polymarket,
    "crypto_news": _normalize_crypto_news,
    "rss": _normalize_rss,
    "whale_alert": _normalize_whale,
    "reddit": _normalize_reddit,
    "fred": _normalize_fred,
}


def normalize_event(event: dict[str, Any]) -> str:
    """Turn any source event dict into episode text for Graphiti."""
    source = event.get("source", "")
    normalizer = _NORMALIZERS.get(source)
    if normalizer:
        return normalizer(event)
    # Fallback: dump the whole dict as a readable string.
    return f"Event from {source}: {event}"


# ── Dedup ─────────────────────────────────────────────────────────────────────


def _dedup_key(event: dict[str, Any]) -> str:
    """Compute a dedup key from event content.

    Uses source + a content hash. Two different sources reporting the same
    underlying story will have different keys (cross-source dedup would
    require LLM similarity — out of scope for now).
    """
    source = event.get("source", "")

    # Use the most stable identifier available per source.
    if source == "polymarket_ws":
        # Polymarket events are high-frequency; dedup by market + type + price.
        raw = f"{event.get('market_id')}:{event.get('event_type')}:{event.get('data', {}).get('price')}"
    elif source in ("crypto_news", "rss"):
        raw = event.get("url") or event.get("title", "")
    elif source == "whale_alert":
        raw = event.get("tx_hash") or str(event.get("data"))
    elif source == "reddit":
        raw = event.get("post_id") or event.get("url", "")
    elif source == "fred":
        raw = f"{event.get('series_id')}:{event.get('timestamp')}"
    else:
        raw = str(event)

    h = hashlib.sha256(f"{source}:{raw}".encode()).hexdigest()[:16]
    return h


class _LRUDedup:
    """Fixed-size LRU set for deduplication."""

    def __init__(self, maxsize: int = DEDUP_MAX_SIZE):
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._maxsize = maxsize

    def is_duplicate(self, key: str) -> bool:
        if key in self._seen:
            self._seen.move_to_end(key)
            return True
        self._seen[key] = None
        if len(self._seen) > self._maxsize:
            self._seen.popitem(last=False)
        return False


# ── Orchestrator ──────────────────────────────────────────────────────────────


@dataclass
class IngestionOrchestrator:
    """Wires all event source connectors to Graphiti episode ingestion.

    Parameters
    ----------
    graph_id : str
        The Graphiti graph to ingest episodes into.
    neo4j_uri / neo4j_user / neo4j_password : str
        Neo4j connection details (defaults from env / standard local).
    polymarket_asset_ids : list[str]
        Initial Polymarket asset IDs to subscribe to.
    enable_polymarket, enable_crypto_news, etc. : bool
        Toggle individual connectors on/off.
    """

    graph_id: str = ""
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    polymarket_asset_ids: list[str] = field(default_factory=list)

    # Connector toggles — all on by default.
    enable_polymarket: bool = True
    enable_crypto_news: bool = True
    enable_rss: bool = True
    enable_whale: bool = True
    enable_reddit: bool = True
    enable_fred: bool = True

    # Internals
    _dedup: _LRUDedup = field(default_factory=_LRUDedup, init=False)
    _connectors: list[Any] = field(default_factory=list, init=False)
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False)
    _client: Any = field(default=None, init=False)
    _ingested_count: int = field(default=0, init=False)
    _deduped_count: int = field(default=0, init=False)
    _error_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.neo4j_uri = self.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = self.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = self.neo4j_password or os.getenv("NEO4J_PASSWORD", "prememora_local")

    # ── Event handling ────────────────────────────────────────────────

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Process a single event: dedup → normalize → ingest."""
        key = _dedup_key(event)
        if self._dedup.is_duplicate(key):
            self._deduped_count += 1
            logger.debug("Dedup: skipping %s event (key=%s)", event.get("source"), key)
            return

        text = normalize_event(event)
        if not text.strip():
            return

        source = event.get("source", "unknown")
        ts = event.get("timestamp", datetime.now(timezone.utc).isoformat())

        try:
            self._client.graph.add(
                graph_id=self.graph_id,
                data=f"[{ts}] [{source}] {text}",
            )
            self._ingested_count += 1
            logger.info("Ingested %s event → graph %s (total: %d)", source, self.graph_id, self._ingested_count)
        except Exception:
            self._error_count += 1
            logger.exception("Failed to ingest %s event into graph %s", source, self.graph_id)

    async def _handle_event_batch(self, events: list[dict[str, Any]]) -> None:
        """Handle a batch of events (used by Reddit and FRED callbacks)."""
        for event in events:
            await self._handle_event(event)

    # ── Connector wiring ──────────────────────────────────────────────

    def _build_connectors(self) -> list[tuple[str, Any]]:
        """Instantiate enabled connectors with orchestrator callbacks."""
        connectors: list[tuple[str, Any]] = []

        if self.enable_polymarket:
            from ingestors.polymarket_ws import PolymarketWSConnector

            c = PolymarketWSConnector(
                asset_ids=list(self.polymarket_asset_ids),
                callback=self._handle_event,
            )
            connectors.append(("polymarket_ws", c))

        if self.enable_crypto_news:
            from ingestors.crypto_news import CryptoNewsConnector

            c = CryptoNewsConnector(callback=self._handle_event)
            connectors.append(("crypto_news", c))

        if self.enable_rss:
            from ingestors.rss_feeds import RSSPoller

            c = RSSPoller(callback=self._handle_event)
            connectors.append(("rss", c))

        if self.enable_whale:
            from ingestors.whale_tracker import WhaleTracker

            c = WhaleTracker(callback=self._handle_event)
            connectors.append(("whale_alert", c))

        if self.enable_reddit:
            from ingestors.reddit_sentiment import RedditSentimentConnector

            c = RedditSentimentConnector(callback=self._handle_event_batch)
            connectors.append(("reddit", c))

        if self.enable_fred:
            from ingestors.fred_macro import FredMacroConnector

            c = FredMacroConnector(callback=self._handle_event_batch)
            connectors.append(("fred", c))

        return connectors

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all enabled connectors and block until stopped."""
        if not self.graph_id:
            raise ValueError("graph_id is required — set it before calling start()")

        from adapter.client import GraphitiZepClient

        self._client = GraphitiZepClient(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
        )

        named_connectors = self._build_connectors()
        if not named_connectors:
            logger.warning("No connectors enabled — nothing to do")
            return

        logger.info(
            "Starting orchestrator — graph=%s, connectors=%s",
            self.graph_id,
            [name for name, _ in named_connectors],
        )

        self._connectors = [c for _, c in named_connectors]

        # Launch each connector as a task. Different connectors have
        # different start methods:
        #   - PolymarketWSConnector.start()  (async, blocks)
        #   - CryptoNewsConnector.start()    (async, blocks)
        #   - RSSPoller.run()                (async, blocks)
        #   - WhaleTracker.start()           (async, blocks)
        #   - RedditSentimentConnector.start() (async, blocks)
        #   - FredMacroConnector.start()     (async, creates background task)
        for name, connector in named_connectors:
            if hasattr(connector, "run") and name == "rss":
                coro = connector.run()
            else:
                coro = connector.start()
            task = asyncio.create_task(coro, name=f"connector:{name}")
            self._tasks.append(task)
            logger.info("Launched connector: %s", name)

        # Wait for all tasks — if one crashes, log it but keep the rest running.
        done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if task.exception():
                logger.error(
                    "Connector %s crashed: %s", task.get_name(), task.exception()
                )

    async def stop(self) -> None:
        """Gracefully stop all connectors."""
        logger.info("Stopping orchestrator (ingested=%d, deduped=%d, errors=%d)",
                     self._ingested_count, self._deduped_count, self._error_count)

        for connector in self._connectors:
            try:
                if hasattr(connector, "stop"):
                    result = connector.stop()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                logger.exception("Error stopping connector %s", type(connector).__name__)

        # Cancel any remaining tasks.
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        self._connectors.clear()

        logger.info("Orchestrator stopped")

    def get_stats(self) -> dict[str, Any]:
        """Return current ingestion statistics."""
        return {
            "ingested": self._ingested_count,
            "deduped": self._deduped_count,
            "errors": self._error_count,
            "active_connectors": len([t for t in self._tasks if not t.done()]),
        }


# ── CLI entry point ───────────────────────────────────────────────────────────


async def main():
    """Standalone mode: run the orchestrator with all connectors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    graph_id = os.getenv("PREMEMORA_GRAPH_ID", "")
    if not graph_id:
        print("Set PREMEMORA_GRAPH_ID env var to the target Graphiti graph ID.")
        print("Example: PREMEMORA_GRAPH_ID=mirofish_abc123 python -m ingestors.orchestrator")
        return

    orchestrator = IngestionOrchestrator(graph_id=graph_id)

    print(f"Starting ingestion orchestrator → graph {graph_id}")
    print("Press Ctrl-C to stop.\n")

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        pass
    finally:
        await orchestrator.stop()
        stats = orchestrator.get_stats()
        print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
