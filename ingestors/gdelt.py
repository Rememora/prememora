"""
GDELT Global Events connector — polls the GDELT DOC 2.0 API for articles
matching keyword groups relevant to prediction markets.

Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
Free, no API key required, updates every 15 minutes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.gdelt")

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

DEFAULT_QUERIES: List[str] = [
    "bitcoin OR cryptocurrency OR ethereum",
    "federal reserve OR interest rate OR inflation",
    "trump OR election OR congress",
    "war OR military OR sanctions",
]


def _parse_article(article: Dict[str, Any], search_query: str) -> Dict[str, Any]:
    """Parse a single GDELT article into a normalised event dict."""
    # GDELT seendate format: "20260404T103000Z"
    raw_date = article.get("seendate", "")
    try:
        dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        timestamp = dt.isoformat()
    except (ValueError, TypeError):
        timestamp = datetime.now(timezone.utc).isoformat()

    tone = 0.0
    raw_tone = article.get("tone")
    if raw_tone is not None:
        try:
            tone = float(raw_tone)
        except (ValueError, TypeError):
            pass

    return {
        "source": "gdelt",
        "timestamp": timestamp,
        "title": article.get("title", ""),
        "url": article.get("url", ""),
        "domain": article.get("domain", ""),
        "language": article.get("language", "English"),
        "tone": tone,
        "search_query": search_query,
    }


@dataclass
class GDELTConnector:
    """Polls the GDELT DOC 2.0 API for global event articles.

    Parameters
    ----------
    callback : EventCallback | None
        Async function called with each new article event dict.
    poll_interval : float
        Seconds between polls (default: 3600 = 1 hour).
    queries : list[str]
        Keyword query groups to search for.
    """

    callback: Optional[EventCallback] = None
    poll_interval: float = 3600.0
    queries: List[str] = field(default_factory=lambda: list(DEFAULT_QUERIES))
    _running: bool = field(default=False, init=False)
    _seen_urls: set = field(default_factory=set, init=False)
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

    async def _fetch_query(self, query: str, max_records: int = 20) -> List[Dict[str, Any]]:
        """Fetch articles for a single query from the GDELT API."""
        session = await self._get_session()
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "timespan": "1h",
        }
        try:
            async with session.get(BASE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("GDELT API returned %d for query '%s'", resp.status, query)
                    return []
                data = await resp.json(content_type=None)
                return data.get("articles", [])
        except Exception as e:
            logger.error("Failed to fetch GDELT data for query '%s': %s", query, e)
            return []

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Run all queries, deduplicate by URL, return list of new events."""
        all_events: List[Dict[str, Any]] = []
        seen_in_batch: set = set()

        for query in self.queries:
            articles = await self._fetch_query(query)
            for article in articles:
                url = article.get("url", "")
                if not url:
                    continue
                # Dedup within this batch (cross-query) and across polls.
                if url in seen_in_batch or url in self._seen_urls:
                    continue
                seen_in_batch.add(url)
                self._seen_urls.add(url)
                all_events.append(_parse_article(article, query))

        return all_events

    async def start(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info("GDELT connector starting, interval=%ss, queries=%d", self.poll_interval, len(self.queries))

        while self._running:
            try:
                events = await self.poll_once()
                if events and self.callback:
                    for event in events:
                        await self.callback(event)
                if events:
                    logger.info("GDELT: fetched %d new articles", len(events))
            except Exception:
                logger.exception("Error during GDELT poll cycle")

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
            f"[{event['timestamp']}] [{event['domain']}] {event['title'][:80]}"
            f" (tone={event['tone']:.1f}, query={event['search_query'][:40]})"
        )

    connector = GDELTConnector(callback=print_event, poll_interval=60)
    print("Polling GDELT — Ctrl-C to stop\n")

    try:
        await connector.start()
    except KeyboardInterrupt:
        connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(_main())
