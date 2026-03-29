"""
RSS Feed Poller — polls wire services and news sources for breaking news.

Fetches AP News, Reuters, CoinDesk, CoinTelegraph, The Block, White House,
and Federal Reserve feeds. Deduplicates by URL/GUID, normalizes into event
dicts, and dispatches new articles via an async callback.

Config via environment variables (or .env):
    RSS_POLL_INTERVAL   — seconds between polls (default: 30)
    RSS_FEEDS           — JSON list of {name, url, category} overrides
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import aiohttp
import feedparser
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("prememora.ingestors.rss")

# ── Default feed list ────────────────────────────────────────────────────────

FeedConfig = Dict[str, str]  # keys: name, url, category

DEFAULT_FEEDS: List[FeedConfig] = [
    {
        "name": "AP News — Top News",
        "url": "https://news.google.com/rss/search?q=when:24h+allinurl:apnews.com&ceid=US:en&hl=en-US&gl=US",
        "category": "general",
    },
    {
        "name": "Reuters — World",
        "url": "https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com&ceid=US:en&hl=en-US&gl=US",
        "category": "general",
    },
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "category": "crypto",
    },
    {
        "name": "CoinTelegraph",
        "url": "https://cointelegraph.com/rss",
        "category": "crypto",
    },
    {
        "name": "The Block",
        "url": "https://www.theblock.co/rss.xml",
        "category": "crypto",
    },
    {
        "name": "White House",
        "url": "https://www.whitehouse.gov/news/feed/",
        "category": "government",
    },
    {
        "name": "Federal Reserve — Press Releases",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "category": "government",
    },
]

# Type alias for the async callback that receives new articles.
ArticleCallback = Callable[[Dict[str, Any]], Awaitable[None]]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_timestamp(entry: feedparser.FeedParserDict) -> str:
    """Return an ISO-8601 timestamp string from an RSS entry.

    Tries ``published_parsed``, ``updated_parsed``, and raw date strings
    before falling back to the current UTC time.
    """
    for attr in ("published_parsed", "updated_parsed"):
        tp = getattr(entry, attr, None)
        if tp is not None:
            try:
                return datetime(*tp[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass

    # Try raw date strings
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                return parsedate_to_datetime(raw).isoformat()
            except Exception:
                pass

    return datetime.now(timezone.utc).isoformat()


def _entry_id(entry: feedparser.FeedParserDict) -> str:
    """Stable dedup key: prefer ``id`` (GUID), then ``link``."""
    return getattr(entry, "id", None) or getattr(entry, "link", "") or ""


def _entry_content(entry: feedparser.FeedParserDict) -> str:
    """Best-effort plain-text content from an RSS entry."""
    # Some feeds put the body in <content:encoded> or <content>
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    return getattr(entry, "summary", "") or getattr(entry, "description", "") or ""


def _normalize_entry(
    entry: feedparser.FeedParserDict,
    feed_name: str,
    feed_url: str,
    category: str,
) -> Dict[str, Any]:
    """Normalize a feedparser entry into a standard event dict."""
    return {
        "source": "rss",
        "timestamp": _parse_timestamp(entry),
        "title": getattr(entry, "title", "(no title)"),
        "content": _entry_content(entry),
        "url": getattr(entry, "link", feed_url),
        "feed_name": feed_name,
        "category": category,
    }


# ── RSS Poller ───────────────────────────────────────────────────────────────


class RSSPoller:
    """Async RSS feed poller with deduplication and configurable callbacks.

    Parameters
    ----------
    feeds : list[FeedConfig] | None
        Override the default feed list.
    poll_interval : float
        Seconds between polling cycles.
    callback : ArticleCallback | None
        Async function invoked for each new article.
    user_agent : str
        HTTP User-Agent header sent with requests.
    """

    def __init__(
        self,
        feeds: Optional[List[FeedConfig]] = None,
        poll_interval: float | None = None,
        callback: Optional[ArticleCallback] = None,
        user_agent: str = "PreMemora-RSS/0.1 (+https://github.com/prememora)",
    ) -> None:
        self.feeds = feeds or _load_feeds_from_env()
        self.poll_interval = poll_interval or float(
            os.getenv("RSS_POLL_INTERVAL", "30")
        )
        self.callback = callback
        self.user_agent = user_agent

        # Dedup state — set of entry IDs already seen.
        self._seen: Set[str] = set()
        self._running = False

    # ── public API ───────────────────────────────────────────────────────

    async def poll_once(self) -> List[Dict[str, Any]]:
        """Fetch all feeds once and return *new* (unseen) articles."""
        new_articles: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession(
            headers={"User-Agent": self.user_agent},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            tasks = [self._fetch_feed(session, feed) for feed in self.feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.error("Feed fetch error: %s", result)
                continue
            for article in result:
                eid = article["url"]
                if eid and eid not in self._seen:
                    self._seen.add(eid)
                    new_articles.append(article)
                    if self.callback:
                        try:
                            await self.callback(article)
                        except Exception:
                            logger.exception(
                                "Callback error for article: %s", article.get("title")
                            )

        return new_articles

    async def run(self) -> None:
        """Start the polling loop. Runs until ``stop()`` is called."""
        self._running = True
        logger.info(
            "RSS poller starting — %d feeds, interval=%ss",
            len(self.feeds),
            self.poll_interval,
        )

        while self._running:
            try:
                new = await self.poll_once()
                if new:
                    logger.info("Fetched %d new articles", len(new))
            except Exception:
                logger.exception("Error during poll cycle")

            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Signal the polling loop to stop after the current cycle."""
        self._running = False

    @property
    def seen_count(self) -> int:
        """Number of unique articles seen since start."""
        return len(self._seen)

    # ── internals ────────────────────────────────────────────────────────

    async def _fetch_feed(
        self, session: aiohttp.ClientSession, feed: FeedConfig
    ) -> List[Dict[str, Any]]:
        """Fetch and parse a single RSS feed, returning normalized articles."""
        url = feed["url"]
        name = feed.get("name", url)
        category = feed.get("category", "unknown")

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Feed %s returned HTTP %s", name, resp.status
                    )
                    return []
                raw = await resp.text()
        except Exception as exc:
            logger.warning("Failed to fetch feed %s: %s", name, exc)
            return []

        parsed = feedparser.parse(raw)
        if parsed.bozo and not parsed.entries:
            logger.warning(
                "Feed %s parse error: %s", name, parsed.bozo_exception
            )
            return []

        articles: List[Dict[str, Any]] = []
        for entry in parsed.entries:
            eid = _entry_id(entry)
            if not eid or eid in self._seen:
                continue
            articles.append(_normalize_entry(entry, name, url, category))

        logger.debug("Feed %s: %d entries, %d new", name, len(parsed.entries), len(articles))
        return articles


# ── Config helpers ───────────────────────────────────────────────────────────


def _load_feeds_from_env() -> List[FeedConfig]:
    """Load feeds from RSS_FEEDS env var (JSON list) or return defaults."""
    raw = os.getenv("RSS_FEEDS")
    if raw:
        try:
            feeds = json.loads(raw)
            if isinstance(feeds, list) and all(isinstance(f, dict) for f in feeds):
                return feeds
            logger.warning("RSS_FEEDS is not a list of dicts, using defaults")
        except json.JSONDecodeError:
            logger.warning("RSS_FEEDS is not valid JSON, using defaults")
    return DEFAULT_FEEDS


# ── __main__ — standalone demo ───────────────────────────────────────────────

async def _main() -> None:
    """Poll feeds and print new articles to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    async def print_article(article: Dict[str, Any]) -> None:
        ts = article["timestamp"]
        title = article["title"]
        url = article["url"]
        feed = article["feed_name"]
        cat = article["category"]
        print(f"[{ts}] [{cat}] {feed}\n  {title}\n  {url}\n")

    poller = RSSPoller(callback=print_article)
    print(
        f"Polling {len(poller.feeds)} feeds every {poller.poll_interval}s — Ctrl+C to stop\n"
    )

    try:
        await poller.run()
    except KeyboardInterrupt:
        poller.stop()
        print(f"\nStopped. Saw {poller.seen_count} unique articles total.")


if __name__ == "__main__":
    asyncio.run(_main())
