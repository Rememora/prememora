"""
Free Crypto News API connector — streams real-time crypto news with sentiment.

Uses the free API at https://cryptocurrency.cv for real-time news,
search, sentiment analysis, and historical archive access.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp

logger = logging.getLogger("prememora.ingestors.crypto_news")

BASE_URL = "https://cryptocurrency.cv"

EventCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class CryptoNewsConnector:
    """Streams and fetches crypto news from the Free Crypto News API."""

    callback: Optional[EventCallback] = None
    poll_interval: float = 30.0
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

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_latest(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch latest news articles via REST."""
        session = await self._get_session()
        try:
            async with session.get(f"{BASE_URL}/api/news", params={"limit": limit}) as resp:
                if resp.status != 200:
                    logger.warning(f"News API returned {resp.status}")
                    return []
                data = await resp.json()
                articles = data if isinstance(data, list) else data.get("articles", data.get("data", []))
                return [self._normalize(a) for a in articles]
        except Exception as e:
            logger.error(f"Failed to fetch latest news: {e}")
            return []

    async def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for articles by keyword."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{BASE_URL}/api/search",
                params={"q": query, "limit": limit},
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Search API returned {resp.status}")
                    return []
                data = await resp.json()
                articles = data if isinstance(data, list) else data.get("articles", data.get("data", []))
                return [self._normalize(a) for a in articles]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Get AI sentiment analysis for text."""
        session = await self._get_session()
        try:
            async with session.post(
                f"{BASE_URL}/api/ai/sentiment",
                json={"text": text},
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"Sentiment API failed: {e}")
            return None

    async def fetch_archive(
        self, start_date: str = "", end_date: str = "", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch historical articles from archive (for backtesting)."""
        session = await self._get_session()
        params: Dict[str, Any] = {"limit": limit}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        try:
            async with session.get(f"{BASE_URL}/api/archive", params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Archive API returned {resp.status}")
                    return []
                data = await resp.json()
                articles = data if isinstance(data, list) else data.get("articles", data.get("data", []))
                return [self._normalize(a) for a in articles]
        except Exception as e:
            logger.error(f"Archive fetch failed: {e}")
            return []

    async def stream(self):
        """Connect to the streaming endpoint for real-time news."""
        session = await self._get_session()
        self._running = True
        backoff = 1

        while self._running:
            try:
                async with session.get(
                    f"{BASE_URL}/api/stream",
                    timeout=aiohttp.ClientTimeout(total=None, sock_read=120),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Stream returned {resp.status}, falling back to polling")
                        await self._poll_loop()
                        return

                    backoff = 1
                    logger.info("Connected to crypto news stream")

                    async for line in resp.content:
                        if not self._running:
                            break
                        decoded = line.decode("utf-8").strip()
                        if not decoded:
                            continue
                        # Handle SSE format
                        if decoded.startswith("data:"):
                            decoded = decoded[5:].strip()
                        if not decoded or decoded == "[DONE]":
                            continue
                        try:
                            article = json.loads(decoded)
                            event = self._normalize(article)
                            url = event.get("url", "")
                            if url and url in self._seen_urls:
                                continue
                            if url:
                                self._seen_urls.add(url)
                            if self.callback:
                                await self.callback(event)
                        except json.JSONDecodeError:
                            logger.debug(f"Non-JSON stream data: {decoded[:100]}")

            except asyncio.TimeoutError:
                logger.info("Stream timeout, reconnecting...")
            except aiohttp.ClientError as e:
                logger.warning(f"Stream connection error: {e}")
            except Exception as e:
                logger.error(f"Stream error: {e}")

            if not self._running:
                break

            logger.info(f"Stream reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def start(self):
        """Start streaming (tries stream endpoint, falls back to polling)."""
        try:
            await self.stream()
        except Exception:
            logger.info("Streaming unavailable, falling back to polling")
            await self._poll_loop()

    async def stop(self):
        """Stop streaming/polling."""
        self._running = False

    async def _poll_loop(self):
        """Fallback: poll the REST endpoint at regular intervals."""
        self._running = True
        while self._running:
            events = await self.fetch_latest(limit=20)
            for event in events:
                url = event.get("url", "")
                if url and url in self._seen_urls:
                    continue
                if url:
                    self._seen_urls.add(url)
                if self.callback:
                    await self.callback(event)
            await asyncio.sleep(self.poll_interval)

    def _normalize(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an article into a standard event dict."""
        # Handle varying field names from the API
        title = article.get("title", "")
        content = article.get("content") or article.get("body") or article.get("description", "")
        url = article.get("url") or article.get("link", "")
        published = (
            article.get("published_at")
            or article.get("publishedAt")
            or article.get("created_at")
            or article.get("timestamp")
            or datetime.now(timezone.utc).isoformat()
        )
        source_name = (
            article.get("source", {}).get("name")
            if isinstance(article.get("source"), dict)
            else article.get("source", "cryptocurrency.cv")
        )

        # Extract entity mentions from title + content
        entities = article.get("entities", [])
        if not entities:
            entities = article.get("currencies", [])
        if not entities:
            entities = article.get("tags", [])

        return {
            "source": "crypto_news",
            "source_name": source_name,
            "timestamp": published,
            "title": title,
            "content": content[:2000] if content else "",
            "url": url,
            "entities": entities,
            "sentiment": article.get("sentiment"),
            "categories": article.get("categories", []),
        }


async def _print_callback(event: Dict[str, Any]):
    """Default callback for standalone testing."""
    ts = event.get("timestamp", "")[:19]
    title = event.get("title", "")[:80]
    sentiment = event.get("sentiment", "")
    entities = event.get("entities", [])[:3]
    src = event.get("source_name", "")
    print(f"[{ts}] [{src}] {title}")
    if sentiment:
        print(f"  sentiment: {sentiment}")
    if entities:
        print(f"  entities: {entities}")
    print()


async def main():
    """Standalone mode: fetch and stream crypto news."""
    logging.basicConfig(level=logging.INFO)

    connector = CryptoNewsConnector(callback=_print_callback)

    print("=== Latest News ===")
    articles = await connector.fetch_latest(limit=5)
    for a in articles:
        await _print_callback(a)

    print("\n=== Starting stream (Ctrl+C to stop) ===")
    try:
        await connector.start()
    except KeyboardInterrupt:
        await connector.stop()
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(main())
