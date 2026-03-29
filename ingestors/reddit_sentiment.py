"""Reddit + social sentiment connector for PreMemora.

Monitors configurable subreddits, streams new and hot posts, tracks volume
spikes, and normalizes everything into event dicts ready for downstream
ingestion into the Graphiti knowledge graph.

Uses PRAW (synchronous) wrapped in asyncio.to_thread for async compatibility.

Env vars required:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT  (defaults to "prememora:reddit-sentiment:v0.1")
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import praw
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SUBREDDITS: list[str] = [
    "Bitcoin",
    "ethereum",
    "CryptoCurrency",
    "politics",
    "news",
    "wallstreetbets",
]

DEFAULT_USER_AGENT = "prememora:reddit-sentiment:v0.1"

# Volume-spike detection: a subreddit's current-window post count must exceed
# the rolling average by this factor to be flagged.
SPIKE_THRESHOLD_FACTOR = 2.0

# Rolling window sizes (number of windows kept for the average).
ROLLING_WINDOW_SIZE = 10

# How many "hot" posts to fetch per poll cycle.
HOT_LIMIT = 25

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EventDict = dict[str, Any]
AsyncCallback = Callable[[list[EventDict]], Coroutine[Any, Any, None]]


@dataclass
class SubredditStats:
    """Tracks per-subreddit post volume for spike detection."""

    # Recent window counts (most recent at the right).
    window_counts: deque[int] = field(
        default_factory=lambda: deque(maxlen=ROLLING_WINDOW_SIZE)
    )
    current_window_count: int = 0

    @property
    def rolling_average(self) -> float:
        if not self.window_counts:
            return 0.0
        return sum(self.window_counts) / len(self.window_counts)

    def close_window(self) -> tuple[int, float, bool]:
        """Close the current window, push count, return (count, avg, is_spike)."""
        count = self.current_window_count
        avg = self.rolling_average
        is_spike = avg > 0 and count > avg * SPIKE_THRESHOLD_FACTOR
        self.window_counts.append(count)
        self.current_window_count = 0
        return count, avg, is_spike

    def increment(self) -> None:
        self.current_window_count += 1


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


def _normalize_submission(submission: praw.models.Submission, subreddit_name: str) -> EventDict:
    """Turn a PRAW Submission into a flat event dict."""
    created_utc = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
    return {
        "source": "reddit",
        "timestamp": created_utc.isoformat(),
        "subreddit": subreddit_name,
        "title": submission.title,
        "content": (submission.selftext or "")[:2000],  # cap large self-posts
        "score": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "url": f"https://reddit.com{submission.permalink}",
        "flair": submission.link_flair_text,
        "post_id": submission.id,
    }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class RedditSentimentConnector:
    """Async connector that polls Reddit for new/hot posts and detects volume spikes.

    Usage::

        connector = RedditSentimentConnector(
            subreddits=["Bitcoin", "ethereum"],
            callback=my_async_handler,
        )
        await connector.start(poll_interval=120)
    """

    def __init__(
        self,
        *,
        subreddits: list[str] | None = None,
        callback: AsyncCallback | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        hot_limit: int = HOT_LIMIT,
    ) -> None:
        self.subreddits = subreddits or DEFAULT_SUBREDDITS
        self.callback = callback
        self.hot_limit = hot_limit

        self._client_id = client_id or os.environ["REDDIT_CLIENT_ID"]
        self._client_secret = client_secret or os.environ["REDDIT_CLIENT_SECRET"]
        self._user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT", DEFAULT_USER_AGENT)

        self._reddit: praw.Reddit | None = None
        self._seen_ids: set[str] = set()
        self._stats: dict[str, SubredditStats] = {
            name: SubredditStats() for name in self.subreddits
        }
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    def _ensure_reddit(self) -> praw.Reddit:
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            # Read-only mode (no user login needed).
            self._reddit.read_only = True
        return self._reddit

    async def start(self, poll_interval: float = 120) -> None:
        """Run the poll loop.  Blocks until ``stop()`` is called or cancelled."""
        self._running = True
        logger.info(
            "Starting Reddit sentiment connector — subreddits=%s, interval=%ss",
            self.subreddits,
            poll_interval,
        )
        while self._running:
            try:
                events = await self.poll()
                if events and self.callback:
                    await self.callback(events)
            except Exception:
                logger.exception("Error during Reddit poll cycle")
            await asyncio.sleep(poll_interval)

    def stop(self) -> None:
        """Signal the poll loop to exit after the current cycle."""
        self._running = False
        logger.info("Reddit sentiment connector stopping")

    # -- polling -------------------------------------------------------------

    async def poll(self) -> list[EventDict]:
        """Run one poll cycle across all subreddits.  Returns new events."""
        all_events: list[EventDict] = []

        for sub_name in self.subreddits:
            try:
                events = await self._poll_subreddit(sub_name)
                all_events.extend(events)
            except Exception:
                logger.exception("Failed to poll r/%s", sub_name)

        if all_events:
            logger.info("Poll cycle produced %d new events", len(all_events))

        return all_events

    async def _poll_subreddit(self, sub_name: str) -> list[EventDict]:
        """Fetch hot + new posts for a single subreddit (in a thread)."""
        reddit = self._ensure_reddit()

        def _fetch() -> list[praw.models.Submission]:
            subreddit = reddit.subreddit(sub_name)
            posts: list[praw.models.Submission] = []
            # Fetch hot posts.
            for submission in subreddit.hot(limit=self.hot_limit):
                posts.append(submission)
            # Fetch newest posts (may overlap with hot).
            for submission in subreddit.new(limit=self.hot_limit):
                posts.append(submission)
            return posts

        submissions = await asyncio.to_thread(_fetch)

        events: list[EventDict] = []
        stats = self._stats[sub_name]

        for submission in submissions:
            if submission.id in self._seen_ids:
                continue
            self._seen_ids.add(submission.id)
            stats.increment()
            events.append(_normalize_submission(submission, sub_name))

        return events

    # -- volume spike detection ----------------------------------------------

    def close_volume_windows(self) -> list[dict[str, Any]]:
        """Close all per-subreddit volume windows and return spike reports.

        Call this periodically (e.g. every N poll cycles) to track trends.
        Returns a list of dicts for subreddits where a spike was detected.
        """
        spikes: list[dict[str, Any]] = []
        for sub_name, stats in self._stats.items():
            count, avg, is_spike = stats.close_window()
            logger.debug(
                "r/%s volume window: count=%d avg=%.1f spike=%s",
                sub_name,
                count,
                avg,
                is_spike,
            )
            if is_spike:
                spike_info = {
                    "source": "reddit_volume_spike",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "subreddit": sub_name,
                    "window_count": count,
                    "rolling_average": round(avg, 2),
                    "spike_factor": round(count / avg, 2) if avg else 0,
                }
                spikes.append(spike_info)
                logger.warning(
                    "Volume spike in r/%s: %d posts vs %.1f avg (%.1fx)",
                    sub_name,
                    count,
                    avg,
                    spike_info["spike_factor"],
                )
        return spikes

    # -- convenience ---------------------------------------------------------

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of per-subreddit volume stats."""
        return {
            name: {
                "current_window_count": stats.current_window_count,
                "rolling_average": round(stats.rolling_average, 2),
                "windows_tracked": len(stats.window_counts),
            }
            for name, stats in self._stats.items()
        }

    @property
    def seen_count(self) -> int:
        return len(self._seen_ids)


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------

async def _main() -> None:
    """Run a single poll cycle and print the events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    collected: list[EventDict] = []

    async def _print_callback(events: list[EventDict]) -> None:
        for e in events:
            logger.info(
                "r/%s | score=%s ratio=%s comments=%s | %s",
                e["subreddit"],
                e["score"],
                e["upvote_ratio"],
                e["num_comments"],
                e["title"][:80],
            )
        collected.extend(events)

    connector = RedditSentimentConnector(callback=_print_callback)

    logger.info("Running single poll cycle...")
    start = time.monotonic()
    events = await connector.poll()
    elapsed = time.monotonic() - start

    if events and connector.callback:
        await connector.callback(events)

    # Close volume windows so we can see stats.
    spikes = connector.close_volume_windows()
    stats = connector.get_stats()

    logger.info("--- Results ---")
    logger.info("Fetched %d events in %.1fs", len(collected), elapsed)
    logger.info("Stats: %s", stats)
    if spikes:
        logger.info("Spikes detected: %s", spikes)
    else:
        logger.info("No volume spikes (first window, need history)")


if __name__ == "__main__":
    asyncio.run(_main())
