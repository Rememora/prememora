"""Historical event collector for backtesting.

Converts Polymarket price history + market metadata into narrative event
dicts suitable for Graphiti graph ingestion.  Each event is a plain dict
with at minimum ``text``, ``timestamp``, and ``source`` keys.

Usage:
    from backtesting.polymarket_history import PricePoint
    events = collect_historical_events(
        market_question="Will BTC hit $100k?",
        market_description="Resolves YES if ...",
        prices=[PricePoint(...)],
        window_hours=24,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backtesting.polymarket_history import PricePoint

logger = logging.getLogger("prememora.backtesting.event_replay")


def _format_ts(ts: int) -> str:
    """Unix timestamp to human-readable UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _price_direction(delta: float) -> str:
    if delta > 0.01:
        return "rose"
    elif delta < -0.01:
        return "fell"
    return "held steady"


def collect_historical_events(
    market_question: str,
    market_description: str,
    prices: list[PricePoint],
    window_hours: int = 24,
) -> list[dict[str, Any]]:
    """Convert price history into chronologically ordered event dicts.

    Parameters
    ----------
    market_question : str
        The market question (e.g. "Will BTC hit $100k by June 2026?").
    market_description : str
        Longer description / resolution criteria.
    prices : list[PricePoint]
        Hourly (or other interval) price points, need not be sorted.
    window_hours : int
        Aggregate price movements into windows of this many hours.
        Produces one narrative event per window.

    Returns
    -------
    list[dict]
        Chronologically sorted event dicts with keys:
        ``text``, ``timestamp`` (int), ``source``, ``event_type``.
    """
    events: list[dict[str, Any]] = []

    # ── Seed event: market metadata ──────────────────────────────────
    if prices:
        seed_ts = min(p.timestamp for p in prices)
    else:
        seed_ts = int(datetime.now(timezone.utc).timestamp())

    seed_text = (
        f"New prediction market opened: \"{market_question}\" "
        f"Description: {market_description[:500]}"
    )
    events.append({
        "text": seed_text,
        "timestamp": seed_ts,
        "source": "polymarket_metadata",
        "event_type": "market_open",
    })

    if not prices:
        return events

    # ── Sort prices chronologically ──────────────────────────────────
    sorted_prices = sorted(prices, key=lambda p: p.timestamp)

    # ── Bucket into windows ──────────────────────────────────────────
    window_seconds = window_hours * 3600
    if window_seconds <= 0:
        window_seconds = 3600  # fallback to 1h

    buckets: list[list[PricePoint]] = []
    current_bucket: list[PricePoint] = []
    bucket_start = sorted_prices[0].timestamp

    for p in sorted_prices:
        if p.timestamp - bucket_start >= window_seconds and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [p]
            bucket_start = p.timestamp
        else:
            current_bucket.append(p)

    if current_bucket:
        buckets.append(current_bucket)

    # ── Generate narrative events from each window ───────────────────
    prev_close: float | None = None

    for bucket in buckets:
        open_price = bucket[0].price
        close_price = bucket[-1].price
        high_price = max(p.price for p in bucket)
        low_price = min(p.price for p in bucket)
        window_start = bucket[0].timestamp
        window_end = bucket[-1].timestamp

        # Delta from previous window (or from open if first window)
        ref_price = prev_close if prev_close is not None else open_price
        delta = close_price - ref_price
        direction = _price_direction(delta)

        # Short label for the question (first 80 chars)
        q_short = market_question[:80]

        text = (
            f"\"{q_short}\" market: YES price {direction} "
            f"from {ref_price:.2f} to {close_price:.2f} "
            f"(high {high_price:.2f}, low {low_price:.2f}) "
            f"over {_format_ts(window_start)} to {_format_ts(window_end)}."
        )

        events.append({
            "text": text,
            "timestamp": window_end,
            "source": "polymarket_price_history",
            "event_type": "price_movement",
            "open": open_price,
            "close": close_price,
            "high": high_price,
            "low": low_price,
        })

        prev_close = close_price

    # ── Sort all events chronologically ──────────────────────────────
    events.sort(key=lambda e: e["timestamp"])

    logger.info(
        "Generated %d events for '%s' (%d price points, %dh windows)",
        len(events), market_question[:60], len(prices), window_hours,
    )

    return events
