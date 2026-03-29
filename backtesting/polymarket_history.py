"""Pull and store historical Polymarket market data for backtesting.

Usage:
    python -m backtesting.polymarket_history discover [--category CAT] [--status STATUS] [--limit N]
    python -m backtesting.polymarket_history fetch --market <id> [--interval 1h] [--start-ts TS] [--end-ts TS]
    python -m backtesting.polymarket_history fetch-all [--category CAT] [--limit 50] [--interval 1h]
    python -m backtesting.polymarket_history stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

VALID_INTERVALS = {"1m", "1h", "1d"}

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "polymarket_history.db"

# Gamma API paginates with offset/limit (max 100 per page)
GAMMA_PAGE_SIZE = 100

# Rate-limit: pause between CLOB price-history requests
CLOB_REQUEST_DELAY = 0.25  # seconds


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Market:
    id: str
    question: str
    description: str
    category: str
    outcomes: str  # JSON array string, e.g. '["Yes","No"]'
    created_at: str
    resolved_at: str | None
    resolution: str | None
    volume: float
    clob_token_ids: str  # JSON array string of token IDs


@dataclass
class PricePoint:
    market_id: str
    token_id: str
    timestamp: int
    price: float
    interval: str


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the SQLite database."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS markets (
            id            TEXT PRIMARY KEY,
            question      TEXT NOT NULL,
            description   TEXT,
            category      TEXT,
            outcomes      TEXT,
            created_at    TEXT,
            resolved_at   TEXT,
            resolution    TEXT,
            volume        REAL DEFAULT 0,
            clob_token_ids TEXT
        );

        CREATE TABLE IF NOT EXISTS prices (
            market_id  TEXT NOT NULL,
            token_id   TEXT NOT NULL,
            timestamp  INTEGER NOT NULL,
            price      REAL NOT NULL,
            interval   TEXT NOT NULL,
            PRIMARY KEY (market_id, token_id, timestamp, interval),
            FOREIGN KEY (market_id) REFERENCES markets(id)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_market
            ON prices(market_id, interval, timestamp);

        CREATE INDEX IF NOT EXISTS idx_markets_category
            ON markets(category);
        """
    )
    conn.commit()


def upsert_market(conn: sqlite3.Connection, m: Market) -> None:
    conn.execute(
        """
        INSERT INTO markets (id, question, description, category, outcomes,
                             created_at, resolved_at, resolution, volume, clob_token_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            question=excluded.question,
            description=excluded.description,
            category=excluded.category,
            outcomes=excluded.outcomes,
            created_at=excluded.created_at,
            resolved_at=excluded.resolved_at,
            resolution=excluded.resolution,
            volume=excluded.volume,
            clob_token_ids=excluded.clob_token_ids
        """,
        (
            m.id,
            m.question,
            m.description,
            m.category,
            m.outcomes,
            m.created_at,
            m.resolved_at,
            m.resolution,
            m.volume,
            m.clob_token_ids,
        ),
    )


def insert_prices(conn: sqlite3.Connection, points: list[PricePoint]) -> int:
    """Insert price points, skipping duplicates. Returns count of new rows."""
    if not points:
        return 0
    inserted = 0
    for p in points:
        try:
            conn.execute(
                """
                INSERT INTO prices (market_id, token_id, timestamp, price, interval)
                VALUES (?, ?, ?, ?, ?)
                """,
                (p.market_id, p.token_id, p.timestamp, p.price, p.interval),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate — skip
    return inserted


def get_last_timestamp(
    conn: sqlite3.Connection, market_id: str, token_id: str, interval: str
) -> int | None:
    """Return the latest stored timestamp for incremental fetching."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM prices WHERE market_id=? AND token_id=? AND interval=?",
        (market_id, token_id, interval),
    ).fetchone()
    if row and row[0] is not None:
        return int(row[0])
    return None


def get_market_row(conn: sqlite3.Connection, market_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM markets WHERE id=?", (market_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Gamma API helpers (market discovery)
# ---------------------------------------------------------------------------


async def _gamma_get(
    session: aiohttp.ClientSession, path: str, params: dict[str, Any] | None = None
) -> Any:
    url = f"{GAMMA_API_BASE}{path}"
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


def _parse_market(raw: dict[str, Any]) -> Market:
    """Parse a Gamma API market object into our Market dataclass."""
    # Extract CLOB token IDs from the nested tokens array
    clob_token_ids: list[str] = []
    tokens = raw.get("clobTokenIds") or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except (json.JSONDecodeError, TypeError):
            tokens = [tokens] if tokens else []
    clob_token_ids = tokens if isinstance(tokens, list) else []

    # Outcomes
    outcomes_raw = raw.get("outcomes") or []
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except (json.JSONDecodeError, TypeError):
            outcomes_raw = [outcomes_raw]

    return Market(
        id=str(raw.get("id", raw.get("conditionId", ""))),
        question=raw.get("question", ""),
        description=raw.get("description", ""),
        category=raw.get("category", "") or "",
        outcomes=json.dumps(outcomes_raw),
        created_at=raw.get("createdAt") or raw.get("created_at") or "",
        resolved_at=raw.get("resolvedAt") or raw.get("resolved_at"),
        resolution=raw.get("resolution"),
        volume=float(raw.get("volume", 0) or 0),
        clob_token_ids=json.dumps(clob_token_ids),
    )


async def discover_markets(
    category: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[Market]:
    """Fetch market metadata from the Gamma API.

    Args:
        category: Filter by category string (e.g. "crypto", "politics").
        status: Filter by status ("active", "resolved", etc.).
        limit: Maximum number of markets to return.
    """
    markets: list[Market] = []
    offset = 0
    async with aiohttp.ClientSession() as session:
        while len(markets) < limit:
            page_size = min(GAMMA_PAGE_SIZE, limit - len(markets))
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
            }
            if category:
                params["tag"] = category
            if status:
                if status == "resolved":
                    params["closed"] = "true"
                elif status == "active":
                    params["active"] = "true"

            data = await _gamma_get(session, "/markets", params)
            if not data:
                break

            for raw in data:
                markets.append(_parse_market(raw))
            if len(data) < page_size:
                break
            offset += page_size

    return markets[:limit]


# ---------------------------------------------------------------------------
# CLOB API helpers (price history)
# ---------------------------------------------------------------------------


async def _clob_get(
    session: aiohttp.ClientSession, path: str, params: dict[str, Any] | None = None
) -> Any:
    url = f"{CLOB_API_BASE}{path}"
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_price_history(
    market_id: str,
    token_id: str,
    interval: str = "1h",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[PricePoint]:
    """Fetch price history for a single token from the CLOB API.

    Args:
        market_id: The market's condition ID (used for local storage).
        token_id: The CLOB token ID.
        interval: Granularity — "1m", "1h", or "1d".
        start_ts: Unix start timestamp (inclusive).
        end_ts: Unix end timestamp (inclusive). Defaults to now.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval {interval!r}, must be one of {VALID_INTERVALS}")

    params: dict[str, Any] = {
        "market": token_id,
        "interval": interval,
    }
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts

    points: list[PricePoint] = []
    async with aiohttp.ClientSession() as session:
        data = await _clob_get(session, "/prices-history", params)
        history = data.get("history") or []
        for entry in history:
            ts = int(entry.get("t", 0))
            price = float(entry.get("p", 0))
            points.append(
                PricePoint(
                    market_id=market_id,
                    token_id=token_id,
                    timestamp=ts,
                    price=price,
                    interval=interval,
                )
            )
    return points


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------


async def cmd_discover(
    category: str | None = None,
    status: str | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> int:
    """Discover markets and store metadata. Returns count of markets stored."""
    logger.info("Discovering markets (category=%s, status=%s, limit=%d)", category, status, limit)
    markets = await discover_markets(category=category, status=status, limit=limit)
    conn = _get_db(db_path)
    try:
        for m in markets:
            upsert_market(conn, m)
        conn.commit()
    finally:
        conn.close()
    logger.info("Stored %d markets", len(markets))
    return len(markets)


async def cmd_fetch(
    market_id: str,
    interval: str = "1h",
    start_ts: int | None = None,
    end_ts: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Fetch price history for a single market (all its tokens).

    Supports incremental updates — only fetches data newer than what is stored.
    Returns total number of new price points inserted.
    """
    conn = _get_db(db_path)
    try:
        row = get_market_row(conn, market_id)
        if not row:
            logger.error("Market %s not found in DB. Run 'discover' first.", market_id)
            return 0

        token_ids: list[str] = json.loads(row["clob_token_ids"] or "[]")
        if not token_ids:
            logger.warning("Market %s has no CLOB token IDs.", market_id)
            return 0

        total_inserted = 0
        for token_id in token_ids:
            # Incremental: use last stored timestamp as start if not overridden
            effective_start = start_ts
            if effective_start is None:
                last_ts = get_last_timestamp(conn, market_id, token_id, interval)
                if last_ts is not None:
                    effective_start = last_ts + 1
                    logger.info(
                        "Incremental fetch for token %s from ts=%d", token_id, effective_start
                    )

            points = await fetch_price_history(
                market_id=market_id,
                token_id=token_id,
                interval=interval,
                start_ts=effective_start,
                end_ts=end_ts,
            )
            inserted = insert_prices(conn, points)
            total_inserted += inserted
            logger.info(
                "Token %s: fetched %d points, inserted %d new",
                token_id,
                len(points),
                inserted,
            )
            await asyncio.sleep(CLOB_REQUEST_DELAY)

        conn.commit()
    finally:
        conn.close()

    return total_inserted


async def cmd_fetch_all(
    category: str | None = None,
    limit: int = 50,
    interval: str = "1h",
    db_path: Path | None = None,
) -> dict[str, int]:
    """Fetch price history for multiple markets matching criteria.

    Returns dict of {market_id: new_points_inserted}.
    """
    conn = _get_db(db_path)
    try:
        query = "SELECT id FROM markets WHERE 1=1"
        params: list[Any] = []
        if category:
            query += " AND LOWER(category) = LOWER(?)"
            params.append(category)
        query += " ORDER BY volume DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    results: dict[str, int] = {}
    for i, row in enumerate(rows):
        mid = row["id"]
        logger.info("Fetching market %d/%d: %s", i + 1, len(rows), mid)
        n = await cmd_fetch(market_id=mid, interval=interval, db_path=db_path)
        results[mid] = n

    return results


def cmd_stats(db_path: Path | None = None) -> dict[str, Any]:
    """Print summary statistics of stored data."""
    conn = _get_db(db_path)
    try:
        total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
        resolved_markets = conn.execute(
            "SELECT COUNT(*) FROM markets WHERE resolved_at IS NOT NULL"
        ).fetchone()[0]
        total_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        markets_with_prices = conn.execute(
            "SELECT COUNT(DISTINCT market_id) FROM prices"
        ).fetchone()[0]

        # Category breakdown
        cat_rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM markets GROUP BY category ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        categories = {r["category"] or "(none)": r["cnt"] for r in cat_rows}

        # Interval breakdown
        int_rows = conn.execute(
            "SELECT interval, COUNT(*) as cnt FROM prices GROUP BY interval ORDER BY cnt DESC"
        ).fetchall()
        intervals = {r["interval"]: r["cnt"] for r in int_rows}

        # Date range of prices
        ts_row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM prices"
        ).fetchone()
        min_ts = ts_row[0]
        max_ts = ts_row[1]
    finally:
        conn.close()

    stats = {
        "total_markets": total_markets,
        "resolved_markets": resolved_markets,
        "total_price_points": total_prices,
        "markets_with_prices": markets_with_prices,
        "categories": categories,
        "price_intervals": intervals,
        "price_range_start": min_ts,
        "price_range_end": max_ts,
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polymarket_history",
        description="Pull and store historical Polymarket market data for backtesting.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite database (default: data/polymarket_history.db)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # discover
    p_discover = sub.add_parser("discover", help="Find and store market metadata")
    p_discover.add_argument("--category", type=str, default=None)
    p_discover.add_argument(
        "--status", type=str, default=None, choices=["active", "resolved"]
    )
    p_discover.add_argument("--limit", type=int, default=100)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch price history for a single market")
    p_fetch.add_argument("--market", type=str, required=True)
    p_fetch.add_argument("--interval", type=str, default="1h", choices=sorted(VALID_INTERVALS))
    p_fetch.add_argument("--start-ts", type=int, default=None)
    p_fetch.add_argument("--end-ts", type=int, default=None)

    # fetch-all
    p_fetch_all = sub.add_parser("fetch-all", help="Batch fetch for multiple markets")
    p_fetch_all.add_argument("--category", type=str, default=None)
    p_fetch_all.add_argument("--limit", type=int, default=50)
    p_fetch_all.add_argument("--interval", type=str, default="1h", choices=sorted(VALID_INTERVALS))

    # stats
    sub.add_parser("stats", help="Show summary of stored data")

    return parser


def _format_stats(stats: dict[str, Any]) -> str:
    lines = [
        "=== Polymarket History Database ===",
        f"Markets stored:       {stats['total_markets']}",
        f"  Resolved:           {stats['resolved_markets']}",
        f"  With price data:    {stats['markets_with_prices']}",
        f"Total price points:   {stats['total_price_points']}",
    ]
    if stats["price_range_start"]:
        from datetime import datetime, timezone

        start = datetime.fromtimestamp(stats["price_range_start"], tz=timezone.utc)
        end = datetime.fromtimestamp(stats["price_range_end"], tz=timezone.utc)
        lines.append(f"Price data range:     {start:%Y-%m-%d %H:%M} - {end:%Y-%m-%d %H:%M} UTC")

    if stats["price_intervals"]:
        lines.append("Price intervals:")
        for iv, cnt in stats["price_intervals"].items():
            lines.append(f"  {iv:>4s}: {cnt:>10,} points")

    if stats["categories"]:
        lines.append("Categories:")
        for cat, cnt in stats["categories"].items():
            lines.append(f"  {cat:>20s}: {cnt}")

    return "\n".join(lines)


async def async_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = args.db

    if args.command == "discover":
        n = await cmd_discover(
            category=args.category,
            status=args.status,
            limit=args.limit,
            db_path=db_path,
        )
        print(f"Discovered and stored {n} markets.")

    elif args.command == "fetch":
        n = await cmd_fetch(
            market_id=args.market,
            interval=args.interval,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            db_path=db_path,
        )
        print(f"Inserted {n} new price points.")

    elif args.command == "fetch-all":
        results = await cmd_fetch_all(
            category=args.category,
            limit=args.limit,
            interval=args.interval,
            db_path=db_path,
        )
        total = sum(results.values())
        print(f"Fetched {len(results)} markets, {total} new price points total.")

    elif args.command == "stats":
        stats = cmd_stats(db_path=db_path)
        print(_format_stats(stats))


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
