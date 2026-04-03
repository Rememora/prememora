"""
Data collector — polls Polymarket for market prices and resolutions.
Stores everything in SQLite so the hindsight oracle can replay with
real historical prices instead of synthetic ones.

Runs two jobs on a loop:
  1. Snapshot active market prices (Yes/No binary markets)
  2. Detect newly resolved markets and record outcomes

The ingestion orchestrator (separate process) fills the knowledge graph
with world events. This module handles market-side data only.

Usage:
    python -m e2e.data_collector start [--interval 3600] [--max-markets 50]
    python -m e2e.data_collector stats
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger("prememora.e2e.data_collector")

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "collector"
DB_PATH = DATA_DIR / "market_history.db"


# ── Database ──────────────────────────────────────────────────────────────────


def _get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            market_id   TEXT NOT NULL,
            question    TEXT NOT NULL,
            yes_price   REAL NOT NULL,
            volume      REAL,
            end_date    TEXT,
            snapshot_at TEXT NOT NULL,
            PRIMARY KEY (market_id, snapshot_at)
        );

        CREATE TABLE IF NOT EXISTS resolutions (
            market_id       TEXT PRIMARY KEY,
            question        TEXT NOT NULL,
            outcome         TEXT NOT NULL,
            resolved_at     TEXT NOT NULL,
            volume          REAL,
            last_yes_price  REAL,
            snapshot_count  INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_market
            ON market_snapshots(market_id, snapshot_at);
    """)
    conn.commit()
    return conn


# ── Market snapshots ─────────────────────────────────────────────────────────


async def snapshot_markets(
    session: aiohttp.ClientSession,
    conn: sqlite3.Connection,
    max_markets: int = 50,
) -> int:
    """Fetch active Yes/No market prices and store as snapshots."""
    params: dict[str, Any] = {
        "limit": max_markets,
        "active": "true",
        "closed": "false",
        "order": "volume",
        "ascending": "false",
    }

    try:
        async with session.get(f"{GAMMA_API_BASE}/markets", params=params) as resp:
            if resp.status != 200:
                logger.warning("Gamma API returned %d", resp.status)
                return 0
            data = await resp.json()
    except Exception as e:
        logger.error("Failed to fetch markets: %s", e)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for m in data:
        # Only Yes/No binary markets
        outcomes = m.get("outcomes") or ""
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                continue
        if set(o.lower() for o in outcomes) != {"yes", "no"}:
            continue

        tokens = m.get("tokens") or []
        yes_price = 0.5
        for t in tokens:
            if (t.get("outcome") or "").upper() == "YES":
                yes_price = float(t.get("price") or 0.5)
                break

        market_id = m.get("condition_id") or m.get("conditionId") or m.get("id", "")

        try:
            conn.execute(
                """INSERT OR IGNORE INTO market_snapshots
                   (market_id, question, yes_price, volume, end_date, snapshot_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    market_id,
                    m.get("question", ""),
                    yes_price,
                    float(m.get("volume") or 0),
                    m.get("endDateIso") or "",
                    now,
                ),
            )
            count += 1
        except Exception as e:
            logger.debug("Snapshot insert failed: %s", e)

    conn.commit()
    logger.info("Snapshotted %d markets", count)
    return count


# ── Resolution tracking ──────────────────────────────────────────────────────


async def check_resolutions(
    session: aiohttp.ClientSession,
    conn: sqlite3.Connection,
) -> int:
    """Check tracked markets for new resolutions."""
    rows = conn.execute("""
        SELECT DISTINCT ms.market_id, ms.question
        FROM market_snapshots ms
        LEFT JOIN resolutions r ON ms.market_id = r.market_id
        WHERE r.market_id IS NULL
    """).fetchall()

    if not rows:
        return 0

    # Batch check via closed markets endpoint
    params: dict[str, Any] = {
        "limit": 100,
        "closed": "true",
        "order": "volume",
        "ascending": "false",
    }

    try:
        async with session.get(f"{GAMMA_API_BASE}/markets", params=params) as resp:
            if resp.status != 200:
                return 0
            closed_markets = await resp.json()
    except Exception as e:
        logger.error("Failed to fetch closed markets: %s", e)
        return 0

    # Build lookup of tracked market IDs
    tracked = {r["market_id"] for r in rows}
    question_map = {r["market_id"]: r["question"] for r in rows}

    count = 0
    for m in closed_markets:
        mid = m.get("condition_id") or m.get("conditionId") or m.get("id", "")
        if mid not in tracked:
            continue

        resolved_by = m.get("resolvedBy") or ""
        if not resolved_by:
            continue

        outcome_prices = m.get("outcomePrices") or ""
        outcomes_raw = m.get("outcomes") or ""

        try:
            prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
            labels = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        except (json.JSONDecodeError, TypeError):
            continue

        if len(prices) < 2 or len(labels) < 2:
            continue

        winner_idx = None
        for i, p in enumerate(prices):
            if str(p) == "1":
                winner_idx = i
                break
        if winner_idx is None:
            continue

        winner = labels[winner_idx].strip().upper()
        if winner in ("YES", "Y"):
            outcome = "YES"
        elif winner in ("NO", "N"):
            outcome = "NO"
        else:
            continue

        # Get the last snapshot price and count for this market
        last_snap = conn.execute(
            """SELECT yes_price, COUNT(*) as cnt FROM market_snapshots
               WHERE market_id = ? ORDER BY snapshot_at DESC LIMIT 1""",
            (mid,),
        ).fetchone()
        last_price = last_snap["yes_price"] if last_snap else None
        snap_count = last_snap["cnt"] if last_snap else 0

        conn.execute(
            """INSERT OR IGNORE INTO resolutions
               (market_id, question, outcome, resolved_at, volume, last_yes_price, snapshot_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                question_map.get(mid, ""),
                outcome,
                m.get("closedTime") or datetime.now(timezone.utc).isoformat(),
                float(m.get("volume") or 0),
                last_price,
                snap_count,
            ),
        )
        count += 1
        logger.info("Resolved: %s → %s (%s)", mid[:20], outcome, question_map.get(mid, "?")[:40])

    conn.commit()
    return count


# ── Main loop ────────────────────────────────────────────────────────────────


async def run_collector(interval: int = 3600, max_markets: int = 50) -> None:
    """Run the data collector loop."""
    conn = _get_db()
    cycle = 0
    stop = False

    def handle_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Data collector starting: interval=%ds, max_markets=%d", interval, max_markets)

    while not stop:
        cycle += 1
        logger.info("--- Cycle %d ---", cycle)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            snaps = await snapshot_markets(session, conn, max_markets)
            resolved = await check_resolutions(session, conn)

        if resolved:
            logger.info("Cycle %d: %d snapshots, %d new resolutions", cycle, snaps, resolved)

        if not stop:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    conn.close()
    logger.info("Collector stopped after %d cycles", cycle)


def show_stats() -> None:
    """Show collection statistics."""
    if not DB_PATH.exists():
        print("No data yet. Run: python -m e2e.data_collector start")
        return

    conn = _get_db()

    snapshots = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    unique = conn.execute("SELECT COUNT(DISTINCT market_id) FROM market_snapshots").fetchone()[0]
    resolutions = conn.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]

    ts_row = conn.execute(
        "SELECT MIN(snapshot_at), MAX(snapshot_at) FROM market_snapshots"
    ).fetchone()

    recent_res = conn.execute(
        "SELECT question, outcome FROM resolutions ORDER BY resolved_at DESC LIMIT 5"
    ).fetchall()

    conn.close()

    print(f"Snapshots:       {snapshots:,}")
    print(f"Unique markets:  {unique:,}")
    print(f"Resolutions:     {resolutions:,}")
    if ts_row[0]:
        print(f"Date range:      {ts_row[0][:19]} → {ts_row[1][:19]}")
    if recent_res:
        print(f"\nRecent resolutions:")
        for r in recent_res:
            print(f"  {r['outcome']:3s} {r['question'][:55]}")


# ── CLI ──────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Market data collector")
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="Start collecting data")
    p_start.add_argument("--interval", type=int, default=3600, help="Seconds between cycles (default: 1h)")
    p_start.add_argument("--max-markets", type=int, default=50)

    sub.add_parser("stats", help="Show collection statistics")

    args = parser.parse_args()

    if args.command == "start":
        await run_collector(interval=args.interval, max_markets=args.max_markets)
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
