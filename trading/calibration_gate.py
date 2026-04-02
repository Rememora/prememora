"""
Calibration gate — oracle-based pre-trade circuit breaker.

Runs the hindsight oracle on recently resolved markets, computes a Brier
score, and decides whether the system is calibrated well enough to trade.
If Brier > threshold (default 0.25 = random guessing), signals are logged
but trades are not executed.

Tracks calibration history in SQLite so we can see trends over time and
break down accuracy by market category.

Usage:
    gate = CalibrationGate()
    result = await gate.check()
    if result.can_trade:
        # execute trades
    else:
        # log-only mode

CLI:
    python -m trading.calibration_gate check          # run calibration check
    python -m trading.calibration_gate history         # show calibration history
    python -m trading.calibration_gate categories      # per-category breakdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("prememora.trading.calibration_gate")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.db"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class GateConfig:
    """Configuration for the calibration gate."""
    max_brier: float = 0.25            # maximum allowed Brier score (0.25 = random)
    min_markets: int = 3               # minimum resolved markets needed to judge
    max_markets: int = 20              # markets to test per calibration run
    min_volume: float = 100            # minimum market volume
    recalibrate_hours: int = 24        # re-run calibration at most every N hours
    use_graph: bool = False            # use graph context in oracle
    category_threshold: bool = True    # enable per-category gating
    db_path: Path | None = None


@dataclass
class CategoryCalibration:
    """Calibration data for a specific market category."""
    category: str
    brier_score: float | None
    market_count: int
    trade_count: int
    win_rate: float | None
    can_trade: bool


@dataclass
class GateResult:
    """Result of a calibration check."""
    can_trade: bool
    brier_score: float | None
    market_count: int
    reason: str
    categories: list[CategoryCalibration] = field(default_factory=list)
    checked_at: str = ""
    duration_s: float = 0.0

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    @property
    def summary(self) -> str:
        status = "TRADING ENABLED" if self.can_trade else "TRADING BLOCKED"
        lines = [
            f"Calibration Gate: {status}",
            f"  Brier score: {self.brier_score:.4f}" if self.brier_score is not None else "  Brier score: N/A",
            f"  Markets tested: {self.market_count}",
            f"  Reason: {self.reason}",
        ]
        if self.categories:
            lines.append("  Categories:")
            for c in sorted(self.categories, key=lambda x: x.brier_score or 999):
                status_icon = "OK" if c.can_trade else "BLOCKED"
                bs = f"{c.brier_score:.3f}" if c.brier_score is not None else "N/A"
                lines.append(
                    f"    [{status_icon:7s}] {c.category:20s} "
                    f"brier={bs} n={c.market_count}"
                )
        return "\n".join(lines)


# ── Database ──────────────────────────────────────────────────────────────────


def _get_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS calibration_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at  TEXT NOT NULL,
            brier_score REAL,
            market_count INTEGER NOT NULL,
            can_trade   INTEGER NOT NULL,
            reason      TEXT,
            duration_s  REAL,
            config_json TEXT
        );

        CREATE TABLE IF NOT EXISTS category_scores (
            run_id      INTEGER NOT NULL,
            category    TEXT NOT NULL,
            brier_score REAL,
            market_count INTEGER NOT NULL,
            trade_count INTEGER NOT NULL,
            win_rate    REAL,
            can_trade   INTEGER NOT NULL,
            PRIMARY KEY (run_id, category),
            FOREIGN KEY (run_id) REFERENCES calibration_runs(id)
        );
    """)
    conn.commit()


# ── Calibration Gate ──────────────────────────────────────────────────────────


class CalibrationGate:
    """Oracle-based pre-trade circuit breaker.

    Before the pipeline executes trades, call ``check()`` to verify
    prediction accuracy meets threshold. If the system's Brier score
    on recent resolved markets exceeds ``max_brier``, trading is blocked.
    """

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        self._conn = _get_db(self.config.db_path)

    def close(self):
        self._conn.close()

    def _last_check_age_hours(self) -> float | None:
        """Hours since the last calibration check, or None if never checked."""
        row = self._conn.execute(
            "SELECT checked_at FROM calibration_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        last = datetime.fromisoformat(row["checked_at"])
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() / 3600

    def get_last_result(self) -> GateResult | None:
        """Get the most recent calibration result without re-running."""
        row = self._conn.execute(
            "SELECT * FROM calibration_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None

        categories = []
        cat_rows = self._conn.execute(
            "SELECT * FROM category_scores WHERE run_id=?", (row["id"],)
        ).fetchall()
        for cr in cat_rows:
            categories.append(CategoryCalibration(
                category=cr["category"],
                brier_score=cr["brier_score"],
                market_count=cr["market_count"],
                trade_count=cr["trade_count"],
                win_rate=cr["win_rate"],
                can_trade=bool(cr["can_trade"]),
            ))

        return GateResult(
            can_trade=bool(row["can_trade"]),
            brier_score=row["brier_score"],
            market_count=row["market_count"],
            reason=row["reason"] or "",
            categories=categories,
            checked_at=row["checked_at"],
            duration_s=row["duration_s"] or 0,
        )

    async def check(self, force: bool = False) -> GateResult:
        """Run a calibration check. Returns GateResult with can_trade flag.

        Skips re-running if a recent check exists (within recalibrate_hours)
        unless force=True.
        """
        cfg = self.config

        # Check if we can reuse a recent result
        if not force:
            age = self._last_check_age_hours()
            if age is not None and age < cfg.recalibrate_hours:
                last = self.get_last_result()
                if last:
                    logger.info(
                        "Reusing calibration from %.1fh ago (threshold: %dh)",
                        age, cfg.recalibrate_hours,
                    )
                    return last

        start = time.monotonic()

        # Run the hindsight oracle
        from backtesting.hindsight import BacktestConfig, HindsightOracle
        from trading.edge_calculator import EdgeConfig

        backtest_cfg = BacktestConfig(
            max_markets=cfg.max_markets,
            min_volume=cfg.min_volume,
            use_graph=cfg.use_graph,
            initial_cash=10_000.0,
        )
        oracle = HindsightOracle(
            config=backtest_cfg,
            edge_config=EdgeConfig(min_edge=0.05),
        )

        report = await oracle.run()
        duration = time.monotonic() - start

        # Evaluate overall result
        brier = report.brier_score
        market_count = len([r for r in report.results if r.our_probability is not None])

        if market_count < cfg.min_markets:
            result = GateResult(
                can_trade=False,
                brier_score=brier,
                market_count=market_count,
                reason=f"insufficient data: {market_count} markets < {cfg.min_markets} minimum",
                duration_s=duration,
            )
        elif brier is None:
            result = GateResult(
                can_trade=False,
                brier_score=None,
                market_count=market_count,
                reason="no Brier score computed (no predictions with outcomes)",
                duration_s=duration,
            )
        elif brier <= cfg.max_brier:
            result = GateResult(
                can_trade=True,
                brier_score=brier,
                market_count=market_count,
                reason=f"brier {brier:.4f} <= {cfg.max_brier:.2f} threshold",
                duration_s=duration,
            )
        else:
            result = GateResult(
                can_trade=False,
                brier_score=brier,
                market_count=market_count,
                reason=f"brier {brier:.4f} > {cfg.max_brier:.2f} threshold — predictions worse than random",
                duration_s=duration,
            )

        # Category breakdown
        if cfg.category_threshold and report.results:
            result.categories = self._compute_category_scores(report, cfg.max_brier)

        # Persist
        self._save_result(result, cfg)

        return result

    def _compute_category_scores(
        self, report: Any, max_brier: float,
    ) -> list[CategoryCalibration]:
        """Compute per-category Brier scores from backtest results."""
        from trading.strategy_review import brier_score

        # Group by category
        by_cat: dict[str, list] = {}
        for r in report.results:
            cat = r.market_id[:8] if not hasattr(r, "category") else "unknown"
            # We don't have category on MarketBacktestResult, so we'll use
            # the oracle report's market data. For now, group all as one.
            cat = "all"
            by_cat.setdefault(cat, []).append(r)

        categories = []
        for cat, results in by_cat.items():
            pairs = []
            traded = 0
            wins = 0
            for r in results:
                if r.our_probability is not None and r.actual_outcome in ("YES", "NO"):
                    actual = r.actual_outcome == "YES"
                    pairs.append((r.our_probability, actual))
                if r.action not in ("SKIP", "ERROR"):
                    traded += 1
                    if r.pnl > 0:
                        wins += 1

            bs = brier_score(pairs)
            win_rate = wins / traded if traded > 0 else None
            can_trade = bs is not None and bs <= max_brier and len(pairs) >= 2

            categories.append(CategoryCalibration(
                category=cat,
                brier_score=bs,
                market_count=len(pairs),
                trade_count=traded,
                win_rate=win_rate,
                can_trade=can_trade,
            ))

        return categories

    def _save_result(self, result: GateResult, cfg: GateConfig) -> None:
        """Persist calibration result to SQLite."""
        cursor = self._conn.execute(
            """INSERT INTO calibration_runs
               (checked_at, brier_score, market_count, can_trade, reason, duration_s, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                result.checked_at,
                result.brier_score,
                result.market_count,
                1 if result.can_trade else 0,
                result.reason,
                result.duration_s,
                json.dumps({
                    "max_brier": cfg.max_brier,
                    "min_markets": cfg.min_markets,
                    "max_markets": cfg.max_markets,
                    "use_graph": cfg.use_graph,
                }),
            ),
        )
        run_id = cursor.lastrowid

        for cat in result.categories:
            self._conn.execute(
                """INSERT INTO category_scores
                   (run_id, category, brier_score, market_count, trade_count, win_rate, can_trade)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    cat.category,
                    cat.brier_score,
                    cat.market_count,
                    cat.trade_count,
                    cat.win_rate,
                    1 if cat.can_trade else 0,
                ),
            )

        self._conn.commit()

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get calibration run history."""
        rows = self._conn.execute(
            "SELECT * FROM calibration_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def is_category_allowed(self, category: str) -> bool:
        """Check if a specific category is allowed based on last calibration.

        Returns True if no category data exists (permissive default).
        """
        row = self._conn.execute(
            """SELECT cs.can_trade FROM category_scores cs
               JOIN calibration_runs cr ON cs.run_id = cr.id
               WHERE cs.category = ?
               ORDER BY cr.id DESC LIMIT 1""",
            (category,),
        ).fetchone()
        return bool(row["can_trade"]) if row else True


# ── CLI ──────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Calibration gate — oracle-based trade gating")
    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="Run calibration check")
    p_check.add_argument("--max-markets", type=int, default=20)
    p_check.add_argument("--max-brier", type=float, default=0.25)
    p_check.add_argument("--force", action="store_true", help="Force re-check even if recent")
    p_check.add_argument("--use-graph", action="store_true")

    sub.add_parser("history", help="Show calibration history")
    sub.add_parser("last", help="Show last calibration result")

    args = parser.parse_args()

    if args.command == "check":
        cfg = GateConfig(
            max_markets=args.max_markets,
            max_brier=args.max_brier,
            use_graph=args.use_graph,
        )
        gate = CalibrationGate(config=cfg)
        result = await gate.check(force=args.force)
        print(result.summary)
        gate.close()

    elif args.command == "history":
        gate = CalibrationGate()
        history = gate.get_history()
        if not history:
            print("No calibration history.")
        else:
            print(f"{'Time':25s} {'Brier':>8s} {'Markets':>8s} {'Trade?':>7s} Reason")
            for h in history:
                bs = f"{h['brier_score']:.4f}" if h["brier_score"] is not None else "N/A"
                trade = "YES" if h["can_trade"] else "NO"
                print(f"{h['checked_at'][:25]:25s} {bs:>8s} {h['market_count']:>8d} {trade:>7s} {h['reason']}")
        gate.close()

    elif args.command == "last":
        gate = CalibrationGate()
        result = gate.get_last_result()
        if result:
            print(result.summary)
        else:
            print("No calibration results yet. Run: python -m trading.calibration_gate check")
        gate.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
