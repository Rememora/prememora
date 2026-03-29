"""
Paper trading engine for Polymarket prediction markets.

Tracks a virtual portfolio of binary-option positions, simulates fills
at midpoint price, calculates fees, and records P&L on market resolution.

All state is persisted in SQLite so the engine survives restarts.

Usage:
    engine = PaperTradingEngine()
    engine.open_position("market_abc", "YES", shares=100, price=0.60, reason="MiroFish 80% confidence")
    engine.close_position("market_abc", price=0.75, reason="taking profit")
    engine.resolve_market("market_abc", outcome="YES")

CLI:
    python -m trading.paper_engine status
    python -m trading.paper_engine trade <market_id> <side> <shares> <price> [--reason TEXT]
    python -m trading.paper_engine close <market_id> <price> [--reason TEXT]
    python -m trading.paper_engine resolve <market_id> <outcome>
    python -m trading.paper_engine history [--market MARKET_ID]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("prememora.trading.paper_engine")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_trading.db"

# Default fee in basis points (Polymarket charges ~2bps on the cheaper side)
DEFAULT_FEE_BPS = 2


# ── Enums & Data Classes ─────────────────────────────────────────────────────


class Side(str, Enum):
    YES = "YES"
    NO = "NO"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


@dataclass
class Position:
    id: str
    market_id: str
    side: str
    shares: float
    entry_price: float
    current_price: float
    status: str
    confidence: float | None
    entry_reason: str
    exit_reason: str
    entry_time: str
    exit_time: str | None
    pnl: float

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.status != PositionStatus.OPEN.value:
            return 0.0
        return self.market_value - self.cost_basis


@dataclass
class Trade:
    id: str
    position_id: str
    market_id: str
    side: str
    shares: float
    price: float
    fee: float
    trade_type: str  # "OPEN", "CLOSE", "RESOLVE"
    reason: str
    timestamp: str


@dataclass
class PortfolioSummary:
    cash: float
    positions: list[Position]
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_trades: int


# ── Fee Calculation ───────────────────────────────────────────────────────────


def calculate_fee(price: float, shares: float, fee_bps: int = DEFAULT_FEE_BPS) -> float:
    """Polymarket fee: bps/10000 * min(price, 1-price) * shares.

    The fee is proportional to the cheaper side of the binary option.
    A 50/50 market has the highest fee rate; lopsided markets pay less.
    """
    return (fee_bps / 10_000) * min(price, 1.0 - price) * shares


# ── Database ──────────────────────────────────────────────────────────────────


def _get_db(db_path: Path | None = None) -> sqlite3.Connection:
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
        CREATE TABLE IF NOT EXISTS portfolio (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            id            TEXT PRIMARY KEY,
            market_id     TEXT NOT NULL,
            side          TEXT NOT NULL,
            shares        REAL NOT NULL,
            entry_price   REAL NOT NULL,
            current_price REAL NOT NULL,
            status        TEXT NOT NULL DEFAULT 'OPEN',
            confidence    REAL,
            entry_reason  TEXT DEFAULT '',
            exit_reason   TEXT DEFAULT '',
            entry_time    TEXT NOT NULL,
            exit_time     TEXT,
            pnl           REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS trades (
            id           TEXT PRIMARY KEY,
            position_id  TEXT NOT NULL,
            market_id    TEXT NOT NULL,
            side         TEXT NOT NULL,
            shares       REAL NOT NULL,
            price        REAL NOT NULL,
            fee          REAL NOT NULL,
            trade_type   TEXT NOT NULL,
            reason       TEXT DEFAULT '',
            timestamp    TEXT NOT NULL,
            FOREIGN KEY (position_id) REFERENCES positions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_positions_market
            ON positions(market_id, status);
        CREATE INDEX IF NOT EXISTS idx_trades_position
            ON trades(position_id);
        CREATE INDEX IF NOT EXISTS idx_trades_market
            ON trades(market_id);
        """
    )
    # Initialize cash if not set
    existing = conn.execute("SELECT value FROM portfolio WHERE key='cash'").fetchone()
    if existing is None:
        conn.execute("INSERT INTO portfolio (key, value) VALUES ('cash', '1000.0')")
    conn.commit()


# ── Engine ────────────────────────────────────────────────────────────────────


class PaperTradingEngine:
    """Paper trading engine for binary prediction markets.

    Parameters
    ----------
    db_path : Path | None
        SQLite database path. Defaults to data/paper_trading.db.
    initial_cash : float
        Starting cash balance (only used on first init).
    fee_bps : int
        Fee in basis points.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        initial_cash: float = 1000.0,
        fee_bps: int = DEFAULT_FEE_BPS,
    ):
        self.fee_bps = fee_bps
        self._conn = _get_db(db_path)
        # Set initial cash only if DB was just created
        row = self._conn.execute("SELECT value FROM portfolio WHERE key='cash'").fetchone()
        if row and float(row["value"]) == 1000.0:
            self._set_cash(initial_cash)

    def _get_cash(self) -> float:
        row = self._conn.execute("SELECT value FROM portfolio WHERE key='cash'").fetchone()
        return float(row["value"])

    def _set_cash(self, amount: float) -> None:
        self._conn.execute(
            "UPDATE portfolio SET value=? WHERE key='cash'",
            (str(round(amount, 6)),),
        )
        self._conn.commit()

    # ── Position lifecycle ────────────────────────────────────────────

    def open_position(
        self,
        market_id: str,
        side: str,
        shares: float,
        price: float,
        reason: str = "",
        confidence: float | None = None,
    ) -> Position:
        """Open a new position. Deducts cost + fee from cash."""
        side = side.upper()
        if side not in ("YES", "NO"):
            raise ValueError(f"side must be YES or NO, got {side}")
        if not (0 < price < 1):
            raise ValueError(f"price must be between 0 and 1, got {price}")
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")

        # Check for existing open position on same market+side
        existing = self._conn.execute(
            "SELECT id FROM positions WHERE market_id=? AND side=? AND status='OPEN'",
            (market_id, side),
        ).fetchone()
        if existing:
            raise ValueError(
                f"Already have an open {side} position on {market_id} (id={existing['id']}). "
                "Close it first or use a different side."
            )

        cost = shares * price
        fee = calculate_fee(price, shares, self.fee_bps)
        total_cost = cost + fee
        cash = self._get_cash()

        if total_cost > cash:
            raise ValueError(
                f"Insufficient cash: need ${total_cost:.2f} (cost ${cost:.2f} + fee ${fee:.4f}), "
                f"have ${cash:.2f}"
            )

        now = datetime.now(timezone.utc).isoformat()
        pos_id = f"pos_{uuid.uuid4().hex[:12]}"
        trade_id = f"trd_{uuid.uuid4().hex[:12]}"

        self._conn.execute(
            """INSERT INTO positions
               (id, market_id, side, shares, entry_price, current_price,
                status, confidence, entry_reason, entry_time)
               VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)""",
            (pos_id, market_id, side, shares, price, price, confidence, reason, now),
        )
        self._conn.execute(
            """INSERT INTO trades
               (id, position_id, market_id, side, shares, price, fee,
                trade_type, reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)""",
            (trade_id, pos_id, market_id, side, shares, price, fee, reason, now),
        )
        self._set_cash(cash - total_cost)

        logger.info(
            "Opened %s %s x%.0f @%.4f on %s (fee=$%.4f, cost=$%.2f)",
            side, market_id, shares, price, pos_id, fee, total_cost,
        )

        return self._get_position(pos_id)

    def close_position(
        self,
        market_id: str,
        price: float,
        reason: str = "",
        side: str | None = None,
    ) -> Position:
        """Close an open position at the given price. Credits proceeds - fee to cash."""
        query = "SELECT * FROM positions WHERE market_id=? AND status='OPEN'"
        params: list[Any] = [market_id]
        if side:
            query += " AND side=?"
            params.append(side.upper())
        row = self._conn.execute(query, params).fetchone()
        if not row:
            raise ValueError(f"No open position on {market_id}" + (f" side={side}" if side else ""))

        pos_id = row["id"]
        shares = row["shares"]
        entry_price = row["entry_price"]

        proceeds = shares * price
        fee = calculate_fee(price, shares, self.fee_bps)
        net_proceeds = proceeds - fee
        pnl = net_proceeds - (shares * entry_price)

        now = datetime.now(timezone.utc).isoformat()
        trade_id = f"trd_{uuid.uuid4().hex[:12]}"

        self._conn.execute(
            """UPDATE positions SET status='CLOSED', current_price=?,
               exit_reason=?, exit_time=?, pnl=? WHERE id=?""",
            (price, reason, now, pnl, pos_id),
        )
        self._conn.execute(
            """INSERT INTO trades
               (id, position_id, market_id, side, shares, price, fee,
                trade_type, reason, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'CLOSE', ?, ?)""",
            (trade_id, pos_id, market_id, row["side"], shares, price, fee, reason, now),
        )
        self._set_cash(self._get_cash() + net_proceeds)

        logger.info(
            "Closed %s on %s @%.4f (pnl=$%.4f, fee=$%.4f)",
            pos_id, market_id, price, pnl, fee,
        )

        return self._get_position(pos_id)

    def resolve_market(self, market_id: str, outcome: str) -> list[Position]:
        """Resolve all open positions on a market. Winners get $1/share, losers $0."""
        outcome = outcome.upper()
        if outcome not in ("YES", "NO"):
            raise ValueError(f"outcome must be YES or NO, got {outcome}")

        rows = self._conn.execute(
            "SELECT * FROM positions WHERE market_id=? AND status='OPEN'",
            (market_id,),
        ).fetchall()

        if not rows:
            raise ValueError(f"No open positions on {market_id}")

        now = datetime.now(timezone.utc).isoformat()
        resolved = []

        for row in rows:
            pos_id = row["id"]
            side = row["side"]
            shares = row["shares"]
            entry_price = row["entry_price"]

            # Winner gets $1/share, loser gets $0
            won = side == outcome
            settlement_price = 1.0 if won else 0.0
            proceeds = shares * settlement_price
            pnl = proceeds - (shares * entry_price)

            trade_id = f"trd_{uuid.uuid4().hex[:12]}"

            self._conn.execute(
                """UPDATE positions SET status='RESOLVED', current_price=?,
                   exit_reason=?, exit_time=?, pnl=? WHERE id=?""",
                (settlement_price, f"resolved:{outcome}", now, pnl, pos_id),
            )
            self._conn.execute(
                """INSERT INTO trades
                   (id, position_id, market_id, side, shares, price, fee,
                    trade_type, reason, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 'RESOLVE', ?, ?)""",
                (trade_id, pos_id, market_id, side, shares, settlement_price,
                 f"resolved:{outcome}", now),
            )
            self._set_cash(self._get_cash() + proceeds)

            logger.info(
                "Resolved %s on %s: %s (pnl=$%.4f)",
                pos_id, market_id, "WON" if won else "LOST", pnl,
            )
            resolved.append(self._get_position(pos_id))

        return resolved

    def update_confidence(self, market_id: str, confidence: float, side: str | None = None) -> None:
        """Update the confidence score on an open position (for exit strategy)."""
        query = "UPDATE positions SET confidence=? WHERE market_id=? AND status='OPEN'"
        params: list[Any] = [confidence, market_id]
        if side:
            query += " AND side=?"
            params.append(side.upper())
        self._conn.execute(query, params)
        self._conn.commit()

    # ── Queries ───────────────────────────────────────────────────────

    def _get_position(self, pos_id: str) -> Position:
        row = self._conn.execute("SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone()
        return Position(**dict(row))

    def get_open_positions(self) -> list[Position]:
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY entry_time DESC"
        ).fetchall()
        return [Position(**dict(r)) for r in rows]

    def get_all_positions(self) -> list[Position]:
        rows = self._conn.execute(
            "SELECT * FROM positions ORDER BY entry_time DESC"
        ).fetchall()
        return [Position(**dict(r)) for r in rows]

    def get_trade_history(self, market_id: str | None = None) -> list[Trade]:
        if market_id:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE market_id=? ORDER BY timestamp DESC",
                (market_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC"
            ).fetchall()
        return [Trade(**dict(r)) for r in rows]

    def get_portfolio(self) -> PortfolioSummary:
        cash = self._get_cash()
        positions = self.get_open_positions()
        unrealized = sum(p.unrealized_pnl for p in positions)
        position_value = sum(p.market_value for p in positions)

        # Realized P&L from closed/resolved positions
        row = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM positions WHERE status IN ('CLOSED', 'RESOLVED')"
        ).fetchone()
        realized = float(row["total"])

        # Total fees paid
        trade_count = self._conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()["cnt"]

        return PortfolioSummary(
            cash=cash,
            positions=positions,
            total_value=cash + position_value,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_trades=trade_count,
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Paper trading engine for Polymarket")
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Show portfolio status")

    # trade
    p_trade = sub.add_parser("trade", help="Open a new position")
    p_trade.add_argument("market_id")
    p_trade.add_argument("side", choices=["YES", "NO", "yes", "no"])
    p_trade.add_argument("shares", type=float)
    p_trade.add_argument("price", type=float)
    p_trade.add_argument("--reason", default="")
    p_trade.add_argument("--confidence", type=float, default=None)

    # close
    p_close = sub.add_parser("close", help="Close a position")
    p_close.add_argument("market_id")
    p_close.add_argument("price", type=float)
    p_close.add_argument("--reason", default="")
    p_close.add_argument("--side", default=None)

    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve a market")
    p_resolve.add_argument("market_id")
    p_resolve.add_argument("outcome", choices=["YES", "NO", "yes", "no"])

    # history
    p_hist = sub.add_parser("history", help="Show trade history")
    p_hist.add_argument("--market", default=None)

    args = parser.parse_args()
    engine = PaperTradingEngine()

    if args.command == "status":
        p = engine.get_portfolio()
        print(f"Cash:           ${p.cash:>10.2f}")
        print(f"Position Value: ${p.total_value - p.cash:>10.2f}")
        print(f"Total Value:    ${p.total_value:>10.2f}")
        print(f"Unrealized P&L: ${p.unrealized_pnl:>10.2f}")
        print(f"Realized P&L:   ${p.realized_pnl:>10.2f}")
        print(f"Total Trades:   {p.total_trades:>10}")
        if p.positions:
            print(f"\nOpen positions ({len(p.positions)}):")
            for pos in p.positions:
                conf = f" conf={pos.confidence:.0%}" if pos.confidence else ""
                print(
                    f"  {pos.side:3s} {pos.market_id[:24]:24s} "
                    f"x{pos.shares:<6.0f} entry={pos.entry_price:.4f} "
                    f"now={pos.current_price:.4f} pnl=${pos.unrealized_pnl:+.2f}{conf}"
                )

    elif args.command == "trade":
        pos = engine.open_position(
            args.market_id, args.side, args.shares, args.price,
            reason=args.reason, confidence=args.confidence,
        )
        print(f"Opened: {pos.id} {pos.side} x{pos.shares:.0f} @{pos.entry_price:.4f}")

    elif args.command == "close":
        pos = engine.close_position(args.market_id, args.price, reason=args.reason, side=args.side)
        print(f"Closed: {pos.id} pnl=${pos.pnl:+.4f}")

    elif args.command == "resolve":
        resolved = engine.resolve_market(args.market_id, args.outcome)
        for pos in resolved:
            won = "WON" if pos.pnl > 0 else "LOST"
            print(f"Resolved: {pos.id} {pos.side} → {won} pnl=${pos.pnl:+.4f}")

    elif args.command == "history":
        trades = engine.get_trade_history(market_id=args.market)
        if not trades:
            print("No trades.")
        for t in trades:
            print(
                f"[{t.timestamp[:19]}] {t.trade_type:7s} {t.side:3s} "
                f"{t.market_id[:24]:24s} x{t.shares:<6.0f} @{t.price:.4f} "
                f"fee=${t.fee:.4f} {t.reason}"
            )

    else:
        parser.print_help()

    engine.close()


if __name__ == "__main__":
    main()
