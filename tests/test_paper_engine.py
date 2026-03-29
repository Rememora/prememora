"""Tests for the paper trading engine."""

import sqlite3
from pathlib import Path

import pytest

from trading.paper_engine import (
    PaperTradingEngine,
    Position,
    PositionStatus,
    calculate_fee,
)


@pytest.fixture
def engine(tmp_path):
    """Create a fresh engine with $1000 starting cash."""
    db = tmp_path / "test_paper.db"
    eng = PaperTradingEngine(db_path=db, initial_cash=1000.0, fee_bps=2)
    yield eng
    eng.close()


# ── Fee calculation ───────────────────────────────────────────────────────────


class TestFees:
    def test_50_50_market(self):
        # min(0.5, 0.5) = 0.5 → fee = 2/10000 * 0.5 * 100 = 0.01
        fee = calculate_fee(0.5, 100, fee_bps=2)
        assert abs(fee - 0.01) < 1e-10

    def test_lopsided_market(self):
        # min(0.9, 0.1) = 0.1 → fee = 2/10000 * 0.1 * 100 = 0.002
        fee = calculate_fee(0.9, 100, fee_bps=2)
        assert abs(fee - 0.002) < 1e-10

    def test_symmetric(self):
        # Fee for YES at 0.3 == fee for NO at 0.7
        assert abs(calculate_fee(0.3, 100) - calculate_fee(0.7, 100)) < 1e-10

    def test_zero_shares(self):
        assert calculate_fee(0.5, 0) == 0.0

    def test_custom_bps(self):
        fee = calculate_fee(0.5, 100, fee_bps=10)
        assert abs(fee - 0.05) < 1e-10


# ── Open position ─────────────────────────────────────────────────────────────


class TestOpenPosition:
    def test_basic_open(self, engine):
        pos = engine.open_position("mkt_1", "YES", 100, 0.60, reason="test")
        assert pos.market_id == "mkt_1"
        assert pos.side == "YES"
        assert pos.shares == 100
        assert pos.entry_price == 0.60
        assert pos.status == "OPEN"
        assert pos.entry_reason == "test"

    def test_cash_deducted(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        portfolio = engine.get_portfolio()
        # Cost = 100 * 0.60 = 60, fee = 2/10000 * min(0.6, 0.4) * 100 = 0.008
        expected_cash = 1000.0 - 60.0 - 0.008
        assert abs(portfolio.cash - expected_cash) < 1e-4

    def test_insufficient_cash(self, engine):
        with pytest.raises(ValueError, match="Insufficient cash"):
            engine.open_position("mkt_1", "YES", 10000, 0.50)

    def test_invalid_side(self, engine):
        with pytest.raises(ValueError, match="side must be YES or NO"):
            engine.open_position("mkt_1", "MAYBE", 100, 0.50)

    def test_invalid_price_zero(self, engine):
        with pytest.raises(ValueError, match="price must be between 0 and 1"):
            engine.open_position("mkt_1", "YES", 100, 0.0)

    def test_invalid_price_one(self, engine):
        with pytest.raises(ValueError, match="price must be between 0 and 1"):
            engine.open_position("mkt_1", "YES", 100, 1.0)

    def test_invalid_shares(self, engine):
        with pytest.raises(ValueError, match="shares must be positive"):
            engine.open_position("mkt_1", "YES", -10, 0.50)

    def test_duplicate_position_rejected(self, engine):
        engine.open_position("mkt_1", "YES", 10, 0.50)
        with pytest.raises(ValueError, match="Already have an open"):
            engine.open_position("mkt_1", "YES", 10, 0.60)

    def test_opposite_side_allowed(self, engine):
        engine.open_position("mkt_1", "YES", 10, 0.50)
        pos = engine.open_position("mkt_1", "NO", 10, 0.50)
        assert pos.side == "NO"

    def test_case_insensitive_side(self, engine):
        pos = engine.open_position("mkt_1", "yes", 10, 0.50)
        assert pos.side == "YES"

    def test_confidence_stored(self, engine):
        pos = engine.open_position("mkt_1", "YES", 10, 0.50, confidence=0.80)
        assert pos.confidence == 0.80

    def test_trade_recorded(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        trades = engine.get_trade_history("mkt_1")
        assert len(trades) == 1
        assert trades[0].trade_type == "OPEN"
        assert trades[0].shares == 100
        assert trades[0].price == 0.60


# ── Close position ────────────────────────────────────────────────────────────


class TestClosePosition:
    def test_close_at_profit(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50)
        pos = engine.close_position("mkt_1", 0.80, reason="profit")
        assert pos.status == "CLOSED"
        assert pos.exit_reason == "profit"
        assert pos.pnl > 0

    def test_close_at_loss(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.80)
        pos = engine.close_position("mkt_1", 0.40, reason="stop loss")
        assert pos.status == "CLOSED"
        assert pos.pnl < 0

    def test_cash_credited(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50)
        cash_after_open = engine.get_portfolio().cash
        engine.close_position("mkt_1", 0.80)
        cash_after_close = engine.get_portfolio().cash
        # Proceeds = 100 * 0.80 = 80 minus fee
        assert cash_after_close > cash_after_open

    def test_close_nonexistent(self, engine):
        with pytest.raises(ValueError, match="No open position"):
            engine.close_position("mkt_999", 0.50)

    def test_close_by_side(self, engine):
        engine.open_position("mkt_1", "YES", 10, 0.50)
        engine.open_position("mkt_1", "NO", 10, 0.50)
        pos = engine.close_position("mkt_1", 0.70, side="YES")
        assert pos.side == "YES"
        assert pos.status == "CLOSED"
        # NO position still open
        open_pos = engine.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0].side == "NO"

    def test_pnl_calculation(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50)
        pos = engine.close_position("mkt_1", 0.70)
        # PnL = (proceeds - fee) - cost = (70 - fee) - 50
        fee = calculate_fee(0.70, 100, 2)
        expected_pnl = (100 * 0.70 - fee) - (100 * 0.50)
        assert abs(pos.pnl - expected_pnl) < 1e-6

    def test_close_trade_recorded(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50)
        engine.close_position("mkt_1", 0.70)
        trades = engine.get_trade_history("mkt_1")
        assert len(trades) == 2
        close_trade = [t for t in trades if t.trade_type == "CLOSE"][0]
        assert close_trade.price == 0.70


# ── Market resolution ─────────────────────────────────────────────────────────


class TestResolution:
    def test_winner_gets_dollar(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        resolved = engine.resolve_market("mkt_1", "YES")
        assert len(resolved) == 1
        pos = resolved[0]
        assert pos.status == "RESOLVED"
        # PnL = (100 * 1.0) - (100 * 0.60) = 40
        assert abs(pos.pnl - 40.0) < 1e-6

    def test_loser_gets_zero(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        resolved = engine.resolve_market("mkt_1", "NO")
        pos = resolved[0]
        # PnL = (100 * 0.0) - (100 * 0.60) = -60
        assert abs(pos.pnl - (-60.0)) < 1e-6

    def test_both_sides_resolved(self, engine):
        engine.open_position("mkt_1", "YES", 50, 0.60)
        engine.open_position("mkt_1", "NO", 50, 0.40)
        resolved = engine.resolve_market("mkt_1", "YES")
        assert len(resolved) == 2
        yes_pos = [p for p in resolved if p.side == "YES"][0]
        no_pos = [p for p in resolved if p.side == "NO"][0]
        assert yes_pos.pnl > 0
        assert no_pos.pnl < 0

    def test_cash_after_win(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        engine.resolve_market("mkt_1", "YES")
        portfolio = engine.get_portfolio()
        # Started with 1000, spent 60 + fee, got back 100
        fee = calculate_fee(0.60, 100, 2)
        expected = 1000.0 - 60.0 - fee + 100.0
        assert abs(portfolio.cash - expected) < 1e-4

    def test_cash_after_loss(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        engine.resolve_market("mkt_1", "NO")
        portfolio = engine.get_portfolio()
        # Started with 1000, spent 60 + fee, got back 0
        fee = calculate_fee(0.60, 100, 2)
        expected = 1000.0 - 60.0 - fee
        assert abs(portfolio.cash - expected) < 1e-4

    def test_resolve_no_positions(self, engine):
        with pytest.raises(ValueError, match="No open positions"):
            engine.resolve_market("mkt_999", "YES")

    def test_invalid_outcome(self, engine):
        with pytest.raises(ValueError, match="outcome must be YES or NO"):
            engine.resolve_market("mkt_1", "DRAW")

    def test_resolve_trade_recorded(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60)
        engine.resolve_market("mkt_1", "YES")
        trades = engine.get_trade_history("mkt_1")
        resolve_trades = [t for t in trades if t.trade_type == "RESOLVE"]
        assert len(resolve_trades) == 1
        assert resolve_trades[0].fee == 0  # no fee on resolution


# ── Portfolio & confidence ────────────────────────────────────────────────────


class TestPortfolio:
    def test_initial_portfolio(self, engine):
        p = engine.get_portfolio()
        assert p.cash == 1000.0
        assert p.total_value == 1000.0
        assert p.positions == []
        assert p.realized_pnl == 0.0

    def test_unrealized_pnl(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50)
        p = engine.get_portfolio()
        # current_price == entry_price at open, so unrealized = 0
        assert p.unrealized_pnl == 0.0

    def test_realized_pnl_after_close(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50)
        engine.close_position("mkt_1", 0.80)
        p = engine.get_portfolio()
        assert p.realized_pnl > 0

    def test_multiple_positions(self, engine):
        engine.open_position("mkt_1", "YES", 50, 0.50)
        engine.open_position("mkt_2", "NO", 50, 0.30)
        p = engine.get_portfolio()
        assert len(p.positions) == 2

    def test_update_confidence(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.50, confidence=0.80)
        engine.update_confidence("mkt_1", 0.60)
        positions = engine.get_open_positions()
        assert positions[0].confidence == 0.60


class TestPersistence:
    def test_survives_restart(self, tmp_path):
        db = tmp_path / "persist.db"
        eng1 = PaperTradingEngine(db_path=db, initial_cash=500.0)
        eng1.open_position("mkt_1", "YES", 50, 0.40)
        eng1.close()

        eng2 = PaperTradingEngine(db_path=db)
        positions = eng2.get_open_positions()
        assert len(positions) == 1
        assert positions[0].market_id == "mkt_1"
        p = eng2.get_portfolio()
        assert p.cash < 500.0  # cash was deducted
        eng2.close()
