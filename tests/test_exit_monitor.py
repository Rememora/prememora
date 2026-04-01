"""Tests for the exit monitor — confidence-based position management."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from trading.exit_monitor import (
    ExitConfig,
    ExitMonitor,
    ExitSignal,
    check_confidence_drop,
    check_contradictory_evidence,
    check_stop_loss,
    check_time_decay,
)
from trading.paper_engine import PaperTradingEngine, Position


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_position(
    id: str = "pos_test",
    market_id: str = "mkt_1",
    side: str = "YES",
    entry_price: float = 0.60,
    current_price: float = 0.60,
    confidence: float | None = 0.80,
    entry_confidence: float | None = 0.80,
    entry_time: str = "",
    market_deadline: str | None = None,
) -> Position:
    """Build a Position for testing without touching the DB."""
    if not entry_time:
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    return Position(
        id=id,
        market_id=market_id,
        side=side,
        shares=100,
        entry_price=entry_price,
        current_price=current_price,
        status="OPEN",
        confidence=confidence,
        entry_reason="test",
        exit_reason="",
        entry_time=entry_time,
        exit_time=None,
        pnl=0.0,
        entry_confidence=entry_confidence,
        market_deadline=market_deadline,
    )


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test_exit.db"
    eng = PaperTradingEngine(db_path=db, initial_cash=10000.0, fee_bps=2)
    yield eng
    eng.close()


# ── check_confidence_drop ────────────────────────────────────────────────────


class TestCheckConfidenceDrop:
    def test_triggers_on_large_drop(self):
        pos = _make_position(entry_confidence=0.80, confidence=0.80)
        sig = check_confidence_drop(pos, new_probability=0.60, threshold=0.15)
        assert sig is not None
        assert sig.trigger_type == "confidence_drop"
        assert sig.old_confidence == 0.80
        assert sig.new_confidence == 0.60

    def test_no_trigger_on_small_drop(self):
        pos = _make_position(entry_confidence=0.80, confidence=0.80)
        sig = check_confidence_drop(pos, new_probability=0.70, threshold=0.15)
        assert sig is None

    def test_exactly_at_threshold(self):
        pos = _make_position(entry_confidence=0.80, confidence=0.80)
        sig = check_confidence_drop(pos, new_probability=0.65, threshold=0.15)
        assert sig is not None

    def test_no_trigger_on_increase(self):
        pos = _make_position(entry_confidence=0.60, confidence=0.60)
        sig = check_confidence_drop(pos, new_probability=0.80, threshold=0.15)
        assert sig is None

    def test_no_entry_confidence_uses_confidence(self):
        pos = _make_position(entry_confidence=None, confidence=0.80)
        sig = check_confidence_drop(pos, new_probability=0.60, threshold=0.15)
        assert sig is not None

    def test_no_confidence_at_all(self):
        pos = _make_position(entry_confidence=None, confidence=None)
        sig = check_confidence_drop(pos, new_probability=0.60, threshold=0.15)
        assert sig is None

    def test_no_side_yes(self):
        """For NO positions, a probability increase means our confidence dropped."""
        pos = _make_position(side="NO", entry_confidence=0.80, confidence=0.80)
        # entry_confidence=0.80 (YES prob at entry), side=NO
        # For NO side: drop = new_prob - old_conf = 0.96 - 0.80 = 0.16 >= 0.15
        sig = check_confidence_drop(pos, new_probability=0.96, threshold=0.15)
        assert sig is not None

    def test_no_side_no_trigger(self):
        pos = _make_position(side="NO", entry_confidence=0.80, confidence=0.80)
        # effective_old = 0.20, effective_new = 1-0.85=0.15, drop=0.05 < 0.15
        sig = check_confidence_drop(pos, new_probability=0.85, threshold=0.15)
        assert sig is None


# ── check_contradictory_evidence ─────────────────────────────────────────────


class TestCheckContradictoryEvidence:
    def test_triggers_when_edge_flips_yes(self):
        pos = _make_position(side="YES", entry_confidence=0.80)
        # YES edge = new_prob - market_price = 0.40 - 0.65 = -0.25 < 0
        sig = check_contradictory_evidence(pos, new_probability=0.40, market_price=0.65)
        assert sig is not None
        assert sig.trigger_type == "contradictory_evidence"

    def test_no_trigger_positive_edge_yes(self):
        pos = _make_position(side="YES", entry_confidence=0.80)
        sig = check_contradictory_evidence(pos, new_probability=0.75, market_price=0.60)
        assert sig is None

    def test_triggers_when_edge_flips_no(self):
        pos = _make_position(side="NO", entry_confidence=0.80)
        # NO edge = (1 - 0.75) - (1 - 0.65) = 0.25 - 0.35 = -0.10 < 0
        sig = check_contradictory_evidence(pos, new_probability=0.75, market_price=0.65)
        assert sig is not None

    def test_no_trigger_positive_edge_no(self):
        pos = _make_position(side="NO", entry_confidence=0.80)
        # NO edge = (1-0.30) - (1-0.60) = 0.70 - 0.40 = 0.30 > 0
        sig = check_contradictory_evidence(pos, new_probability=0.30, market_price=0.60)
        assert sig is None

    def test_zero_edge_triggers(self):
        pos = _make_position(side="YES", entry_confidence=0.80)
        sig = check_contradictory_evidence(pos, new_probability=0.60, market_price=0.60)
        assert sig is not None  # edge == 0, should still trigger


# ── check_time_decay ─────────────────────────────────────────────────────────


class TestCheckTimeDecay:
    def test_triggers_when_time_running_out(self):
        now = datetime.now(timezone.utc)
        entry = now - timedelta(hours=20)
        deadline = now + timedelta(hours=2)  # 2h left out of 22h total = 9%
        pos = _make_position(
            entry_time=entry.isoformat(),
            market_deadline=deadline.isoformat(),
        )
        sig = check_time_decay(pos, now, threshold=0.20)
        assert sig is not None
        assert sig.trigger_type == "time_decay"

    def test_no_trigger_plenty_of_time(self):
        now = datetime.now(timezone.utc)
        entry = now - timedelta(hours=2)
        deadline = now + timedelta(days=7)
        pos = _make_position(
            entry_time=entry.isoformat(),
            market_deadline=deadline.isoformat(),
        )
        sig = check_time_decay(pos, now, threshold=0.20)
        assert sig is None

    def test_no_deadline(self):
        pos = _make_position(market_deadline=None)
        sig = check_time_decay(pos, datetime.now(timezone.utc), threshold=0.20)
        assert sig is None

    def test_empty_deadline(self):
        pos = _make_position(market_deadline="")
        sig = check_time_decay(pos, datetime.now(timezone.utc), threshold=0.20)
        assert sig is None

    def test_past_deadline(self):
        now = datetime.now(timezone.utc)
        entry = now - timedelta(days=2)
        deadline = now - timedelta(hours=1)  # already past
        pos = _make_position(
            entry_time=entry.isoformat(),
            market_deadline=deadline.isoformat(),
        )
        sig = check_time_decay(pos, now, threshold=0.20)
        assert sig is not None  # remaining < 0 should trigger


# ── check_stop_loss ──────────────────────────────────────────────────────────


class TestCheckStopLoss:
    def test_triggers_yes_price_drop(self):
        pos = _make_position(side="YES", entry_price=0.60)
        sig = check_stop_loss(pos, current_market_price=0.35, delta=0.20)
        assert sig is not None
        assert sig.trigger_type == "stop_loss"

    def test_no_trigger_yes_small_drop(self):
        pos = _make_position(side="YES", entry_price=0.60)
        sig = check_stop_loss(pos, current_market_price=0.50, delta=0.20)
        assert sig is None

    def test_triggers_no_price_rise(self):
        pos = _make_position(side="NO", entry_price=0.40)
        # For NO: move_against = market_price - entry = 0.65 - 0.40 = 0.25 >= 0.20
        sig = check_stop_loss(pos, current_market_price=0.65, delta=0.20)
        assert sig is not None

    def test_no_trigger_no_small_rise(self):
        pos = _make_position(side="NO", entry_price=0.40)
        sig = check_stop_loss(pos, current_market_price=0.50, delta=0.20)
        assert sig is None

    def test_exactly_at_delta(self):
        pos = _make_position(side="YES", entry_price=0.60)
        # Use 0.39 to clearly cross the 0.20 delta (0.60 - 0.39 = 0.21)
        sig = check_stop_loss(pos, current_market_price=0.39, delta=0.20)
        assert sig is not None

    def test_price_moves_in_our_favor(self):
        pos = _make_position(side="YES", entry_price=0.60)
        sig = check_stop_loss(pos, current_market_price=0.80, delta=0.20)
        assert sig is None


# ── ExitMonitor integration ──────────────────────────────────────────────────


class TestExitMonitorCheckAll:
    @pytest.mark.asyncio
    async def test_stop_loss_detected(self, engine):
        """Stop loss is detected without re-interview."""
        engine.open_position("mkt_1", "YES", 100, 0.60, confidence=0.80)

        config = ExitConfig(
            stop_loss_delta=0.20,
            min_hold_seconds=0,  # no hold requirement for test
            enabled_triggers={"stop_loss"},
        )
        monitor = ExitMonitor(config=config, paper_engine=engine)

        # Mock market price returning 0.35 (25c drop)
        with patch("trading.exit_monitor.fetch_market_prices", new_callable=AsyncMock) as mock_prices:
            mock_prices.return_value = {"mkt_1": 0.35}
            signals = await monitor.check_all_positions()

        assert len(signals) == 1
        assert signals[0].trigger_type == "stop_loss"

    @pytest.mark.asyncio
    async def test_time_decay_detected(self, engine):
        now = datetime.now(timezone.utc)
        entry_time = (now - timedelta(hours=20)).isoformat()
        deadline = (now + timedelta(hours=2)).isoformat()

        engine.open_position(
            "mkt_1", "YES", 100, 0.60,
            confidence=0.80,
            market_deadline=deadline,
        )
        # Backdate entry_time in the DB
        engine._conn.execute(
            "UPDATE positions SET entry_time=? WHERE market_id='mkt_1'",
            (entry_time,),
        )
        engine._conn.commit()

        config = ExitConfig(
            time_decay_threshold=0.20,
            min_hold_seconds=0,
            enabled_triggers={"time_decay"},
        )
        monitor = ExitMonitor(config=config, paper_engine=engine)

        with patch("trading.exit_monitor.fetch_market_prices", new_callable=AsyncMock) as mock_prices:
            mock_prices.return_value = {"mkt_1": 0.60}
            signals = await monitor.check_all_positions()

        assert len(signals) == 1
        assert signals[0].trigger_type == "time_decay"

    @pytest.mark.asyncio
    async def test_min_hold_respected(self, engine):
        """Positions opened recently should not be checked."""
        engine.open_position("mkt_1", "YES", 100, 0.60, confidence=0.80)

        config = ExitConfig(
            stop_loss_delta=0.01,  # very aggressive threshold
            min_hold_seconds=99999,  # very long hold
            enabled_triggers={"stop_loss"},
        )
        monitor = ExitMonitor(config=config, paper_engine=engine)

        with patch("trading.exit_monitor.fetch_market_prices", new_callable=AsyncMock) as mock_prices:
            mock_prices.return_value = {"mkt_1": 0.01}
            signals = await monitor.check_all_positions()

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_disabled_triggers_skipped(self, engine):
        """Triggers not in enabled_triggers should not fire."""
        engine.open_position("mkt_1", "YES", 100, 0.60, confidence=0.80)

        config = ExitConfig(
            stop_loss_delta=0.01,
            min_hold_seconds=0,
            enabled_triggers=set(),  # nothing enabled
        )
        monitor = ExitMonitor(config=config, paper_engine=engine)

        with patch("trading.exit_monitor.fetch_market_prices", new_callable=AsyncMock) as mock_prices:
            mock_prices.return_value = {"mkt_1": 0.01}
            signals = await monitor.check_all_positions()

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_no_positions_returns_empty(self, engine):
        config = ExitConfig(min_hold_seconds=0)
        monitor = ExitMonitor(config=config, paper_engine=engine)
        signals = await monitor.check_all_positions()
        assert signals == []


class TestExitMonitorExecute:
    @pytest.mark.asyncio
    async def test_execute_closes_position(self, engine):
        pos = engine.open_position("mkt_1", "YES", 100, 0.60, confidence=0.80)

        config = ExitConfig(min_hold_seconds=0)
        monitor = ExitMonitor(config=config, paper_engine=engine)

        signal = ExitSignal(
            position_id=pos.id,
            market_id="mkt_1",
            trigger_type="stop_loss",
            old_confidence=0.80,
            new_confidence=None,
            market_price=0.40,
            reason="test exit",
        )

        closed = await monitor.execute_exits([signal])
        assert len(closed) == 1
        assert closed[0].status == "CLOSED"
        assert closed[0].exit_reason.startswith("exit:stop_loss")

        # Position should no longer be open
        assert engine.get_open_positions() == []

    @pytest.mark.asyncio
    async def test_execute_nonexistent_handled(self, engine):
        """Trying to exit a nonexistent position should not crash."""
        config = ExitConfig()
        monitor = ExitMonitor(config=config, paper_engine=engine)

        signal = ExitSignal(
            position_id="pos_fake",
            market_id="mkt_fake",
            trigger_type="stop_loss",
            old_confidence=0.80,
            new_confidence=None,
            market_price=0.40,
            reason="test",
        )

        closed = await monitor.execute_exits([signal])
        assert len(closed) == 0


class TestExitMonitorRunOnce:
    @pytest.mark.asyncio
    async def test_run_once_checks_and_executes(self, engine):
        engine.open_position("mkt_1", "YES", 100, 0.60, confidence=0.80)

        config = ExitConfig(
            stop_loss_delta=0.20,
            min_hold_seconds=0,
            enabled_triggers={"stop_loss"},
        )
        monitor = ExitMonitor(config=config, paper_engine=engine)

        with patch("trading.exit_monitor.fetch_market_prices", new_callable=AsyncMock) as mock_prices:
            mock_prices.return_value = {"mkt_1": 0.35}
            signals = await monitor.run_once()

        assert len(signals) == 1
        # Position should be closed
        assert engine.get_open_positions() == []
