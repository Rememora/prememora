"""Tests for trading.calibration_gate — oracle-based pre-trade gating."""

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.calibration_gate import (
    CalibrationGate,
    CategoryCalibration,
    GateConfig,
    GateResult,
)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_calibration.db"


# ── GateResult tests ─────────────────────────────────────────────────────────


class TestGateResult:
    def test_summary_enabled(self):
        result = GateResult(
            can_trade=True,
            brier_score=0.15,
            market_count=10,
            reason="brier 0.1500 <= 0.25 threshold",
        )
        assert "TRADING ENABLED" in result.summary
        assert "0.1500" in result.summary

    def test_summary_blocked(self):
        result = GateResult(
            can_trade=False,
            brier_score=0.35,
            market_count=5,
            reason="brier 0.3500 > 0.25 threshold",
        )
        assert "TRADING BLOCKED" in result.summary
        assert "0.3500" in result.summary

    def test_summary_with_categories(self):
        result = GateResult(
            can_trade=True,
            brier_score=0.15,
            market_count=10,
            reason="ok",
            categories=[
                CategoryCalibration("crypto", 0.12, 5, 3, 0.67, True),
                CategoryCalibration("politics", 0.30, 5, 2, 0.50, False),
            ],
        )
        summary = result.summary
        assert "crypto" in summary
        assert "politics" in summary
        assert "BLOCKED" in summary

    def test_auto_timestamp(self):
        result = GateResult(can_trade=True, brier_score=0.1, market_count=5, reason="ok")
        assert result.checked_at != ""


# ── CalibrationGate tests ────────────────────────────────────────────────────


class TestCalibrationGate:
    def test_no_history(self, tmp_db):
        gate = CalibrationGate(config=GateConfig(db_path=tmp_db))
        assert gate.get_last_result() is None
        assert gate.get_history() == []
        gate.close()

    def test_save_and_retrieve(self, tmp_db):
        gate = CalibrationGate(config=GateConfig(db_path=tmp_db))
        result = GateResult(
            can_trade=True,
            brier_score=0.15,
            market_count=10,
            reason="test",
            duration_s=5.0,
        )
        gate._save_result(result, gate.config)

        last = gate.get_last_result()
        assert last is not None
        assert last.can_trade is True
        assert last.brier_score == 0.15
        assert last.market_count == 10
        gate.close()

    def test_save_with_categories(self, tmp_db):
        gate = CalibrationGate(config=GateConfig(db_path=tmp_db))
        result = GateResult(
            can_trade=True,
            brier_score=0.15,
            market_count=10,
            reason="test",
            categories=[
                CategoryCalibration("crypto", 0.12, 5, 3, 0.67, True),
                CategoryCalibration("politics", 0.30, 5, 2, 0.50, False),
            ],
        )
        gate._save_result(result, gate.config)

        last = gate.get_last_result()
        assert len(last.categories) == 2
        assert last.categories[0].category in ("crypto", "politics")
        gate.close()

    def test_history_order(self, tmp_db):
        gate = CalibrationGate(config=GateConfig(db_path=tmp_db))
        for i, bs in enumerate([0.30, 0.20, 0.15]):
            result = GateResult(
                can_trade=bs <= 0.25,
                brier_score=bs,
                market_count=5,
                reason=f"run {i}",
            )
            gate._save_result(result, gate.config)

        history = gate.get_history()
        assert len(history) == 3
        # Most recent first
        assert history[0]["brier_score"] == 0.15
        assert history[2]["brier_score"] == 0.30
        gate.close()

    def test_is_category_allowed_no_data(self, tmp_db):
        gate = CalibrationGate(config=GateConfig(db_path=tmp_db))
        # No data → permissive default
        assert gate.is_category_allowed("crypto") is True
        gate.close()

    def test_is_category_allowed_with_data(self, tmp_db):
        gate = CalibrationGate(config=GateConfig(db_path=tmp_db))
        result = GateResult(
            can_trade=True,
            brier_score=0.15,
            market_count=10,
            reason="test",
            categories=[
                CategoryCalibration("crypto", 0.12, 5, 3, 0.67, True),
                CategoryCalibration("politics", 0.30, 5, 2, 0.50, False),
            ],
        )
        gate._save_result(result, gate.config)

        assert gate.is_category_allowed("crypto") is True
        assert gate.is_category_allowed("politics") is False
        assert gate.is_category_allowed("unknown") is True  # no data → allowed
        gate.close()


# ── Check method tests (mocked oracle) ───────────────────────────────────────


class TestCalibrationGateCheck:
    def _mock_backtest_report(self, brier=0.15, n_results=5):
        """Create a mock BacktestReport."""
        report = MagicMock()
        report.brier_score = brier
        results = []
        for i in range(n_results):
            r = MagicMock()
            r.our_probability = 0.7
            r.actual_outcome = "YES" if i % 2 == 0 else "NO"
            r.action = "BUY_YES" if i % 3 == 0 else "SKIP"
            r.pnl = 10.0 if i % 2 == 0 else -5.0
            r.market_id = f"market_{i}"
            results.append(r)
        report.results = results
        return report

    def test_check_passes(self, tmp_db):
        async def run():
            with patch("backtesting.hindsight.HindsightOracle") as MockOracle:
                mock_oracle = MagicMock()
                mock_oracle.run = AsyncMock(
                    return_value=self._mock_backtest_report(brier=0.15, n_results=5)
                )
                MockOracle.return_value = mock_oracle

                gate = CalibrationGate(config=GateConfig(db_path=tmp_db, max_brier=0.25))
                result = await gate.check(force=True)

                assert result.can_trade is True
                assert result.brier_score == 0.15
                gate.close()

        asyncio.run(run())

    def test_check_blocks_high_brier(self, tmp_db):
        async def run():
            with patch("backtesting.hindsight.HindsightOracle") as MockOracle:
                mock_oracle = MagicMock()
                mock_oracle.run = AsyncMock(
                    return_value=self._mock_backtest_report(brier=0.35, n_results=5)
                )
                MockOracle.return_value = mock_oracle

                gate = CalibrationGate(config=GateConfig(db_path=tmp_db, max_brier=0.25))
                result = await gate.check(force=True)

                assert result.can_trade is False
                assert "worse than random" in result.reason
                gate.close()

        asyncio.run(run())

    def test_check_blocks_insufficient_data(self, tmp_db):
        async def run():
            with patch("backtesting.hindsight.HindsightOracle") as MockOracle:
                mock_oracle = MagicMock()
                mock_oracle.run = AsyncMock(
                    return_value=self._mock_backtest_report(brier=0.10, n_results=1)
                )
                MockOracle.return_value = mock_oracle

                gate = CalibrationGate(config=GateConfig(
                    db_path=tmp_db, min_markets=3,
                ))
                result = await gate.check(force=True)

                assert result.can_trade is False
                assert "insufficient" in result.reason
                gate.close()

        asyncio.run(run())

    def test_check_reuses_recent(self, tmp_db):
        """If a recent check exists, don't re-run the oracle."""
        async def run():
            with patch("backtesting.hindsight.HindsightOracle") as MockOracle:
                mock_oracle = MagicMock()
                mock_oracle.run = AsyncMock(
                    return_value=self._mock_backtest_report(brier=0.15, n_results=5)
                )
                MockOracle.return_value = mock_oracle

                gate = CalibrationGate(config=GateConfig(
                    db_path=tmp_db, recalibrate_hours=24,
                ))

                # First check runs oracle
                result1 = await gate.check(force=True)
                assert mock_oracle.run.call_count == 1

                # Second check reuses result
                result2 = await gate.check(force=False)
                assert mock_oracle.run.call_count == 1  # not called again
                assert result2.can_trade == result1.can_trade
                gate.close()

        asyncio.run(run())

    def test_check_force_reruns(self, tmp_db):
        """force=True always re-runs the oracle."""
        async def run():
            with patch("backtesting.hindsight.HindsightOracle") as MockOracle:
                mock_oracle = MagicMock()
                mock_oracle.run = AsyncMock(
                    return_value=self._mock_backtest_report(brier=0.15, n_results=5)
                )
                MockOracle.return_value = mock_oracle

                gate = CalibrationGate(config=GateConfig(db_path=tmp_db))

                await gate.check(force=True)
                await gate.check(force=True)
                assert mock_oracle.run.call_count == 2
                gate.close()

        asyncio.run(run())
