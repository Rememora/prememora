"""Tests for the strategy review / feedback loop."""

import json
from pathlib import Path

import pytest

from trading.paper_engine import PaperTradingEngine
from trading.strategy_review import (
    CalibrationBucket,
    StrategyReview,
    brier_score,
    calibration_buckets,
    extract_signal_sources,
    read_signal_log,
)


# ── Brier score ───────────────────────────────────────────────────────────────


class TestBrierScore:
    def test_perfect_predictions(self):
        # Always right: predicted 1.0 and won, predicted 0.0 and lost
        pairs = [(1.0, True), (0.0, False), (0.9, True), (0.1, False)]
        bs = brier_score(pairs)
        assert bs < 0.01  # near-perfect

    def test_random_predictions(self):
        # Always predict 0.5 — Brier should be exactly 0.25
        pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
        bs = brier_score(pairs)
        assert abs(bs - 0.25) < 1e-10

    def test_terrible_predictions(self):
        # Always wrong: predicted 1.0 but lost, predicted 0.0 but won
        pairs = [(1.0, False), (0.0, True)]
        bs = brier_score(pairs)
        assert abs(bs - 1.0) < 1e-10

    def test_empty(self):
        assert brier_score([]) is None

    def test_single_prediction(self):
        bs = brier_score([(0.8, True)])
        # (0.8 - 1.0)² = 0.04
        assert abs(bs - 0.04) < 1e-10


# ── Calibration ───────────────────────────────────────────────────────────────


class TestCalibration:
    def test_basic_buckets(self):
        pairs = [
            (0.2, False), (0.3, True), (0.3, False),   # low bucket
            (0.7, True), (0.8, True), (0.8, False),     # high bucket
        ]
        buckets = calibration_buckets(pairs, n_buckets=2)
        assert len(buckets) == 2
        # Low bucket: predicted ~0.27, won 1/3 = 0.33
        assert buckets[0].predicted_mean < 0.5
        # High bucket: predicted ~0.77, won 2/3 = 0.67
        assert buckets[1].predicted_mean > 0.5

    def test_empty(self):
        assert calibration_buckets([]) == []

    def test_single_bucket(self):
        pairs = [(0.7, True), (0.8, True)]
        buckets = calibration_buckets(pairs, n_buckets=1)
        assert len(buckets) == 1
        assert buckets[0].actual_win_rate == 1.0

    def test_error_calculation(self):
        pairs = [(0.9, True), (0.9, False)]  # predicted 90%, actual 50%
        buckets = calibration_buckets(pairs, n_buckets=1)
        assert abs(buckets[0].error - 0.4) < 1e-10  # 0.9 - 0.5 = 0.4


# ── Source extraction ─────────────────────────────────────────────────────────


class TestExtractSources:
    def test_mirofish(self):
        sources = extract_signal_sources("mirofish: prob=0.82, edge=+0.10")
        assert "mirofish" in sources

    def test_whale_alert(self):
        sources = extract_signal_sources("Whale outflow from Binance detected")
        assert "whale_alert" in sources

    def test_multiple_sources(self):
        sources = extract_signal_sources("mirofish sim + whale alert + reddit sentiment")
        assert "mirofish" in sources
        assert "whale_alert" in sources
        assert "reddit" in sources

    def test_graph_context(self):
        sources = extract_signal_sources("Aggregated from 5 agents with 8 graph facts")
        assert "graph_context" in sources

    def test_unknown(self):
        sources = extract_signal_sources("some random reason")
        assert sources == ["unknown"]


# ── Signal log reader ─────────────────────────────────────────────────────────


class TestSignalLog:
    def test_read_valid(self, tmp_path):
        log_path = tmp_path / "signals.jsonl"
        entries = [
            {"cycle_time": "2026-03-30T10:00:00", "market_id": "mkt_1",
             "question": "Will BTC hit $100k?", "market_price": 0.65,
             "our_probability": 0.82, "action": "BUY_YES", "side": "YES",
             "edge": 0.17, "reason": "mirofish", "executed": True,
             "position_id": "pos_abc", "context_facts": 5},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        records = read_signal_log(log_path)
        assert len(records) == 1
        assert records[0].market_id == "mkt_1"
        assert records[0].our_probability == 0.82
        assert records[0].executed is True

    def test_read_empty(self, tmp_path):
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")
        assert read_signal_log(log_path) == []

    def test_read_missing(self, tmp_path):
        assert read_signal_log(tmp_path / "nope.jsonl") == []


# ── Strategy report integration ───────────────────────────────────────────────


class TestStrategyReport:
    @pytest.fixture
    def setup(self, tmp_path):
        """Create engine with trades and a matching signal log."""
        db = tmp_path / "test.db"
        engine = PaperTradingEngine(db_path=db, initial_cash=1000.0)

        # Open and resolve positions
        engine.open_position("mkt_1", "YES", 100, 0.60, reason="mirofish signal", confidence=0.82)
        engine.open_position("mkt_2", "NO", 50, 0.40, reason="whale alert + reddit", confidence=0.55)
        engine.open_position("mkt_3", "YES", 80, 0.70, reason="mirofish + graph facts", confidence=0.88)

        engine.resolve_market("mkt_1", "YES")  # win
        engine.resolve_market("mkt_2", "YES")  # lose (we bet NO)
        # mkt_3 still open

        # Signal log
        log_path = tmp_path / "signals.jsonl"
        signals = [
            {"cycle_time": "2026-03-30T10:00:00", "market_id": "mkt_1",
             "question": "BTC $100k?", "market_price": 0.60,
             "our_probability": 0.82, "action": "BUY_YES", "side": "YES",
             "edge": 0.22, "reason": "mirofish: prob=0.82",
             "executed": True, "position_id": "pos_1", "context_facts": 3},
            {"cycle_time": "2026-03-30T10:01:00", "market_id": "mkt_2",
             "question": "Fed rate cut?", "market_price": 0.40,
             "our_probability": 0.35, "action": "BUY_NO", "side": "NO",
             "edge": 0.25, "reason": "whale alert + reddit sentiment",
             "executed": True, "position_id": "pos_2", "context_facts": 0},
            {"cycle_time": "2026-03-30T10:02:00", "market_id": "mkt_3",
             "question": "ETH $5k?", "market_price": 0.70,
             "our_probability": 0.88, "action": "BUY_YES", "side": "YES",
             "edge": 0.18, "reason": "mirofish + 8 graph facts",
             "executed": True, "position_id": "pos_3", "context_facts": 8},
        ]
        log_path.write_text("\n".join(json.dumps(s) for s in signals))

        review = StrategyReview(paper_engine=engine, signal_log_path=log_path)
        yield review, engine
        engine.close()

    def test_report_overview(self, setup):
        review, _ = setup
        report = review.generate_report()
        assert report.total_signals == 3
        assert report.executed_signals == 3
        assert report.resolved_trades == 2
        assert report.open_trades == 1

    def test_report_has_pnl(self, setup):
        review, _ = setup
        report = review.generate_report()
        # mkt_1 won (YES@0.60 → $1 = +$40), mkt_2 lost (NO@0.40 → $0 = -$20)
        assert report.total_pnl != 0
        assert report.win_rate is not None

    def test_report_has_brier(self, setup):
        review, _ = setup
        report = review.generate_report()
        # We have 2 resolved trades with probability data
        assert report.brier_score is not None

    def test_report_source_stats(self, setup):
        review, _ = setup
        report = review.generate_report()
        source_names = [s.source for s in report.source_stats]
        assert "mirofish" in source_names

    def test_report_summary_string(self, setup):
        review, _ = setup
        report = review.generate_report()
        summary = report.summary
        assert "Strategy Review" in summary
        assert "Brier score" in summary

    def test_report_recommendations(self, setup):
        review, _ = setup
        report = review.generate_report()
        # With only 2 resolved trades, should recommend more data
        has_more_data_rec = any("more data" in r.lower() for r in report.recommendations)
        assert has_more_data_rec


class TestRecommendations:
    @pytest.fixture
    def engine_with_many_trades(self, tmp_path):
        db = tmp_path / "many.db"
        engine = PaperTradingEngine(db_path=db, initial_cash=10000.0)

        # Create 12 resolved trades to get past the minimum threshold
        for i in range(12):
            market = f"mkt_{i}"
            engine.open_position(market, "YES", 10, 0.50, reason="mirofish signal")
            # Alternate wins and losses — 50% win rate at 50% confidence = bad
            outcome = "YES" if i % 2 == 0 else "NO"
            engine.resolve_market(market, outcome)

        log_path = tmp_path / "signals.jsonl"
        signals = []
        for i in range(12):
            signals.append({
                "cycle_time": f"2026-03-30T{10+i}:00:00", "market_id": f"mkt_{i}",
                "question": f"Q{i}?", "market_price": 0.50,
                "our_probability": 0.80,  # always predict 80% YES
                "action": "BUY_YES", "side": "YES",
                "edge": 0.30, "reason": "mirofish: prob=0.80",
                "executed": True, "position_id": f"pos_{i}", "context_facts": 0,
            })
        log_path.write_text("\n".join(json.dumps(s) for s in signals))

        review = StrategyReview(paper_engine=engine, signal_log_path=log_path)
        yield review
        engine.close()

    def test_overconfidence_detected(self, engine_with_many_trades):
        report = engine_with_many_trades.generate_report()
        # We predicted 80% but only won 50% — should flag overconfidence
        assert report.brier_score is not None
        assert report.brier_score > 0.05  # definitely not well-calibrated
        # Should have recommendation about overconfidence or poor Brier
        has_relevant_rec = any(
            "overconfident" in r.lower() or "brier" in r.lower() or "worse" in r.lower()
            for r in report.recommendations
        )
        assert has_relevant_rec, f"Expected overconfidence rec, got: {report.recommendations}"
