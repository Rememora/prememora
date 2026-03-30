"""Tests for the edge calculator."""

import pytest

from trading.edge_calculator import (
    EdgeCalculator,
    EdgeConfig,
    ProbabilityEstimate,
    RiskLimits,
    TradeSignal,
    kelly_fraction,
)


# ── Kelly fraction ────────────────────────────────────────────────────────────


class TestKellyFraction:
    def test_positive_edge(self):
        # our_prob=0.80, price=0.60 → edge=0.20, kelly = 0.20/0.40 = 0.50
        kf = kelly_fraction(0.80, 0.60)
        assert abs(kf - 0.50) < 1e-10

    def test_no_edge(self):
        # our_prob == market_price → kelly = 0
        assert kelly_fraction(0.60, 0.60) == 0.0

    def test_negative_edge(self):
        # our_prob < market_price → kelly = 0
        assert kelly_fraction(0.50, 0.70) == 0.0

    def test_small_edge(self):
        # our_prob=0.55, price=0.50 → edge=0.05, kelly = 0.05/0.50 = 0.10
        kf = kelly_fraction(0.55, 0.50)
        assert abs(kf - 0.10) < 1e-10

    def test_large_edge(self):
        # our_prob=0.95, price=0.50 → edge=0.45, kelly = 0.45/0.50 = 0.90
        kf = kelly_fraction(0.95, 0.50)
        assert abs(kf - 0.90) < 1e-10

    def test_edge_at_boundary_price_0(self):
        assert kelly_fraction(0.80, 0.0) == 0.0

    def test_edge_at_boundary_price_1(self):
        assert kelly_fraction(0.80, 1.0) == 0.0

    def test_high_price(self):
        # our_prob=0.95, price=0.90 → edge=0.05, kelly = 0.05/0.10 = 0.50
        kf = kelly_fraction(0.95, 0.90)
        assert abs(kf - 0.50) < 1e-10


# ── Edge calculator — basic signals ──────────────────────────────────────────


class TestEdgeCalculatorBasic:
    @pytest.fixture
    def calc(self):
        return EdgeCalculator(
            config=EdgeConfig(min_edge=0.10),
            portfolio_value=10000.0,
        )

    def test_yes_signal(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.82, source="mirofish")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.action == "BUY_YES"
        assert signal.side == "YES"
        assert abs(signal.edge - 0.17) < 1e-10
        assert signal.shares > 0

    def test_no_signal(self, calc):
        # our_prob=0.20 means we think NO is 0.80, market YES=0.65 → NO price=0.35
        # NO edge = (1-0.20) - (1-0.65) = 0.80 - 0.35 = 0.45
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.20, source="mirofish")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.action == "BUY_NO"
        assert signal.side == "NO"
        assert signal.edge > 0

    def test_skip_no_edge(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.65, source="test")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.action == "SKIP"

    def test_skip_below_threshold(self, calc):
        # edge = 0.72 - 0.65 = 0.07, below 10% threshold
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.72, source="test")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.action == "SKIP"
        assert "below threshold" in signal.reason

    def test_skip_extreme_low_prob(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.02, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        assert signal.action == "SKIP"
        assert "outside" in signal.reason

    def test_skip_extreme_high_prob(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.98, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        assert signal.action == "SKIP"
        assert "outside" in signal.reason

    def test_skip_invalid_market_price(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.80, source="test")
        signal = calc.evaluate(est, market_price=0.0)
        assert signal.action == "SKIP"

    def test_signal_includes_source(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.85, source="mirofish")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.source == "mirofish"

    def test_signal_has_timestamp(self, calc):
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.85, source="test")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.timestamp != ""


# ── Kelly capping ─────────────────────────────────────────────────────────────


class TestKellyCapping:
    def test_kelly_capped_at_quarter(self):
        calc = EdgeCalculator(
            config=EdgeConfig(min_edge=0.05, risk=RiskLimits(max_kelly_fraction=0.25)),
            portfolio_value=10000.0,
        )
        # our_prob=0.95, price=0.50 → raw kelly=0.90, should be capped to 0.25
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.95, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        assert signal.action == "BUY_YES"
        assert signal.kelly_fraction > 0.25  # raw kelly is high
        # But shares should reflect capped amount
        max_bet = 10000.0 * 0.25
        max_position = 10000.0 * 0.10  # default max_position_pct
        expected_bet = min(max_bet, max_position)
        expected_shares = expected_bet / 0.50
        assert abs(signal.shares - round(expected_shares, 2)) < 1


# ── Position sizing ───────────────────────────────────────────────────────────


class TestPositionSizing:
    def test_max_position_cap(self):
        calc = EdgeCalculator(
            config=EdgeConfig(
                min_edge=0.05,
                risk=RiskLimits(max_position_pct=0.05, max_kelly_fraction=1.0),
            ),
            portfolio_value=10000.0,
        )
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.90, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        # Max position = 10000 * 0.05 = 500. At price 0.50 → 1000 shares max
        assert signal.shares <= 1000 + 1

    def test_small_portfolio(self):
        calc = EdgeCalculator(
            config=EdgeConfig(min_edge=0.10),
            portfolio_value=50.0,
        )
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.85, source="test")
        signal = calc.evaluate(est, market_price=0.65)
        # With $50 portfolio, max position = $5, might be too small
        assert signal.shares >= 0


# ── Risk checks ───────────────────────────────────────────────────────────────


class TestRiskChecks:
    def test_drawdown_blocks_trading(self):
        calc = EdgeCalculator(
            config=EdgeConfig(min_edge=0.05, risk=RiskLimits(max_drawdown_pct=0.20)),
            portfolio_value=7500.0,
            peak_portfolio_value=10000.0,  # 25% drawdown
        )
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.90, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        assert signal.action == "SKIP"
        assert "drawdown" in signal.reason

    def test_no_drawdown_allows_trading(self):
        calc = EdgeCalculator(
            config=EdgeConfig(min_edge=0.05, risk=RiskLimits(max_drawdown_pct=0.20)),
            portfolio_value=9500.0,
            peak_portfolio_value=10000.0,  # 5% drawdown, under limit
        )
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.90, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        assert signal.action != "SKIP"

    def test_exposure_limit(self):
        calc = EdgeCalculator(
            config=EdgeConfig(
                min_edge=0.05,
                risk=RiskLimits(max_portfolio_exposure_pct=0.10),
            ),
            portfolio_value=10000.0,
            current_exposure=950.0,  # already near 10% limit
        )
        est = ProbabilityEstimate(market_id="mkt_1", probability=0.90, source="test")
        signal = calc.evaluate(est, market_price=0.50)
        assert signal.action == "SKIP"
        assert "exposure" in signal.reason


# ── Portfolio updates ─────────────────────────────────────────────────────────


class TestPortfolioUpdate:
    def test_update_portfolio(self):
        calc = EdgeCalculator(portfolio_value=1000.0)
        calc.update_portfolio(portfolio_value=1500.0, current_exposure=200.0)
        assert calc.portfolio_value == 1500.0
        assert calc.current_exposure == 200.0
        assert calc.peak_portfolio_value == 1500.0

    def test_peak_tracks_high_water(self):
        calc = EdgeCalculator(portfolio_value=1000.0)
        calc.update_portfolio(portfolio_value=1500.0, current_exposure=0)
        calc.update_portfolio(portfolio_value=1200.0, current_exposure=0)
        assert calc.peak_portfolio_value == 1500.0  # still the peak


# ── Batch evaluation ──────────────────────────────────────────────────────────


class TestBatchEval:
    def test_batch(self):
        calc = EdgeCalculator(
            config=EdgeConfig(min_edge=0.10),
            portfolio_value=10000.0,
        )
        estimates = [
            (ProbabilityEstimate(market_id="mkt_1", probability=0.85, source="test"), 0.65),
            (ProbabilityEstimate(market_id="mkt_2", probability=0.60, source="test"), 0.55),
            (ProbabilityEstimate(market_id="mkt_3", probability=0.50, source="test"), 0.50),
        ]
        signals = calc.evaluate_batch(estimates)
        assert len(signals) == 3
        actions = [s.action for s in signals]
        assert "BUY_YES" in actions  # mkt_1 has edge
        assert "SKIP" in actions     # mkt_3 has no edge


# ── Integration with paper engine ─────────────────────────────────────────────


class TestIntegration:
    def test_signal_to_paper_engine(self, tmp_path):
        from trading.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine(db_path=tmp_path / "test.db", initial_cash=10000.0)
        calc = EdgeCalculator(
            config=EdgeConfig(min_edge=0.10),
            portfolio_value=10000.0,
        )

        est = ProbabilityEstimate(market_id="btc_100k", probability=0.85, source="mirofish")
        signal = calc.evaluate(est, market_price=0.65)
        assert signal.action == "BUY_YES"

        # Execute the signal
        pos = engine.open_position(
            signal.market_id,
            signal.side,
            signal.shares,
            signal.price,
            reason=signal.reason,
            confidence=signal.our_probability,
        )
        assert pos.market_id == "btc_100k"
        assert pos.side == "YES"
        assert pos.shares == signal.shares
        assert pos.confidence == 0.85

        portfolio = engine.get_portfolio()
        assert len(portfolio.positions) == 1
        assert portfolio.cash < 10000.0
        engine.close()
