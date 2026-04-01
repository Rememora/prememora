"""Tests for the Hindsight Oracle backtesting module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backtesting.event_replay import collect_historical_events
from backtesting.polymarket_history import Market, PricePoint
from backtesting.hindsight import (
    BacktestConfig,
    BacktestReport,
    HindsightOracle,
    MarketBacktestResult,
    llm_estimate_probability,
    _strip_think_tags,
    _synthetic_estimate,
)
from trading.strategy_review import CalibrationBucket


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_prices():
    """Generate 48 hourly price points with a gradual uptrend."""
    base_ts = 1711699200  # 2024-03-29 08:00 UTC
    prices = []
    for i in range(48):
        price = 0.50 + i * 0.005  # 0.50 → 0.735
        prices.append(PricePoint(
            market_id="mkt_test",
            token_id="tok_yes",
            timestamp=base_ts + i * 3600,
            price=round(price, 4),
            interval="1h",
        ))
    return prices


@pytest.fixture
def sample_market():
    return Market(
        id="mkt_test",
        question="Will BTC hit $100k by June 2026?",
        description="Resolves YES if Bitcoin trades above $100,000.",
        category="crypto",
        outcomes='["Yes", "No"]',
        created_at="2026-01-01T00:00:00Z",
        resolved_at="2026-06-15T00:00:00Z",
        resolution="Yes",
        volume=500_000.0,
        clob_token_ids='["tok_yes", "tok_no"]',
    )


@pytest.fixture
def resolved_markets():
    """Multiple resolved markets for full backtest."""
    return [
        Market(
            id=f"mkt_{i}",
            question=f"Test market {i}?",
            description=f"Description {i}",
            category="crypto",
            outcomes='["Yes", "No"]',
            created_at="2026-01-01T00:00:00Z",
            resolved_at="2026-03-01T00:00:00Z",
            resolution="Yes" if i % 2 == 0 else "No",
            volume=100_000.0,
            clob_token_ids=f'["tok_{i}_yes", "tok_{i}_no"]',
        )
        for i in range(5)
    ]


# ── Event Replay ─────────────────────────────────────────────────────────────


class TestCollectHistoricalEvents:
    def test_produces_seed_event(self, sample_prices):
        events = collect_historical_events(
            market_question="Will BTC hit $100k?",
            market_description="Resolves YES if ...",
            prices=sample_prices,
            window_hours=24,
        )
        seed = [e for e in events if e["event_type"] == "market_open"]
        assert len(seed) == 1
        assert "Will BTC hit $100k?" in seed[0]["text"]
        assert seed[0]["source"] == "polymarket_metadata"

    def test_produces_price_movement_events(self, sample_prices):
        events = collect_historical_events(
            market_question="Test?",
            market_description="Desc",
            prices=sample_prices,
            window_hours=24,
        )
        movements = [e for e in events if e["event_type"] == "price_movement"]
        # 48 hours of data, 24h windows → should get 2 windows
        assert len(movements) == 2

    def test_events_are_chronological(self, sample_prices):
        events = collect_historical_events(
            market_question="Test?",
            market_description="",
            prices=sample_prices,
            window_hours=12,
        )
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)

    def test_events_have_required_keys(self, sample_prices):
        events = collect_historical_events(
            market_question="Test?",
            market_description="",
            prices=sample_prices,
            window_hours=24,
        )
        for event in events:
            assert "text" in event
            assert "timestamp" in event
            assert "source" in event
            assert "event_type" in event
            assert isinstance(event["text"], str)
            assert isinstance(event["timestamp"], int)

    def test_price_movement_has_ohlc(self, sample_prices):
        events = collect_historical_events(
            market_question="Test?",
            market_description="",
            prices=sample_prices,
            window_hours=24,
        )
        movements = [e for e in events if e["event_type"] == "price_movement"]
        for m in movements:
            assert "open" in m
            assert "close" in m
            assert "high" in m
            assert "low" in m
            assert m["high"] >= m["low"]

    def test_empty_prices(self):
        events = collect_historical_events(
            market_question="Test?",
            market_description="Desc",
            prices=[],
            window_hours=24,
        )
        assert len(events) == 1  # just the seed
        assert events[0]["event_type"] == "market_open"

    def test_single_price_point(self):
        prices = [PricePoint("mkt", "tok", 1711699200, 0.65, "1h")]
        events = collect_historical_events(
            market_question="Test?",
            market_description="",
            prices=prices,
            window_hours=24,
        )
        movements = [e for e in events if e["event_type"] == "price_movement"]
        assert len(movements) == 1

    def test_small_window(self, sample_prices):
        events = collect_historical_events(
            market_question="Test?",
            market_description="",
            prices=sample_prices,
            window_hours=6,
        )
        movements = [e for e in events if e["event_type"] == "price_movement"]
        # 48h data, 6h windows → 8 windows
        assert len(movements) == 8

    def test_direction_text(self, sample_prices):
        events = collect_historical_events(
            market_question="Test?",
            market_description="",
            prices=sample_prices,
            window_hours=24,
        )
        movements = [e for e in events if e["event_type"] == "price_movement"]
        # Prices trend upward, so second event should say "rose"
        assert "rose" in movements[1]["text"]


# ── LLM Estimator ───────────────────────────────────────────────────────────


class TestStripThinkTags:
    def test_strip(self):
        text = "Hello <think>reasoning here</think> world"
        assert _strip_think_tags(text) == "Hello  world"

    def test_no_tags(self):
        assert _strip_think_tags("just text") == "just text"

    def test_multiline_think(self):
        text = "Before <think>\nline1\nline2\n</think> After"
        assert _strip_think_tags(text) == "Before  After"


class TestSyntheticEstimate:
    def test_returns_float(self):
        result = _synthetic_estimate("Will BTC hit $100k?", 0.65)
        assert isinstance(result, float)
        assert 0.05 <= result <= 0.95

    def test_deterministic(self):
        a = _synthetic_estimate("Same question?", 0.50)
        b = _synthetic_estimate("Same question?", 0.50)
        assert a == b

    def test_different_questions_differ(self):
        a = _synthetic_estimate("Question A?", 0.50)
        b = _synthetic_estimate("Question B?", 0.50)
        # Very unlikely to be exactly equal
        assert a != b


class TestLLMEstimateProbability:
    @pytest.mark.asyncio
    async def test_fallback_without_api_key(self):
        """Without MINIMAX_API_KEY, should return a synthetic estimate."""
        with patch.dict("os.environ", {"MINIMAX_API_KEY": ""}, clear=False):
            result = await llm_estimate_probability(
                question="Will BTC hit $100k?",
                context_facts=["BTC at $95k"],
                market_price=0.65,
            )
        assert result is not None
        assert 0.05 <= result <= 0.95

    @pytest.mark.asyncio
    async def test_successful_llm_call(self):
        """Mock a successful MiniMax API response."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "<think>Let me think...</think> Based on analysis, Probability: 72%",
                }
            }]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"MINIMAX_API_KEY": "test_key"}, clear=False):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await llm_estimate_probability(
                    question="Will BTC hit $100k?",
                    context_facts=["BTC at $95k"],
                    market_price=0.65,
                )

        assert result is not None
        assert abs(result - 0.72) < 0.01

    @pytest.mark.asyncio
    async def test_api_error_falls_back(self):
        """API returning non-200 should fall back to synthetic."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"MINIMAX_API_KEY": "test_key"}, clear=False):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await llm_estimate_probability(
                    question="Test?",
                    context_facts=[],
                    market_price=0.50,
                )

        assert result is not None
        assert 0.05 <= result <= 0.95

    @pytest.mark.asyncio
    async def test_context_facts_in_prompt(self):
        """Verify context facts are included in the API call."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": "Probability: 80%",
                }
            }]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"MINIMAX_API_KEY": "test_key"}, clear=False):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                await llm_estimate_probability(
                    question="Will BTC hit $100k?",
                    context_facts=["BTC surged to $95k", "Whale alert: 5000 BTC moved"],
                    market_price=0.65,
                )

        # Verify the call was made with context in the payload
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        user_msg = payload["messages"][1]["content"]
        assert "BTC surged to $95k" in user_msg
        assert "knowledge graph" in user_msg


# ── Backtest Report ──────────────────────────────────────────────────────────


class TestBacktestReport:
    def test_summary_formatting(self):
        report = BacktestReport(
            config=BacktestConfig(use_graph=True),
            results=[
                MarketBacktestResult(
                    market_id="mkt_1",
                    question="Will BTC hit $100k?",
                    actual_outcome="YES",
                    market_price_at_eval=0.65,
                    our_probability=0.80,
                    edge=0.15,
                    action="BUY_YES",
                    pnl=35.0,
                    graph_facts_used=5,
                ),
                MarketBacktestResult(
                    market_id="mkt_2",
                    question="Will ETH hit $5k?",
                    actual_outcome="NO",
                    market_price_at_eval=0.70,
                    our_probability=0.75,
                    edge=0.05,
                    action="SKIP",
                    pnl=0.0,
                    graph_facts_used=3,
                ),
            ],
            brier_score=0.15,
            calibration=[],
            total_pnl=35.0,
            win_rate=1.0,
            portfolio_final=10035.0,
            duration_s=12.5,
        )
        summary = report.summary
        assert "HINDSIGHT ORACLE" in summary
        assert "Graph enabled:     yes" in summary
        assert "0.1500" in summary  # brier score
        assert "BUY_YES" in summary
        assert "$100k" in summary

    def test_summary_with_no_trades(self):
        report = BacktestReport(
            config=BacktestConfig(use_graph=False),
            results=[],
            brier_score=None,
            calibration=[],
            total_pnl=0.0,
            win_rate=None,
            portfolio_final=10000.0,
            duration_s=1.0,
        )
        summary = report.summary
        assert "HINDSIGHT ORACLE" in summary
        assert "Graph enabled:     no" in summary
        assert "0 markets" not in summary or "Markets tested:    0" in summary

    def test_summary_with_calibration(self):
        report = BacktestReport(
            config=BacktestConfig(),
            results=[],
            brier_score=0.20,
            calibration=[
                CalibrationBucket(
                    bucket_low=0.0,
                    bucket_high=0.5,
                    predicted_mean=0.3,
                    actual_win_rate=0.4,
                    count=10,
                    error=-0.1,
                ),
            ],
            total_pnl=0.0,
            win_rate=None,
            portfolio_final=10000.0,
            duration_s=5.0,
        )
        summary = report.summary
        assert "Calibration" in summary
        assert "n=10" in summary


# ── HindsightOracle ─────────────────────────────────────────────────────────


class TestHindsightOracle:
    @pytest.mark.asyncio
    async def test_run_no_markets(self, tmp_path):
        """When no markets match, should return empty report."""
        with patch(
            "backtesting.hindsight.discover_markets",
            new_callable=AsyncMock,
            return_value=[],
        ):
            oracle = HindsightOracle()
            cfg = BacktestConfig(
                max_markets=5,
                db_path=tmp_path / "test.db",
            )
            report = await oracle.run(cfg)

        assert len(report.results) == 0
        assert report.total_pnl == 0.0
        assert report.portfolio_final == cfg.initial_cash

    @pytest.mark.asyncio
    async def test_run_with_mocked_apis(self, tmp_path, resolved_markets, sample_prices):
        """Full run with mocked Gamma and CLOB APIs."""

        async def mock_discover(**kwargs):
            return resolved_markets

        async def mock_fetch_prices(market_id, token_id, **kwargs):
            # Return prices with market-specific IDs
            return [
                PricePoint(
                    market_id=market_id,
                    token_id=token_id,
                    timestamp=p.timestamp,
                    price=p.price,
                    interval=p.interval,
                )
                for p in sample_prices
            ]

        with patch(
            "backtesting.hindsight.discover_markets",
            side_effect=mock_discover,
        ), patch(
            "backtesting.hindsight.fetch_price_history",
            side_effect=mock_fetch_prices,
        ), patch(
            "backtesting.hindsight.llm_estimate_probability",
            new_callable=AsyncMock,
            return_value=0.70,
        ):
            oracle = HindsightOracle()
            cfg = BacktestConfig(
                max_markets=5,
                db_path=tmp_path / "test.db",
                use_graph=False,
                initial_cash=10_000.0,
            )
            report = await oracle.run(cfg)

        assert len(report.results) == 5
        assert report.duration_s > 0
        # At least some should have probability estimates
        estimated = [r for r in report.results if r.our_probability is not None]
        assert len(estimated) > 0

    @pytest.mark.asyncio
    async def test_run_with_graph_enabled(self, tmp_path, resolved_markets, sample_prices):
        """Graph-enabled run produces context facts."""

        async def mock_discover(**kwargs):
            return resolved_markets

        async def mock_fetch_prices(market_id, token_id, **kwargs):
            return [
                PricePoint(
                    market_id=market_id,
                    token_id=token_id,
                    timestamp=p.timestamp,
                    price=p.price,
                    interval=p.interval,
                )
                for p in sample_prices
            ]

        with patch(
            "backtesting.hindsight.discover_markets",
            side_effect=mock_discover,
        ), patch(
            "backtesting.hindsight.fetch_price_history",
            side_effect=mock_fetch_prices,
        ), patch(
            "backtesting.hindsight.llm_estimate_probability",
            new_callable=AsyncMock,
            return_value=0.75,
        ):
            oracle = HindsightOracle()
            cfg = BacktestConfig(
                max_markets=3,
                db_path=tmp_path / "test_graph.db",
                use_graph=True,
                initial_cash=10_000.0,
            )
            report = await oracle.run(cfg)

        assert len(report.results) == 3
        # With graph enabled, should have context facts
        facts_used = sum(r.graph_facts_used for r in report.results)
        assert facts_used > 0

    @pytest.mark.asyncio
    async def test_handles_price_fetch_failure(self, tmp_path):
        """Markets with failed price fetches should be skipped gracefully."""
        market = Market(
            id="mkt_fail",
            question="Test fail?",
            description="",
            category="",
            outcomes='["Yes", "No"]',
            created_at="2026-01-01T00:00:00Z",
            resolved_at="2026-03-01T00:00:00Z",
            resolution="Yes",
            volume=100_000.0,
            clob_token_ids='["tok_fail"]',
        )

        async def mock_discover(**kwargs):
            return [market]

        async def mock_fetch_prices(**kwargs):
            raise aiohttp.ClientError("Connection refused")

        import aiohttp

        with patch(
            "backtesting.hindsight.discover_markets",
            side_effect=mock_discover,
        ), patch(
            "backtesting.hindsight.fetch_price_history",
            side_effect=mock_fetch_prices,
        ), patch(
            "backtesting.hindsight.llm_estimate_probability",
            new_callable=AsyncMock,
            return_value=0.60,
        ):
            oracle = HindsightOracle()
            cfg = BacktestConfig(
                max_markets=1,
                db_path=tmp_path / "test_fail.db",
                use_graph=False,
            )
            report = await oracle.run(cfg)

        # Should still produce a result (with fallback price)
        assert len(report.results) == 1

    @pytest.mark.asyncio
    async def test_scoring_produces_brier_score(self, tmp_path, sample_prices):
        """With enough estimates, should produce a Brier score."""
        markets = [
            Market(
                id=f"mkt_s{i}",
                question=f"Scoring test {i}?",
                description="",
                category="",
                outcomes='["Yes", "No"]',
                created_at="2026-01-01T00:00:00Z",
                resolved_at="2026-03-01T00:00:00Z",
                resolution="Yes" if i < 3 else "No",
                volume=100_000.0,
                clob_token_ids=f'["tok_s{i}"]',
            )
            for i in range(5)
        ]

        async def mock_discover(**kwargs):
            return markets

        async def mock_fetch_prices(market_id, token_id, **kwargs):
            return [
                PricePoint(market_id=market_id, token_id=token_id,
                           timestamp=p.timestamp, price=p.price, interval=p.interval)
                for p in sample_prices
            ]

        with patch(
            "backtesting.hindsight.discover_markets",
            side_effect=mock_discover,
        ), patch(
            "backtesting.hindsight.fetch_price_history",
            side_effect=mock_fetch_prices,
        ), patch(
            "backtesting.hindsight.llm_estimate_probability",
            new_callable=AsyncMock,
            return_value=0.65,
        ):
            oracle = HindsightOracle()
            cfg = BacktestConfig(
                max_markets=5,
                db_path=tmp_path / "test_score.db",
                use_graph=False,
            )
            report = await oracle.run(cfg)

        # Should have a brier score since all markets have probabilities
        assert report.brier_score is not None
        assert 0 <= report.brier_score <= 1


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestGetEvalPrice:
    def test_eval_price_from_history(self, sample_prices):
        oracle = HindsightOracle()
        price = oracle._get_eval_price(sample_prices, hours_before_close=24)
        assert price is not None
        assert 0.01 <= price <= 0.99

    def test_eval_price_empty(self):
        oracle = HindsightOracle()
        price = oracle._get_eval_price([], hours_before_close=24)
        assert price is None

    def test_eval_price_single_point(self):
        prices = [PricePoint("mkt", "tok", 1711699200, 0.65, "1h")]
        oracle = HindsightOracle()
        price = oracle._get_eval_price(prices, hours_before_close=24)
        assert price == 0.65

    def test_eval_price_clamped(self):
        # Price exactly 0 should be clamped to 0.01
        prices = [PricePoint("mkt", "tok", 1711699200, 0.0, "1h")]
        oracle = HindsightOracle()
        price = oracle._get_eval_price(prices, hours_before_close=0)
        assert price == 0.01
