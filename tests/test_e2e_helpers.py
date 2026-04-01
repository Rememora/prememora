"""Tests for e2e.helpers — synthetic probabilities, report formatting, timing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from e2e.helpers import (
    E2EReport,
    ResolvedMarket,
    StageResult,
    fetch_resolved_markets,
    generate_synthetic_probability,
    timed_stage,
)


# ── Synthetic probability tests ──────────────────────────────────────────────


class TestSyntheticProbability:
    def test_deterministic(self):
        """Same question + price + outcome always gives same probability."""
        p1 = generate_synthetic_probability("Will BTC hit 100k?", 0.7, "YES")
        p2 = generate_synthetic_probability("Will BTC hit 100k?", 0.7, "YES")
        assert p1 == p2

    def test_different_questions_different_probs(self):
        """Different questions produce different probabilities."""
        p1 = generate_synthetic_probability("Will BTC hit 100k?", 0.5, "YES")
        p2 = generate_synthetic_probability("Will ETH hit 10k?", 0.5, "YES")
        assert p1 != p2

    def test_bounded(self):
        """Probabilities are always between 0.05 and 0.95."""
        for q in ["a", "b", "c", "question about markets", "xyz123"]:
            for outcome in ["YES", "NO", None]:
                p = generate_synthetic_probability(q, 0.5, outcome)
                assert 0.05 <= p <= 0.95, f"Out of bounds: {p} for {q}/{outcome}"

    def test_smoke_mode_biased_toward_correct(self):
        """When outcome is known, probability should tend toward correct answer."""
        yes_probs = []
        no_probs = []
        for i in range(50):
            q = f"test question {i}"
            yes_probs.append(generate_synthetic_probability(q, 0.5, "YES"))
            no_probs.append(generate_synthetic_probability(q, 0.5, "NO"))

        # On average, YES-outcome probs should be higher than NO-outcome probs
        assert sum(yes_probs) / len(yes_probs) > sum(no_probs) / len(no_probs)

    def test_live_mode_around_market_price(self):
        """Without outcome, probability should be near market price."""
        probs = [generate_synthetic_probability(f"q{i}", 0.6, None) for i in range(50)]
        avg = sum(probs) / len(probs)
        # Should be within ±0.2 of market price on average
        assert 0.4 <= avg <= 0.8


# ── Report tests ─────────────────────────────────────────────────────────────


class TestE2EReport:
    def test_all_passed_true(self):
        report = E2EReport(mode="smoke")
        report.stages = [
            StageResult(name="A", success=True, duration_s=1.0),
            StageResult(name="B", success=True, duration_s=2.0),
        ]
        assert report.all_passed is True

    def test_all_passed_false(self):
        report = E2EReport(mode="smoke")
        report.stages = [
            StageResult(name="A", success=True, duration_s=1.0),
            StageResult(name="B", success=False, duration_s=2.0, error="boom"),
        ]
        assert report.all_passed is False

    def test_summary_contains_mode(self):
        report = E2EReport(mode="smoke")
        report.stages = [StageResult(name="Test", success=True, duration_s=0.5)]
        assert "smoke" in report.summary

    def test_summary_shows_pnl(self):
        report = E2EReport(mode="smoke", total_pnl=42.50)
        report.stages = [StageResult(name="Test", success=True, duration_s=0.5)]
        assert "$+42.50" in report.summary or "+42.50" in report.summary

    def test_summary_shows_brier(self):
        report = E2EReport(mode="smoke", brier_score=0.1234)
        report.stages = [StageResult(name="Test", success=True, duration_s=0.5)]
        assert "0.1234" in report.summary


# ── Timed stage tests ────────────────────────────────────────────────────────


class TestTimedStage:
    def test_successful_stage(self):
        report = E2EReport(mode="test")
        with timed_stage("my stage", report) as stage:
            stage.detail = "did something"
        assert len(report.stages) == 1
        assert report.stages[0].success is True
        assert report.stages[0].name == "my stage"
        assert report.stages[0].duration_s >= 0

    def test_failed_stage(self):
        report = E2EReport(mode="test")
        with timed_stage("bad stage", report) as stage:
            raise ValueError("test error")
        assert len(report.stages) == 1
        assert report.stages[0].success is False
        assert "test error" in report.stages[0].error


# ── Fetch resolved markets tests ─────────────────────────────────────────────


class TestFetchResolvedMarkets:
    def test_parses_response(self):
        mock_data = [
            {
                "conditionId": "cond_123",
                "question": "Will BTC hit 100k?",
                "outcomePrices": '["1","0"]',
                "outcomes": '["Yes","No"]',
                "resolvedBy": "0xABC123",
                "closedTime": "2026-01-01T00:00:00Z",
                "volume": "50000",
                "lastTradePrice": 0.85,
                "groupSlug": "crypto",
            }
        ]

        async def run():
            with patch("e2e.helpers.aiohttp.ClientSession") as mock_session_cls:
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(return_value=mock_data)
                mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_resp.__aexit__ = AsyncMock(return_value=False)

                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                mock_session_cls.return_value = mock_session

                markets = await fetch_resolved_markets(max_markets=5)
                assert len(markets) == 1
                assert markets[0].condition_id == "cond_123"
                assert markets[0].outcome == "YES"
                assert 0.55 <= markets[0].yes_price <= 0.85  # synthetic price for YES outcome

        asyncio.run(run())

    def test_filters_low_volume(self):
        mock_data = [
            {
                "conditionId": "low_vol",
                "question": "Low volume market?",
                "outcomePrices": '["1","0"]',
                "outcomes": '["Yes","No"]',
                "resolvedBy": "0xABC123",
                "closedTime": "2026-01-01T00:00:00Z",
                "volume": "10",  # below default min_volume of 100
                "lastTradePrice": 0.5,
            }
        ]

        async def run():
            with patch("e2e.helpers.aiohttp.ClientSession") as mock_session_cls:
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(return_value=mock_data)
                mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_resp.__aexit__ = AsyncMock(return_value=False)

                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                mock_session_cls.return_value = mock_session

                markets = await fetch_resolved_markets(max_markets=5)
                assert len(markets) == 0

        asyncio.run(run())
