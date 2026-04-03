"""Tests for the pipeline trigger."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.trigger import (
    ActiveMarket,
    PipelineConfig,
    PipelineTrigger,
    aggregate_probabilities,
    parse_probability,
)


# ── Probability parsing ──────────────────────────────────────────────────────


class TestParseProbability:
    def test_percentage(self):
        assert parse_probability("I think there's a 70% chance") == 0.70

    def test_percentage_with_decimal(self):
        assert parse_probability("About 72.5%") == 0.72  # int match only in pattern

    def test_percent_word(self):
        assert parse_probability("around 65 percent") == 0.65

    def test_probability_colon(self):
        assert parse_probability("probability: 0.75") == 0.75

    def test_probability_of(self):
        assert parse_probability("probability of 0.80") == 0.80

    def test_range_midpoint(self):
        assert parse_probability("I'd say 60-80%") == 0.70

    def test_bare_decimal(self):
        assert parse_probability("I estimate 0.65 for this outcome") == 0.65

    def test_high_value_normalized(self):
        # "probability: 75" → should be 75/100 = 0.75
        assert parse_probability("probability: 75") == 0.75

    def test_no_match(self):
        assert parse_probability("I have no idea about this market") is None

    def test_empty_string(self):
        assert parse_probability("") is None

    def test_none_input(self):
        assert parse_probability(None) is None

    def test_case_insensitive(self):
        assert parse_probability("PROBABILITY: 0.82") == 0.82


# ── Aggregation ───────────────────────────────────────────────────────────────


class TestAggregateProbabilities:
    def test_simple_mean(self):
        # < 4 agents → simple mean
        result = aggregate_probabilities([0.70, 0.80])
        assert abs(result - 0.75) < 1e-10

    def test_single(self):
        assert aggregate_probabilities([0.65]) == 0.65

    def test_trimmed_mean(self):
        # >= 4 agents → drop min and max
        # [0.50, 0.60, 0.70, 0.80, 0.90] → trim → [0.60, 0.70, 0.80] → mean 0.70
        result = aggregate_probabilities([0.50, 0.60, 0.70, 0.80, 0.90])
        assert abs(result - 0.70) < 1e-10

    def test_trimmed_removes_outliers(self):
        # One very low outlier
        result = aggregate_probabilities([0.10, 0.70, 0.72, 0.68, 0.75])
        # Trimmed: [0.68, 0.70, 0.72] → mean ~0.70
        assert 0.68 <= result <= 0.72

    def test_empty(self):
        assert aggregate_probabilities([]) is None

    def test_four_agents(self):
        # Exactly 4 → trim → 2 middle values
        result = aggregate_probabilities([0.50, 0.60, 0.70, 0.80])
        assert abs(result - 0.65) < 1e-10


# ── Relevance filter ──────────────────────────────────────────────────────────


class TestRelevanceFilter:
    def _make_market(self, question):
        return ActiveMarket(
            id="0x1", question=question, token_ids=["t1"],
            current_price=0.5, volume=100, category="",
        )

    def test_crypto_market_relevant(self):
        config = PipelineConfig(relevance_keywords=["bitcoin", "ethereum"])
        trigger = PipelineTrigger(config=config)
        assert trigger._is_market_relevant(self._make_market("Will Bitcoin hit $100k?"))

    def test_sports_market_not_relevant(self):
        config = PipelineConfig(relevance_keywords=["bitcoin", "ethereum"])
        trigger = PipelineTrigger(config=config)
        assert not trigger._is_market_relevant(self._make_market("Will the Lakers win tonight?"))

    def test_case_insensitive(self):
        config = PipelineConfig(relevance_keywords=["bitcoin"])
        trigger = PipelineTrigger(config=config)
        assert trigger._is_market_relevant(self._make_market("BITCOIN price prediction"))

    def test_empty_keywords_allows_all(self):
        config = PipelineConfig(relevance_keywords=[])
        trigger = PipelineTrigger(config=config)
        assert trigger._is_market_relevant(self._make_market("Anything at all"))

    def test_geopolitics_relevant(self):
        config = PipelineConfig(relevance_keywords=["war", "iran", "israel"])
        trigger = PipelineTrigger(config=config)
        assert trigger._is_market_relevant(self._make_market("Will Israel strike Iran?"))

    def test_word_in_sentence(self):
        config = PipelineConfig(relevance_keywords=["bitcoin"])
        trigger = PipelineTrigger(config=config)
        assert trigger._is_market_relevant(self._make_market("Price of bitcoin on April 5"))

    def test_no_substring_false_positive(self):
        """'ai' should not match inside 'Taishan'."""
        config = PipelineConfig(relevance_keywords=["ai"])
        trigger = PipelineTrigger(config=config)
        assert not trigger._is_market_relevant(self._make_market("Shandong Taishan FC (-1.5)"))

    def test_multi_word_keyword(self):
        config = PipelineConfig(relevance_keywords=["interest rate"])
        trigger = PipelineTrigger(config=config)
        assert trigger._is_market_relevant(self._make_market("Will interest rate stay at 4%?"))


# ── Active market ─────────────────────────────────────────────────────────────


class TestActiveMarket:
    def test_create(self):
        m = ActiveMarket(
            id="0xabc",
            question="Will BTC hit $100k?",
            token_ids=["tok_yes", "tok_no"],
            current_price=0.65,
            volume=1_000_000,
            category="crypto",
        )
        assert m.question == "Will BTC hit $100k?"
        assert m.current_price == 0.65


# ── Pipeline trigger ──────────────────────────────────────────────────────────


class TestPipelineTrigger:
    @pytest.fixture
    def config(self, tmp_path):
        return PipelineConfig(
            mirofish_url="http://localhost:5001",
            simulation_id="sim_test",
            signal_log_path=tmp_path / "signals.jsonl",
        )

    @pytest.mark.asyncio
    async def test_run_once_no_markets(self, config):
        trigger = PipelineTrigger(config=config)
        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            signals = await trigger.run_once()
            assert signals == []

    @pytest.mark.asyncio
    async def test_run_once_no_simulation_id(self, config, tmp_path):
        config.simulation_id = ""
        trigger = PipelineTrigger(config=config)

        market = ActiveMarket(
            id="0xabc", question="Will X happen?", token_ids=["t1"],
            current_price=0.65, volume=100000, category="test",
        )
        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [market]
            signals = await trigger.run_once()
            assert len(signals) == 1
            assert signals[0]["action"] == "SKIP"
            assert "no simulation_id" in signals[0]["reason"]

    @pytest.mark.asyncio
    async def test_run_once_with_responses(self, config, tmp_path):
        trigger = PipelineTrigger(config=config)

        market = ActiveMarket(
            id="0xabc", question="Will BTC hit $100k?", token_ids=["t1"],
            current_price=0.55, volume=100000, category="crypto",
        )

        mock_responses = [
            {"response": "I think there's an 80% chance", "agent_id": 0},
            {"response": "probability: 0.85", "agent_id": 1},
            {"response": "Around 75%", "agent_id": 2},
        ]

        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch, \
             patch("pipeline.trigger.interview_agents", new_callable=AsyncMock) as mock_interview:
            mock_fetch.return_value = [market]
            mock_interview.return_value = mock_responses

            signals = await trigger.run_once()
            assert len(signals) == 1
            s = signals[0]
            assert s["our_probability"] is not None
            assert 0.75 <= s["our_probability"] <= 0.85
            # Edge = ~0.80 - 0.55 = ~0.25, should trigger BUY_YES
            assert s["action"] == "BUY_YES"

    @pytest.mark.asyncio
    async def test_run_once_unparseable_responses(self, config, tmp_path):
        trigger = PipelineTrigger(config=config)

        market = ActiveMarket(
            id="0xabc", question="Will X happen?", token_ids=["t1"],
            current_price=0.65, volume=100000, category="test",
        )

        mock_responses = [
            {"response": "I really don't know about this", "agent_id": 0},
            {"response": "It's complicated", "agent_id": 1},
        ]

        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch, \
             patch("pipeline.trigger.interview_agents", new_callable=AsyncMock) as mock_interview:
            mock_fetch.return_value = [market]
            mock_interview.return_value = mock_responses

            signals = await trigger.run_once()
            assert len(signals) == 1
            assert signals[0]["action"] == "SKIP"
            assert "could not parse" in signals[0]["reason"]

    @pytest.mark.asyncio
    async def test_signal_logged(self, config, tmp_path):
        trigger = PipelineTrigger(config=config)

        market = ActiveMarket(
            id="0xabc", question="Test?", token_ids=["t1"],
            current_price=0.55, volume=100000, category="test",
        )
        mock_responses = [{"response": "85%", "agent_id": 0}]

        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch, \
             patch("pipeline.trigger.interview_agents", new_callable=AsyncMock) as mock_interview:
            mock_fetch.return_value = [market]
            mock_interview.return_value = mock_responses

            await trigger.run_once()

        log_path = config.signal_log_path
        assert log_path.exists()
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["market_id"] == "0xabc"

    @pytest.mark.asyncio
    async def test_trade_executed(self, config, tmp_path):
        from trading.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine(db_path=tmp_path / "test.db", initial_cash=10000.0)
        trigger = PipelineTrigger(config=config, paper_engine=engine)

        market = ActiveMarket(
            id="0xabc", question="Will BTC hit $100k?", token_ids=["t1"],
            current_price=0.55, volume=100000, category="crypto",
        )
        mock_responses = [
            {"response": "80%", "agent_id": 0},
            {"response": "75%", "agent_id": 1},
            {"response": "85%", "agent_id": 2},
        ]

        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch, \
             patch("pipeline.trigger.interview_agents", new_callable=AsyncMock) as mock_interview:
            mock_fetch.return_value = [market]
            mock_interview.return_value = mock_responses

            signals = await trigger.run_once()

        s = signals[0]
        assert s.get("executed") is True
        assert s.get("position_id") is not None

        portfolio = engine.get_portfolio()
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].market_id == "0xabc"
        engine.close()

    @pytest.mark.asyncio
    async def test_dry_run_no_execution(self, config, tmp_path):
        trigger = PipelineTrigger(config=config, paper_engine=None)

        market = ActiveMarket(
            id="0xabc", question="Test?", token_ids=["t1"],
            current_price=0.55, volume=100000, category="crypto",
        )
        mock_responses = [{"response": "85%", "agent_id": 0}]

        with patch("pipeline.trigger.fetch_active_markets", new_callable=AsyncMock) as mock_fetch, \
             patch("pipeline.trigger.interview_agents", new_callable=AsyncMock) as mock_interview:
            mock_fetch.return_value = [market]
            mock_interview.return_value = mock_responses

            signals = await trigger.run_once()

        assert signals[0]["action"] == "BUY_YES"
        assert signals[0]["executed"] is False
