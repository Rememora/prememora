"""Tests for the graph context builder."""

import pytest

from pipeline.context import (
    ContextBuilder,
    MarketContext,
    build_enriched_prompt,
    extract_search_terms,
)


# ── Search term extraction ────────────────────────────────────────────────────


class TestExtractSearchTerms:
    def test_bitcoin_question(self):
        terms = extract_search_terms("Will Bitcoin exceed $100,000 by April 2026?")
        lower = [t.lower() for t in terms]
        assert "bitcoin" in lower

    def test_fed_question(self):
        terms = extract_search_terms("Will the Federal Reserve cut interest rates in June?")
        # Should capture "Federal Reserve" as a compound
        assert any("Federal Reserve" in t for t in terms)

    def test_ethereum_with_entity(self):
        terms = extract_search_terms("Will Ethereum price go above $5,000?")
        lower = [t.lower() for t in terms]
        assert "ethereum" in lower

    def test_removes_stop_words(self):
        terms = extract_search_terms("Will the price of Bitcoin go above the current level?")
        lower = [t.lower() for t in terms]
        assert "the" not in lower
        assert "will" not in lower
        assert "of" not in lower

    def test_preserves_proper_nouns(self):
        terms = extract_search_terms("Will Donald Trump win the election?")
        assert any("Donald Trump" in t for t in terms) or (
            any("donald" in t.lower() for t in terms) and
            any("trump" in t.lower() for t in terms)
        )

    def test_empty_question(self):
        assert extract_search_terms("") == []

    def test_all_stop_words(self):
        terms = extract_search_terms("Will this be the one?")
        # All meaningful words are stop words
        assert len(terms) == 0 or all(len(t) > 2 for t in terms)

    def test_deduplication(self):
        terms = extract_search_terms("Bitcoin Bitcoin BTC bitcoin")
        lower = [t.lower() for t in terms]
        assert lower.count("bitcoin") == 1

    def test_multi_entity(self):
        terms = extract_search_terms("Will Elon Musk buy Twitter?")
        assert any("Elon Musk" in t for t in terms) or any("elon" in t.lower() for t in terms)
        assert any("twitter" in t.lower() for t in terms)


# ── MarketContext ─────────────────────────────────────────────────────────────


class TestMarketContext:
    def test_with_facts(self):
        ctx = MarketContext(
            question="Will BTC hit $100k?",
            search_terms=["BTC"],
            facts=["BTC surged 5%", "Whale moved 10k BTC"],
        )
        assert ctx.fact_count == 2
        assert "BTC surged 5%" in ctx.prompt_section
        assert "Whale moved 10k BTC" in ctx.prompt_section
        assert "recent facts" in ctx.prompt_section

    def test_without_facts(self):
        ctx = MarketContext(
            question="Will BTC hit $100k?",
            search_terms=["BTC"],
            facts=[],
        )
        assert ctx.fact_count == 0
        assert "no specific recent intelligence" in ctx.prompt_section


# ── Enriched prompt ───────────────────────────────────────────────────────────


class TestBuildEnrichedPrompt:
    def test_with_context(self):
        ctx = MarketContext(
            question="Will BTC exceed $100k?",
            search_terms=["BTC"],
            facts=["Bitcoin surged past $95k", "ETF inflows hit $2B"],
        )
        prompt = build_enriched_prompt("Will BTC exceed $100,000?", ctx)
        # Should contain the facts
        assert "Bitcoin surged past $95k" in prompt
        assert "ETF inflows hit $2B" in prompt
        # Should contain the probability question
        assert "probability" in prompt.lower()
        assert "Will BTC exceed $100,000?" in prompt

    def test_without_context(self):
        ctx = MarketContext(
            question="Will BTC exceed $100k?",
            search_terms=["BTC"],
            facts=[],
        )
        prompt = build_enriched_prompt("Will BTC exceed $100,000?", ctx)
        # Should still ask for probability
        assert "probability" in prompt.lower()
        # No facts section
        assert "recent facts" not in prompt

    def test_step_by_step_instruction(self):
        ctx = MarketContext(
            question="test", search_terms=["test"],
            facts=["fact1"],
        )
        prompt = build_enriched_prompt("test?", ctx)
        assert "step by step" in prompt.lower()


# ── ContextBuilder unit tests (mocked graph) ─────────────────────────────────


class TestContextBuilderUnit:
    def test_build_context_no_terms(self):
        """If no search terms extracted, return empty context."""
        builder = ContextBuilder(graph_id="test")
        # Override _get_client to avoid real connection
        builder._client = None
        ctx = builder.build_context("the")  # all stop words
        assert ctx.facts == []
        assert ctx.search_terms == []
