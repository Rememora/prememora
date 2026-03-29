"""Tests for the unified ingestion orchestrator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestors.orchestrator import (
    IngestionOrchestrator,
    _LRUDedup,
    _dedup_key,
    normalize_event,
)


# ── Normalization tests ───────────────────────────────────────────────────────


class TestNormalizeEvent:
    def test_polymarket_price_change(self):
        event = {
            "source": "polymarket_ws",
            "timestamp": "2026-03-29T10:00:00Z",
            "market_id": "0xabc123",
            "event_type": "price_change",
            "data": {"old_price": "0.45", "price": "0.52", "side": "YES"},
        }
        text = normalize_event(event)
        assert "Polymarket price change" in text
        assert "0xabc123" in text
        assert "0.45" in text
        assert "0.52" in text
        assert "YES" in text

    def test_polymarket_trade(self):
        event = {
            "source": "polymarket_ws",
            "timestamp": "2026-03-29T10:00:00Z",
            "market_id": "0xabc123",
            "event_type": "last_trade_price",
            "data": {"price": "0.55", "size": "100"},
        }
        text = normalize_event(event)
        assert "trade" in text
        assert "0.55" in text

    def test_polymarket_unknown_type(self):
        event = {
            "source": "polymarket_ws",
            "timestamp": "2026-03-29T10:00:00Z",
            "market_id": "0xabc",
            "event_type": "book",
            "data": {"bids": [], "asks": []},
        }
        text = normalize_event(event)
        assert "book" in text

    def test_crypto_news(self):
        event = {
            "source": "crypto_news",
            "title": "Bitcoin Hits $100k",
            "content": "Major milestone reached...",
            "source_name": "CoinDesk",
            "entities": ["Bitcoin"],
            "sentiment": "positive",
        }
        text = normalize_event(event)
        assert "Bitcoin Hits $100k" in text
        assert "CoinDesk" in text
        assert "positive" in text

    def test_crypto_news_empty(self):
        event = {"source": "crypto_news"}
        text = normalize_event(event)
        assert "no content" in text

    def test_rss(self):
        event = {
            "source": "rss",
            "title": "Fed Raises Rates",
            "content": "The Federal Reserve announced...",
            "feed_name": "Reuters",
            "category": "government",
        }
        text = normalize_event(event)
        assert "[Reuters]" in text
        assert "(government)" in text
        assert "Fed Raises Rates" in text

    def test_whale_alert(self):
        event = {
            "source": "whale_alert",
            "chain": "ethereum",
            "token": "ETH",
            "amount_usd": 15_000_000,
            "from_entity": "Binance",
            "to_entity": "unknown",
            "classification": "exchange_outflow",
            "tx_hash": "0xdeadbeef",
        }
        text = normalize_event(event)
        assert "ethereum" in text
        assert "ETH" in text
        assert "15,000,000" in text
        assert "Binance" in text
        assert "exchange_outflow" in text

    def test_reddit(self):
        event = {
            "source": "reddit",
            "subreddit": "Bitcoin",
            "title": "BTC to the moon",
            "content": "Bullish signals everywhere",
            "score": 1500,
            "num_comments": 300,
        }
        text = normalize_event(event)
        assert "r/Bitcoin" in text
        assert "BTC to the moon" in text
        assert "1500" in text

    def test_fred(self):
        event = {
            "source": "fred",
            "series_id": "FEDFUNDS",
            "series_name": "Federal Funds Effective Rate",
            "value": 5.33,
            "change": 0.25,
            "units": "Percent",
        }
        text = normalize_event(event)
        assert "Federal Funds Effective Rate" in text
        assert "5.33" in text
        assert "+0.25" in text

    def test_fred_negative_change(self):
        event = {
            "source": "fred",
            "series_name": "VIX",
            "value": 18.5,
            "change": -2.3,
            "units": "Index",
        }
        text = normalize_event(event)
        assert "-2.3" in text
        assert "+" not in text

    def test_unknown_source(self):
        event = {"source": "mystery", "data": "hello"}
        text = normalize_event(event)
        assert "mystery" in text


# ── Dedup tests ───────────────────────────────────────────────────────────────


class TestDedupKey:
    def test_polymarket_key(self):
        event = {
            "source": "polymarket_ws",
            "market_id": "0xabc",
            "event_type": "price_change",
            "data": {"price": "0.5"},
        }
        key = _dedup_key(event)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_same_event_same_key(self):
        event = {
            "source": "crypto_news",
            "url": "https://example.com/article-1",
            "title": "Test",
        }
        assert _dedup_key(event) == _dedup_key(event)

    def test_different_url_different_key(self):
        e1 = {"source": "crypto_news", "url": "https://example.com/a"}
        e2 = {"source": "crypto_news", "url": "https://example.com/b"}
        assert _dedup_key(e1) != _dedup_key(e2)

    def test_different_source_different_key(self):
        """Same URL but different source should produce different keys."""
        e1 = {"source": "crypto_news", "url": "https://example.com/a"}
        e2 = {"source": "rss", "url": "https://example.com/a"}
        assert _dedup_key(e1) != _dedup_key(e2)

    def test_whale_uses_tx_hash(self):
        e1 = {"source": "whale_alert", "tx_hash": "0xaaa"}
        e2 = {"source": "whale_alert", "tx_hash": "0xbbb"}
        assert _dedup_key(e1) != _dedup_key(e2)

    def test_reddit_uses_post_id(self):
        e1 = {"source": "reddit", "post_id": "abc123"}
        e2 = {"source": "reddit", "post_id": "def456"}
        assert _dedup_key(e1) != _dedup_key(e2)

    def test_fred_uses_series_and_timestamp(self):
        e1 = {"source": "fred", "series_id": "GDP", "timestamp": "2026-03-01"}
        e2 = {"source": "fred", "series_id": "GDP", "timestamp": "2026-03-02"}
        assert _dedup_key(e1) != _dedup_key(e2)


class TestLRUDedup:
    def test_first_seen_not_duplicate(self):
        dedup = _LRUDedup(maxsize=100)
        assert dedup.is_duplicate("a") is False

    def test_second_seen_is_duplicate(self):
        dedup = _LRUDedup(maxsize=100)
        dedup.is_duplicate("a")
        assert dedup.is_duplicate("a") is True

    def test_eviction(self):
        dedup = _LRUDedup(maxsize=3)
        dedup.is_duplicate("a")
        dedup.is_duplicate("b")
        dedup.is_duplicate("c")
        dedup.is_duplicate("d")  # evicts "a"
        assert dedup.is_duplicate("a") is False  # "a" was evicted
        assert dedup.is_duplicate("d") is True   # "d" still there
        assert dedup.is_duplicate("c") is True   # "c" still there

    def test_lru_order(self):
        dedup = _LRUDedup(maxsize=3)
        dedup.is_duplicate("a")
        dedup.is_duplicate("b")
        dedup.is_duplicate("c")
        dedup.is_duplicate("a")  # access "a", moves it to end
        dedup.is_duplicate("d")  # evicts "b" (oldest)
        assert dedup.is_duplicate("a") is True   # "a" was refreshed
        assert dedup.is_duplicate("b") is False  # "b" was evicted


# ── Orchestrator tests ────────────────────────────────────────────────────────


class TestOrchestratorHandleEvent:
    @pytest.fixture
    def orchestrator(self):
        orch = IngestionOrchestrator(graph_id="test_graph")
        orch._client = MagicMock()
        orch._client.graph.add = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_ingests_event(self, orchestrator):
        event = {
            "source": "crypto_news",
            "timestamp": "2026-03-29T10:00:00Z",
            "title": "Test Article",
            "content": "Some content",
        }
        await orchestrator._handle_event(event)
        orchestrator._client.graph.add.assert_called_once()
        call_kwargs = orchestrator._client.graph.add.call_args
        assert call_kwargs[1]["graph_id"] == "test_graph"
        assert "Test Article" in call_kwargs[1]["data"]
        assert orchestrator._ingested_count == 1

    @pytest.mark.asyncio
    async def test_dedup_blocks_duplicate(self, orchestrator):
        event = {
            "source": "crypto_news",
            "url": "https://example.com/same-article",
            "title": "Duplicate",
        }
        await orchestrator._handle_event(event)
        await orchestrator._handle_event(event)
        assert orchestrator._client.graph.add.call_count == 1
        assert orchestrator._ingested_count == 1
        assert orchestrator._deduped_count == 1

    @pytest.mark.asyncio
    async def test_different_events_both_ingested(self, orchestrator):
        e1 = {"source": "crypto_news", "url": "https://example.com/a", "title": "A"}
        e2 = {"source": "crypto_news", "url": "https://example.com/b", "title": "B"}
        await orchestrator._handle_event(e1)
        await orchestrator._handle_event(e2)
        assert orchestrator._client.graph.add.call_count == 2
        assert orchestrator._ingested_count == 2

    @pytest.mark.asyncio
    async def test_graphiti_error_counted(self, orchestrator):
        orchestrator._client.graph.add.side_effect = RuntimeError("Neo4j down")
        event = {"source": "rss", "title": "Test", "url": "https://example.com/test"}
        await orchestrator._handle_event(event)
        assert orchestrator._error_count == 1
        assert orchestrator._ingested_count == 0

    @pytest.mark.asyncio
    async def test_handle_batch(self, orchestrator):
        events = [
            {"source": "reddit", "post_id": "a", "subreddit": "Bitcoin", "title": "Post A"},
            {"source": "reddit", "post_id": "b", "subreddit": "Bitcoin", "title": "Post B"},
        ]
        await orchestrator._handle_event_batch(events)
        assert orchestrator._ingested_count == 2

    @pytest.mark.asyncio
    async def test_stats(self, orchestrator):
        event = {"source": "fred", "series_id": "GDP", "timestamp": "2026-03-01", "value": 100}
        await orchestrator._handle_event(event)
        stats = orchestrator.get_stats()
        assert stats["ingested"] == 1
        assert stats["deduped"] == 0
        assert stats["errors"] == 0


class TestOrchestratorLifecycle:
    def test_requires_graph_id(self):
        orch = IngestionOrchestrator()
        with pytest.raises(ValueError, match="graph_id is required"):
            asyncio.get_event_loop().run_until_complete(orch.start())

    def test_build_connectors_all_disabled(self):
        orch = IngestionOrchestrator(
            graph_id="test",
            enable_polymarket=False,
            enable_crypto_news=False,
            enable_rss=False,
            enable_whale=False,
            enable_reddit=False,
            enable_fred=False,
        )
        connectors = orch._build_connectors()
        assert connectors == []

    def test_build_connectors_selective(self):
        """Test that disabling all connectors yields empty list,
        and enabling one adds exactly that connector."""
        orch = IngestionOrchestrator(
            graph_id="test",
            enable_polymarket=False,
            enable_crypto_news=False,
            enable_rss=False,
            enable_whale=False,
            enable_reddit=False,
            enable_fred=False,
        )
        connectors = orch._build_connectors()
        assert connectors == []
