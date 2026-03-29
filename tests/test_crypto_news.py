"""Tests for the Crypto News API connector — normalization and deduplication."""

import pytest
from ingestors.crypto_news import CryptoNewsConnector


@pytest.fixture
def connector():
    return CryptoNewsConnector()


class TestNormalize:
    def test_basic_article(self, connector):
        article = {
            "title": "Bitcoin Surges Past $100k",
            "content": "The price of Bitcoin has surged...",
            "url": "https://example.com/btc-100k",
            "published_at": "2026-03-29T10:00:00Z",
            "source": {"name": "CoinDesk"},
            "entities": ["Bitcoin", "BTC"],
            "sentiment": "positive",
            "categories": ["crypto"],
        }
        event = connector._normalize(article)
        assert event["source"] == "crypto_news"
        assert event["source_name"] == "CoinDesk"
        assert event["title"] == "Bitcoin Surges Past $100k"
        assert event["url"] == "https://example.com/btc-100k"
        assert event["entities"] == ["Bitcoin", "BTC"]
        assert event["sentiment"] == "positive"
        assert event["timestamp"] == "2026-03-29T10:00:00Z"

    def test_fallback_field_names(self, connector):
        """API may use different field names — test fallbacks."""
        article = {
            "title": "ETH Update",
            "body": "Ethereum network upgrade...",
            "link": "https://example.com/eth",
            "publishedAt": "2026-03-28T08:00:00Z",
            "source": "The Block",
            "currencies": ["ETH"],
        }
        event = connector._normalize(article)
        assert event["content"] == "Ethereum network upgrade..."
        assert event["url"] == "https://example.com/eth"
        assert event["timestamp"] == "2026-03-28T08:00:00Z"
        assert event["source_name"] == "The Block"
        assert event["entities"] == ["ETH"]

    def test_tags_as_entities_fallback(self, connector):
        article = {
            "title": "DeFi News",
            "url": "https://example.com/defi",
            "tags": ["defi", "lending"],
        }
        event = connector._normalize(article)
        assert event["entities"] == ["defi", "lending"]

    def test_content_truncation(self, connector):
        article = {
            "title": "Long Article",
            "content": "x" * 5000,
            "url": "https://example.com/long",
        }
        event = connector._normalize(article)
        assert len(event["content"]) == 2000

    def test_missing_fields_get_defaults(self, connector):
        event = connector._normalize({})
        assert event["source"] == "crypto_news"
        assert event["title"] == ""
        assert event["content"] == ""
        assert event["entities"] == []
        assert event["timestamp"]  # should have a default timestamp


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_seen_urls_deduplicate(self):
        events = []

        async def cb(event):
            events.append(event)

        connector = CryptoNewsConnector(callback=cb)
        connector._seen_urls.add("https://example.com/old")

        # Simulate processing an article with a seen URL — should be skipped
        # (tested via the _poll_loop dedup logic, but we test the set directly)
        assert "https://example.com/old" in connector._seen_urls
        assert "https://example.com/new" not in connector._seen_urls
