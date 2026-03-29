"""Tests for the RSS feed poller — normalization, dedup, config."""

import pytest
from unittest.mock import MagicMock
from ingestors.rss_feeds import (
    RSSPoller,
    _parse_timestamp,
    _entry_id,
    _entry_content,
    _normalize_entry,
    _load_feeds_from_env,
    DEFAULT_FEEDS,
)


class TestParseTimestamp:
    def test_published_parsed(self):
        entry = MagicMock()
        entry.published_parsed = (2026, 3, 29, 10, 0, 0, 0, 0, 0)
        entry.updated_parsed = None
        entry.published = None
        entry.updated = None
        ts = _parse_timestamp(entry)
        assert ts.startswith("2026-03-29T10:00:00")

    def test_raw_published_string(self):
        entry = MagicMock()
        entry.published_parsed = None
        entry.updated_parsed = None
        entry.published = "Sat, 29 Mar 2026 10:00:00 GMT"
        entry.updated = None
        ts = _parse_timestamp(entry)
        assert "2026" in ts

    def test_fallback_to_now(self):
        entry = MagicMock()
        entry.published_parsed = None
        entry.updated_parsed = None
        entry.published = None
        entry.updated = None
        ts = _parse_timestamp(entry)
        assert ts  # should return current time, not empty


class TestEntryId:
    def test_prefers_id(self):
        entry = MagicMock()
        entry.id = "guid-123"
        entry.link = "https://example.com/article"
        assert _entry_id(entry) == "guid-123"

    def test_falls_back_to_link(self):
        entry = MagicMock(spec=[])
        entry.id = None
        entry.link = "https://example.com/article"
        assert _entry_id(entry) == "https://example.com/article"


class TestEntryContent:
    def test_content_encoded(self):
        entry = MagicMock()
        entry.content = [{"value": "<p>Full article body</p>"}]
        assert _entry_content(entry) == "<p>Full article body</p>"

    def test_summary_fallback(self):
        entry = MagicMock()
        entry.content = None
        entry.summary = "Short summary"
        entry.description = "Description"
        assert _entry_content(entry) == "Short summary"


class TestNormalizeEntry:
    def test_basic(self):
        entry = MagicMock()
        entry.published_parsed = (2026, 3, 29, 10, 0, 0, 0, 0, 0)
        entry.updated_parsed = None
        entry.published = None
        entry.updated = None
        entry.title = "Breaking News"
        entry.content = None
        entry.summary = "Summary of breaking news"
        entry.description = None
        entry.link = "https://apnews.com/article/123"
        entry.id = "guid-123"

        event = _normalize_entry(entry, "AP News", "https://rss.ap.org/rss/apf-topnews", "general")
        assert event["source"] == "rss"
        assert event["title"] == "Breaking News"
        assert event["feed_name"] == "AP News"
        assert event["category"] == "general"
        assert event["url"] == "https://apnews.com/article/123"


class TestRSSPoller:
    def test_default_feeds(self):
        poller = RSSPoller()
        assert len(poller.feeds) == len(DEFAULT_FEEDS)
        assert poller.poll_interval == 30.0

    def test_custom_feeds(self):
        custom = [{"name": "Test", "url": "https://example.com/rss", "category": "test"}]
        poller = RSSPoller(feeds=custom, poll_interval=10.0)
        assert len(poller.feeds) == 1
        assert poller.poll_interval == 10.0

    def test_dedup_tracking(self):
        poller = RSSPoller()
        poller._seen.add("https://example.com/article-1")
        assert "https://example.com/article-1" in poller._seen
        assert poller.seen_count == 1


class TestLoadFeedsFromEnv:
    def test_defaults_when_no_env(self, monkeypatch):
        monkeypatch.delenv("RSS_FEEDS", raising=False)
        feeds = _load_feeds_from_env()
        assert feeds == DEFAULT_FEEDS

    def test_custom_json(self, monkeypatch):
        import json
        custom = [{"name": "Custom", "url": "https://x.com/rss", "category": "test"}]
        monkeypatch.setenv("RSS_FEEDS", json.dumps(custom))
        feeds = _load_feeds_from_env()
        assert len(feeds) == 1
        assert feeds[0]["name"] == "Custom"

    def test_invalid_json_falls_back(self, monkeypatch):
        monkeypatch.setenv("RSS_FEEDS", "not-json")
        feeds = _load_feeds_from_env()
        assert feeds == DEFAULT_FEEDS
