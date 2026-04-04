"""Tests for the GDELT connector — API parsing, multi-query dedup, event format, tone extraction."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from ingestors.gdelt import (
    GDELTConnector,
    DEFAULT_QUERIES,
    _parse_article,
)


# ── Article parsing ──────────────────────────────────────────────────────────


class TestParseArticle:
    def test_basic_article(self):
        article = {
            "url": "https://reuters.com/article/fed-rate-cut",
            "title": "Fed signals rate cut amid slowing inflation",
            "seendate": "20260404T103000Z",
            "domain": "reuters.com",
            "language": "English",
            "tone": "-2.5",
        }
        event = _parse_article(article, "federal reserve OR interest rate")
        assert event["source"] == "gdelt"
        assert event["title"] == "Fed signals rate cut amid slowing inflation"
        assert event["url"] == "https://reuters.com/article/fed-rate-cut"
        assert event["domain"] == "reuters.com"
        assert event["language"] == "English"
        assert event["tone"] == -2.5
        assert event["search_query"] == "federal reserve OR interest rate"
        assert "2026-04-04T10:30:00" in event["timestamp"]

    def test_positive_tone(self):
        article = {
            "url": "https://example.com/good-news",
            "title": "Markets rally",
            "seendate": "20260404T120000Z",
            "tone": "5.3",
        }
        event = _parse_article(article, "bitcoin OR cryptocurrency")
        assert event["tone"] == 5.3

    def test_missing_tone(self):
        article = {
            "url": "https://example.com/no-tone",
            "title": "Article without tone",
            "seendate": "20260404T120000Z",
        }
        event = _parse_article(article, "test")
        assert event["tone"] == 0.0

    def test_invalid_tone(self):
        article = {
            "url": "https://example.com/bad-tone",
            "title": "Bad tone data",
            "seendate": "20260404T120000Z",
            "tone": "not-a-number",
        }
        event = _parse_article(article, "test")
        assert event["tone"] == 0.0

    def test_missing_seendate_fallback(self):
        article = {
            "url": "https://example.com/no-date",
            "title": "No date article",
        }
        event = _parse_article(article, "test")
        assert event["timestamp"]  # should have a fallback timestamp

    def test_invalid_seendate_fallback(self):
        article = {
            "url": "https://example.com/bad-date",
            "title": "Bad date article",
            "seendate": "not-a-date",
        }
        event = _parse_article(article, "test")
        assert event["timestamp"]  # should have a fallback timestamp

    def test_missing_fields_defaults(self):
        event = _parse_article({}, "test query")
        assert event["source"] == "gdelt"
        assert event["title"] == ""
        assert event["url"] == ""
        assert event["domain"] == ""
        assert event["language"] == "English"
        assert event["tone"] == 0.0
        assert event["search_query"] == "test query"

    def test_numeric_tone_float(self):
        """Tone may arrive as a float rather than a string."""
        article = {
            "url": "https://example.com/float-tone",
            "title": "Float tone",
            "seendate": "20260404T120000Z",
            "tone": -1.7,
        }
        event = _parse_article(article, "test")
        assert event["tone"] == -1.7


# ── Connector tests ──────────────────────────────────────────────────────────


SAMPLE_GDELT_RESPONSE = {
    "articles": [
        {
            "url": "https://reuters.com/article/btc-surge",
            "title": "Bitcoin Surges Past $100k",
            "seendate": "20260404T100000Z",
            "domain": "reuters.com",
            "language": "English",
            "tone": "3.2",
        },
        {
            "url": "https://bbc.com/news/crypto-regulation",
            "title": "EU Crypto Regulation Update",
            "seendate": "20260404T093000Z",
            "domain": "bbc.com",
            "language": "English",
            "tone": "-1.1",
        },
    ]
}

SAMPLE_GDELT_RESPONSE_2 = {
    "articles": [
        {
            "url": "https://apnews.com/article/fed-meeting",
            "title": "Fed Holds Rates Steady",
            "seendate": "20260404T110000Z",
            "domain": "apnews.com",
            "language": "English",
            "tone": "0.5",
        },
        {
            # Duplicate URL from first query — should be deduped.
            "url": "https://reuters.com/article/btc-surge",
            "title": "Bitcoin Surges Past $100k",
            "seendate": "20260404T100000Z",
            "domain": "reuters.com",
            "language": "English",
            "tone": "3.2",
        },
    ]
}


def _make_mock_session_multi(responses: list, status: int = 200):
    """Create a mock session that returns different data for each .get() call."""
    mock_resps = []
    for resp_data in responses:
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=resp_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_resps.append(mock_resp)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(side_effect=mock_resps)
    mock_session.closed = False
    return mock_session


def _make_mock_session(response_data, status=200):
    """Create a mock aiohttp session that returns the same data for every call."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    return mock_session


class TestGDELTConnector:
    def test_defaults(self):
        connector = GDELTConnector()
        assert connector.poll_interval == 3600.0
        assert connector.callback is None
        assert connector.queries == DEFAULT_QUERIES
        assert len(connector._seen_urls) == 0

    def test_custom_queries(self):
        custom = ["bitcoin", "ethereum"]
        connector = GDELTConnector(queries=custom)
        assert connector.queries == custom

    @pytest.mark.asyncio
    async def test_poll_once_basic(self):
        """Single query returns all articles."""
        connector = GDELTConnector(queries=["bitcoin OR cryptocurrency"])
        connector._session = _make_mock_session(SAMPLE_GDELT_RESPONSE)

        events = await connector.poll_once()
        assert len(events) == 2
        assert events[0]["source"] == "gdelt"
        assert events[0]["title"] == "Bitcoin Surges Past $100k"
        assert events[0]["url"] == "https://reuters.com/article/btc-surge"
        assert events[0]["tone"] == 3.2
        assert events[1]["title"] == "EU Crypto Regulation Update"
        assert events[1]["tone"] == -1.1

    @pytest.mark.asyncio
    async def test_poll_once_cross_query_dedup(self):
        """Articles with the same URL from different queries are deduplicated."""
        connector = GDELTConnector(queries=["query1", "query2"])
        connector._session = _make_mock_session_multi([
            SAMPLE_GDELT_RESPONSE,
            SAMPLE_GDELT_RESPONSE_2,
        ])

        events = await connector.poll_once()
        urls = [e["url"] for e in events]
        # reuters btc-surge appears in both responses — should appear once
        assert urls.count("https://reuters.com/article/btc-surge") == 1
        # Total: 2 from first query + 1 unique from second query = 3
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_poll_once_cross_poll_dedup(self):
        """Articles seen in a previous poll are not returned again."""
        connector = GDELTConnector(queries=["test"])
        connector._session = _make_mock_session(SAMPLE_GDELT_RESPONSE)

        events1 = await connector.poll_once()
        assert len(events1) == 2

        # Reset session mock to return same data.
        connector._session = _make_mock_session(SAMPLE_GDELT_RESPONSE)
        events2 = await connector.poll_once()
        assert len(events2) == 0  # all URLs already seen

    @pytest.mark.asyncio
    async def test_poll_once_skips_articles_without_url(self):
        """Articles without a URL are skipped."""
        response = {
            "articles": [
                {"title": "No URL article", "seendate": "20260404T100000Z"},
                {"url": "https://example.com/valid", "title": "Valid", "seendate": "20260404T100000Z"},
            ]
        }
        connector = GDELTConnector(queries=["test"])
        connector._session = _make_mock_session(response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["url"] == "https://example.com/valid"

    @pytest.mark.asyncio
    async def test_poll_once_api_failure(self):
        """If API returns non-200, poll_once returns empty list."""
        connector = GDELTConnector(queries=["test"])
        connector._session = _make_mock_session({}, status=500)

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_once_empty_articles(self):
        """If API returns no articles, poll_once returns empty list."""
        connector = GDELTConnector(queries=["test"])
        connector._session = _make_mock_session({"articles": []})

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_once_missing_articles_key(self):
        """If API response lacks 'articles' key, return empty."""
        connector = GDELTConnector(queries=["test"])
        connector._session = _make_mock_session({"error": "bad request"})

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """Verify callback is called for each new event."""
        received = []

        async def cb(event):
            received.append(event)

        connector = GDELTConnector(callback=cb, queries=["test"])
        connector._session = _make_mock_session(SAMPLE_GDELT_RESPONSE)

        events = await connector.poll_once()
        for event in events:
            await connector.callback(event)

        assert len(received) == 2
        assert received[0]["source"] == "gdelt"

    @pytest.mark.asyncio
    async def test_search_query_preserved(self):
        """Each event should carry the query that found it."""
        connector = GDELTConnector(queries=["bitcoin OR cryptocurrency"])
        connector._session = _make_mock_session(SAMPLE_GDELT_RESPONSE)

        events = await connector.poll_once()
        for event in events:
            assert event["search_query"] == "bitcoin OR cryptocurrency"

    @pytest.mark.asyncio
    async def test_close(self):
        connector = GDELTConnector()
        await connector.close()  # safe with no session

    def test_stop(self):
        connector = GDELTConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False
