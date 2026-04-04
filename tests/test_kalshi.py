"""Tests for the Kalshi connector — price extraction, change detection, event format."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from ingestors.kalshi import (
    KalshiConnector,
    _extract_yes_price,
    _market_fingerprint,
    _build_event,
)


# ── Helper function tests ────────────────────────────────────────────────────


class TestExtractYesPrice:
    def test_yes_price_normalized(self):
        """yes_price as a 0-1 float."""
        market = {"yes_price": 0.72}
        assert _extract_yes_price(market) == 0.72

    def test_yes_price_cents(self):
        """yes_price in cents (>1) should be normalized to 0-1."""
        market = {"yes_price": 72}
        assert _extract_yes_price(market) == 0.72

    def test_last_price(self):
        market = {"last_price": 0.65}
        assert _extract_yes_price(market) == 0.65

    def test_last_price_cents(self):
        market = {"last_price": 65}
        assert _extract_yes_price(market) == 0.65

    def test_yes_bid_cents(self):
        """yes_bid is always in cents."""
        market = {"yes_bid": 55}
        assert _extract_yes_price(market) == 0.55

    def test_yes_ask_cents(self):
        market = {"yes_ask": 80}
        assert _extract_yes_price(market) == 0.80

    def test_no_price_fields(self):
        assert _extract_yes_price({}) is None

    def test_priority_yes_price_over_bid(self):
        """yes_price should be preferred over yes_bid."""
        market = {"yes_price": 0.72, "yes_bid": 70}
        assert _extract_yes_price(market) == 0.72

    def test_zero_price(self):
        """Zero is a valid price (market at 0%)."""
        market = {"yes_price": 0}
        # 0 is falsy but not None, however 0 > 1 is False so stays 0
        assert _extract_yes_price(market) == 0.0


class TestMarketFingerprint:
    def test_deterministic(self):
        fp1 = _market_fingerprint("KXBTC-26APR04", 0.72)
        fp2 = _market_fingerprint("KXBTC-26APR04", 0.72)
        assert fp1 == fp2

    def test_different_price_different_fingerprint(self):
        fp1 = _market_fingerprint("KXBTC-26APR04", 0.72)
        fp2 = _market_fingerprint("KXBTC-26APR04", 0.75)
        assert fp1 != fp2

    def test_different_ticker_different_fingerprint(self):
        fp1 = _market_fingerprint("KXBTC-A", 0.72)
        fp2 = _market_fingerprint("KXBTC-B", 0.72)
        assert fp1 != fp2

    def test_none_price(self):
        fp = _market_fingerprint("KXBTC-A", None)
        assert "none" in fp

    def test_rounding_same_fingerprint(self):
        """Prices rounded to 2 decimals — tiny differences should match."""
        fp1 = _market_fingerprint("KXBTC-A", 0.721)
        fp2 = _market_fingerprint("KXBTC-A", 0.724)
        assert fp1 == fp2

    def test_rounding_different_fingerprint(self):
        """Large enough difference should differ."""
        fp1 = _market_fingerprint("KXBTC-A", 0.72)
        fp2 = _market_fingerprint("KXBTC-A", 0.73)
        assert fp1 != fp2


class TestBuildEvent:
    def test_basic(self):
        market = {
            "ticker": "KXBTC-26APR04-T64400",
            "title": "Bitcoin above $64,400 on April 4?",
            "volume": 50000,
            "category": "Crypto",
        }
        event = _build_event(market, 0.72)
        assert event["source"] == "kalshi"
        assert event["market_id"] == "KXBTC-26APR04-T64400"
        assert event["market_title"] == "Bitcoin above $64,400 on April 4?"
        assert event["yes_price"] == 0.72
        assert event["volume"] == 50000
        assert event["category"] == "Crypto"
        assert event["timestamp"]

    def test_event_title_fallback(self):
        """If title is missing, try event_title."""
        market = {
            "ticker": "TEST",
            "event_title": "Fallback Title",
        }
        event = _build_event(market, 0.50)
        assert event["market_title"] == "Fallback Title"

    def test_volume_fallback(self):
        """If volume is missing, try volume_24h."""
        market = {
            "ticker": "TEST",
            "volume_24h": 12345,
        }
        event = _build_event(market, 0.50)
        assert event["volume"] == 12345

    def test_category_fallback(self):
        """If category is missing, try series_ticker."""
        market = {
            "ticker": "TEST",
            "series_ticker": "ECON",
        }
        event = _build_event(market, 0.50)
        assert event["category"] == "ECON"

    def test_missing_fields(self):
        event = _build_event({}, 0.5)
        assert event["market_id"] == ""
        assert event["market_title"] == ""
        assert event["volume"] == 0
        assert event["category"] == ""


# ── Connector tests ──────────────────────────────────────────────────────────


SAMPLE_API_RESPONSE = {
    "markets": [
        {
            "ticker": "KXBTC-26APR04-T64400",
            "title": "Bitcoin above $64,400 on April 4?",
            "yes_price": 0.72,
            "volume": 50000,
            "category": "Crypto",
        },
        {
            "ticker": "KXFED-26MAY01-T525",
            "title": "Fed rate above 5.25% on May 1?",
            "yes_price": 0.45,
            "volume": 30000,
            "category": "Economics",
        },
    ]
}


def _make_mock_session(response_data, status=200):
    """Create a mock aiohttp session that returns the given data."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    return mock_session


class TestKalshiConnector:
    def test_defaults(self):
        connector = KalshiConnector()
        assert connector.poll_interval == 1800.0
        assert connector.callback is None
        assert connector._last_fingerprints == {}

    @pytest.mark.asyncio
    async def test_poll_once_first_call_returns_all(self):
        """First poll should return all markets with valid prices."""
        connector = KalshiConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        events = await connector.poll_once()
        assert len(events) == 2
        assert events[0]["source"] == "kalshi"
        assert events[0]["market_id"] == "KXBTC-26APR04-T64400"
        assert events[0]["yes_price"] == 0.72
        assert events[1]["market_id"] == "KXFED-26MAY01-T525"
        assert events[1]["yes_price"] == 0.45

    @pytest.mark.asyncio
    async def test_poll_once_no_change_returns_empty(self):
        """Second poll with same data should return empty (no changes)."""
        connector = KalshiConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        events1 = await connector.poll_once()
        assert len(events1) == 2

        events2 = await connector.poll_once()
        assert len(events2) == 0

    @pytest.mark.asyncio
    async def test_poll_once_detects_price_change(self):
        """If one market's price changes, only that market is emitted."""
        connector = KalshiConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        await connector.poll_once()

        changed_response = {
            "markets": [
                {
                    "ticker": "KXBTC-26APR04-T64400",
                    "title": "Bitcoin above $64,400 on April 4?",
                    "yes_price": 0.78,  # changed from 0.72
                    "volume": 55000,
                    "category": "Crypto",
                },
                {
                    "ticker": "KXFED-26MAY01-T525",
                    "title": "Fed rate above 5.25% on May 1?",
                    "yes_price": 0.45,  # unchanged
                    "volume": 30000,
                    "category": "Economics",
                },
            ]
        }
        connector._session = _make_mock_session(changed_response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["market_id"] == "KXBTC-26APR04-T64400"
        assert events[0]["yes_price"] == 0.78

    @pytest.mark.asyncio
    async def test_poll_once_api_failure(self):
        """If API returns non-200, poll_once returns empty list."""
        connector = KalshiConnector()
        connector._session = _make_mock_session({}, status=500)

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_once_empty_markets(self):
        """If API returns no markets, poll_once returns empty list."""
        connector = KalshiConnector()
        connector._session = _make_mock_session({"markets": []})

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_market_without_ticker_skipped(self):
        """Markets without a ticker should be skipped."""
        response = {
            "markets": [
                {
                    "title": "No Ticker Market",
                    "yes_price": 0.50,
                },
                {
                    "ticker": "VALID-MKT",
                    "title": "Valid Market",
                    "yes_price": 0.80,
                },
            ]
        }
        connector = KalshiConnector()
        connector._session = _make_mock_session(response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["market_id"] == "VALID-MKT"

    @pytest.mark.asyncio
    async def test_market_without_price_skipped(self):
        """Markets without any price field should be skipped."""
        response = {
            "markets": [
                {
                    "ticker": "NO-PRICE",
                    "title": "No price data",
                },
                {
                    "ticker": "HAS-PRICE",
                    "title": "Has price",
                    "yes_price": 0.60,
                },
            ]
        }
        connector = KalshiConnector()
        connector._session = _make_mock_session(response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["market_id"] == "HAS-PRICE"

    @pytest.mark.asyncio
    async def test_callback_invoked_for_each_market(self):
        """Verify callback is called once per changed market."""
        events = []

        async def cb(event):
            events.append(event)

        connector = KalshiConnector(callback=cb)
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        changed_events = await connector.poll_once()
        for event in changed_events:
            await connector.callback(event)

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_volume_change_only_not_detected(self):
        """Volume-only changes should not trigger a new event (price-based fingerprint)."""
        connector = KalshiConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)
        await connector.poll_once()

        # Only volume changed, price same.
        same_price_response = {
            "markets": [
                {
                    "ticker": "KXBTC-26APR04-T64400",
                    "title": "Bitcoin above $64,400 on April 4?",
                    "yes_price": 0.72,  # same price
                    "volume": 99999,  # different volume
                    "category": "Crypto",
                },
            ]
        }
        connector._session = _make_mock_session(same_price_response)
        events = await connector.poll_once()
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_close(self):
        connector = KalshiConnector()
        await connector.close()  # should be safe with no session

    def test_stop(self):
        connector = KalshiConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False
