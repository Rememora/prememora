"""Tests for the CoinGecko connector — trending parsing, price extraction, event format."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from ingestors.coingecko import (
    CoinGeckoConnector,
    DEFAULT_PRICE_IDS,
    _parse_trending,
    _parse_prices,
)


# ── Parsing tests ────────────────────────────────────────────────────────────


class TestParseTrending:
    def test_basic_trending(self):
        data = {
            "coins": [
                {
                    "item": {
                        "name": "Pepe",
                        "symbol": "PEPE",
                        "market_cap_rank": 25,
                    }
                },
                {
                    "item": {
                        "name": "Dogwifhat",
                        "symbol": "WIF",
                        "market_cap_rank": 80,
                    }
                },
            ]
        }
        event = _parse_trending(data)
        assert event["source"] == "coingecko"
        assert event["event_type"] == "trending"
        assert len(event["coins"]) == 2
        assert event["coins"][0]["name"] == "Pepe"
        assert event["coins"][0]["symbol"] == "PEPE"
        assert event["coins"][0]["market_cap_rank"] == 25
        assert event["coins"][1]["name"] == "Dogwifhat"
        assert event["coins"][1]["symbol"] == "WIF"
        assert event["timestamp"]

    def test_empty_coins(self):
        event = _parse_trending({"coins": []})
        assert event["event_type"] == "trending"
        assert event["coins"] == []

    def test_missing_coins_key(self):
        event = _parse_trending({})
        assert event["coins"] == []

    def test_missing_item_fields(self):
        data = {
            "coins": [
                {"item": {}},
            ]
        }
        event = _parse_trending(data)
        assert len(event["coins"]) == 1
        assert event["coins"][0]["name"] == ""
        assert event["coins"][0]["symbol"] == ""
        assert event["coins"][0]["market_cap_rank"] is None

    def test_missing_item_key(self):
        data = {"coins": [{}]}
        event = _parse_trending(data)
        assert len(event["coins"]) == 1
        assert event["coins"][0]["name"] == ""


class TestParsePrices:
    def test_basic_prices(self):
        data = {
            "bitcoin": {"usd": 67000, "usd_24h_change": 2.5},
            "ethereum": {"usd": 3500, "usd_24h_change": -1.2},
        }
        event = _parse_prices(data)
        assert event["source"] == "coingecko"
        assert event["event_type"] == "prices"
        assert event["prices"]["bitcoin"]["usd"] == 67000
        assert event["prices"]["bitcoin"]["usd_24h_change"] == 2.5
        assert event["prices"]["ethereum"]["usd"] == 3500
        assert event["prices"]["ethereum"]["usd_24h_change"] == -1.2
        assert event["timestamp"]

    def test_empty_data(self):
        event = _parse_prices({})
        assert event["prices"] == {}

    def test_missing_change_field(self):
        data = {"bitcoin": {"usd": 67000}}
        event = _parse_prices(data)
        assert event["prices"]["bitcoin"]["usd"] == 67000
        assert event["prices"]["bitcoin"]["usd_24h_change"] is None

    def test_zero_price(self):
        data = {"testcoin": {"usd": 0, "usd_24h_change": 0.0}}
        event = _parse_prices(data)
        assert event["prices"]["testcoin"]["usd"] == 0
        assert event["prices"]["testcoin"]["usd_24h_change"] == 0.0


# ── Connector tests ──────────────────────────────────────────────────────────


SAMPLE_TRENDING_RESPONSE = {
    "coins": [
        {
            "item": {
                "name": "Pepe",
                "symbol": "PEPE",
                "market_cap_rank": 25,
            }
        },
        {
            "item": {
                "name": "Bitcoin",
                "symbol": "BTC",
                "market_cap_rank": 1,
            }
        },
    ]
}

SAMPLE_PRICE_RESPONSE = {
    "bitcoin": {"usd": 67000, "usd_24h_change": 2.5},
    "ethereum": {"usd": 3500, "usd_24h_change": -1.2},
    "solana": {"usd": 150, "usd_24h_change": 5.0},
}


def _make_mock_session_sequential(responses: list, statuses: list = None):
    """Create a mock session that returns different data for sequential .get() calls."""
    if statuses is None:
        statuses = [200] * len(responses)

    mock_resps = []
    for resp_data, status in zip(responses, statuses):
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


class TestCoinGeckoConnector:
    def test_defaults(self):
        connector = CoinGeckoConnector()
        assert connector.poll_interval == 3600.0
        assert connector.callback is None
        assert connector.price_ids == DEFAULT_PRICE_IDS

    def test_custom_price_ids(self):
        connector = CoinGeckoConnector(price_ids=["bitcoin", "cardano"])
        assert connector.price_ids == ["bitcoin", "cardano"]

    @pytest.mark.asyncio
    async def test_poll_once_returns_two_events(self):
        """poll_once should return both trending and prices events."""
        connector = CoinGeckoConnector()
        connector._session = _make_mock_session_sequential([
            SAMPLE_TRENDING_RESPONSE,
            SAMPLE_PRICE_RESPONSE,
        ])

        events = await connector.poll_once()
        assert len(events) == 2

        trending = events[0]
        assert trending["event_type"] == "trending"
        assert len(trending["coins"]) == 2
        assert trending["coins"][0]["name"] == "Pepe"

        prices = events[1]
        assert prices["event_type"] == "prices"
        assert prices["prices"]["bitcoin"]["usd"] == 67000
        assert prices["prices"]["ethereum"]["usd_24h_change"] == -1.2

    @pytest.mark.asyncio
    async def test_poll_once_trending_failure_still_returns_prices(self):
        """If trending fails, prices should still be returned."""
        connector = CoinGeckoConnector()
        connector._session = _make_mock_session_sequential(
            [SAMPLE_TRENDING_RESPONSE, SAMPLE_PRICE_RESPONSE],
            statuses=[500, 200],
        )

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["event_type"] == "prices"

    @pytest.mark.asyncio
    async def test_poll_once_prices_failure_still_returns_trending(self):
        """If prices fails, trending should still be returned."""
        connector = CoinGeckoConnector()
        connector._session = _make_mock_session_sequential(
            [SAMPLE_TRENDING_RESPONSE, SAMPLE_PRICE_RESPONSE],
            statuses=[200, 500],
        )

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["event_type"] == "trending"

    @pytest.mark.asyncio
    async def test_poll_once_both_fail(self):
        """If both APIs fail, poll_once returns empty list."""
        connector = CoinGeckoConnector()
        connector._session = _make_mock_session_sequential(
            [{}, {}],
            statuses=[500, 500],
        )

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_once_empty_price_ids(self):
        """With no price IDs, only trending is fetched."""
        connector = CoinGeckoConnector(price_ids=[])
        connector._session = _make_mock_session(SAMPLE_TRENDING_RESPONSE)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["event_type"] == "trending"

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """Verify callback is called for each event."""
        received = []

        async def cb(event):
            received.append(event)

        connector = CoinGeckoConnector(callback=cb)
        connector._session = _make_mock_session_sequential([
            SAMPLE_TRENDING_RESPONSE,
            SAMPLE_PRICE_RESPONSE,
        ])

        events = await connector.poll_once()
        for event in events:
            await connector.callback(event)

        assert len(received) == 2
        assert received[0]["source"] == "coingecko"
        assert received[1]["source"] == "coingecko"

    @pytest.mark.asyncio
    async def test_trending_coins_structure(self):
        """Verify the structure of individual coin entries in trending."""
        connector = CoinGeckoConnector(price_ids=[])
        connector._session = _make_mock_session(SAMPLE_TRENDING_RESPONSE)

        events = await connector.poll_once()
        coins = events[0]["coins"]
        for coin in coins:
            assert "name" in coin
            assert "symbol" in coin
            assert "market_cap_rank" in coin

    @pytest.mark.asyncio
    async def test_prices_all_tracked_coins(self):
        """All requested coin IDs should appear in price data."""
        connector = CoinGeckoConnector(price_ids=["bitcoin", "ethereum", "solana"])
        connector._session = _make_mock_session_sequential([
            SAMPLE_TRENDING_RESPONSE,
            SAMPLE_PRICE_RESPONSE,
        ])

        events = await connector.poll_once()
        prices_event = [e for e in events if e["event_type"] == "prices"][0]
        for coin_id in ["bitcoin", "ethereum", "solana"]:
            assert coin_id in prices_event["prices"]
            assert "usd" in prices_event["prices"][coin_id]
            assert "usd_24h_change" in prices_event["prices"][coin_id]

    @pytest.mark.asyncio
    async def test_close(self):
        connector = CoinGeckoConnector()
        await connector.close()  # safe with no session

    def test_stop(self):
        connector = CoinGeckoConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False
