"""Tests for the PredictIt connector — parsing, change detection, event format."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from ingestors.predictit import (
    PredictItConnector,
    _extract_contracts,
    _market_fingerprint,
    _build_market_event,
)


# ── Helper function tests ────────────────────────────────────────────────────


class TestExtractContracts:
    def test_basic_contracts(self):
        market = {
            "contracts": [
                {"name": "Alice", "lastTradePrice": 0.62, "totalSharesTraded": 1234},
                {"name": "Bob", "lastTradePrice": 0.35, "totalSharesTraded": 567},
            ]
        }
        contracts = _extract_contracts(market)
        assert len(contracts) == 2
        assert contracts[0]["name"] == "Alice"
        assert contracts[0]["price"] == 0.62
        assert contracts[0]["volume"] == 1234
        assert contracts[1]["name"] == "Bob"
        assert contracts[1]["price"] == 0.35

    def test_missing_fields(self):
        market = {"contracts": [{"name": "Only Name"}]}
        contracts = _extract_contracts(market)
        assert len(contracts) == 1
        assert contracts[0]["name"] == "Only Name"
        assert contracts[0]["price"] is None
        assert contracts[0]["volume"] == 0

    def test_no_contracts_key(self):
        contracts = _extract_contracts({})
        assert contracts == []

    def test_empty_contracts_list(self):
        contracts = _extract_contracts({"contracts": []})
        assert contracts == []


class TestMarketFingerprint:
    def test_deterministic(self):
        contracts = [
            {"name": "Alice", "price": 0.62},
            {"name": "Bob", "price": 0.35},
        ]
        fp1 = _market_fingerprint(contracts)
        fp2 = _market_fingerprint(contracts)
        assert fp1 == fp2

    def test_order_independent(self):
        """Fingerprint should be the same regardless of contract order."""
        c1 = [{"name": "Alice", "price": 0.62}, {"name": "Bob", "price": 0.35}]
        c2 = [{"name": "Bob", "price": 0.35}, {"name": "Alice", "price": 0.62}]
        assert _market_fingerprint(c1) == _market_fingerprint(c2)

    def test_different_prices_different_fingerprint(self):
        c1 = [{"name": "Alice", "price": 0.62}]
        c2 = [{"name": "Alice", "price": 0.65}]
        assert _market_fingerprint(c1) != _market_fingerprint(c2)

    def test_empty_contracts(self):
        assert _market_fingerprint([]) == ""


class TestBuildMarketEvent:
    def test_basic(self):
        market = {
            "id": 7456,
            "name": "Who will win the 2026 Senate race in Pennsylvania?",
            "contracts": [
                {"name": "John Fetterman", "lastTradePrice": 0.62, "totalSharesTraded": 1234},
            ],
        }
        event = _build_market_event(market, "2026-04-01T10:00:00+00:00")
        assert event["source"] == "predictit"
        assert event["market_id"] == "7456"
        assert event["market_name"] == "Who will win the 2026 Senate race in Pennsylvania?"
        assert event["timestamp"] == "2026-04-01T10:00:00+00:00"
        assert len(event["contracts"]) == 1
        assert event["contracts"][0]["name"] == "John Fetterman"
        assert event["contracts"][0]["price"] == 0.62


# ── Connector tests ──────────────────────────────────────────────────────────


SAMPLE_API_RESPONSE = {
    "markets": [
        {
            "id": 100,
            "name": "Market A",
            "contracts": [
                {"name": "Yes", "lastTradePrice": 0.60, "totalSharesTraded": 500},
                {"name": "No", "lastTradePrice": 0.40, "totalSharesTraded": 300},
            ],
        },
        {
            "id": 200,
            "name": "Market B",
            "contracts": [
                {"name": "Candidate X", "lastTradePrice": 0.75, "totalSharesTraded": 1000},
                {"name": "Candidate Y", "lastTradePrice": 0.25, "totalSharesTraded": 800},
            ],
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


class TestPredictItConnector:
    def test_defaults(self):
        connector = PredictItConnector()
        assert connector.poll_interval == 1800.0
        assert connector.callback is None
        assert connector._last_fingerprints == {}

    @pytest.mark.asyncio
    async def test_poll_once_first_call_returns_all(self):
        """First poll should return all markets (no prior fingerprints)."""
        connector = PredictItConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        events = await connector.poll_once()
        assert len(events) == 2
        assert events[0]["source"] == "predictit"
        assert events[0]["market_id"] == "100"
        assert events[1]["market_id"] == "200"

    @pytest.mark.asyncio
    async def test_poll_once_no_change_returns_empty(self):
        """Second poll with same data should return empty (no changes)."""
        connector = PredictItConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        # First poll — populates fingerprints.
        events1 = await connector.poll_once()
        assert len(events1) == 2

        # Second poll — same data, no changes.
        events2 = await connector.poll_once()
        assert len(events2) == 0

    @pytest.mark.asyncio
    async def test_poll_once_detects_price_change(self):
        """If one market's price changes, only that market is emitted."""
        connector = PredictItConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        # First poll.
        await connector.poll_once()

        # Modify Market A's price.
        changed_response = {
            "markets": [
                {
                    "id": 100,
                    "name": "Market A",
                    "contracts": [
                        {"name": "Yes", "lastTradePrice": 0.65, "totalSharesTraded": 550},
                        {"name": "No", "lastTradePrice": 0.35, "totalSharesTraded": 320},
                    ],
                },
                {
                    "id": 200,
                    "name": "Market B",
                    "contracts": [
                        {"name": "Candidate X", "lastTradePrice": 0.75, "totalSharesTraded": 1000},
                        {"name": "Candidate Y", "lastTradePrice": 0.25, "totalSharesTraded": 800},
                    ],
                },
            ]
        }
        connector._session = _make_mock_session(changed_response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["market_id"] == "100"
        # Volume change in Market B doesn't matter — price is the same.

    @pytest.mark.asyncio
    async def test_poll_once_api_failure(self):
        """If API returns non-200, poll_once returns empty list."""
        connector = PredictItConnector()
        connector._session = _make_mock_session({}, status=500)

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_once_empty_markets(self):
        """If API returns no markets, poll_once returns empty list."""
        connector = PredictItConnector()
        connector._session = _make_mock_session({"markets": []})

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_callback_invoked_for_each_market(self):
        """Verify callback is called once per changed market."""
        events = []

        async def cb(event):
            events.append(event)

        connector = PredictItConnector(callback=cb)
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        changed_events = await connector.poll_once()
        for event in changed_events:
            await connector.callback(event)

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_market_without_id_skipped(self):
        """Markets without an ID should be skipped."""
        response = {
            "markets": [
                {
                    "name": "No ID Market",
                    "contracts": [{"name": "Yes", "lastTradePrice": 0.50}],
                },
                {
                    "id": 999,
                    "name": "Valid Market",
                    "contracts": [{"name": "Yes", "lastTradePrice": 0.80}],
                },
            ]
        }
        connector = PredictItConnector()
        connector._session = _make_mock_session(response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["market_id"] == "999"

    @pytest.mark.asyncio
    async def test_close(self):
        connector = PredictItConnector()
        await connector.close()  # should be safe with no session

    def test_stop(self):
        connector = PredictItConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False

    @pytest.mark.asyncio
    async def test_volume_change_only_detected(self):
        """Volume-only changes still change the fingerprint since volume
        is not part of fingerprint — only price is. Verify this."""
        connector = PredictItConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)
        await connector.poll_once()

        # Only volume changed, price same.
        same_price_response = {
            "markets": [
                {
                    "id": 100,
                    "name": "Market A",
                    "contracts": [
                        {"name": "Yes", "lastTradePrice": 0.60, "totalSharesTraded": 9999},
                        {"name": "No", "lastTradePrice": 0.40, "totalSharesTraded": 9999},
                    ],
                },
            ]
        }
        connector._session = _make_mock_session(same_price_response)
        events = await connector.poll_once()
        # Fingerprint is based on price only, so no change detected.
        assert len(events) == 0
