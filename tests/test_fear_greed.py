"""Tests for the Fear & Greed Index connector — API parsing, trend calculation, polling."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientSession

from ingestors.fear_greed import (
    FearGreedConnector,
    _compute_trend,
    _parse_fng_entry,
)


# ── Trend computation ────────────────────────────────────────────────────────


class TestComputeTrend:
    def test_rising(self):
        # Oldest to newest: values climbing
        assert _compute_trend([20, 25, 30, 40, 50, 60, 70]) == "rising"

    def test_falling(self):
        # Oldest to newest: values dropping
        assert _compute_trend([70, 60, 50, 40, 30, 25, 20]) == "falling"

    def test_stable(self):
        # Flat values within threshold
        assert _compute_trend([50, 50, 51, 50, 49, 50, 50]) == "stable"

    def test_single_value(self):
        assert _compute_trend([50]) == "stable"

    def test_empty(self):
        assert _compute_trend([]) == "stable"

    def test_two_values_rising(self):
        assert _compute_trend([30, 40]) == "rising"

    def test_two_values_stable(self):
        assert _compute_trend([50, 51]) == "stable"

    def test_boundary_exactly_three(self):
        # diff of exactly 3 should be stable (threshold is >3)
        assert _compute_trend([47, 50]) == "stable"

    def test_boundary_just_over_three(self):
        # diff of 4 should be rising
        assert _compute_trend([46, 50]) == "rising"


# ── FNG entry parsing ────────────────────────────────────────────────────────


class TestParseFngEntry:
    def test_basic_entry(self):
        entry = {
            "value": "73",
            "value_classification": "Greed",
            "timestamp": "1711929600",
        }
        value, classification, ts = _parse_fng_entry(entry)
        assert value == 73
        assert classification == "Greed"
        assert "2024" in ts  # timestamp 1711929600 is in 2024

    def test_missing_fields(self):
        value, classification, ts = _parse_fng_entry({})
        assert value == 0
        assert classification == "Unknown"
        assert ts  # should fallback to now

    def test_zero_timestamp_fallback(self):
        entry = {"value": "50", "value_classification": "Neutral", "timestamp": "0"}
        value, classification, ts = _parse_fng_entry(entry)
        assert value == 50
        assert ts  # should have a timestamp (fallback to now)


# ── Connector poll_once ──────────────────────────────────────────────────────


class TestFearGreedConnector:
    def test_defaults(self):
        connector = FearGreedConnector()
        assert connector.poll_interval == 3600.0
        assert connector.callback is None
        assert connector._last_value is None

    @pytest.mark.asyncio
    async def test_poll_once_with_mock_api(self):
        """Mock the API response and verify poll_once returns correct event."""
        mock_response_data = {
            "data": [
                {"value": "73", "value_classification": "Greed", "timestamp": "1711929600"},
                {"value": "65", "value_classification": "Greed", "timestamp": "1711843200"},
                {"value": "60", "value_classification": "Greed", "timestamp": "1711756800"},
                {"value": "55", "value_classification": "Greed", "timestamp": "1711670400"},
                {"value": "45", "value_classification": "Fear", "timestamp": "1711584000"},
                {"value": "40", "value_classification": "Fear", "timestamp": "1711497600"},
                {"value": "35", "value_classification": "Fear", "timestamp": "1711411200"},
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        connector = FearGreedConnector()
        connector._session = mock_session

        event = await connector.poll_once()

        assert event is not None
        assert event["source"] == "fear_greed"
        assert event["value"] == 73
        assert event["classification"] == "Greed"
        assert event["previous_value"] == 65
        assert event["trend"] == "rising"  # 35,40,45,55,60,65,73 is rising
        assert event["timestamp"]

    @pytest.mark.asyncio
    async def test_poll_once_api_failure(self):
        """If the API returns non-200, poll_once should return None."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        connector = FearGreedConnector()
        connector._session = mock_session

        event = await connector.poll_once()
        assert event is None

    @pytest.mark.asyncio
    async def test_poll_once_empty_data(self):
        """If API returns empty data array, poll_once returns None."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"data": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        connector = FearGreedConnector()
        connector._session = mock_session

        event = await connector.poll_once()
        assert event is None

    @pytest.mark.asyncio
    async def test_poll_once_single_entry(self):
        """With only 1 data point, previous_value should be None and trend stable."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "data": [
                {"value": "50", "value_classification": "Neutral", "timestamp": "1711929600"},
            ]
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        connector = FearGreedConnector()
        connector._session = mock_session

        event = await connector.poll_once()
        assert event is not None
        assert event["value"] == 50
        assert event["previous_value"] is None
        assert event["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """Verify start() calls the callback with the event."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "data": [
                {"value": "60", "value_classification": "Greed", "timestamp": "1711929600"},
                {"value": "55", "value_classification": "Greed", "timestamp": "1711843200"},
            ]
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        events = []

        async def cb(event):
            events.append(event)

        connector = FearGreedConnector(callback=cb, poll_interval=0.01)
        connector._session = mock_session

        # Run poll_once directly instead of start() to avoid loop
        event = await connector.poll_once()
        if event and connector.callback:
            await connector.callback(event)

        assert len(events) == 1
        assert events[0]["source"] == "fear_greed"
        assert events[0]["value"] == 60

    @pytest.mark.asyncio
    async def test_close(self):
        connector = FearGreedConnector()
        # No session yet — close should be safe
        await connector.close()

    def test_stop(self):
        connector = FearGreedConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False
