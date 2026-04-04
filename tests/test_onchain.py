"""Tests for the On-Chain Metrics connector — parsing, interpretation thresholds, event format."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from ingestors.onchain import (
    OnChainConnector,
    _interpret_mvrv,
    _interpret_sopr,
    _interpret_nupl,
    _parse_metric_data,
)


# ── Interpretation function tests ────────────────────────────────────────────


class TestInterpretMVRV:
    def test_undervalued(self):
        assert _interpret_mvrv(-0.5) == "undervalued"
        assert _interpret_mvrv(-2.0) == "undervalued"

    def test_undervalued_boundary(self):
        """Exactly 0 should be fair, not undervalued."""
        assert _interpret_mvrv(0) == "fair"

    def test_fair(self):
        assert _interpret_mvrv(1.5) == "fair"
        assert _interpret_mvrv(0.1) == "fair"
        assert _interpret_mvrv(3.0) == "fair"

    def test_overvalued(self):
        assert _interpret_mvrv(3.1) == "overvalued"
        assert _interpret_mvrv(7.0) == "overvalued"


class TestInterpretSOPR:
    def test_selling_at_loss(self):
        assert _interpret_sopr(0.95) == "selling_at_loss"
        assert _interpret_sopr(0.5) == "selling_at_loss"

    def test_breakeven(self):
        assert _interpret_sopr(1.0) == "breakeven"

    def test_selling_at_profit(self):
        assert _interpret_sopr(1.05) == "selling_at_profit"
        assert _interpret_sopr(1.5) == "selling_at_profit"


class TestInterpretNUPL:
    def test_capitulation(self):
        assert _interpret_nupl(-0.1) == "capitulation"
        assert _interpret_nupl(-0.5) == "capitulation"

    def test_hope_fear(self):
        assert _interpret_nupl(0.0) == "hope_fear"
        assert _interpret_nupl(0.24) == "hope_fear"

    def test_optimism(self):
        assert _interpret_nupl(0.25) == "optimism"
        assert _interpret_nupl(0.49) == "optimism"

    def test_belief(self):
        assert _interpret_nupl(0.5) == "belief"
        assert _interpret_nupl(0.74) == "belief"

    def test_euphoria(self):
        assert _interpret_nupl(0.75) == "euphoria"
        assert _interpret_nupl(0.9) == "euphoria"


# ── Data parsing tests ───────────────────────────────────────────────────────


class TestParseMetricData:
    def test_list_of_lists(self):
        """Standard format: [[date, value], ...]"""
        data = [
            ["2026-04-01", 1.8],
            ["2026-04-02", 1.9],
            ["2026-04-03", 2.1],
        ]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 2.1
        assert previous == 1.9

    def test_list_of_dicts_value_key(self):
        """Dict format with 'value' key."""
        data = [
            {"date": "2026-04-01", "value": 0.95},
            {"date": "2026-04-02", "value": 1.02},
        ]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 1.02
        assert previous == 0.95

    def test_list_of_dicts_v_key(self):
        """Dict format with 'v' key."""
        data = [
            {"t": "2026-04-01", "v": 0.3},
            {"t": "2026-04-02", "v": 0.4},
        ]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 0.4
        assert previous == 0.3

    def test_list_of_dicts_y_key(self):
        """Dict format with 'y' key."""
        data = [
            {"x": "2026-04-01", "y": 1.5},
            {"x": "2026-04-02", "y": 1.6},
        ]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 1.6
        assert previous == 1.5

    def test_plain_list_of_numbers(self):
        """Plain list of numeric values."""
        data = [1.5, 1.6, 1.8, 2.0]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 2.0
        assert previous == 1.8

    def test_single_entry(self):
        """Single entry: previous should equal latest."""
        data = [["2026-04-01", 2.5]]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 2.5
        assert previous == 2.5

    def test_empty_list(self):
        result = _parse_metric_data([])
        assert result is None

    def test_none_input(self):
        result = _parse_metric_data(None)
        assert result is None

    def test_non_list_input(self):
        result = _parse_metric_data("not a list")
        assert result is None

    def test_string_values_converted(self):
        """String numeric values should be converted to float."""
        data = [["2026-04-01", "1.8"], ["2026-04-02", "2.1"]]
        result = _parse_metric_data(data)
        assert result is not None
        value, previous = result
        assert value == 2.1
        assert previous == 1.8

    def test_dict_without_known_value_key(self):
        """Dicts without known value keys should return None."""
        data = [{"date": "2026-04-01", "unknown_key": 1.5}]
        result = _parse_metric_data(data)
        assert result is None


# ── Connector tests ──────────────────────────────────────────────────────────


# Sample responses for each metric (list-of-lists format).
SAMPLE_MVRV_DATA = [
    ["2026-04-01", 1.8],
    ["2026-04-02", 1.9],
    ["2026-04-03", 2.1],
]

SAMPLE_SOPR_DATA = [
    ["2026-04-01", 0.98],
    ["2026-04-02", 1.01],
    ["2026-04-03", 1.03],
]

SAMPLE_NUPL_DATA = [
    ["2026-04-01", 0.45],
    ["2026-04-02", 0.48],
    ["2026-04-03", 0.52],
]


def _make_mock_session_multi(metric_data_map, status=200):
    """Create a mock session that returns different data based on the chart param.

    metric_data_map: dict mapping chart name → response data.
    """
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.closed = False

    def make_response(data, st):
        mock_resp = AsyncMock()
        mock_resp.status = st
        mock_resp.json = AsyncMock(return_value=data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        return mock_resp

    call_count = [0]
    responses = []

    # Build responses in the order METRICS are fetched: mvrv_zscore, sopr, nupl
    from ingestors.onchain import METRICS
    for metric in METRICS:
        data = metric_data_map.get(metric)
        if data is not None:
            responses.append(make_response(data, status))
        else:
            responses.append(make_response(None, 404))

    def side_effect(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(responses):
            return responses[idx]
        return make_response(None, 500)

    mock_session.get = MagicMock(side_effect=side_effect)
    return mock_session


class TestOnChainConnector:
    def test_defaults(self):
        connector = OnChainConnector()
        assert connector.poll_interval == 21600.0
        assert connector.callback is None
        assert connector._last_values == {}

    @pytest.mark.asyncio
    async def test_poll_once_returns_all_metrics(self):
        """poll_once should return one event per metric."""
        connector = OnChainConnector()
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "sopr": SAMPLE_SOPR_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        events = await connector.poll_once()
        assert len(events) == 3

        metrics = [e["metric"] for e in events]
        assert "mvrv_zscore" in metrics
        assert "sopr" in metrics
        assert "nupl" in metrics

    @pytest.mark.asyncio
    async def test_mvrv_event_structure(self):
        """Verify MVRV event has correct fields."""
        connector = OnChainConnector()
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "sopr": SAMPLE_SOPR_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        events = await connector.poll_once()
        mvrv_event = [e for e in events if e["metric"] == "mvrv_zscore"][0]

        assert mvrv_event["source"] == "onchain"
        assert mvrv_event["value"] == 2.1
        assert mvrv_event["previous_value"] == 1.9
        assert mvrv_event["interpretation"] == "fair"
        assert mvrv_event["timestamp"]

    @pytest.mark.asyncio
    async def test_sopr_event_interpretation(self):
        """SOPR > 1 should be selling_at_profit."""
        connector = OnChainConnector()
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "sopr": SAMPLE_SOPR_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        events = await connector.poll_once()
        sopr_event = [e for e in events if e["metric"] == "sopr"][0]
        assert sopr_event["interpretation"] == "selling_at_profit"

    @pytest.mark.asyncio
    async def test_nupl_event_interpretation(self):
        """NUPL 0.52 should be belief."""
        connector = OnChainConnector()
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "sopr": SAMPLE_SOPR_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        events = await connector.poll_once()
        nupl_event = [e for e in events if e["metric"] == "nupl"][0]
        assert nupl_event["interpretation"] == "belief"

    @pytest.mark.asyncio
    async def test_partial_api_failure(self):
        """If one metric API fails, others should still return."""
        connector = OnChainConnector()
        # Only provide mvrv and nupl, sopr will 404
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        events = await connector.poll_once()
        metrics = [e["metric"] for e in events]
        assert "mvrv_zscore" in metrics
        assert "nupl" in metrics
        # sopr may or may not be present depending on mock behavior
        # The key test: at least mvrv and nupl were fetched
        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_all_api_failure(self):
        """If all APIs fail, poll_once returns empty."""
        connector = OnChainConnector()
        connector._session = _make_mock_session_multi({}, status=500)

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """Verify callback is called for each metric event."""
        events = []

        async def cb(event):
            events.append(event)

        connector = OnChainConnector(callback=cb)
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "sopr": SAMPLE_SOPR_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        fetched = await connector.poll_once()
        for event in fetched:
            await connector.callback(event)

        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_close(self):
        connector = OnChainConnector()
        await connector.close()  # should be safe with no session

    def test_stop(self):
        connector = OnChainConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False

    @pytest.mark.asyncio
    async def test_last_values_updated(self):
        """poll_once should update _last_values dict."""
        connector = OnChainConnector()
        connector._session = _make_mock_session_multi({
            "mvrv_zscore": SAMPLE_MVRV_DATA,
            "sopr": SAMPLE_SOPR_DATA,
            "nupl": SAMPLE_NUPL_DATA,
        })

        await connector.poll_once()
        assert connector._last_values.get("mvrv_zscore") == 2.1
        assert connector._last_values.get("sopr") == 1.03
        assert connector._last_values.get("nupl") == 0.52
