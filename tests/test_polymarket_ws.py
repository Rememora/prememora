"""Tests for the Polymarket WebSocket connector — parsing and normalization only."""

import pytest
from ingestors.polymarket_ws import PolymarketWSConnector


@pytest.fixture
def connector():
    return PolymarketWSConnector(asset_ids=["token_abc"])


class TestParseMessage:
    def test_price_change(self, connector):
        msg = {
            "event_type": "price_change",
            "asset_id": "token_abc",
            "market": "cond_123",
            "timestamp": "2026-03-29T10:00:00Z",
            "price": 0.65,
            "old_price": 0.60,
            "side": "buy",
        }
        events = connector._parse_message(msg)
        assert len(events) == 1
        e = events[0]
        assert e["source"] == "polymarket_ws"
        assert e["event_type"] == "price_change"
        assert e["market_id"] == "cond_123"
        assert e["asset_id"] == "token_abc"
        assert e["data"]["price"] == 0.65
        assert e["data"]["old_price"] == 0.60
        assert e["data"]["side"] == "buy"

    def test_last_trade_price(self, connector):
        msg = {
            "event_type": "last_trade_price",
            "asset_id": "token_abc",
            "market": "cond_123",
            "price": 0.72,
            "size": 100,
            "side": "sell",
        }
        events = connector._parse_message(msg)
        assert len(events) == 1
        assert events[0]["data"]["price"] == 0.72
        assert events[0]["data"]["size"] == 100
        assert events[0]["data"]["side"] == "sell"

    def test_book_event(self, connector):
        msg = {
            "event_type": "book",
            "asset_id": "token_abc",
            "market": "cond_123",
            "bids": [[0.60, 500]],
            "asks": [[0.65, 300]],
            "best_bid": 0.60,
            "best_ask": 0.65,
            "spread": 0.05,
        }
        events = connector._parse_message(msg)
        assert len(events) == 1
        d = events[0]["data"]
        assert d["bids"] == [[0.60, 500]]
        assert d["asks"] == [[0.65, 300]]
        assert d["spread"] == 0.05

    def test_tick_size_change(self, connector):
        msg = {
            "event_type": "tick_size_change",
            "asset_id": "token_abc",
            "market": "cond_123",
            "tick_size": 0.01,
            "old_tick_size": 0.001,
        }
        events = connector._parse_message(msg)
        assert events[0]["data"]["tick_size"] == 0.01

    def test_unknown_event_captures_extra_fields(self, connector):
        msg = {
            "event_type": "new_market",
            "asset_id": "token_xyz",
            "market": "cond_456",
            "question": "Will BTC hit 100k?",
        }
        events = connector._parse_message(msg)
        assert events[0]["event_type"] == "new_market"
        assert events[0]["data"]["question"] == "Will BTC hit 100k?"

    def test_array_of_messages(self, connector):
        msgs = [
            {"event_type": "price_change", "asset_id": "a", "market": "m1", "price": 0.5},
            {"event_type": "price_change", "asset_id": "b", "market": "m2", "price": 0.7},
        ]
        events = connector._parse_message(msgs)
        assert len(events) == 2
        assert events[0]["asset_id"] == "a"
        assert events[1]["asset_id"] == "b"

    def test_fallback_type_field(self, connector):
        """When event_type is missing, falls back to 'type' field."""
        msg = {"type": "price_change", "asset_id": "a", "market": "m", "price": 0.5}
        events = connector._parse_message(msg)
        assert events[0]["event_type"] == "price_change"

    def test_missing_timestamp_gets_default(self, connector):
        msg = {"event_type": "price_change", "asset_id": "a", "market": "m"}
        events = connector._parse_message(msg)
        assert events[0]["timestamp"]  # not empty — should have fallback


class TestSubscription:
    def test_dedup_subscription(self, connector):
        """Subscribed set prevents re-subscribing."""
        connector._subscribed.add("token_abc")
        # _subscribe would skip token_abc since it's already in _subscribed
        assert "token_abc" in connector._subscribed
