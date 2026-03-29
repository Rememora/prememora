"""Tests for the whale tracker — classification, normalization, config."""

import pytest
from ingestors.whale_tracker import (
    WhaleTracker,
    WhaleTrackerConfig,
    TxClassification,
    classify_transaction,
    _normalize_chain,
    _is_exchange,
    _build_event,
)


class TestNormalizeChain:
    def test_canonical_names(self):
        assert _normalize_chain("bitcoin") == "bitcoin"
        assert _normalize_chain("Bitcoin") == "bitcoin"
        assert _normalize_chain("BTC") == "bitcoin"
        assert _normalize_chain("ethereum") == "ethereum"
        assert _normalize_chain("ETH") == "ethereum"
        assert _normalize_chain("solana") == "solana"
        assert _normalize_chain("SOL") == "solana"

    def test_unknown_chain(self):
        assert _normalize_chain("polygon") == "polygon"


class TestIsExchange:
    def test_known_binance_eth(self):
        assert _is_exchange("0x28c6c06298d514db089934071355e5743bf21d60", "ethereum")

    def test_case_insensitive(self):
        assert _is_exchange("0x28C6C06298D514DB089934071355E5743BF21D60", "ethereum")

    def test_unknown_address(self):
        assert not _is_exchange("0xdeadbeef", "ethereum")

    def test_empty_address(self):
        assert not _is_exchange("", "ethereum")


class TestClassifyTransaction:
    def test_exchange_inflow(self):
        result = classify_transaction(
            from_address="0xunknown",
            to_address="0x28c6c06298d514db089934071355e5743bf21d60",
            chain="ethereum",
        )
        assert result == TxClassification.EXCHANGE_INFLOW

    def test_exchange_outflow(self):
        result = classify_transaction(
            from_address="0x28c6c06298d514db089934071355e5743bf21d60",
            to_address="0xunknown",
            chain="ethereum",
        )
        assert result == TxClassification.EXCHANGE_OUTFLOW

    def test_exchange_to_exchange_by_owner_type(self):
        result = classify_transaction(
            from_address="0xaaa",
            to_address="0xbbb",
            chain="ethereum",
            from_owner="exchange",
            to_owner="exchange",
        )
        assert result == TxClassification.EXCHANGE_TO_EXCHANGE

    def test_whale_transfer(self):
        result = classify_transaction(
            from_address="0xaaa",
            to_address="0xbbb",
            chain="ethereum",
        )
        assert result == TxClassification.WHALE_TRANSFER

    def test_owner_type_exchange_inflow(self):
        result = classify_transaction(
            from_address="0xaaa",
            to_address="0xbbb",
            chain="ethereum",
            from_owner="unknown",
            to_owner="exchange",
        )
        assert result == TxClassification.EXCHANGE_INFLOW


class TestBuildEvent:
    def test_basic_transaction(self):
        tx = {
            "blockchain": "ethereum",
            "timestamp": 1711699200,
            "from": {"address": "0xaaa", "owner": "Whale", "owner_type": "unknown"},
            "to": {"address": "0xbbb", "owner": "Binance", "owner_type": "exchange"},
            "amount_usd": 5_000_000,
            "symbol": "eth",
            "hash": "0xdeadbeef",
        }
        event = _build_event(tx)
        assert event["source"] == "whale_alert"
        assert event["chain"] == "ethereum"
        assert event["amount_usd"] == 5_000_000
        assert event["token"] == "ETH"
        assert event["tx_hash"] == "0xdeadbeef"
        assert event["from_entity"] == "Whale"
        assert event["to_entity"] == "Binance"
        assert event["classification"] == "exchange_inflow"

    def test_missing_from_to(self):
        tx = {"blockchain": "bitcoin", "timestamp": 1711699200}
        event = _build_event(tx)
        assert event["from_entity"] == "unknown"
        assert event["to_entity"] == "unknown"
        assert event["classification"] == "whale_transfer"


class TestWhaleTrackerConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("WHALE_ALERT_API_KEY", raising=False)
        config = WhaleTrackerConfig()
        assert config.api_key == ""
        assert config.min_value_usd == 1_000_000
        assert "bitcoin" in config.chains

    def test_chain_normalization(self):
        config = WhaleTrackerConfig(chains=["BTC", "ETH"])
        assert config.chains == ["bitcoin", "ethereum"]


class TestWhaleTrackerNoKey:
    @pytest.mark.asyncio
    async def test_poll_without_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("WHALE_ALERT_API_KEY", raising=False)
        tracker = WhaleTracker(config=WhaleTrackerConfig(api_key=""))
        events = await tracker.poll_once()
        assert events == []
        await tracker.close()
