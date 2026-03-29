"""Tests for the Polymarket historical data module — DB operations and parsing."""

import json
import sqlite3
import pytest
from pathlib import Path

from backtesting.polymarket_history import (
    Market,
    PricePoint,
    _get_db,
    upsert_market,
    insert_prices,
    get_last_timestamp,
    get_market_row,
    cmd_stats,
    _parse_market,
)


@pytest.fixture
def db(tmp_path):
    """Create a temporary SQLite database."""
    db_path = tmp_path / "test.db"
    conn = _get_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def sample_market():
    return Market(
        id="cond_abc123",
        question="Will BTC hit $100k by June 2026?",
        description="Resolves YES if Bitcoin trades above $100,000.",
        category="crypto",
        outcomes='["Yes", "No"]',
        created_at="2026-01-01T00:00:00Z",
        resolved_at=None,
        resolution=None,
        volume=1_500_000.0,
        clob_token_ids='["token_yes", "token_no"]',
    )


class TestDatabaseOps:
    def test_upsert_and_retrieve(self, db, sample_market):
        upsert_market(db, sample_market)
        db.commit()
        row = get_market_row(db, "cond_abc123")
        assert row is not None
        assert row["question"] == "Will BTC hit $100k by June 2026?"
        assert row["category"] == "crypto"
        assert row["volume"] == 1_500_000.0

    def test_upsert_updates_existing(self, db, sample_market):
        upsert_market(db, sample_market)
        db.commit()
        sample_market.volume = 2_000_000.0
        upsert_market(db, sample_market)
        db.commit()
        row = get_market_row(db, "cond_abc123")
        assert row["volume"] == 2_000_000.0

    def test_get_nonexistent_market(self, db):
        assert get_market_row(db, "nonexistent") is None

    def test_insert_prices(self, db, sample_market):
        upsert_market(db, sample_market)
        db.commit()
        points = [
            PricePoint("cond_abc123", "token_yes", 1711699200, 0.65, "1h"),
            PricePoint("cond_abc123", "token_yes", 1711702800, 0.67, "1h"),
            PricePoint("cond_abc123", "token_no", 1711699200, 0.35, "1h"),
        ]
        inserted = insert_prices(db, points)
        db.commit()
        assert inserted == 3

    def test_insert_prices_dedup(self, db, sample_market):
        upsert_market(db, sample_market)
        db.commit()
        points = [PricePoint("cond_abc123", "token_yes", 1711699200, 0.65, "1h")]
        insert_prices(db, points)
        db.commit()
        # Insert again — should skip duplicate
        inserted = insert_prices(db, points)
        db.commit()
        assert inserted == 0

    def test_get_last_timestamp(self, db, sample_market):
        upsert_market(db, sample_market)
        points = [
            PricePoint("cond_abc123", "token_yes", 1711699200, 0.65, "1h"),
            PricePoint("cond_abc123", "token_yes", 1711702800, 0.67, "1h"),
        ]
        insert_prices(db, points)
        db.commit()
        last = get_last_timestamp(db, "cond_abc123", "token_yes", "1h")
        assert last == 1711702800

    def test_get_last_timestamp_none(self, db):
        last = get_last_timestamp(db, "nonexistent", "token", "1h")
        assert last is None


class TestParseMarket:
    def test_basic_gamma_response(self):
        raw = {
            "id": "0x123",
            "question": "Will X happen?",
            "description": "Some description",
            "category": "politics",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["tok_a", "tok_b"]',
            "createdAt": "2026-01-01T00:00:00Z",
            "resolvedAt": None,
            "resolution": None,
            "volume": "250000",
        }
        m = _parse_market(raw)
        assert m.id == "0x123"
        assert m.question == "Will X happen?"
        assert m.category == "politics"
        assert m.volume == 250000.0
        assert json.loads(m.clob_token_ids) == ["tok_a", "tok_b"]

    def test_outcomes_as_list(self):
        raw = {
            "id": "0x456",
            "question": "Test?",
            "outcomes": ["A", "B", "C"],
            "clobTokenIds": [],
        }
        m = _parse_market(raw)
        assert json.loads(m.outcomes) == ["A", "B", "C"]

    def test_missing_fields(self):
        raw = {"conditionId": "0x789"}
        m = _parse_market(raw)
        assert m.id == "0x789"
        assert m.question == ""
        assert m.volume == 0.0


class TestStats:
    def test_empty_db(self, tmp_path):
        stats = cmd_stats(db_path=tmp_path / "empty.db")
        assert stats["total_markets"] == 0
        assert stats["total_price_points"] == 0
