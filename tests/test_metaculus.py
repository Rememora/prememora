"""Tests for the Metaculus connector — probability extraction, change detection, event format."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from ingestors.metaculus import (
    MetaculusConnector,
    _extract_probability,
    _question_fingerprint,
    _build_event,
)


# ── Helper function tests ────────────────────────────────────────────────────


class TestExtractProbability:
    def test_standard_path(self):
        question = {
            "aggregations": {
                "recency_weighted": {
                    "latest": {
                        "centers": [0.35],
                    }
                }
            }
        }
        assert _extract_probability(question) == 0.35

    def test_multiple_centers_takes_first(self):
        question = {
            "aggregations": {
                "recency_weighted": {
                    "latest": {
                        "centers": [0.42, 0.58],
                    }
                }
            }
        }
        assert _extract_probability(question) == 0.42

    def test_empty_centers(self):
        question = {
            "aggregations": {
                "recency_weighted": {
                    "latest": {
                        "centers": [],
                    }
                }
            }
        }
        assert _extract_probability(question) is None

    def test_missing_aggregations(self):
        assert _extract_probability({}) is None

    def test_missing_recency_weighted(self):
        question = {"aggregations": {}}
        assert _extract_probability(question) is None

    def test_missing_latest(self):
        question = {"aggregations": {"recency_weighted": {}}}
        assert _extract_probability(question) is None

    def test_missing_centers_key(self):
        question = {"aggregations": {"recency_weighted": {"latest": {}}}}
        assert _extract_probability(question) is None

    def test_none_in_centers(self):
        """If centers contains None, float() should fail gracefully."""
        question = {
            "aggregations": {
                "recency_weighted": {
                    "latest": {
                        "centers": [None],
                    }
                }
            }
        }
        assert _extract_probability(question) is None

    def test_string_value_in_centers(self):
        """String values should be converted to float."""
        question = {
            "aggregations": {
                "recency_weighted": {
                    "latest": {
                        "centers": ["0.75"],
                    }
                }
            }
        }
        assert _extract_probability(question) == 0.75


class TestQuestionFingerprint:
    def test_deterministic(self):
        fp1 = _question_fingerprint(123, 0.35)
        fp2 = _question_fingerprint(123, 0.35)
        assert fp1 == fp2

    def test_different_probability_different_fingerprint(self):
        fp1 = _question_fingerprint(123, 0.35)
        fp2 = _question_fingerprint(123, 0.40)
        assert fp1 != fp2

    def test_different_question_different_fingerprint(self):
        fp1 = _question_fingerprint(123, 0.35)
        fp2 = _question_fingerprint(456, 0.35)
        assert fp1 != fp2

    def test_none_probability(self):
        fp = _question_fingerprint(123, None)
        assert "none" in fp

    def test_tiny_jitter_same_fingerprint(self):
        """Probability rounded to 4 decimals, so tiny jitter should match."""
        fp1 = _question_fingerprint(123, 0.350001)
        fp2 = _question_fingerprint(123, 0.350002)
        assert fp1 == fp2


class TestBuildEvent:
    def test_basic(self):
        question = {
            "id": 12345,
            "title": "Will AI pass the Turing test by 2030?",
            "number_of_forecasters": 450,
        }
        event = _build_event(question, 0.35)
        assert event["source"] == "metaculus"
        assert event["question_id"] == 12345
        assert event["question_title"] == "Will AI pass the Turing test by 2030?"
        assert event["community_probability"] == 0.35
        assert event["num_forecasters"] == 450
        assert "12345" in event["url"]
        assert event["timestamp"]

    def test_missing_fields(self):
        event = _build_event({}, 0.5)
        assert event["question_id"] == 0
        assert event["question_title"] == ""
        assert event["num_forecasters"] == 0


# ── Connector tests ──────────────────────────────────────────────────────────


SAMPLE_API_RESPONSE = {
    "results": [
        {
            "id": 100,
            "title": "Will Bitcoin reach $100k by 2026?",
            "number_of_forecasters": 500,
            "aggregations": {
                "recency_weighted": {
                    "latest": {"centers": [0.65]}
                }
            },
        },
        {
            "id": 200,
            "title": "Will there be a US recession in 2026?",
            "number_of_forecasters": 300,
            "aggregations": {
                "recency_weighted": {
                    "latest": {"centers": [0.30]}
                }
            },
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


class TestMetaculusConnector:
    def test_defaults(self):
        connector = MetaculusConnector()
        assert connector.poll_interval == 21600.0
        assert connector.callback is None
        assert connector._last_fingerprints == {}

    @pytest.mark.asyncio
    async def test_poll_once_first_call_returns_all(self):
        """First poll should return all questions with valid probabilities."""
        connector = MetaculusConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        events = await connector.poll_once()
        assert len(events) == 2
        assert events[0]["source"] == "metaculus"
        assert events[0]["question_id"] == 100
        assert events[0]["community_probability"] == 0.65
        assert events[1]["question_id"] == 200
        assert events[1]["community_probability"] == 0.30

    @pytest.mark.asyncio
    async def test_poll_once_no_change_returns_empty(self):
        """Second poll with same data should return empty (no changes)."""
        connector = MetaculusConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        events1 = await connector.poll_once()
        assert len(events1) == 2

        events2 = await connector.poll_once()
        assert len(events2) == 0

    @pytest.mark.asyncio
    async def test_poll_once_detects_probability_change(self):
        """If one question's probability changes, only that question is emitted."""
        connector = MetaculusConnector()
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        await connector.poll_once()

        changed_response = {
            "results": [
                {
                    "id": 100,
                    "title": "Will Bitcoin reach $100k by 2026?",
                    "number_of_forecasters": 510,
                    "aggregations": {
                        "recency_weighted": {
                            "latest": {"centers": [0.70]}  # changed from 0.65
                        }
                    },
                },
                {
                    "id": 200,
                    "title": "Will there be a US recession in 2026?",
                    "number_of_forecasters": 300,
                    "aggregations": {
                        "recency_weighted": {
                            "latest": {"centers": [0.30]}  # unchanged
                        }
                    },
                },
            ]
        }
        connector._session = _make_mock_session(changed_response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["question_id"] == 100
        assert events[0]["community_probability"] == 0.70

    @pytest.mark.asyncio
    async def test_poll_once_api_failure(self):
        """If API returns non-200, poll_once returns empty list."""
        connector = MetaculusConnector()
        connector._session = _make_mock_session({}, status=500)

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_poll_once_empty_results(self):
        """If API returns no results, poll_once returns empty list."""
        connector = MetaculusConnector()
        connector._session = _make_mock_session({"results": []})

        events = await connector.poll_once()
        assert events == []

    @pytest.mark.asyncio
    async def test_question_without_id_skipped(self):
        """Questions without an ID should be skipped."""
        response = {
            "results": [
                {
                    "title": "No ID Question",
                    "aggregations": {
                        "recency_weighted": {
                            "latest": {"centers": [0.50]}
                        }
                    },
                },
                {
                    "id": 999,
                    "title": "Valid Question",
                    "aggregations": {
                        "recency_weighted": {
                            "latest": {"centers": [0.80]}
                        }
                    },
                },
            ]
        }
        connector = MetaculusConnector()
        connector._session = _make_mock_session(response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["question_id"] == 999

    @pytest.mark.asyncio
    async def test_question_without_probability_skipped(self):
        """Questions without valid probability data should be skipped."""
        response = {
            "results": [
                {
                    "id": 111,
                    "title": "No probability data",
                    "aggregations": {},
                },
                {
                    "id": 222,
                    "title": "Valid",
                    "aggregations": {
                        "recency_weighted": {
                            "latest": {"centers": [0.55]}
                        }
                    },
                },
            ]
        }
        connector = MetaculusConnector()
        connector._session = _make_mock_session(response)

        events = await connector.poll_once()
        assert len(events) == 1
        assert events[0]["question_id"] == 222

    @pytest.mark.asyncio
    async def test_callback_invoked_for_each_question(self):
        """Verify callback is called once per changed question."""
        events = []

        async def cb(event):
            events.append(event)

        connector = MetaculusConnector(callback=cb)
        connector._session = _make_mock_session(SAMPLE_API_RESPONSE)

        changed_events = await connector.poll_once()
        for event in changed_events:
            await connector.callback(event)

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_close(self):
        connector = MetaculusConnector()
        await connector.close()  # should be safe with no session

    def test_stop(self):
        connector = MetaculusConnector()
        connector._running = True
        connector.stop()
        assert connector._running is False
