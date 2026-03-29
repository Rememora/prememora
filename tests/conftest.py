"""Shared fixtures for ingestor tests."""

import pytest


@pytest.fixture
def collected_events():
    """Returns a list that callbacks can append to, plus an async callback."""
    events = []

    async def callback(event):
        events.append(event)

    return events, callback


@pytest.fixture
def collected_batches():
    """Returns a list for batch callbacks (Reddit-style: callback receives a list)."""
    batches = []

    async def callback(event_list):
        batches.extend(event_list)

    return batches, callback
