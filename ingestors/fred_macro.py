"""
FRED API macro-economic context connector.

Polls the Federal Reserve Economic Data API for key macro indicators and
emits normalised event dicts suitable for downstream graph ingestion.

Env vars
--------
FRED_API_KEY   – API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_INTERVAL  – polling interval in seconds (default 3600 = 1 hour)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Sequence

from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

logger = logging.getLogger(__name__)

# ── default series ──────────────────────────────────────────────────
DEFAULT_SERIES: list[str] = [
    "FEDFUNDS",  # Federal Funds Effective Rate
    "CPIAUCSL",  # Consumer Price Index (All Urban Consumers)
    "UNRATE",    # Unemployment Rate
    "GDP",       # Gross Domestic Product
    "DGS10",     # 10-Year Treasury Constant Maturity Rate
    "DGS2",      # 2-Year Treasury Constant Maturity Rate
    "DEXUSEU",   # USD/EUR Exchange Rate
    "VIXCLS",    # CBOE Volatility Index (VIX)
]

# Friendly names for series (FRED provides these via info, but we cache a
# local mapping so we don't need an extra API call per series each cycle).
SERIES_NAMES: dict[str, str] = {
    "FEDFUNDS": "Federal Funds Effective Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE":   "Unemployment Rate",
    "GDP":      "Gross Domestic Product",
    "DGS10":    "10-Year Treasury Rate",
    "DGS2":     "2-Year Treasury Rate",
    "DEXUSEU":  "USD/EUR Exchange Rate",
    "VIXCLS":   "CBOE Volatility Index (VIX)",
}

# Type alias for the async callback that receives a batch of events.
EventCallback = Callable[[list[dict[str, Any]]], Coroutine[Any, Any, None]]


# ── helpers ─────────────────────────────────────────────────────────

def _series_units(fred: Fred, series_id: str) -> str:
    """Return the *units* string from FRED series info."""
    try:
        info = fred.get_series_info(series_id)
        return str(info.get("units", ""))
    except Exception:
        return ""


def _build_event(
    series_id: str,
    observation_date: datetime,
    value: float,
    previous_value: float | None,
    units: str,
) -> dict[str, Any]:
    """Return a normalised event dict."""
    change: float | None = None
    if previous_value is not None:
        change = round(value - previous_value, 6)

    return {
        "source": "fred",
        "timestamp": observation_date.isoformat(),
        "series_id": series_id,
        "series_name": SERIES_NAMES.get(series_id, series_id),
        "value": value,
        "previous_value": previous_value,
        "change": change,
        "units": units,
    }


# ── core connector ──────────────────────────────────────────────────

class FredMacroConnector:
    """Polls FRED for macro-economic observations and fires callbacks."""

    def __init__(
        self,
        api_key: str | None = None,
        series: Sequence[str] | None = None,
        interval: int | None = None,
        callback: EventCallback | None = None,
    ) -> None:
        self.api_key = api_key or os.environ["FRED_API_KEY"]
        self.fred = Fred(api_key=self.api_key)
        self.series: list[str] = list(series or DEFAULT_SERIES)
        self.interval: int = interval or int(os.getenv("FRED_INTERVAL", "3600"))
        self.callback = callback

        # last-seen observation date per series (ISO date string → avoids re-emitting)
        self._last_seen: dict[str, str] = {}
        # cached units per series
        self._units_cache: dict[str, str] = {}
        # running task handle
        self._task: asyncio.Task | None = None

    # ── public API ──────────────────────────────────────────────────

    def fetch_latest(self) -> list[dict[str, Any]]:
        """Synchronously fetch the latest observation for every tracked series.

        Returns a list of event dicts (one per series that has data).
        Does *not* update ``_last_seen`` — use :meth:`poll_once` for that.
        """
        events: list[dict[str, Any]] = []
        for sid in self.series:
            try:
                event = self._fetch_series(sid, track=False)
                if event is not None:
                    events.append(event)
            except Exception:
                logger.exception("Error fetching %s", sid)
        return events

    def poll_once(self) -> list[dict[str, Any]]:
        """Fetch latest observations, emit only *new* ones, update state."""
        events: list[dict[str, Any]] = []
        for sid in self.series:
            try:
                event = self._fetch_series(sid, track=True)
                if event is not None:
                    events.append(event)
            except Exception:
                logger.exception("Error fetching %s", sid)
        return events

    async def start(self) -> None:
        """Begin periodic polling in the background.  Requires a running event loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "FredMacroConnector started — polling %d series every %ds",
            len(self.series),
            self.interval,
        )

    async def stop(self) -> None:
        """Cancel the background polling task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("FredMacroConnector stopped")

    # ── internals ───────────────────────────────────────────────────

    def _get_units(self, series_id: str) -> str:
        if series_id not in self._units_cache:
            self._units_cache[series_id] = _series_units(self.fred, series_id)
        return self._units_cache[series_id]

    def _fetch_series(
        self,
        series_id: str,
        *,
        track: bool,
    ) -> dict[str, Any] | None:
        """Fetch the two most recent observations for *series_id*.

        If *track* is True, only return an event when the observation is newer
        than the last-seen date (and update ``_last_seen``).
        """
        # Get last 2 observations so we can compute change
        data = self.fred.get_series(series_id, observation_start=None)
        data = data.dropna()
        if data.empty:
            return None

        latest_date = data.index[-1]
        latest_value = float(data.iloc[-1])
        obs_date_str = latest_date.strftime("%Y-%m-%d")

        if track:
            prev = self._last_seen.get(series_id)
            if prev == obs_date_str:
                return None  # no new observation
            self._last_seen[series_id] = obs_date_str

        previous_value: float | None = None
        if len(data) >= 2:
            previous_value = float(data.iloc[-2])

        units = self._get_units(series_id)
        obs_dt = datetime(
            latest_date.year,
            latest_date.month,
            latest_date.day,
            tzinfo=timezone.utc,
        )

        return _build_event(series_id, obs_dt, latest_value, previous_value, units)

    async def _poll_loop(self) -> None:
        """Infinite loop: poll → callback → sleep."""
        while True:
            events = self.poll_once()
            if events and self.callback is not None:
                try:
                    await self.callback(events)
                except Exception:
                    logger.exception("Callback error")
            if events:
                logger.info(
                    "Emitted %d FRED events: %s",
                    len(events),
                    [e["series_id"] for e in events],
                )
            await asyncio.sleep(self.interval)


# ── CLI helper ──────────────────────────────────────────────────────

def _print_events(events: list[dict[str, Any]]) -> None:
    for e in events:
        change_str = ""
        if e["change"] is not None:
            sign = "+" if e["change"] >= 0 else ""
            change_str = f"  ({sign}{e['change']})"
        print(
            f"[{e['timestamp']}] {e['series_id']:10s} "
            f"{e['series_name']:40s} "
            f"{e['value']:>12.4f}{change_str}"
            f"  {e['units']}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = FredMacroConnector()
    events = connector.fetch_latest()
    if events:
        _print_events(events)
    else:
        print("No data returned — check FRED_API_KEY")
