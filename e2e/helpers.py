"""
E2E test helpers — Gamma API queries, synthetic probability generation,
timing utilities, and result formatting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger("prememora.e2e.helpers")

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ResolvedMarket:
    """A Polymarket market that has already resolved."""
    condition_id: str
    question: str
    outcome: str          # "YES" or "NO"
    yes_price: float      # final YES token price before resolution
    volume: float
    category: str
    resolved_at: str


@dataclass
class StageResult:
    """Result of one pipeline stage execution."""
    name: str
    success: bool
    duration_s: float
    detail: str = ""
    error: str = ""


@dataclass
class E2EReport:
    """Full E2E test report."""
    mode: str
    stages: list[StageResult] = field(default_factory=list)
    markets_tested: int = 0
    trades_opened: int = 0
    trades_resolved: int = 0
    brier_score: float | None = None
    total_pnl: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(s.success for s in self.stages)

    @property
    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  E2E TEST REPORT ({self.mode} mode)",
            f"{'=' * 60}",
            "",
        ]

        for s in self.stages:
            icon = "PASS" if s.success else "FAIL"
            lines.append(f"  [{icon}] {s.name:<30s} ({s.duration_s:.1f}s)")
            if s.detail:
                lines.append(f"         {s.detail}")
            if s.error:
                lines.append(f"         ERROR: {s.error}")

        lines.append("")
        lines.append(f"  Markets tested:   {self.markets_tested}")
        lines.append(f"  Trades opened:    {self.trades_opened}")
        lines.append(f"  Trades resolved:  {self.trades_resolved}")
        if self.brier_score is not None:
            lines.append(f"  Brier score:      {self.brier_score:.4f}")
        lines.append(f"  Total P&L:        ${self.total_pnl:+.2f}")
        lines.append("")

        verdict = "ALL STAGES PASSED" if self.all_passed else "SOME STAGES FAILED"
        lines.append(f"  Result: {verdict}")
        lines.append(f"{'=' * 60}")

        return "\n".join(lines)


# ── Gamma API helpers ────────────────────────────────────────────────────────


async def fetch_resolved_markets(
    max_markets: int = 5,
    min_volume: float = 100,
) -> list[ResolvedMarket]:
    """Fetch recently resolved markets from the Polymarket Gamma API.

    These are markets where we already know the outcome, making them
    ideal for smoke testing the full pipeline in one shot.
    """
    params: dict[str, Any] = {
        "limit": max_markets * 3,  # fetch extra, filter later
        "closed": "true",
        "order": "volume",
        "ascending": "false",
    }

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        try:
            async with session.get(
                f"{GAMMA_API_BASE}/markets", params=params
            ) as resp:
                if resp.status != 200:
                    logger.warning("Gamma API returned %d", resp.status)
                    return []
                data = await resp.json()
        except Exception as e:
            logger.error("Failed to fetch resolved markets: %s", e)
            return []

    markets = []
    for m in data:
        volume = float(m.get("volume") or m.get("volumeNum") or 0)
        if volume < min_volume:
            continue

        # Gamma API doesn't use resolution/resolvedAt. Instead:
        # - outcomePrices: ["1","0"] or ["0","1"] — winner gets "1"
        # - resolvedBy: non-empty address means resolved
        # - outcomes: ["Yes","No"] or custom labels
        resolved_by = m.get("resolvedBy") or ""
        outcome_prices = m.get("outcomePrices") or ""
        outcomes = m.get("outcomes") or ""
        closed_time = m.get("closedTime") or ""

        if not resolved_by or not outcome_prices or not closed_time:
            continue

        # Parse outcome prices and outcomes (stored as JSON strings or lists)
        try:
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices
            if isinstance(outcomes, str):
                labels = json.loads(outcomes)
            else:
                labels = outcomes
        except (json.JSONDecodeError, TypeError):
            continue

        if len(prices) < 2 or len(labels) < 2:
            continue

        # Find which outcome won (price = "1")
        winner_idx = None
        for i, p in enumerate(prices):
            if str(p) == "1":
                winner_idx = i
                break
        if winner_idx is None:
            continue

        # Only include Yes/No binary markets for smoke testing
        winner_label = labels[winner_idx].strip().upper()
        if winner_label in ("YES", "Y"):
            outcome = "YES"
        elif winner_label in ("NO", "N"):
            outcome = "NO"
        else:
            continue

        # For resolved markets, lastTradePrice is often 0 or 1 (extreme).
        # Generate a synthetic "evaluation price" that's biased toward
        # the correct outcome but not extreme — this simulates what the
        # market looked like before resolution.
        h = int(hashlib.sha256(m.get("question", "").encode()).hexdigest()[:4], 16)
        noise = (h % 100) / 100.0 * 0.3  # 0 to 0.3
        if outcome == "YES":
            yes_price = 0.55 + noise  # 0.55 to 0.85
        else:
            yes_price = 0.15 + noise  # 0.15 to 0.45

        markets.append(ResolvedMarket(
            condition_id=m.get("conditionId") or m.get("condition_id") or m.get("id", ""),
            question=m.get("question", ""),
            outcome=outcome,
            yes_price=yes_price,
            volume=volume,
            category=m.get("groupSlug", m.get("category", "")),
            resolved_at=closed_time,
        ))

        if len(markets) >= max_markets:
            break

    logger.info("Fetched %d resolved markets", len(markets))
    return markets


async def fetch_active_markets_for_live(
    max_markets: int = 3,
) -> list[dict[str, Any]]:
    """Fetch active markets for live mode testing.

    Returns raw market dicts (the pipeline trigger handles parsing).
    """
    params: dict[str, Any] = {
        "limit": max_markets,
        "active": "true",
        "closed": "false",
        "order": "volume",
        "ascending": "false",
    }

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        try:
            async with session.get(
                f"{GAMMA_API_BASE}/markets", params=params
            ) as resp:
                if resp.status != 200:
                    return []
                return await resp.json()
        except Exception as e:
            logger.error("Failed to fetch active markets: %s", e)
            return []


# ── Synthetic probabilities ──────────────────────────────────────────────────


def generate_synthetic_probability(
    question: str,
    market_price: float,
    outcome: str | None = None,
) -> float:
    """Generate a deterministic synthetic probability for a market.

    Uses a hash of the question to produce a stable probability that's
    biased toward the correct answer (if outcome is known). This lets us
    validate the pipeline mechanics without needing a live MiroFish simulation.

    For smoke tests (outcome known): biases toward correct answer with noise.
    For live tests (outcome unknown): adds random-ish spread around market price.
    """
    h = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
    noise = (h % 1000) / 1000.0  # 0.0 to 0.999

    if outcome is not None:
        # Smoke mode: bias toward correct answer
        correct_prob = 1.0 if outcome == "YES" else 0.0
        # Mix: 60% toward correct answer + 40% noise
        base = correct_prob * 0.6 + noise * 0.4
    else:
        # Live mode: spread around market price
        spread = (noise - 0.5) * 0.3  # ±15%
        base = market_price + spread

    return max(0.05, min(0.95, base))


# ── Timing utility ───────────────────────────────────────────────────────────


@contextmanager
def timed_stage(name: str, report: E2EReport):
    """Context manager that times a pipeline stage and records the result."""
    start = time.monotonic()
    result = StageResult(name=name, success=False, duration_s=0)
    try:
        yield result
        result.success = True
    except Exception as e:
        result.error = str(e)
        logger.error("Stage '%s' failed: %s", name, e)
    finally:
        result.duration_s = time.monotonic() - start
        report.stages.append(result)
