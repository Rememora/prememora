"""
Strategy review — closes the feedback loop by scoring predictions against
actual market outcomes and generating actionable recommendations.

Components:
  1. Auto-resolver: polls Polymarket for resolved markets, settles positions
  2. Scoreboard: Brier score, calibration, source attribution, P&L analysis
  3. Recommendations: specific parameter adjustments based on findings

Usage:
    review = StrategyReview(paper_engine=engine, signal_log_path=Path("data/signal_log.jsonl"))
    await review.auto_resolve()     # check for resolved markets, settle positions
    report = review.generate_report()
    print(report.summary)

CLI:
    python -m trading.strategy_review resolve     # auto-resolve markets
    python -m trading.strategy_review report      # generate full report
    python -m trading.strategy_review calibration # show calibration curve
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from trading.paper_engine import PaperTradingEngine

logger = logging.getLogger("prememora.trading.strategy_review")

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DEFAULT_SIGNAL_LOG = Path(__file__).resolve().parent.parent / "data" / "signal_log.jsonl"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class SignalRecord:
    """A parsed signal log entry."""
    cycle_time: str
    market_id: str
    question: str
    market_price: float
    our_probability: float | None
    action: str
    side: str
    edge: float | None
    reason: str
    context_facts: int
    executed: bool
    position_id: str


@dataclass
class CalibrationBucket:
    """Calibration data for a probability range."""
    bucket_low: float
    bucket_high: float
    predicted_mean: float
    actual_win_rate: float
    count: int
    error: float  # predicted - actual


@dataclass
class SourceStats:
    """P&L statistics attributed to a signal source pattern."""
    source: str
    trade_count: int
    total_pnl: float
    win_count: int
    avg_edge: float


@dataclass
class StrategyReport:
    """Complete strategy review report."""
    # Overview
    total_signals: int
    executed_signals: int
    resolved_trades: int
    open_trades: int

    # Scoring
    brier_score: float | None       # 0 = perfect, 0.25 = random
    win_rate: float | None
    avg_pnl_per_trade: float | None
    total_pnl: float

    # Calibration
    calibration: list[CalibrationBucket]

    # Source attribution
    source_stats: list[SourceStats]

    # Recommendations
    recommendations: list[str]

    @property
    def summary(self) -> str:
        lines = ["=== Strategy Review ===", ""]

        # Overview
        lines.append(f"Signals generated: {self.total_signals}")
        lines.append(f"Trades executed:   {self.executed_signals}")
        lines.append(f"Resolved:          {self.resolved_trades}")
        lines.append(f"Still open:        {self.open_trades}")
        lines.append("")

        # Scoring
        if self.brier_score is not None:
            quality = "excellent" if self.brier_score < 0.1 else "good" if self.brier_score < 0.2 else "fair" if self.brier_score < 0.25 else "poor"
            lines.append(f"Brier score:       {self.brier_score:.4f} ({quality})")
        if self.win_rate is not None:
            lines.append(f"Win rate:          {self.win_rate:.1%}")
        if self.avg_pnl_per_trade is not None:
            lines.append(f"Avg P&L per trade: ${self.avg_pnl_per_trade:+.2f}")
        lines.append(f"Total P&L:         ${self.total_pnl:+.2f}")
        lines.append("")

        # Calibration
        if self.calibration:
            lines.append("Calibration (predicted → actual):")
            for b in self.calibration:
                bar = "█" * int(b.actual_win_rate * 20) + "░" * (20 - int(b.actual_win_rate * 20))
                lines.append(
                    f"  {b.bucket_low:.0%}-{b.bucket_high:.0%}: "
                    f"predicted={b.predicted_mean:.0%} actual={b.actual_win_rate:.0%} "
                    f"n={b.count} {bar}"
                )
            lines.append("")

        # Source attribution
        if self.source_stats:
            lines.append("Source attribution:")
            for s in sorted(self.source_stats, key=lambda x: x.total_pnl, reverse=True):
                lines.append(
                    f"  {s.source:20s} trades={s.trade_count:<4d} "
                    f"pnl=${s.total_pnl:>+8.2f} wins={s.win_count} "
                    f"avg_edge={s.avg_edge:+.1%}"
                )
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  → {r}")
        else:
            lines.append("No recommendations (need more data)")

        return "\n".join(lines)


# ── Signal log reader ─────────────────────────────────────────────────────────


def read_signal_log(path: Path) -> list[SignalRecord]:
    """Read and parse the JSONL signal log."""
    if not path.exists():
        return []

    records = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(SignalRecord(
                cycle_time=d.get("cycle_time", ""),
                market_id=d.get("market_id", ""),
                question=d.get("question", ""),
                market_price=d.get("market_price", 0),
                our_probability=d.get("our_probability"),
                action=d.get("action", "SKIP"),
                side=d.get("side", ""),
                edge=d.get("edge"),
                reason=d.get("reason", ""),
                context_facts=d.get("context_facts", 0),
                executed=d.get("executed", False),
                position_id=d.get("position_id", ""),
            ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Skipping malformed signal log entry: %s", e)

    return records


# ── Scoring functions ─────────────────────────────────────────────────────────


def brier_score(predictions: list[tuple[float, bool]]) -> float | None:
    """Compute the Brier score for a set of (predicted_probability, actual_outcome) pairs.

    Brier score = mean((predicted - actual)²)
    Range: 0 (perfect) to 1 (always wrong). 0.25 = random guessing.
    """
    if not predictions:
        return None
    total = sum((pred - (1.0 if actual else 0.0)) ** 2 for pred, actual in predictions)
    return total / len(predictions)


def calibration_buckets(
    predictions: list[tuple[float, bool]],
    n_buckets: int = 5,
) -> list[CalibrationBucket]:
    """Group predictions into probability buckets and compare predicted vs actual.

    A well-calibrated model has predicted_mean ≈ actual_win_rate in each bucket.
    """
    if not predictions:
        return []

    # Sort by predicted probability
    sorted_preds = sorted(predictions, key=lambda x: x[0])

    bucket_size = max(1, len(sorted_preds) // n_buckets)
    buckets = []

    for i in range(0, len(sorted_preds), bucket_size):
        chunk = sorted_preds[i:i + bucket_size]
        if not chunk:
            continue
        probs = [p for p, _ in chunk]
        wins = [a for _, a in chunk]

        predicted_mean = sum(probs) / len(probs)
        actual_rate = sum(1 for w in wins if w) / len(wins)

        buckets.append(CalibrationBucket(
            bucket_low=min(probs),
            bucket_high=max(probs),
            predicted_mean=predicted_mean,
            actual_win_rate=actual_rate,
            count=len(chunk),
            error=predicted_mean - actual_rate,
        ))

    return buckets


def extract_signal_sources(reason: str) -> list[str]:
    """Extract signal source tags from a trade reason string.

    Looks for keywords that indicate which data sources contributed.
    """
    sources = []
    reason_lower = reason.lower()

    source_keywords = {
        "mirofish": "mirofish",
        "whale": "whale_alert",
        "rss": "rss_feeds",
        "reddit": "reddit",
        "fred": "fred_macro",
        "crypto news": "crypto_news",
        "graph fact": "graph_context",
    }

    for keyword, source in source_keywords.items():
        if keyword in reason_lower:
            sources.append(source)

    # If "context facts" mentioned, add graph_context
    if "context fact" in reason_lower or "graph" in reason_lower:
        if "graph_context" not in sources:
            sources.append("graph_context")

    return sources or ["unknown"]


# ── Auto-resolver ─────────────────────────────────────────────────────────────


async def check_resolved_markets(
    market_ids: list[str],
) -> dict[str, str]:
    """Check Polymarket Gamma API for resolved markets.

    Returns dict of {market_id: outcome} for markets that have resolved.
    Outcome is "YES" or "NO".
    """
    if not market_ids:
        return {}

    resolved = {}

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        for market_id in market_ids:
            try:
                # Try looking up by condition_id in the markets list
                async with session.get(
                    f"{GAMMA_API_BASE}/markets",
                    params={"condition_id": market_id, "limit": 1},
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    if not data:
                        continue
                    market = data[0] if isinstance(data, list) else data

                    resolved_by = market.get("resolvedBy") or ""
                    if not resolved_by:
                        continue

                    outcome_prices = market.get("outcomePrices") or ""
                    outcomes_raw = market.get("outcomes") or ""

                    try:
                        if isinstance(outcome_prices, str):
                            prices = json.loads(outcome_prices)
                        else:
                            prices = outcome_prices
                        if isinstance(outcomes_raw, str):
                            labels = json.loads(outcomes_raw)
                        else:
                            labels = outcomes_raw
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if len(prices) < 2 or len(labels) < 2:
                        continue

                    winner_idx = None
                    for i, p in enumerate(prices):
                        if str(p) == "1":
                            winner_idx = i
                            break
                    if winner_idx is None:
                        continue

                    winner_label = labels[winner_idx].strip().upper()
                    if winner_label in ("YES", "Y"):
                        resolved[market_id] = "YES"
                    elif winner_label in ("NO", "N"):
                        resolved[market_id] = "NO"
            except Exception as e:
                logger.debug("Failed to check market %s: %s", market_id[:16], e)

    return resolved


# ── Strategy Review ───────────────────────────────────────────────────────────


class StrategyReview:
    """Analyzes trading performance and generates improvement recommendations.

    Parameters
    ----------
    paper_engine : PaperTradingEngine
        The paper trading engine with position/trade data.
    signal_log_path : Path
        Path to the JSONL signal log.
    """

    def __init__(
        self,
        paper_engine: PaperTradingEngine,
        signal_log_path: Path = DEFAULT_SIGNAL_LOG,
    ):
        self.engine = paper_engine
        self.signal_log_path = signal_log_path

    async def auto_resolve(self) -> list[dict[str, Any]]:
        """Check for resolved markets and settle open positions.

        Returns list of resolution records.
        """
        open_positions = self.engine.get_open_positions()
        if not open_positions:
            logger.info("No open positions to resolve")
            return []

        market_ids = list({p.market_id for p in open_positions})
        logger.info("Checking %d markets for resolution", len(market_ids))

        resolved = await check_resolved_markets(market_ids)
        if not resolved:
            logger.info("No markets have resolved yet")
            return []

        results = []
        for market_id, outcome in resolved.items():
            try:
                positions = self.engine.resolve_market(market_id, outcome)
                for p in positions:
                    results.append({
                        "market_id": market_id,
                        "outcome": outcome,
                        "side": p.side,
                        "pnl": p.pnl,
                        "won": p.pnl > 0,
                    })
                    logger.info(
                        "Resolved %s → %s: %s pnl=$%.2f",
                        market_id[:16], outcome,
                        "WON" if p.pnl > 0 else "LOST", p.pnl,
                    )
            except Exception as e:
                logger.warning("Failed to resolve %s: %s", market_id[:16], e)

        return results

    def generate_report(self) -> StrategyReport:
        """Generate a full strategy review report."""
        signals = read_signal_log(self.signal_log_path)
        positions = self.engine.get_all_positions()
        portfolio = self.engine.get_portfolio()

        # Categorize positions
        resolved = [p for p in positions if p.status == "RESOLVED"]
        closed = [p for p in positions if p.status == "CLOSED"]
        open_pos = [p for p in positions if p.status == "OPEN"]
        settled = resolved + closed  # all positions with known outcomes

        # Build prediction pairs for scoring: (our_predicted_prob, did_we_win)
        prediction_pairs: list[tuple[float, bool]] = []
        signal_by_market: dict[str, SignalRecord] = {}
        for s in signals:
            if s.our_probability is not None and s.executed:
                signal_by_market[s.market_id] = s

        for p in resolved:
            sig = signal_by_market.get(p.market_id)
            if sig and sig.our_probability is not None:
                won = p.pnl > 0
                # Use our probability on the side we bet
                if p.side == "YES":
                    prediction_pairs.append((sig.our_probability, won))
                else:
                    prediction_pairs.append((1 - sig.our_probability, won))

        # Scoring
        bs = brier_score(prediction_pairs)
        cal = calibration_buckets(prediction_pairs)

        win_rate = None
        if settled:
            winners = sum(1 for p in settled if p.pnl > 0)
            win_rate = winners / len(settled)

        total_pnl = sum(p.pnl for p in settled)
        avg_pnl = total_pnl / len(settled) if settled else None

        # Source attribution
        source_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "pnl": 0.0, "wins": 0, "edges": []}
        )
        for p in settled:
            sig = signal_by_market.get(p.market_id)
            reason = sig.reason if sig else p.entry_reason
            edge = sig.edge if sig else None
            sources = extract_signal_sources(reason)
            for src in sources:
                source_data[src]["count"] += 1
                source_data[src]["pnl"] += p.pnl
                if p.pnl > 0:
                    source_data[src]["wins"] += 1
                if edge is not None:
                    source_data[src]["edges"].append(edge)

        source_stats = [
            SourceStats(
                source=src,
                trade_count=d["count"],
                total_pnl=d["pnl"],
                win_count=d["wins"],
                avg_edge=sum(d["edges"]) / len(d["edges"]) if d["edges"] else 0,
            )
            for src, d in source_data.items()
        ]

        # Recommendations
        recs = self._generate_recommendations(
            bs, cal, win_rate, source_stats, len(settled), len(prediction_pairs),
        )

        executed_count = sum(1 for s in signals if s.executed)

        return StrategyReport(
            total_signals=len(signals),
            executed_signals=executed_count,
            resolved_trades=len(resolved),
            open_trades=len(open_pos),
            brier_score=bs,
            win_rate=win_rate,
            avg_pnl_per_trade=avg_pnl,
            total_pnl=total_pnl,
            calibration=cal,
            source_stats=source_stats,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        bs: float | None,
        cal: list[CalibrationBucket],
        win_rate: float | None,
        source_stats: list[SourceStats],
        settled_count: int,
        scored_count: int,
    ) -> list[str]:
        """Generate actionable recommendations from the data."""
        recs: list[str] = []

        if settled_count < 10:
            recs.append(f"Only {settled_count} settled trades — need more data for reliable analysis")
            return recs

        # Brier score feedback
        if bs is not None:
            if bs > 0.25:
                recs.append(
                    f"Brier score {bs:.3f} is worse than random (0.25). "
                    "Predictions are actively harmful — consider widening min_edge threshold."
                )
            elif bs > 0.20:
                recs.append(
                    f"Brier score {bs:.3f} is marginal. "
                    "Consider increasing min_edge from 10% to 15%."
                )

        # Calibration feedback
        for b in cal:
            if b.count >= 3:
                if b.error > 0.15:
                    recs.append(
                        f"Overconfident in {b.bucket_low:.0%}-{b.bucket_high:.0%} range: "
                        f"predicted {b.predicted_mean:.0%} but won {b.actual_win_rate:.0%}. "
                        f"Reduce kelly_multiplier for high-confidence trades."
                    )
                elif b.error < -0.15:
                    recs.append(
                        f"Underconfident in {b.bucket_low:.0%}-{b.bucket_high:.0%} range: "
                        f"predicted {b.predicted_mean:.0%} but won {b.actual_win_rate:.0%}. "
                        f"Could increase position size in this range."
                    )

        # Source attribution feedback
        for s in source_stats:
            if s.trade_count >= 5:
                if s.total_pnl < 0:
                    recs.append(
                        f"'{s.source}' signals are losing money "
                        f"(${s.total_pnl:+.2f} over {s.trade_count} trades). "
                        f"Consider reducing weight or adding filters."
                    )
                elif s.total_pnl > 0 and s.win_count / s.trade_count > 0.6:
                    recs.append(
                        f"'{s.source}' signals are performing well "
                        f"(${s.total_pnl:+.2f}, {s.win_count}/{s.trade_count} wins). "
                        f"Could increase position size for these signals."
                    )

        # Win rate feedback
        if win_rate is not None and win_rate < 0.4:
            recs.append(
                f"Win rate is low ({win_rate:.0%}). "
                "Either increase min_edge threshold or improve probability estimation."
            )

        return recs


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Strategy review & feedback loop")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("resolve", help="Auto-resolve markets and settle positions")
    sub.add_parser("report", help="Generate full strategy report")
    sub.add_parser("calibration", help="Show calibration curve only")

    args = parser.parse_args()
    engine = PaperTradingEngine()
    review = StrategyReview(paper_engine=engine)

    if args.command == "resolve":
        results = await review.auto_resolve()
        if results:
            for r in results:
                status = "WON" if r["won"] else "LOST"
                print(f"  {r['market_id'][:24]:24s} {r['outcome']} {r['side']} {status} pnl=${r['pnl']:+.2f}")
        else:
            print("No markets resolved.")

    elif args.command == "report":
        report = review.generate_report()
        print(report.summary)

    elif args.command == "calibration":
        report = review.generate_report()
        if report.calibration:
            print("Calibration curve:")
            for b in report.calibration:
                print(
                    f"  {b.bucket_low:.0%}-{b.bucket_high:.0%}: "
                    f"predicted={b.predicted_mean:.1%} actual={b.actual_win_rate:.1%} "
                    f"(n={b.count}, error={b.error:+.1%})"
                )
        else:
            print("No resolved trades with probability data for calibration.")

    else:
        parser.print_help()

    engine.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
