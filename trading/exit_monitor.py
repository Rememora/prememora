"""
Exit monitor — confidence-based position management with early exit triggers.

Monitors open positions and generates exit signals when:
  1. Confidence drop: agent re-interview shows probability dropped significantly
  2. Contradictory evidence: new probability flips edge negative
  3. Time decay: market deadline approaching with uncertain position
  4. Stop loss: market price moved against us beyond threshold

Each trigger can be individually enabled/disabled via ExitConfig.

Usage:
    monitor = ExitMonitor(config=ExitConfig(), paper_engine=engine)
    signals = await monitor.run_once()   # check + execute exits

CLI:
    python -m trading.exit_monitor check   # dry run
    python -m trading.exit_monitor run     # execute exits
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from trading.paper_engine import PaperTradingEngine, Position

logger = logging.getLogger("prememora.trading.exit_monitor")

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


# ── Config & Signals ─────────────────────────────────────────────────────────


@dataclass
class ExitConfig:
    """Configuration for exit monitoring thresholds."""
    confidence_drop_threshold: float = 0.15   # 15% absolute drop triggers exit
    time_decay_threshold: float = 0.20        # exit if <20% time remaining
    stop_loss_delta: float = 0.20             # exit if price moves 20c against us
    min_hold_seconds: int = 1800              # don't exit within 30min of entry
    enabled_triggers: set[str] = field(
        default_factory=lambda: {
            "confidence_drop",
            "contradictory_evidence",
            "time_decay",
            "stop_loss",
        }
    )


@dataclass
class ExitSignal:
    """A recommendation to exit a position."""
    position_id: str
    market_id: str
    trigger_type: str        # "confidence_drop", "contradictory_evidence", "time_decay", "stop_loss"
    old_confidence: float
    new_confidence: float | None
    market_price: float
    reason: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── Trigger check functions (module-level, independently testable) ───────────


def check_confidence_drop(
    position: Position,
    new_probability: float,
    threshold: float,
) -> ExitSignal | None:
    """Check if confidence has dropped enough to warrant an exit.

    Compares the new probability estimate against the entry_confidence
    (or confidence if entry_confidence is unavailable). If the drop exceeds
    the threshold, returns an ExitSignal.
    """
    old_conf = position.entry_confidence if position.entry_confidence is not None else position.confidence
    if old_conf is None:
        return None

    # For YES positions: conviction drops when new_prob < old_conf
    # For NO positions: conviction drops when new_prob > old_conf (YES became more likely)
    if position.side == "NO":
        drop = new_probability - old_conf
    else:
        drop = old_conf - new_probability

    if drop >= threshold:
        return ExitSignal(
            position_id=position.id,
            market_id=position.market_id,
            trigger_type="confidence_drop",
            old_confidence=old_conf,
            new_confidence=new_probability,
            market_price=position.current_price,
            reason=(
                f"Confidence dropped {drop:.0%}: "
                f"entry={old_conf:.2f} -> now={new_probability:.2f}"
            ),
        )
    return None


def check_contradictory_evidence(
    position: Position,
    new_probability: float,
    market_price: float,
) -> ExitSignal | None:
    """Check if new evidence has flipped our edge to negative.

    For a YES position: edge = new_prob - market_price.
    For a NO position: edge = (1 - new_prob) - (1 - market_price).
    If edge is now <= 0, the trade thesis is invalidated.
    """
    if position.side == "YES":
        edge = new_probability - market_price
    else:
        edge = (1.0 - new_probability) - (1.0 - market_price)

    old_conf = position.entry_confidence if position.entry_confidence is not None else position.confidence

    if edge <= 0:
        return ExitSignal(
            position_id=position.id,
            market_id=position.market_id,
            trigger_type="contradictory_evidence",
            old_confidence=old_conf or 0.0,
            new_confidence=new_probability,
            market_price=market_price,
            reason=(
                f"Edge flipped negative: {position.side} edge={edge:+.2f} "
                f"(new_prob={new_probability:.2f}, mkt={market_price:.2f})"
            ),
        )
    return None


def check_time_decay(
    position: Position,
    current_time: datetime,
    threshold: float,
) -> ExitSignal | None:
    """Check if the market deadline is approaching with too little time left.

    Triggers when remaining_time / total_time < threshold.
    Only applies if the position has a market_deadline set.
    """
    if not position.market_deadline:
        return None

    try:
        deadline = datetime.fromisoformat(position.market_deadline.replace("Z", "+00:00"))
        entry = datetime.fromisoformat(position.entry_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    # Ensure timezone-aware comparison
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    if entry.tzinfo is None:
        entry = entry.replace(tzinfo=timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    total = (deadline - entry).total_seconds()
    remaining = (deadline - current_time).total_seconds()

    if total <= 0:
        return None

    fraction_remaining = remaining / total

    if remaining <= 0 or fraction_remaining < threshold:
        old_conf = position.entry_confidence if position.entry_confidence is not None else position.confidence
        return ExitSignal(
            position_id=position.id,
            market_id=position.market_id,
            trigger_type="time_decay",
            old_confidence=old_conf or 0.0,
            new_confidence=None,
            market_price=position.current_price,
            reason=(
                f"Time decay: {fraction_remaining:.0%} remaining "
                f"(threshold={threshold:.0%}, deadline={position.market_deadline})"
            ),
        )
    return None


def check_stop_loss(
    position: Position,
    current_market_price: float,
    delta: float,
) -> ExitSignal | None:
    """Check if the market has moved against our position beyond the stop loss.

    For YES positions: trigger if market_price dropped delta below entry_price.
    For NO positions: trigger if market_price rose delta above entry_price.
    """
    if position.side == "YES":
        move_against = position.entry_price - current_market_price
    else:
        move_against = current_market_price - position.entry_price

    if move_against >= delta:
        old_conf = position.entry_confidence if position.entry_confidence is not None else position.confidence
        return ExitSignal(
            position_id=position.id,
            market_id=position.market_id,
            trigger_type="stop_loss",
            old_confidence=old_conf or 0.0,
            new_confidence=None,
            market_price=current_market_price,
            reason=(
                f"Stop loss: price moved {move_against:.2f} against {position.side} "
                f"(entry={position.entry_price:.2f}, now={current_market_price:.2f}, "
                f"delta={delta:.2f})"
            ),
        )
    return None


# ── Market price fetching ────────────────────────────────────────────────────


async def fetch_market_prices(
    session: aiohttp.ClientSession,
    market_ids: list[str],
) -> dict[str, float]:
    """Fetch current YES prices for a batch of markets from Gamma API.

    Returns dict of {market_id: yes_price}.
    """
    prices: dict[str, float] = {}

    for market_id in market_ids:
        try:
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
                tokens = market.get("tokens", [])
                for t in tokens:
                    if t.get("outcome", "").upper() == "YES":
                        prices[market_id] = float(t.get("price", 0))
                        break
        except Exception as e:
            logger.debug("Failed to fetch price for %s: %s", market_id[:16], e)

    return prices


# ── ExitMonitor ──────────────────────────────────────────────────────────────


class ExitMonitor:
    """Monitors open positions and generates/executes exit signals.

    Parameters
    ----------
    config : ExitConfig
        Exit monitoring configuration.
    paper_engine : PaperTradingEngine
        Paper trading engine with position data.
    mirofish_url : str
        MiroFish backend URL for agent re-interviews.
    simulation_id : str
        Simulation ID for agent interviews.
    graph_id : str
        Graph ID for context enrichment.
    """

    def __init__(
        self,
        config: ExitConfig,
        paper_engine: PaperTradingEngine,
        mirofish_url: str = "",
        simulation_id: str = "",
        graph_id: str = "",
    ):
        self.config = config
        self.engine = paper_engine
        self.mirofish_url = mirofish_url
        self.simulation_id = simulation_id
        self.graph_id = graph_id

    async def check_all_positions(self) -> list[ExitSignal]:
        """Check all open positions for exit triggers.

        Returns list of triggered ExitSignals (does NOT execute exits).
        """
        positions = self.engine.get_open_positions()
        if not positions:
            return []

        now = datetime.now(timezone.utc)

        # Filter out positions within min_hold_seconds
        eligible = []
        for p in positions:
            try:
                entry = datetime.fromisoformat(p.entry_time.replace("Z", "+00:00"))
                if entry.tzinfo is None:
                    entry = entry.replace(tzinfo=timezone.utc)
                held_seconds = (now - entry).total_seconds()
                if held_seconds >= self.config.min_hold_seconds:
                    eligible.append(p)
            except (ValueError, AttributeError):
                eligible.append(p)  # can't parse entry_time, include it

        if not eligible:
            logger.debug("No positions past min_hold_seconds (%d)", self.config.min_hold_seconds)
            return []

        # Fetch current market prices
        market_ids = list({p.market_id for p in eligible})
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            prices = await fetch_market_prices(session, market_ids)

        signals: list[ExitSignal] = []

        # Checks that don't need re-interview (fast)
        for p in eligible:
            market_price = prices.get(p.market_id, p.current_price)

            # Stop loss
            if "stop_loss" in self.config.enabled_triggers:
                sig = check_stop_loss(p, market_price, self.config.stop_loss_delta)
                if sig:
                    signals.append(sig)
                    continue  # no need to check further for this position

            # Time decay
            if "time_decay" in self.config.enabled_triggers:
                sig = check_time_decay(p, now, self.config.time_decay_threshold)
                if sig:
                    signals.append(sig)
                    continue

        # Positions that need re-evaluation (confidence_drop / contradictory_evidence)
        already_signaled = {s.position_id for s in signals}
        needs_reeval = [
            p for p in eligible
            if p.id not in already_signaled
            and (
                "confidence_drop" in self.config.enabled_triggers
                or "contradictory_evidence" in self.config.enabled_triggers
            )
        ]

        if needs_reeval and self.simulation_id and self.mirofish_url:
            new_probs = await self._re_interview_positions(needs_reeval)

            for p in needs_reeval:
                new_prob = new_probs.get(p.market_id)
                if new_prob is None:
                    continue

                market_price = prices.get(p.market_id, p.current_price)

                # Update confidence on the position
                self.engine.update_confidence(p.market_id, new_prob, side=p.side)

                # Confidence drop
                if "confidence_drop" in self.config.enabled_triggers:
                    sig = check_confidence_drop(
                        p, new_prob, self.config.confidence_drop_threshold,
                    )
                    if sig:
                        sig.market_price = market_price
                        signals.append(sig)
                        continue

                # Contradictory evidence
                if "contradictory_evidence" in self.config.enabled_triggers:
                    sig = check_contradictory_evidence(p, new_prob, market_price)
                    if sig:
                        signals.append(sig)

        return signals

    async def _re_interview_positions(
        self, positions: list[Position],
    ) -> dict[str, float]:
        """Re-interview agents for the given positions' markets.

        Returns {market_id: new_probability}.
        """
        from pipeline.trigger import interview_agents, parse_probability, aggregate_probabilities

        results: dict[str, float] = {}
        seen_markets: set[str] = set()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
        ) as session:
            for p in positions:
                if p.market_id in seen_markets:
                    continue
                seen_markets.add(p.market_id)

                question = (
                    f"What probability (0-100%) do you assign to market {p.market_id}? "
                    f"Please give a specific number."
                )

                responses = await interview_agents(
                    session,
                    self.mirofish_url,
                    self.simulation_id,
                    question,
                    timeout=60,
                )

                probs = []
                for resp in responses:
                    text = resp.get("response", "")
                    prob = parse_probability(text)
                    if prob is not None and 0 <= prob <= 1:
                        probs.append(prob)

                agg = aggregate_probabilities(probs)
                if agg is not None:
                    results[p.market_id] = agg

        return results

    async def execute_exits(self, signals: list[ExitSignal]) -> list[Position]:
        """Execute exit signals by closing positions.

        Returns list of closed Position objects.
        """
        closed: list[Position] = []

        for sig in signals:
            try:
                pos = self.engine.close_position(
                    market_id=sig.market_id,
                    price=sig.market_price,
                    reason=f"exit:{sig.trigger_type} - {sig.reason}",
                )
                closed.append(pos)
                logger.info(
                    "Exited %s on %s: trigger=%s, price=%.4f, pnl=$%.4f",
                    sig.position_id, sig.market_id, sig.trigger_type,
                    sig.market_price, pos.pnl,
                )
            except Exception as e:
                logger.warning(
                    "Failed to exit %s on %s: %s",
                    sig.position_id, sig.market_id, e,
                )

        return closed

    async def run_once(self) -> list[ExitSignal]:
        """Check all positions and execute any triggered exits.

        Returns the list of triggered ExitSignals.
        """
        signals = await self.check_all_positions()
        if signals:
            logger.info("Exit monitor found %d exit signals", len(signals))
            await self.execute_exits(signals)
        else:
            logger.debug("Exit monitor: no exits triggered")
        return signals


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    parser = argparse.ArgumentParser(description="Exit monitor for open positions")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="Dry run: show what would be exited")
    sub.add_parser("run", help="Check and execute exits")

    args = parser.parse_args()

    engine = PaperTradingEngine()
    config = ExitConfig()
    monitor = ExitMonitor(config=config, paper_engine=engine)

    if args.command == "check":
        signals = await monitor.check_all_positions()
        if signals:
            for s in signals:
                print(
                    f"  [{s.trigger_type:24s}] {s.market_id[:24]:24s} "
                    f"conf={s.old_confidence:.2f}->{s.new_confidence or '?':>5} "
                    f"mkt={s.market_price:.4f} {s.reason}"
                )
        else:
            print("No exits triggered.")

    elif args.command == "run":
        signals = await monitor.run_once()
        if signals:
            for s in signals:
                print(f"  EXITED {s.position_id} on {s.market_id[:24]} ({s.trigger_type})")
        else:
            print("No exits triggered.")

    else:
        parser.print_help()

    engine.close()


if __name__ == "__main__":
    asyncio.run(main())
