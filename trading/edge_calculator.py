"""
Edge calculator — compares probability estimates against market prices
to identify trading opportunities.

Accepts probability estimates (from MiroFish or any other source),
compares against current market prices, calculates edge and Kelly-optimal
position size, applies risk checks, and emits trade signals.

Designed to be market-agnostic: works with Polymarket, stocks, or any
market where you have a probability estimate and a price.

Usage:
    calc = EdgeCalculator(config=EdgeConfig(min_edge=0.10))
    signal = calc.evaluate(
        estimate=ProbabilityEstimate(market_id="btc_100k", probability=0.82, source="mirofish"),
        market_price=0.72,
    )
    if signal and signal.action != "SKIP":
        paper_engine.open_position(signal.market_id, signal.side, signal.shares, signal.price)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("prememora.trading.edge_calculator")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ProbabilityEstimate:
    """A probability estimate from any source (MiroFish, manual, etc.)."""
    market_id: str
    probability: float          # 0-1, our estimated probability of YES
    source: str = ""            # e.g. "mirofish", "manual", "ensemble"
    confidence: float = 1.0     # 0-1, how confident we are in this estimate
    reasoning: str = ""         # why we think this
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class TradeSignal:
    """Output of the edge calculator — a recommended trade or skip."""
    market_id: str
    action: str                 # "BUY_YES", "BUY_NO", "SKIP"
    side: str                   # "YES" or "NO"
    edge: float                 # our_prob - market_price (for the chosen side)
    our_probability: float      # our estimate for YES
    market_price: float         # current market price for YES
    kelly_fraction: float       # optimal fraction of bankroll to risk
    shares: float               # recommended shares (after risk limits)
    price: float                # price to buy at
    reason: str                 # human-readable explanation
    source: str                 # signal source
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class RiskLimits:
    """Portfolio-level risk constraints."""
    max_position_pct: float = 0.10      # max % of portfolio in one position
    max_portfolio_exposure_pct: float = 0.50  # max % of portfolio in all positions
    max_kelly_fraction: float = 0.25    # cap Kelly to avoid over-betting (quarter-Kelly)
    max_drawdown_pct: float = 0.20      # stop trading if portfolio drops this much


@dataclass
class EdgeConfig:
    """Configuration for the edge calculator."""
    min_edge: float = 0.10              # minimum edge to consider trading (10%)
    min_probability: float = 0.05       # ignore extreme probabilities below this
    max_probability: float = 0.95       # ignore extreme probabilities above this
    risk: RiskLimits = field(default_factory=RiskLimits)


# ── Kelly criterion ───────────────────────────────────────────────────────────


def kelly_fraction(our_prob: float, market_price: float) -> float:
    """Calculate the Kelly-optimal bet fraction for a binary option.

    For a binary option bought at `market_price` that pays $1 on win:
      - Win probability: our_prob
      - Profit if win: (1 - market_price) per share
      - Loss if lose: market_price per share
      - Kelly f = (p * (1-price) - (1-p) * price) / (1-price) * price
        Simplifies to: f = (our_prob - market_price) / (1 - market_price)
        i.e. f = edge / odds

    Returns 0 if edge is non-positive.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    edge = our_prob - market_price
    if edge <= 0:
        return 0.0
    return edge / (1.0 - market_price)


# ── Edge Calculator ───────────────────────────────────────────────────────────


class EdgeCalculator:
    """Evaluates probability estimates against market prices.

    Parameters
    ----------
    config : EdgeConfig
        Thresholds and risk limits.
    portfolio_value : float
        Current portfolio value (cash + positions). Used for position sizing.
    current_exposure : float
        Current total position value. Used for exposure checks.
    peak_portfolio_value : float
        Highest portfolio value seen (for drawdown check).
    """

    def __init__(
        self,
        config: EdgeConfig | None = None,
        portfolio_value: float = 1000.0,
        current_exposure: float = 0.0,
        peak_portfolio_value: float | None = None,
    ):
        self.config = config or EdgeConfig()
        self.portfolio_value = portfolio_value
        self.current_exposure = current_exposure
        self.peak_portfolio_value = peak_portfolio_value or portfolio_value

    def update_portfolio(
        self,
        portfolio_value: float,
        current_exposure: float,
        peak_portfolio_value: float | None = None,
    ) -> None:
        """Update portfolio state (call before evaluate)."""
        self.portfolio_value = portfolio_value
        self.current_exposure = current_exposure
        if peak_portfolio_value is not None:
            self.peak_portfolio_value = peak_portfolio_value
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

    def evaluate(
        self,
        estimate: ProbabilityEstimate,
        market_price: float,
    ) -> TradeSignal:
        """Evaluate a probability estimate against a market price.

        Returns a TradeSignal with action BUY_YES, BUY_NO, or SKIP.
        """
        our_prob = estimate.probability
        cfg = self.config

        # ── Validate inputs ───────────────────────────────────────────
        if not (0 < market_price < 1):
            return self._skip(estimate, market_price, "market_price out of range (0,1)")

        if not (0 <= our_prob <= 1):
            return self._skip(estimate, market_price, "probability out of range [0,1]")

        # ── Check extreme probabilities ───────────────────────────────
        if our_prob < cfg.min_probability or our_prob > cfg.max_probability:
            return self._skip(estimate, market_price,
                              f"probability {our_prob:.2f} outside [{cfg.min_probability}, {cfg.max_probability}]")

        # ── Determine side and edge ───────────────────────────────────
        # Check both YES and NO sides, pick the one with positive edge
        yes_edge = our_prob - market_price
        no_edge = (1 - our_prob) - (1 - market_price)  # = market_price - our_prob

        if yes_edge >= abs(no_edge) and yes_edge > 0:
            side = "YES"
            edge = yes_edge
            buy_price = market_price
        elif no_edge > 0:
            side = "NO"
            edge = no_edge
            buy_price = 1.0 - market_price
        else:
            return self._skip(estimate, market_price, "no positive edge on either side")

        # ── Minimum edge threshold ────────────────────────────────────
        if edge < cfg.min_edge:
            return self._skip(estimate, market_price,
                              f"edge {edge:.2%} below threshold {cfg.min_edge:.2%}")

        # ── Kelly sizing ──────────────────────────────────────────────
        if side == "YES":
            kf = kelly_fraction(our_prob, market_price)
        else:
            kf = kelly_fraction(1 - our_prob, 1 - market_price)

        # Cap Kelly fraction
        capped_kf = min(kf, cfg.risk.max_kelly_fraction)

        # ── Risk checks ──────────────────────────────────────────────
        risk_reason = self._check_risk(capped_kf, buy_price)
        if risk_reason:
            return self._skip(estimate, market_price, risk_reason)

        # ── Position sizing ───────────────────────────────────────────
        bet_amount = self.portfolio_value * capped_kf
        # Also cap by max position size
        max_position = self.portfolio_value * cfg.risk.max_position_pct
        bet_amount = min(bet_amount, max_position)

        shares = bet_amount / buy_price if buy_price > 0 else 0

        if shares < 1:
            return self._skip(estimate, market_price, "position too small (< 1 share)")

        shares = round(shares, 2)

        action = f"BUY_{side}"
        reason = (
            f"{side} edge={edge:.2%} (our={our_prob:.2%} vs market={market_price:.2%}), "
            f"kelly={kf:.2%} → capped={capped_kf:.2%}, "
            f"size=${bet_amount:.2f} ({shares:.0f} shares @{buy_price:.4f})"
        )

        logger.info("Signal: %s %s — %s", action, estimate.market_id, reason)

        return TradeSignal(
            market_id=estimate.market_id,
            action=action,
            side=side,
            edge=edge,
            our_probability=our_prob,
            market_price=market_price,
            kelly_fraction=kf,
            shares=shares,
            price=buy_price,
            reason=reason,
            source=estimate.source,
        )

    def _check_risk(self, kelly_fraction: float, buy_price: float) -> str | None:
        """Return a skip reason if any risk limit is breached, else None."""
        cfg = self.config.risk

        # Drawdown check
        if self.peak_portfolio_value > 0:
            drawdown = 1 - (self.portfolio_value / self.peak_portfolio_value)
            if drawdown >= cfg.max_drawdown_pct:
                return f"drawdown {drawdown:.2%} exceeds limit {cfg.max_drawdown_pct:.2%}"

        # Exposure check
        new_exposure = self.current_exposure + (self.portfolio_value * kelly_fraction * buy_price)
        max_exposure = self.portfolio_value * cfg.max_portfolio_exposure_pct
        if new_exposure > max_exposure:
            return (f"exposure ${new_exposure:.2f} would exceed "
                    f"limit ${max_exposure:.2f} ({cfg.max_portfolio_exposure_pct:.0%})")

        return None

    def _skip(
        self,
        estimate: ProbabilityEstimate,
        market_price: float,
        reason: str,
    ) -> TradeSignal:
        """Create a SKIP signal."""
        logger.debug("Skip %s: %s", estimate.market_id, reason)
        return TradeSignal(
            market_id=estimate.market_id,
            action="SKIP",
            side="",
            edge=estimate.probability - market_price,
            our_probability=estimate.probability,
            market_price=market_price,
            kelly_fraction=0.0,
            shares=0,
            price=market_price,
            reason=reason,
            source=estimate.source,
        )

    # ── Batch evaluation ──────────────────────────────────────────────

    def evaluate_batch(
        self,
        estimates: list[tuple[ProbabilityEstimate, float]],
    ) -> list[TradeSignal]:
        """Evaluate multiple (estimate, market_price) pairs.

        Returns all signals (including SKIPs). Filter by action != "SKIP"
        to get actionable signals.
        """
        return [self.evaluate(est, price) for est, price in estimates]
