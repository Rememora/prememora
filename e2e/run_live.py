"""
Live end-to-end test runner for the PreMemora prediction pipeline.

Validates the full pipeline: market discovery → context enrichment →
probability estimation → edge calculation → paper trading → resolution → scoring.

Two modes:

  Smoke (default):
    Uses already-resolved Polymarket markets. Since outcomes are known,
    the entire cycle runs in one shot: discover → estimate → trade → resolve → score.
    No waiting, no external dependencies beyond the Gamma API.

  Live:
    Uses real active markets. Optionally interviews MiroFish agents.
    Positions stay open until markets resolve. Re-run with --resolve-only
    to score after resolution.

Usage:
    python -m e2e.run_live                       # smoke test
    python -m e2e.run_live --mode live           # live test (active markets)
    python -m e2e.run_live --mode live --resolve-only  # score after waiting
    python -m e2e.run_live --max-markets 3       # limit markets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from e2e.helpers import (
    E2EReport,
    ResolvedMarket,
    fetch_active_markets_for_live,
    fetch_resolved_markets,
    generate_synthetic_probability,
    timed_stage,
)

logger = logging.getLogger("prememora.e2e")

# E2E data goes to data/e2e/ to avoid polluting production data
E2E_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "e2e"
E2E_DB_PATH = E2E_DATA_DIR / "e2e_paper_trading.db"
E2E_SIGNAL_LOG = E2E_DATA_DIR / "e2e_signal_log.jsonl"


# ── Smoke test ───────────────────────────────────────────────────────────────


async def run_smoke(
    max_markets: int = 5,
    graph_id: str = "",
) -> E2EReport:
    """Run the smoke test against resolved markets.

    Stages:
      1. Fetch resolved markets from Gamma API
      2. (Optional) Build graph context for each market
      3. Generate synthetic probabilities
      4. Run edge calculator
      5. Open paper positions
      6. Resolve positions (we know the outcome)
      7. Generate strategy report with Brier score
    """
    report = E2EReport(mode="smoke")
    E2E_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous E2E data
    if E2E_DB_PATH.exists():
        E2E_DB_PATH.unlink()
    if E2E_SIGNAL_LOG.exists():
        E2E_SIGNAL_LOG.unlink()

    # ── Stage 1: Fetch resolved markets ──────────────────────────────
    markets: list[ResolvedMarket] = []
    with timed_stage("Fetch resolved markets", report) as stage:
        markets = await fetch_resolved_markets(max_markets=max_markets)
        if not markets:
            stage.error = "No resolved markets found"
            stage.success = False
            print(report.summary)
            return report
        stage.detail = f"{len(markets)} markets fetched"
        report.markets_tested = len(markets)

    print(f"\n  Found {len(markets)} resolved markets:")
    for m in markets:
        print(f"    {m.outcome:3s} | price={m.yes_price:.2f} | {m.question[:60]}")
    print()

    # ── Stage 2: Graph context (optional) ────────────────────────────
    context_map: dict[str, Any] = {}
    if graph_id:
        with timed_stage("Build graph context", report) as stage:
            try:
                from pipeline.context import ContextBuilder
                builder = ContextBuilder(graph_id=graph_id)
                for m in markets:
                    ctx = await asyncio.to_thread(builder.build_context, m.question)
                    context_map[m.condition_id] = ctx
                total_facts = sum(len(c.facts) for c in context_map.values())
                stage.detail = f"{total_facts} facts across {len(context_map)} markets"
            except Exception as e:
                stage.error = str(e)
                stage.success = False
                logger.warning("Graph context failed (continuing without): %s", e)
    else:
        from e2e.helpers import StageResult
        report.stages.append(StageResult(
            name="Build graph context",
            success=True,
            duration_s=0,
            detail="skipped (no graph_id)",
        ))

    # ── Stage 3: Generate probabilities ──────────────────────────────
    estimates: list[dict[str, Any]] = []
    with timed_stage("Generate probabilities", report) as stage:
        for m in markets:
            prob = generate_synthetic_probability(
                m.question, m.yes_price, outcome=m.outcome,
            )
            ctx = context_map.get(m.condition_id)
            estimates.append({
                "market": m,
                "our_probability": prob,
                "context_facts": len(ctx.facts) if ctx else 0,
            })
        stage.detail = f"{len(estimates)} probabilities generated"

    # ── Stage 4: Edge calculation ────────────────────────────────────
    from trading.edge_calculator import EdgeCalculator, EdgeConfig, ProbabilityEstimate

    signals: list[dict[str, Any]] = []
    with timed_stage("Edge calculation", report) as stage:
        # Use relaxed thresholds for E2E testing — we want to see trades happen
        config = EdgeConfig(min_edge=0.05, min_probability=0.05, max_probability=0.95)
        calc = EdgeCalculator(config=config, portfolio_value=1000.0)

        for est in estimates:
            m = est["market"]
            estimate = ProbabilityEstimate(
                market_id=m.condition_id,
                probability=est["our_probability"],
                source="e2e_synthetic",
                reasoning=f"Synthetic probability for smoke test (known outcome: {m.outcome})",
            )
            signal = calc.evaluate(estimate, m.yes_price)
            signals.append({
                "market": m,
                "signal": signal,
                "our_probability": est["our_probability"],
            })

        actionable = [s for s in signals if s["signal"].action != "SKIP"]
        stage.detail = f"{len(actionable)}/{len(signals)} actionable signals"

    # ── Stage 5: Open paper positions ────────────────────────────────
    from trading.paper_engine import PaperTradingEngine

    engine = PaperTradingEngine(db_path=E2E_DB_PATH, initial_cash=1000.0)

    with timed_stage("Open paper positions", report) as stage:
        opened = 0
        for s in signals:
            sig = s["signal"]
            m = s["market"]
            if sig.action == "SKIP":
                continue
            try:
                engine.open_position(
                    market_id=sig.market_id,
                    side=sig.side,
                    shares=sig.shares,
                    price=sig.price,
                    reason=sig.reason,
                    confidence=s["our_probability"],
                )
                opened += 1

                # Log signal
                log_entry = {
                    "cycle_time": datetime.now(timezone.utc).isoformat(),
                    "market_id": sig.market_id,
                    "question": m.question,
                    "market_price": m.yes_price,
                    "our_probability": s["our_probability"],
                    "action": sig.action,
                    "side": sig.side,
                    "edge": sig.edge,
                    "kelly_fraction": sig.kelly_fraction,
                    "shares": sig.shares,
                    "price": sig.price,
                    "reason": sig.reason,
                    "executed": True,
                }
                E2E_SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(E2E_SIGNAL_LOG, "a") as f:
                    f.write(json.dumps(log_entry, default=str) + "\n")
            except Exception as e:
                logger.warning("Failed to open position on %s: %s", sig.market_id[:16], e)

        stage.detail = f"{opened} positions opened"
        report.trades_opened = opened

    if report.trades_opened == 0:
        print("\n  No trades were opened — edge calculator filtered everything.")
        print("  This can happen if synthetic probabilities are close to market prices.")
        print(report.summary)
        engine.close()
        return report

    # ── Stage 6: Resolve positions ───────────────────────────────────
    with timed_stage("Resolve positions", report) as stage:
        resolved_count = 0
        for s in signals:
            sig = s["signal"]
            m = s["market"]
            if sig.action == "SKIP":
                continue
            try:
                engine.resolve_market(m.condition_id, m.outcome)
                resolved_count += 1
            except Exception as e:
                logger.warning("Failed to resolve %s: %s", m.condition_id[:16], e)

        stage.detail = f"{resolved_count} positions resolved"
        report.trades_resolved = resolved_count

    # ── Stage 7: Strategy report ─────────────────────────────────────
    from trading.strategy_review import StrategyReview

    with timed_stage("Generate strategy report", report) as stage:
        review = StrategyReview(paper_engine=engine, signal_log_path=E2E_SIGNAL_LOG)
        strategy_report = review.generate_report()

        report.brier_score = strategy_report.brier_score
        report.total_pnl = strategy_report.total_pnl

        detail_parts = []
        if strategy_report.brier_score is not None:
            detail_parts.append(f"brier={strategy_report.brier_score:.4f}")
        detail_parts.append(f"pnl=${strategy_report.total_pnl:+.2f}")
        if strategy_report.win_rate is not None:
            detail_parts.append(f"win_rate={strategy_report.win_rate:.0%}")
        stage.detail = ", ".join(detail_parts)

        # Print the full strategy report
        print(f"\n{strategy_report.summary}\n")

    portfolio = engine.get_portfolio()
    print(f"  Final portfolio: ${portfolio.total_value:.2f} (started $1000.00)")
    engine.close()

    return report


# ── Live test ────────────────────────────────────────────────────────────────


async def run_live(
    max_markets: int = 3,
    simulation_id: str = "",
    graph_id: str = "",
    resolve_only: bool = False,
) -> E2EReport:
    """Run the live test against active markets.

    If resolve_only=True, just checks for resolved markets and scores.
    """
    report = E2EReport(mode="live")
    E2E_DATA_DIR.mkdir(parents=True, exist_ok=True)

    from trading.paper_engine import PaperTradingEngine
    engine = PaperTradingEngine(db_path=E2E_DB_PATH, initial_cash=1000.0)

    if resolve_only:
        # Just check for resolutions and score
        with timed_stage("Auto-resolve markets", report) as stage:
            from trading.strategy_review import StrategyReview
            review = StrategyReview(paper_engine=engine, signal_log_path=E2E_SIGNAL_LOG)
            results = await review.auto_resolve()
            stage.detail = f"{len(results)} positions resolved"
            report.trades_resolved = len(results)

            if results:
                for r in results:
                    status = "WON" if r["won"] else "LOST"
                    print(f"  {r['market_id'][:24]:24s} {r['outcome']} {status} pnl=${r['pnl']:+.2f}")

        with timed_stage("Generate strategy report", report) as stage:
            strategy_report = review.generate_report()
            report.brier_score = strategy_report.brier_score
            report.total_pnl = strategy_report.total_pnl
            stage.detail = f"brier={strategy_report.brier_score}, pnl=${strategy_report.total_pnl:+.2f}"
            print(f"\n{strategy_report.summary}\n")

        engine.close()
        return report

    # Full live run
    from pipeline.trigger import PipelineConfig, PipelineTrigger

    with timed_stage("Fetch active markets", report) as stage:
        raw_markets = await fetch_active_markets_for_live(max_markets=max_markets)
        if not raw_markets:
            stage.error = "No active markets found"
            stage.success = False
            print(report.summary)
            engine.close()
            return report
        stage.detail = f"{len(raw_markets)} active markets"
        report.markets_tested = len(raw_markets)

        print(f"\n  Found {len(raw_markets)} active markets:")
        for m in raw_markets:
            q = m.get("question", "")[:60]
            print(f"    {q}")
        print()

    with timed_stage("Run pipeline trigger", report) as stage:
        config = PipelineConfig(
            simulation_id=simulation_id,
            graph_id=graph_id,
            max_markets=max_markets,
            signal_log_path=E2E_SIGNAL_LOG,
        )

        # If no simulation_id, use synthetic probabilities
        if not simulation_id:
            # Run manually with synthetic probs instead of the trigger
            from trading.edge_calculator import EdgeCalculator, EdgeConfig, ProbabilityEstimate
            calc_config = EdgeConfig(min_edge=0.05)
            calc = EdgeCalculator(config=calc_config, portfolio_value=1000.0)

            opened = 0
            for m in raw_markets:
                tokens = m.get("tokens", [])
                yes_price = 0.5
                for t in tokens:
                    if (t.get("outcome") or "").upper() == "YES":
                        yes_price = float(t.get("price") or 0.5)
                        break

                condition_id = m.get("condition_id", m.get("id", ""))
                question = m.get("question", "")
                prob = generate_synthetic_probability(question, yes_price)

                estimate = ProbabilityEstimate(
                    market_id=condition_id,
                    probability=prob,
                    source="e2e_synthetic",
                    reasoning="Synthetic probability for live E2E test",
                )
                signal = calc.evaluate(estimate, yes_price)

                if signal.action != "SKIP":
                    try:
                        engine.open_position(
                            market_id=signal.market_id,
                            side=signal.side,
                            shares=signal.shares,
                            price=signal.price,
                            reason=signal.reason,
                            confidence=prob,
                        )
                        opened += 1
                        log_entry = {
                            "cycle_time": datetime.now(timezone.utc).isoformat(),
                            "market_id": condition_id,
                            "question": question,
                            "market_price": yes_price,
                            "our_probability": prob,
                            "action": signal.action,
                            "side": signal.side,
                            "edge": signal.edge,
                            "executed": True,
                        }
                        with open(E2E_SIGNAL_LOG, "a") as f:
                            f.write(json.dumps(log_entry, default=str) + "\n")
                    except Exception as e:
                        logger.warning("Failed to open: %s", e)

            stage.detail = f"{opened} positions opened (synthetic probs, no MiroFish)"
            report.trades_opened = opened
        else:
            # Use the full pipeline trigger with MiroFish interviews
            trigger = PipelineTrigger(config=config, paper_engine=engine)
            signals = await trigger.run_once()
            actionable = [s for s in signals if s.get("executed")]
            stage.detail = f"{len(actionable)} trades from {len(signals)} evaluations"
            report.trades_opened = len(actionable)

    portfolio = engine.get_portfolio()
    print(f"\n  Portfolio after trading: ${portfolio.total_value:.2f}")
    print(f"  Open positions: {len(portfolio.positions)}")
    if portfolio.positions:
        for pos in portfolio.positions:
            print(f"    {pos.side:3s} {pos.market_id[:24]:24s} x{pos.shares:.0f} @{pos.entry_price:.4f}")
    print(f"\n  Re-run with --resolve-only to check for resolutions and score.\n")

    engine.close()
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="PreMemora E2E test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m e2e.run_live                          # smoke test (resolved markets)
  python -m e2e.run_live --mode live              # live test (active markets)
  python -m e2e.run_live --mode live --resolve-only  # check resolutions
  python -m e2e.run_live --graph-id mirofish_abc  # with graph context
        """,
    )
    parser.add_argument(
        "--mode", choices=["smoke", "live"], default="smoke",
        help="Test mode (default: smoke)",
    )
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--simulation-id", default="")
    parser.add_argument("--graph-id", default="")
    parser.add_argument("--resolve-only", action="store_true",
                        help="Only check for resolutions (live mode)")

    args = parser.parse_args()

    # Allow env vars as fallback
    graph_id = args.graph_id or os.getenv("PREMEMORA_GRAPH_ID", "")
    simulation_id = args.simulation_id or os.getenv("MIROFISH_SIMULATION_ID", "")

    print(f"\n  PreMemora E2E Test — mode={args.mode}, markets={args.max_markets}")
    if graph_id:
        print(f"  Graph: {graph_id}")
    if simulation_id:
        print(f"  Simulation: {simulation_id}")
    print()

    if args.mode == "smoke":
        report = await run_smoke(
            max_markets=args.max_markets,
            graph_id=graph_id,
        )
    else:
        report = await run_live(
            max_markets=args.max_markets,
            simulation_id=simulation_id,
            graph_id=graph_id,
            resolve_only=args.resolve_only,
        )

    print(report.summary)

    # Save report to data/e2e/
    report_path = E2E_DATA_DIR / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report.summary)
    print(f"\n  Report saved to: {report_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
