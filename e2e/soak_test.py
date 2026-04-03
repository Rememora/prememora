"""
Soak test — run the pipeline continuously in paper-trading mode against
real Polymarket markets. Periodically checks calibration, runs exit
monitor, and attempts to auto-resolve settled markets.

This is the real validation: predictions on active markets scored against
actual outcomes after resolution. Unlike the hindsight oracle (which uses
synthetic eval prices), this captures real market prices at evaluation time.

Usage:
    python -m e2e.soak_test start [--hours 72] [--interval 1800]
    python -m e2e.soak_test status
    python -m e2e.soak_test report
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("prememora.e2e.soak_test")

SOAK_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soak"
SOAK_DB_PATH = SOAK_DATA_DIR / "soak_paper_trading.db"
SOAK_SIGNAL_LOG = SOAK_DATA_DIR / "soak_signal_log.jsonl"
SOAK_STATUS_FILE = SOAK_DATA_DIR / "soak_status.json"


def _write_status(status: dict) -> None:
    SOAK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOAK_STATUS_FILE.write_text(json.dumps(status, default=str, indent=2))


def _read_status() -> dict | None:
    if SOAK_STATUS_FILE.exists():
        return json.loads(SOAK_STATUS_FILE.read_text())
    return None


async def run_soak(
    hours: float = 72,
    interval: int = 1800,
    max_markets: int = 10,
    simulation_id: str = "",
    graph_id: str = "",
    gate_enabled: bool = True,
) -> None:
    """Run the soak test.

    Starts the pipeline trigger in continuous mode with:
    - Paper trading engine (isolated DB)
    - Calibration gate (if enabled)
    - Exit monitor
    - Periodic auto-resolve checks
    """
    from pipeline.trigger import PipelineConfig, PipelineTrigger
    from trading.paper_engine import PaperTradingEngine
    from trading.strategy_review import StrategyReview

    SOAK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous soak data only if explicitly requested
    engine = PaperTradingEngine(db_path=SOAK_DB_PATH, initial_cash=1000.0)

    from pipeline.trigger import DEFAULT_RELEVANCE_KEYWORDS

    config = PipelineConfig(
        simulation_id=simulation_id,
        graph_id=graph_id,
        interval_seconds=interval,
        max_markets=max_markets,
        signal_log_path=SOAK_SIGNAL_LOG,
        calibration_gate=gate_enabled,
        relevance_keywords=DEFAULT_RELEVANCE_KEYWORDS,
    )

    trigger = PipelineTrigger(config=config, paper_engine=engine)
    review = StrategyReview(paper_engine=engine, signal_log_path=SOAK_SIGNAL_LOG)

    end_time = datetime.now(timezone.utc).timestamp() + (hours * 3600)
    cycle = 0
    stop = False

    def handle_signal(*_):
        nonlocal stop
        stop = True
        logger.info("Received stop signal, finishing current cycle...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(
        "Soak test starting: %d hours, %ds interval, %d markets, gate=%s",
        hours, interval, max_markets, gate_enabled,
    )

    _write_status({
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target_hours": hours,
        "interval": interval,
        "cycles": 0,
    })

    while not stop and datetime.now(timezone.utc).timestamp() < end_time:
        cycle += 1
        logger.info("=== Soak cycle %d ===", cycle)

        try:
            # Run pipeline evaluation
            signals = await trigger.run_once()
            actionable = [s for s in signals if s.get("action") not in ("SKIP", None)]
            executed = [s for s in signals if s.get("executed")]
            blocked = [s for s in signals if s.get("gate_blocked")]

            logger.info(
                "Cycle %d: %d markets, %d actionable, %d executed, %d blocked",
                cycle, len(signals), len(actionable), len(executed), len(blocked),
            )

            # Try auto-resolving settled markets every cycle
            try:
                resolved = await review.auto_resolve()
                if resolved:
                    logger.info("Auto-resolved %d positions", len(resolved))
                    for r in resolved:
                        status = "WON" if r["won"] else "LOST"
                        logger.info(
                            "  %s %s pnl=$%.2f",
                            r["market_id"][:20], status, r["pnl"],
                        )
            except Exception:
                logger.exception("Auto-resolve failed")

            # Update status file
            portfolio = engine.get_portfolio()
            _write_status({
                "state": "running",
                "started_at": _read_status().get("started_at", "") if _read_status() else "",
                "target_hours": hours,
                "interval": interval,
                "cycles": cycle,
                "last_cycle": datetime.now(timezone.utc).isoformat(),
                "portfolio_value": portfolio.total_value,
                "cash": portfolio.cash,
                "open_positions": len(portfolio.positions),
                "total_trades": portfolio.total_trades,
                "realized_pnl": portfolio.realized_pnl,
            })

        except Exception:
            logger.exception("Soak cycle %d failed", cycle)

        # Wait for next cycle
        if not stop and datetime.now(timezone.utc).timestamp() < end_time:
            logger.info("Sleeping %ds until next cycle...", interval)
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    # Final report
    logger.info("Soak test ending after %d cycles", cycle)
    report = review.generate_report()
    print(f"\n{report.summary}\n")

    # Save final status
    _write_status({
        "state": "completed",
        "started_at": _read_status().get("started_at", "") if _read_status() else "",
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "cycles": cycle,
        "portfolio_value": engine.get_portfolio().total_value,
        "brier_score": report.brier_score,
        "total_pnl": report.total_pnl,
    })

    engine.close()


def show_status() -> None:
    """Show current soak test status."""
    status = _read_status()
    if not status:
        print("No soak test data found. Run: python -m e2e.soak_test start")
        return

    print(f"State:           {status.get('state', 'unknown')}")
    print(f"Started:         {status.get('started_at', 'N/A')}")
    print(f"Cycles:          {status.get('cycles', 0)}")
    print(f"Last cycle:      {status.get('last_cycle', 'N/A')}")
    print(f"Portfolio:       ${status.get('portfolio_value', 0):.2f}")
    print(f"Cash:            ${status.get('cash', 0):.2f}")
    print(f"Open positions:  {status.get('open_positions', 0)}")
    print(f"Total trades:    {status.get('total_trades', 0)}")
    print(f"Realized P&L:    ${status.get('realized_pnl', 0):+.2f}")
    if status.get("brier_score") is not None:
        print(f"Brier score:     {status['brier_score']:.4f}")


def show_report() -> None:
    """Generate strategy report from soak test data."""
    if not SOAK_DB_PATH.exists():
        print("No soak test data. Run: python -m e2e.soak_test start")
        return

    from trading.paper_engine import PaperTradingEngine
    from trading.strategy_review import StrategyReview

    engine = PaperTradingEngine(db_path=SOAK_DB_PATH)
    review = StrategyReview(paper_engine=engine, signal_log_path=SOAK_SIGNAL_LOG)
    report = review.generate_report()
    print(report.summary)
    engine.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Soak test — continuous paper trading")
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="Start the soak test")
    p_start.add_argument("--hours", type=float, default=72)
    p_start.add_argument("--interval", type=int, default=1800)
    p_start.add_argument("--max-markets", type=int, default=20)
    p_start.add_argument("--simulation-id", default="")
    p_start.add_argument("--graph-id", default="")
    p_start.add_argument("--no-gate", action="store_true", help="Disable calibration gate")

    sub.add_parser("status", help="Show soak test status")
    sub.add_parser("report", help="Generate strategy report")

    args = parser.parse_args()

    if args.command == "start":
        sim_id = args.simulation_id or os.getenv("MIROFISH_SIMULATION_ID", "")
        gid = args.graph_id or os.getenv("PREMEMORA_GRAPH_ID", "")
        await run_soak(
            hours=args.hours,
            interval=args.interval,
            max_markets=args.max_markets,
            simulation_id=sim_id,
            graph_id=gid,
            gate_enabled=not args.no_gate,
        )

    elif args.command == "status":
        show_status()

    elif args.command == "report":
        show_report()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
