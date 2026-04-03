"""
Pipeline trigger — automatically evaluates Polymarket markets using
MiroFish agent interviews, feeds probability estimates to the edge
calculator, and executes trades via the paper trading engine.

Flow:
    1. Fetch active Polymarket markets (Gamma API)
    2. For each market, interview MiroFish simulation agents
    3. Parse agent responses into probability estimates
    4. Run through EdgeCalculator
    5. Execute actionable signals via PaperTradingEngine
    6. Log everything for strategy review

Usage:
    trigger = PipelineTrigger(config=PipelineConfig())
    await trigger.run_once()       # single evaluation cycle
    await trigger.run_loop()       # continuous loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("prememora.pipeline.trigger")

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

DEFAULT_SIGNAL_LOG = Path(__file__).resolve().parent.parent / "data" / "signal_log.jsonl"


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Configuration for the pipeline trigger."""
    mirofish_url: str = ""
    simulation_id: str = ""
    graph_id: str = ""                    # Graphiti graph for context enrichment
    interval_seconds: int = 1800          # 30 minutes
    max_markets: int = 10                 # top N markets by volume
    market_category: str = ""             # filter by category (empty = all)
    interview_timeout: int = 120          # seconds
    signal_log_path: Path = DEFAULT_SIGNAL_LOG
    exit_config: Any = None               # ExitConfig for exit monitoring (None = disabled)
    calibration_gate: bool = False         # enable oracle-based calibration gate
    gate_max_brier: float = 0.25          # max Brier score to allow trading

    def __post_init__(self):
        self.mirofish_url = self.mirofish_url or os.getenv("MIROFISH_BACKEND", "http://localhost:5001")
        self.simulation_id = self.simulation_id or os.getenv("MIROFISH_SIMULATION_ID", "")
        self.graph_id = self.graph_id or os.getenv("PREMEMORA_GRAPH_ID", "")


# ── Market fetching ───────────────────────────────────────────────────────────


@dataclass
class ActiveMarket:
    """A Polymarket market we want to evaluate."""
    id: str
    question: str
    token_ids: list[str]
    current_price: float   # YES token price (0-1)
    volume: float
    category: str
    end_date: str = ""     # ISO date when market closes (from Gamma endDateIso)


async def fetch_active_markets(
    session: aiohttp.ClientSession,
    max_markets: int = 10,
    category: str = "",
) -> list[ActiveMarket]:
    """Fetch top active markets from the Polymarket Gamma API."""
    params: dict[str, Any] = {
        "limit": max_markets,
        "active": "true",
        "order": "volume",
        "ascending": "false",
        "closed": "false",
    }
    if category:
        params["tag"] = category

    try:
        async with session.get(f"{GAMMA_API_BASE}/markets", params=params) as resp:
            if resp.status != 200:
                logger.warning("Gamma API returned %d", resp.status)
                return []
            data = await resp.json()
    except Exception as e:
        logger.error("Failed to fetch markets: %s", e)
        return []

    markets = []
    for m in data:
        tokens = m.get("tokens", [])
        token_ids = [t.get("token_id", "") for t in tokens]

        # YES token price
        yes_price = None
        for t in tokens:
            if t.get("outcome", "").upper() == "YES":
                yes_price = float(t.get("price", 0))
                break
        if yes_price is None and tokens:
            yes_price = float(tokens[0].get("price", 0.5))

        markets.append(ActiveMarket(
            id=m.get("condition_id", m.get("id", "")),
            question=m.get("question", ""),
            token_ids=token_ids,
            current_price=yes_price or 0.5,
            volume=float(m.get("volume", 0)),
            category=m.get("groupSlug", m.get("category", "")),
            end_date=m.get("endDateIso", ""),
        ))

    logger.info("Fetched %d active markets", len(markets))
    return markets


# ── MiroFish interview ────────────────────────────────────────────────────────


async def interview_agents(
    session: aiohttp.ClientSession,
    mirofish_url: str,
    simulation_id: str,
    question: str,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Interview all MiroFish agents with a probability question.

    Returns list of agent response dicts.
    """
    url = f"{mirofish_url}/api/simulation/interview/all"
    payload = {
        "simulation_id": simulation_id,
        "prompt": question,
        "timeout": timeout,
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout + 30)) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Interview API returned %d: %s", resp.status, body[:200])
                return []
            data = await resp.json()
            if not data.get("success"):
                logger.warning("Interview failed: %s", data.get("error", "unknown"))
                return []

            results = data.get("data", {}).get("result", {}).get("results", {})
            return list(results.values()) if isinstance(results, dict) else results
    except asyncio.TimeoutError:
        logger.warning("Interview timed out after %ds", timeout)
        return []
    except Exception as e:
        logger.error("Interview request failed: %s", e)
        return []


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_probability(response_text: str) -> float | None:
    """Extract a probability (0-1) from an agent's text response.

    Looks for patterns like:
        - "70%" or "70 percent"
        - "probability: 0.70"
        - "I'd estimate 0.7"
        - "around 65-75%" → takes midpoint
    """
    if not response_text:
        return None

    text = response_text.lower()

    # Pattern: "probability: 0.XX" or "probability of 0.XX"
    m = re.search(r'probability[:\s]+(?:of\s+)?(\d*\.?\d+)', text)
    if m:
        val = float(m.group(1))
        return val if val <= 1 else val / 100

    # Pattern: range "XX-YY%" → midpoint (must check before single %)
    m = re.search(r'(\d{1,3})\s*[-–]\s*(\d{1,3})\s*%', text)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return (low + high) / 200

    # Pattern: "XX%" or "XX percent"
    m = re.search(r'(\d{1,3})(?:\.\d+)?[%\s]*(?:percent|%)', text)
    if m:
        return float(m.group(1)) / 100

    # Pattern: bare decimal "0.XX"
    m = re.search(r'\b(0\.\d+)\b', text)
    if m:
        return float(m.group(1))

    return None


def aggregate_probabilities(probs: list[float]) -> float | None:
    """Aggregate multiple agent probability estimates into one.

    Uses trimmed mean: drop highest and lowest, average the rest.
    Falls back to simple mean if < 4 agents.
    """
    if not probs:
        return None
    if len(probs) < 4:
        return sum(probs) / len(probs)

    sorted_probs = sorted(probs)
    trimmed = sorted_probs[1:-1]
    return sum(trimmed) / len(trimmed)


# ── Signal logging ────────────────────────────────────────────────────────────


def log_signal(signal_log_path: Path, signal_data: dict[str, Any]) -> None:
    """Append a signal record to the JSONL log."""
    signal_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(signal_log_path, "a") as f:
        f.write(json.dumps(signal_data, default=str) + "\n")


# ── Pipeline trigger ──────────────────────────────────────────────────────────


class PipelineTrigger:
    """Orchestrates the full prediction pipeline cycle.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    paper_engine : PaperTradingEngine | None
        Paper trading engine instance. If None, signals are logged but not executed.
    edge_config : EdgeConfig | None
        Edge calculator configuration.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        paper_engine: Any = None,
        edge_config: Any = None,
    ):
        self.config = config or PipelineConfig()
        self.paper_engine = paper_engine
        self._edge_config = edge_config
        self._cycle_count = 0
        self._context_builder = None
        self._exit_monitor = None
        self._calibration_gate = None
        self._gate_passed = None  # None = not checked, True/False = last result

        if self.config.graph_id:
            from pipeline.context import ContextBuilder
            self._context_builder = ContextBuilder(graph_id=self.config.graph_id)

        if self.config.calibration_gate:
            from trading.calibration_gate import CalibrationGate, GateConfig
            self._calibration_gate = CalibrationGate(config=GateConfig(
                max_brier=self.config.gate_max_brier,
            ))

        if self.config.exit_config and self.paper_engine:
            from trading.exit_monitor import ExitMonitor
            self._exit_monitor = ExitMonitor(
                config=self.config.exit_config,
                paper_engine=self.paper_engine,
                mirofish_url=self.config.mirofish_url,
                simulation_id=self.config.simulation_id,
                graph_id=self.config.graph_id,
            )

    def _get_edge_calculator(self):
        from trading.edge_calculator import EdgeCalculator, EdgeConfig
        config = self._edge_config or EdgeConfig()

        portfolio_value = 1000.0
        current_exposure = 0.0
        if self.paper_engine:
            p = self.paper_engine.get_portfolio()
            portfolio_value = p.total_value
            current_exposure = p.total_value - p.cash

        return EdgeCalculator(
            config=config,
            portfolio_value=portfolio_value,
            current_exposure=current_exposure,
        )

    async def run_once(self) -> list[dict[str, Any]]:
        """Run a single evaluation cycle. Returns list of signal records."""
        self._cycle_count += 1
        cycle_time = datetime.now(timezone.utc).isoformat()
        logger.info("Pipeline cycle #%d starting", self._cycle_count)

        # Calibration gate check — block trade execution if poorly calibrated
        execute_trades = True
        if self._calibration_gate:
            try:
                gate_result = await self._calibration_gate.check()
                self._gate_passed = gate_result.can_trade
                execute_trades = gate_result.can_trade
                if not execute_trades:
                    logger.warning(
                        "Calibration gate BLOCKED trading: %s", gate_result.reason,
                    )
                else:
                    logger.info(
                        "Calibration gate passed (brier=%.4f)",
                        gate_result.brier_score or 0,
                    )
            except Exception:
                logger.exception("Calibration gate check failed — defaulting to no-trade")
                execute_trades = False

        signals_log: list[dict[str, Any]] = []

        async with aiohttp.ClientSession() as session:
            # 1. Fetch active markets
            markets = await fetch_active_markets(
                session,
                max_markets=self.config.max_markets,
                category=self.config.market_category,
            )

            if not markets:
                logger.warning("No active markets found")
                return signals_log

            calc = self._get_edge_calculator()

            for market in markets:
                record = await self._evaluate_market(session, market, calc, cycle_time, execute_trades)
                signals_log.append(record)
                log_signal(self.config.signal_log_path, record)

        actionable = [s for s in signals_log if s.get("action") not in ("SKIP", None)]
        logger.info(
            "Pipeline cycle #%d complete: %d markets evaluated, %d actionable signals",
            self._cycle_count, len(signals_log), len(actionable),
        )

        # Run exit monitoring on existing positions
        if self._exit_monitor:
            try:
                exit_signals = await self._exit_monitor.run_once()
                for es in exit_signals:
                    exit_record = {
                        "cycle_time": cycle_time,
                        "market_id": es.market_id,
                        "action": "EXIT",
                        "reason": es.reason,
                        "trigger_type": es.trigger_type,
                        "position_id": es.position_id,
                        "old_confidence": es.old_confidence,
                        "new_confidence": es.new_confidence,
                        "market_price": es.market_price,
                        "executed": True,
                    }
                    signals_log.append(exit_record)
                    log_signal(self.config.signal_log_path, exit_record)
                if exit_signals:
                    logger.info("Exit monitor triggered %d exits", len(exit_signals))
            except Exception:
                logger.exception("Exit monitor failed")

        return signals_log

    async def _evaluate_market(
        self,
        session: aiohttp.ClientSession,
        market: ActiveMarket,
        calc: Any,
        cycle_time: str,
        execute: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a single market: interview → parse → edge calc → maybe trade."""
        from trading.edge_calculator import ProbabilityEstimate

        record: dict[str, Any] = {
            "cycle_time": cycle_time,
            "market_id": market.id,
            "question": market.question,
            "market_price": market.current_price,
            "action": "SKIP",
            "reason": "",
        }

        # Build context from knowledge graph (if graph_id configured)
        context = None
        if self.config.graph_id and self._context_builder:
            try:
                context = await asyncio.to_thread(
                    self._context_builder.build_context, market.question
                )
                record["context_facts"] = len(context.facts) if context else 0
                record["search_terms"] = context.search_terms if context else []
            except Exception as e:
                logger.warning("Context building failed for %s: %s", market.id[:16], e)

        # Get probability estimate — MiroFish interview or LLM fallback
        our_prob = None
        context_facts = context.facts if context else []

        if self.config.simulation_id:
            # Build interview prompt (enriched with graph context if available)
            if context and context.facts:
                from pipeline.context import build_enriched_prompt
                question = build_enriched_prompt(market.question, context)
            else:
                question = (
                    f"What probability (0-100%) do you assign to the following: "
                    f"{market.question} "
                    f"Please give a specific number."
                )

            responses = await interview_agents(
                session,
                self.config.mirofish_url,
                self.config.simulation_id,
                question,
                timeout=self.config.interview_timeout,
            )

            if not responses:
                record["reason"] = "no agent responses"
                return record

            probs = []
            for resp in responses:
                text = resp.get("response", "")
                p = parse_probability(text)
                if p is not None and 0 <= p <= 1:
                    probs.append(p)

            record["agent_responses"] = len(responses)
            record["parsed_probabilities"] = probs

            if not probs:
                record["reason"] = f"could not parse probability from {len(responses)} responses"
                return record

            our_prob = aggregate_probabilities(probs)
            record["source"] = "mirofish"
        else:
            # LLM fallback — single call to estimate probability
            try:
                from backtesting.hindsight import llm_estimate_probability
                our_prob = await llm_estimate_probability(
                    question=market.question,
                    context_facts=context_facts,
                    market_price=market.current_price,
                )
                record["source"] = "llm_direct"
            except Exception as e:
                record["reason"] = f"LLM fallback failed: {e}"
                return record

        if our_prob is None:
            record["reason"] = "probability estimation failed"
            return record

        record["our_probability"] = our_prob

        # Edge calculation
        context_note = ""
        if context and context.facts:
            context_note = f" with {len(context.facts)} graph facts"
        source = record.get("source", "mirofish")
        if source == "mirofish":
            reasoning = f"Aggregated from {len(record.get('parsed_probabilities', []))} agent estimates{context_note}"
        else:
            reasoning = f"LLM direct estimate{context_note}"

        estimate = ProbabilityEstimate(
            market_id=market.id,
            probability=our_prob,
            source=source,
            reasoning=reasoning,
        )
        signal = calc.evaluate(estimate, market.current_price)

        record["action"] = signal.action
        record["side"] = signal.side
        record["edge"] = signal.edge
        record["kelly_fraction"] = signal.kelly_fraction
        record["shares"] = signal.shares
        record["price"] = signal.price
        record["reason"] = signal.reason

        # Execute if actionable and gate allows it
        if signal.action != "SKIP" and self.paper_engine and execute:
            try:
                pos = self.paper_engine.open_position(
                    market_id=signal.market_id,
                    side=signal.side,
                    shares=signal.shares,
                    price=signal.price,
                    reason=signal.reason,
                    confidence=our_prob,
                    market_deadline=market.end_date or None,
                )
                record["executed"] = True
                record["position_id"] = pos.id
                logger.info("Executed: %s %s x%.0f @%.4f on %s",
                            signal.side, signal.market_id[:16], signal.shares, signal.price, pos.id)
            except Exception as e:
                record["executed"] = False
                record["execution_error"] = str(e)
                logger.warning("Failed to execute signal on %s: %s", market.id[:16], e)
        else:
            record["executed"] = False
            if signal.action != "SKIP" and not execute:
                record["gate_blocked"] = True
                logger.info("Signal blocked by calibration gate: %s %s", signal.action, market.id[:16])

        return record

    async def run_loop(self) -> None:
        """Run evaluation cycles on the configured interval."""
        logger.info(
            "Pipeline starting — interval=%ds, max_markets=%d, mirofish=%s",
            self.config.interval_seconds,
            self.config.max_markets,
            self.config.mirofish_url,
        )

        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("Pipeline cycle failed")

            await asyncio.sleep(self.config.interval_seconds)


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="PreMemora prediction pipeline")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between cycles")
    parser.add_argument("--max-markets", type=int, default=10)
    parser.add_argument("--simulation-id", default="")
    parser.add_argument("--graph-id", default="", help="Graphiti graph ID for context enrichment")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute trades")
    args = parser.parse_args()

    config = PipelineConfig(
        simulation_id=args.simulation_id,
        graph_id=args.graph_id,
        interval_seconds=args.interval,
        max_markets=args.max_markets,
    )

    engine = None
    if not args.dry_run:
        from trading.paper_engine import PaperTradingEngine
        engine = PaperTradingEngine()

    trigger = PipelineTrigger(config=config, paper_engine=engine)

    if args.once:
        signals = await trigger.run_once()
        for s in signals:
            action = s.get("action", "SKIP")
            q = s.get("question", "")[:60]
            prob = s.get("our_probability", "?")
            mp = s.get("market_price", "?")
            print(f"[{action:8s}] {q}  (ours={prob}, market={mp})")
    else:
        await trigger.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
