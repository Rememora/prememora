# Architecture Insights

Design decisions and lessons learned while building the PreMemora prediction pipeline. Each insight captures the *why* behind a choice so future work builds on it rather than re-discovering it.

## Query-Time vs Ingestion-Time Classification

**Decision**: Don't classify events by market at ingestion time. Instead, query the knowledge graph at prediction time.

**Why**: LLM relevance scoring at ingestion is unreliable — it hallucinates connections, misses subtle ones, and is inconsistent across runs. If you build the whole pipeline on a flaky classifier, predictions are garbage regardless of how good the simulation is.

**What we do instead**: Graphiti already extracts entities (Bitcoin, Binance, Federal Reserve) when ingesting events. At prediction time, we extract key terms from the market question ("Will BTC hit $100k?" → [Bitcoin, BTC]), search the graph for episodes connected to those entities, and feed those as context to agents.

**Benefits**:
- Entity extraction is what Graphiti is built for — it's good at it
- Graph traversal is deterministic — no LLM flakiness at query time
- Relationship context comes free (BTC → Binance → whale alert chain)
- No wasted LLM calls at ingestion time (already bottlenecked there)
- No O(events × markets) classification — only O(markets) graph searches when evaluating

## Adapter Event Loop Isolation

**Decision**: Run Graphiti ingestion calls in `asyncio.to_thread()` from the orchestrator.

**Why**: The adapter uses a background event loop via `run_coroutine_threadsafe` to bridge sync MiroFish code with async Graphiti. When the orchestrator (itself async) calls `graph.add()`, Graphiti's Neo4j driver binds to the wrong loop, causing "Future attached to a different loop" errors.

**Fix**: `await asyncio.to_thread(client.graph.add, ...)` runs the sync adapter call in a thread pool, where it creates its own event loop context cleanly.

## Brier Score Over Accuracy

**Decision**: Use Brier score (not accuracy) as the primary prediction quality metric.

**Why**: Accuracy just measures "did we get the direction right?" but ignores confidence. A model that says 51% on everything it gets right scores the same accuracy as one saying 95%. Brier score penalizes overconfidence — saying 90% and being wrong costs more than saying 55% and being wrong. This is critical for Kelly sizing, since overbetting on false confidence is how you blow up.

**Formula**: `Brier = mean((predicted - actual)²)`. Range: 0 (perfect) to 1 (always wrong). 0.25 = random guessing.

## Calibration Curves Over Aggregate Scores

**Decision**: Report calibration by probability bucket, not just aggregate Brier score.

**Why**: A Brier of 0.18 could mean "well-calibrated everywhere" or "perfect at 50% but terrible at 90%". The calibration curve catches this. If the 80-90% bucket shows 50% actual win rate, we know to shrink position sizes for high-confidence signals specifically — the aggregate score wouldn't tell us that.

## Quarter-Kelly Position Sizing

**Decision**: Default to 0.25× Kelly fraction, not full Kelly or even half-Kelly.

**Why**: Full Kelly is mathematically optimal for known probabilities, but our probabilities are *estimated* by LLM agents — they're noisy. Full Kelly on noisy estimates leads to massive variance and potential ruin. Half-Kelly (0.5×) is standard in quantitative finance for uncertain edge; quarter-Kelly (0.25×) is conservative for a system that hasn't been validated yet. The strategy review can recommend increasing this once calibration proves out.

## Source Attribution as the Improvement Lever

**Decision**: Track P&L per signal source (whale alerts, RSS, reddit, graph context) in the strategy review.

**Why**: Knowing "we lost $50 total" doesn't tell you what to fix. Knowing "whale alert signals made +$112 on 4 trades while reddit signals lost -$120 on 12 trades" tells you exactly where to invest engineering effort. If whale signals are gold, add more chains. If graph context hurts predictions, the context builder is injecting noise — fix search term extraction.

## LRU Dedup Over Unbounded Sets

**Decision**: Use an LRU (OrderedDict-based) dedup set in the ingestion orchestrator instead of a plain set.

**Why**: The orchestrator runs continuously. A plain `set()` grows unbounded as events accumulate over days/weeks, eventually consuming significant memory. LRU eviction at 10K entries means old events naturally fall off while recent ones stay deduped. O(1) for both lookup and eviction.

## Dual Callback Signatures

**Decision**: The orchestrator provides both `_handle_event(dict)` and `_handle_event_batch(list[dict])` callbacks.

**Why**: The 6 ingestors were built independently with two callback patterns — 4 pass a single event dict, while Reddit and FRED pass a list. Rather than refactoring all connectors (touching tested code for no functional gain), the orchestrator adapts to both. The batch handler just loops over `_handle_event`.

## Prediction Market Fee Model

**Decision**: Fee formula is `bps/10000 * min(price, 1-price) * shares`.

**Why**: In a binary market, YES at $0.60 means NO is $0.40. The fee is charged on the cheaper side because that represents the "risk" being priced. At $0.50 (max uncertainty), both sides pay equally. At $0.95, the fee is tiny because the market is near-certain. No fee on resolution — Polymarket only charges on trades, not settlements.

## No Fee on Market Resolution

Winners get exactly $1/share, losers get $0. The entry fee is the only drag on a winning position. This is why the paper trading engine records `fee=0` for RESOLVE trade records.

## Calibration Gate as Circuit Breaker

**Decision**: Block all trade execution when prediction Brier score exceeds 0.25 (random guessing baseline).

**Why**: The hindsight oracle revealed that while LLM probability estimates can have good calibration (Brier 0.10), the trading outcomes can still be negative (-$1,857 in backtest). Running the oracle before trading acts like a circuit breaker — if the system's predictions degrade (model drift, API changes, bad context), trading stops automatically rather than bleeding money.

**How it works**: Before each pipeline cycle, the gate runs the oracle on recently resolved markets. Results are cached in SQLite (default: 24h TTL). If Brier > threshold, all signals are logged but `executed=False` with `gate_blocked=True` in the signal log.

## Agent Expertise Matching

**Decision**: Filter markets by keyword relevance before interviewing agents.

**Why**: MiroFish agents have specific personas (BlackRock, Federal Reserve, MicroStrategy, etc.) with crypto/finance/geopolitics expertise. When asked about golf tournaments or basketball games, every agent returns 50% — no edge, wasted interview time (~5 minutes for 22 agents). Word-boundary keyword matching (`\bbitcoin\b`, `\bwar\b`) skips irrelevant markets before the expensive LLM interview step.

**Result**: On real Polymarket data, agents gave strong signals on geopolitics (2% for "Will Israel strike 14 countries" vs market 50%) and nuanced estimates on tech (32% for "DeepSeek V4 by April 7" vs market 50%), while correctly abstaining on sports.

## Swarm Over Single LLM

**Decision**: MiroFish agent interviews are required — no LLM fallback.

**Why**: We briefly added a single-LLM fallback for when MiroFish isn't running. It was removed because: (a) 22 agents with diverse personas produce a range of estimates (e.g., 20-85% on the Hormuz question) that captures genuine uncertainty — a single LLM gives one point estimate; (b) trimmed mean aggregation filters outliers that would mislead a single model; (c) the whole value proposition of PreMemora is swarm intelligence, not yet-another-LLM-wrapper.

## Brier Score vs P&L Divergence

**Discovery**: Good Brier score (0.10) does not guarantee profitable trading (-$1,857).

**Why**: Brier measures probability calibration ("are your 70% predictions right 70% of the time?"). P&L depends on edge size, position sizing, and market selection. You can have perfect calibration and still lose money if you bet on tiny edges that happen to go wrong, or if the eval prices don't reflect real market conditions.

**Implication**: The calibration gate uses Brier (prevents garbage predictions), but profitability requires additional tuning of min_edge threshold, Kelly fraction, and market selection. These should be tuned using the soak test data, not just Brier.

## CLOB Price History Gap

**Discovery**: Polymarket's CLOB API doesn't serve price history for resolved markets.

**Why**: Once a market resolves, the CLOB order book is closed and historical data becomes unavailable. This means the hindsight oracle can't get real pre-resolution prices for backtesting — it has to use synthetic eval prices, making backtest P&L unreliable.

**Fix**: The data collector (`e2e/data_collector.py`) polls Polymarket every 15 minutes and stores price snapshots in SQLite while markets are active. When markets resolve, we have their actual price history from our own observations. This gives the oracle real eval prices for future backtests.

## Data-First, Trade-Later

**Decision**: Run in data-collection mode before enabling trading.

**Why**: The system needs historical data to validate itself. Without price history, the oracle uses synthetic prices. Without accumulated graph events, agents lack context. Without resolved predictions, we can't compute meaningful Brier scores. The right sequence is: (1) collect market prices + news events for weeks, (2) run the oracle with real historical data, (3) validate prediction accuracy, (4) enable paper trading only when calibration proves out, (5) live trading only after paper trading is profitable.

**What's running continuously**: Ingestion orchestrator (RSS + crypto news → graph) + data collector (market prices + resolutions → SQLite). Everything else is on-demand.
