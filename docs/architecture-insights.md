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
