"""
Graph context builder — queries the Graphiti knowledge graph for facts
relevant to a market question, then formats them into context for
agent interviews.

The key insight: rather than classifying events at ingestion time (which
is unreliable), we use Graphiti's entity extraction + semantic search at
query time. When we need to evaluate "Will BTC hit $100k?", we search
the graph for BTC-related facts and feed those to the agents.

Usage:
    builder = ContextBuilder(graph_id="my_graph", neo4j_uri="bolt://localhost:7687")
    context = builder.build_context("Will Bitcoin exceed $100,000 by April 2026?")
    # context.facts = ["Bitcoin surged past $95k...", "Binance saw 50k BTC outflow...", ...]
    # context.prompt_section = "Based on the following recent events:\n- ..."
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("prememora.pipeline.context")

# Common terms to extract from market questions for search queries.
# We search the graph for each relevant term separately and merge results.
_STOP_WORDS = frozenset({
    "will", "would", "could", "should", "does", "do", "did", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "the", "a",
    "an", "and", "or", "but", "if", "of", "at", "by", "for", "with", "in",
    "on", "to", "from", "up", "about", "into", "than", "that", "this",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "before", "after", "above", "below", "between", "during", "through",
    "against", "each", "every", "both", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "too", "very", "just", "because",
    "not", "no", "nor", "can", "may", "might", "shall", "then", "there",
    "these", "those", "his", "her", "its", "our", "their", "your", "my",
    "end", "any", "all", "over", "under", "exceed", "reach", "hit", "pass",
    "remain", "stay", "drop", "fall", "rise", "go", "get", "price",
    "market", "percent", "yes", "no", "happen", "occur",
})


@dataclass
class MarketContext:
    """Context gathered from the knowledge graph for a specific market."""
    question: str
    search_terms: list[str]
    facts: list[str]
    fact_count: int = 0
    prompt_section: str = ""

    def __post_init__(self):
        self.fact_count = len(self.facts)
        if self.facts:
            bullets = "\n".join(f"- {f}" for f in self.facts)
            self.prompt_section = (
                f"Here are {self.fact_count} recent facts from our knowledge graph "
                f"that may be relevant:\n{bullets}"
            )
        else:
            self.prompt_section = (
                "We have no specific recent intelligence on this topic. "
                "Use your general knowledge and reasoning."
            )


def extract_search_terms(question: str) -> list[str]:
    """Extract meaningful search terms from a market question.

    Pulls out entities, numbers, and key nouns while filtering stop words.
    Keeps multi-word terms that look like proper nouns (e.g. "Federal Reserve").
    """
    # Remove punctuation except hyphens and dollar signs
    cleaned = re.sub(r'[^\w\s\-$%.]', ' ', question)
    words = cleaned.split()

    terms = []
    i = 0
    while i < len(words):
        word = words[i]

        # Skip pure numbers and dates unless they have $ or %
        if re.match(r'^\d+\.?\d*$', word) and i + 1 < len(words):
            # Check if next word makes it meaningful (e.g. "100k", "2026")
            next_word = words[i + 1].lower() if i + 1 < len(words) else ""
            if word.startswith('$') or next_word in ('%',):
                terms.append(word)
            i += 1
            continue

        if word.lower() in _STOP_WORDS:
            i += 1
            continue

        # Keep words with $ prefix
        if word.startswith('$'):
            terms.append(word)
            i += 1
            continue

        # Multi-word proper nouns: consecutive capitalized words
        if word[0].isupper() and len(word) > 1:
            compound = [word]
            j = i + 1
            while j < len(words) and words[j][0:1].isupper() and words[j].lower() not in _STOP_WORDS:
                compound.append(words[j])
                j += 1
            if len(compound) > 1:
                terms.append(" ".join(compound))
                # Also add individual words if they're meaningful
                for w in compound:
                    if len(w) > 2 and w.lower() not in _STOP_WORDS:
                        terms.append(w)
                i = j
                continue

        # Single meaningful word
        if len(word) > 2 and word.lower() not in _STOP_WORDS:
            terms.append(word)

        i += 1

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


class ContextBuilder:
    """Queries the Graphiti knowledge graph for context relevant to a market.

    Parameters
    ----------
    graph_id : str
        The Graphiti graph to search.
    neo4j_uri, neo4j_user, neo4j_password : str
        Neo4j connection details.
    max_facts : int
        Maximum number of facts to include in context.
    max_facts_per_term : int
        Maximum search results per term (prevents one term from dominating).
    """

    def __init__(
        self,
        graph_id: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "prememora_local",
        max_facts: int = 15,
        max_facts_per_term: int = 5,
    ):
        self.graph_id = graph_id
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.max_facts = max_facts
        self.max_facts_per_term = max_facts_per_term
        self._client = None

    def _get_client(self):
        if self._client is None:
            from adapter.client import GraphitiZepClient
            self._client = GraphitiZepClient(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
            )
        return self._client

    def build_context(self, question: str) -> MarketContext:
        """Build context for a market question by searching the graph.

        Extracts key terms from the question, searches the graph for each,
        deduplicates and ranks the results.
        """
        terms = extract_search_terms(question)
        if not terms:
            logger.debug("No search terms extracted from: %s", question[:80])
            return MarketContext(question=question, search_terms=[], facts=[])

        logger.info("Searching graph for terms: %s", terms)

        all_facts: list[str] = []
        seen_facts: set[str] = set()

        client = self._get_client()

        # Also search with the full question for semantic matching
        search_queries = terms + [question]

        for query in search_queries:
            try:
                results = client.graph.search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=self.max_facts_per_term,
                )
                for edge in results.edges:
                    fact = edge.fact.strip() if edge.fact else ""
                    if not fact or len(fact) < 10:
                        continue
                    # Deduplicate by normalized text
                    norm = fact.lower()[:100]
                    if norm not in seen_facts:
                        seen_facts.add(norm)
                        all_facts.append(fact)
            except Exception as e:
                logger.warning("Graph search failed for '%s': %s", query[:40], e)

        # Truncate to max
        facts = all_facts[:self.max_facts]

        logger.info(
            "Built context for '%s': %d terms → %d unique facts (from %d total)",
            question[:60], len(terms), len(facts), len(all_facts),
        )

        return MarketContext(question=question, search_terms=terms, facts=facts)


def build_enriched_prompt(question: str, context: MarketContext) -> str:
    """Build an interview prompt that includes graph context.

    This is the prompt sent to MiroFish agents. It includes:
    1. The relevant facts from our knowledge graph
    2. The specific probability question
    3. Instructions for structured response
    """
    parts = []

    if context.facts:
        parts.append(context.prompt_section)
        parts.append("")

    parts.append(
        f"Given the above context, what probability (0-100%) do you assign to "
        f"the following outcome?\n\n"
        f"  {question}\n\n"
        f"Think step by step about the evidence, then give a specific probability "
        f"as a percentage. Be precise — don't just say 'likely' or 'unlikely'."
    )

    return "\n".join(parts)
