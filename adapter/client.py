"""
GraphitiZepClient — drop-in replacement for zep_cloud.client.Zep.

Maps Zep Cloud API surface to local Graphiti + Neo4j operations.
MiroFish code can import this as `Zep` and use the same interface.
"""

import asyncio
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.time import DateTime as Neo4jDateTime


def _sanitize_neo4j_props(d: dict) -> dict:
    """Convert Neo4j-specific types to JSON-serializable Python types."""
    result = {}
    for k, v in d.items():
        if k.endswith("_embedding"):
            continue  # skip large embedding vectors
        elif isinstance(v, Neo4jDateTime):
            result[k] = v.isoformat()
        elif isinstance(v, (list, tuple)):
            result[k] = [str(x) if isinstance(x, Neo4jDateTime) else x for x in v]
        else:
            result[k] = v
    return result

from .zep_types import (
    EdgeResult,
    EpisodeData,
    EpisodeResult,
    NodeResult,
    SearchResults,
)

logger = logging.getLogger("prememora.adapter")

# Graphiti is async — we need a persistent event loop so the Neo4j async driver
# doesn't get attached to a loop that gets destroyed between calls.
import threading

_loop = None
_loop_thread = None
_loop_lock = threading.Lock()


def _get_loop():
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
            _loop_thread.start()
    return _loop


def _run_async(coro):
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)


class _GraphNamespace:
    """Mimics client.graph.* API from Zep Cloud SDK."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, llm_config: Optional[Dict] = None):
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._llm_config = llm_config or {}
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._graphiti_instances: Dict[str, Any] = {}

        self.node = _NodeNamespace(self._driver)
        self.edge = _EdgeNamespace(self._driver)
        self.episode = _EpisodeNamespace(self._driver)

    def _get_graphiti(self, graph_id: str):
        """Lazy-init a Graphiti instance per graph_id."""
        if graph_id not in self._graphiti_instances:
            from graphiti_core import Graphiti
            from graphiti_core.llm_client import LLMConfig
            from .minimax_llm_client import MiniMaxLLMClient

            api_key = os.environ.get("MINIMAX_API_KEY")
            if not api_key:
                raise ValueError("MINIMAX_API_KEY not set")

            minimax_llm_config = LLMConfig(
                api_key=api_key,
                base_url="https://api.minimaxi.chat/v1",
                model=os.environ.get("GRAPHITI_MODEL", "MiniMax-M2.5"),
                small_model=os.environ.get("GRAPHITI_SMALL_MODEL", "MiniMax-M2.5"),
            )

            llm_client = MiniMaxLLMClient(minimax_llm_config)

            from .local_embedder import LocalEmbedder
            embedder = LocalEmbedder()

            from .minimax_reranker import MiniMaxReranker
            reranker = MiniMaxReranker(config=minimax_llm_config)

            g = Graphiti(
                self._neo4j_uri, self._neo4j_user, self._neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=reranker,
            )
            _run_async(g.build_indices_and_constraints())
            self._graphiti_instances[graph_id] = g
        return self._graphiti_instances[graph_id]

    def create(self, graph_id: str, name: str = "", description: str = ""):
        """Create a graph (in Neo4j, we use labels/properties to namespace)."""
        with self._driver.session() as session:
            session.run(
                "MERGE (g:Graph {graph_id: $graph_id}) "
                "SET g.name = $name, g.description = $description, g.created_at = $now",
                graph_id=graph_id, name=name, description=description,
                now=datetime.now(timezone.utc).isoformat(),
            )
        logger.info(f"Created graph: {graph_id}")

    def delete(self, graph_id: str):
        """Delete all data associated with a graph."""
        with self._driver.session() as session:
            session.run(
                "MATCH (n) WHERE n.graph_id = $graph_id DETACH DELETE n",
                graph_id=graph_id,
            )
        self._graphiti_instances.pop(graph_id, None)
        logger.info(f"Deleted graph: {graph_id}")

    def add(self, graph_id: str, type: str = "text", data: str = ""):
        """Add a single episode (text) to the graph via Graphiti."""
        g = self._get_graphiti(graph_id)
        ep_uuid = uuid.uuid4().hex[:16]
        _run_async(g.add_episode(
            name=f"episode_{ep_uuid}",
            episode_body=data,
            source_description="mirofish_simulation",
            group_id=graph_id,
            reference_time=datetime.now(timezone.utc),
        ))
        # Track episode in Neo4j for status queries
        with self._driver.session() as session:
            session.run(
                "CREATE (e:Episode {uuid: $uuid, graph_id: $graph_id, processed: true, "
                "created_at: $now, type: $type})",
                uuid=ep_uuid, graph_id=graph_id,
                now=datetime.now(timezone.utc).isoformat(), type=type,
            )
        return EpisodeResult(uuid_=ep_uuid, processed=True)

    def add_batch(self, graph_id: str, episodes: List[EpisodeData]) -> List[EpisodeResult]:
        """Add multiple episodes using Graphiti's bulk API."""
        from graphiti_core.utils.bulk_utils import RawEpisode
        from graphiti_core.nodes import EpisodeType

        g = self._get_graphiti(graph_id)
        raw_episodes = []
        ep_uuids = []
        now = datetime.now(timezone.utc)

        for ep in episodes:
            ep_uuid = uuid.uuid4().hex[:16]
            ep_uuids.append(ep_uuid)
            raw_episodes.append(RawEpisode(
                name=f"episode_{ep_uuid}",
                content=ep.data,
                source=EpisodeType.text,
                source_description="mirofish_graph_build",
                reference_time=now,
            ))

        _run_async(g.add_episode_bulk(raw_episodes, group_id=graph_id))

        # Track all episodes in Neo4j
        results = []
        with self._driver.session() as session:
            for ep_uuid in ep_uuids:
                session.run(
                    "CREATE (e:Episode {uuid: $uuid, graph_id: $graph_id, processed: true, "
                    "created_at: $now, type: 'text'})",
                    uuid=ep_uuid, graph_id=graph_id,
                    now=now.isoformat(),
                )
                results.append(EpisodeResult(uuid_=ep_uuid, processed=True))

        return results

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,
    ) -> SearchResults:
        """Search the graph using Graphiti's hybrid search."""
        g = self._get_graphiti(graph_id)
        results = _run_async(g.search(query=query, num_results=limit, group_ids=[graph_id]))

        edges = []
        for r in results:
            edges.append(EdgeResult(
                uuid_=getattr(r, "uuid", str(uuid.uuid4())),
                name=getattr(r, "name", ""),
                fact=getattr(r, "fact", str(r)),
                source_node_uuid=getattr(r, "source_node_uuid", ""),
                target_node_uuid=getattr(r, "target_node_uuid", ""),
                valid_at=str(getattr(r, "valid_at", "")) if getattr(r, "valid_at", None) else None,
                invalid_at=str(getattr(r, "invalid_at", "")) if getattr(r, "invalid_at", None) else None,
            ))

        return SearchResults(edges=edges, nodes=[])

    def set_ontology(self, graph_ids: List[str], entities=None, edges=None):
        """
        Set ontology for graphs.
        Graphiti handles ontology differently — it learns from data.
        We store the ontology definition in Neo4j for reference.
        """
        for gid in graph_ids:
            with self._driver.session() as session:
                session.run(
                    "MATCH (g:Graph {graph_id: $graph_id}) "
                    "SET g.ontology_entities = $entities, g.ontology_edges = $edges",
                    graph_id=gid,
                    entities=str(list(entities.keys())) if entities else "[]",
                    edges=str(list(edges.keys())) if edges else "[]",
                )
        logger.info(f"Ontology set for graphs: {graph_ids}")


class _NodeNamespace:
    """Mimics client.graph.node.* from Zep Cloud SDK."""

    def __init__(self, driver):
        self._driver = driver
        self._ontology_cache: Dict[str, List[str]] = {}

    def _get_ontology_types(self, graph_id: str) -> List[str]:
        """Get stored ontology entity types for a graph."""
        if graph_id not in self._ontology_cache:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (g:Graph {graph_id: $graph_id}) RETURN g.ontology_entities as types",
                    graph_id=graph_id,
                )
                record = result.single()
                if record and record["types"]:
                    import ast
                    try:
                        self._ontology_cache[graph_id] = ast.literal_eval(record["types"])
                    except Exception:
                        self._ontology_cache[graph_id] = []
                else:
                    self._ontology_cache[graph_id] = []
        return self._ontology_cache[graph_id]

    def _infer_entity_type(self, name: str, summary: str, ontology_types: List[str]) -> Optional[str]:
        """Simple heuristic to infer entity type from name/summary and ontology."""
        if not ontology_types:
            return None
        name_lower = name.lower()
        text = f"{name} {summary}".lower()
        # Keyword mappings for common ontology types (ordered by specificity)
        type_keywords = {
            "CompanyCEO": ["ceo", "chief executive", "saylor"],
            "CentralBankOfficial": ["chair", "governor", "official", "powell", "yellen"],
            "CentralBank": ["central bank", "federal reserve", "ecb", "boj"],
            "AssetManagementFirm": ["blackrock", "fidelity", "vanguard", "asset management"],
            "PublicCompany": ["microstrategy", "tesla", "company", "inc", "corp"],
            "PredictionMarket": ["polymarket", "prediction market", "kalshi"],
            "BlockchainProtocol": ["ethereum", "bitcoin", "solana", "protocol", "blockchain", "upgrade"],
            "CryptoAnalyst": ["analyst", "researcher"],
            "Person": ["person"],
            "Organization": ["organization"],
        }
        # First pass: match against entity NAME only (more precise)
        priority_order = [
            "CompanyCEO", "CentralBankOfficial", "CentralBank",
            "AssetManagementFirm", "PublicCompany", "PredictionMarket",
            "BlockchainProtocol", "CryptoAnalyst", "Person", "Organization",
        ]
        for etype in priority_order:
            if etype not in ontology_types:
                continue
            keywords = type_keywords.get(etype, [etype.lower()])
            for kw in keywords:
                if kw in name_lower:
                    return etype
        # Second pass: match against name+summary (broader)
        for etype in priority_order:
            if etype not in ontology_types:
                continue
            keywords = type_keywords.get(etype, [etype.lower()])
            for kw in keywords:
                if kw in text:
                    return etype
        # Fallback: try matching the ontology type name itself
        for etype in ontology_types:
            if etype not in type_keywords and etype.lower() in text:
                return etype
        return None

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None) -> List[NodeResult]:
        query = (
            "MATCH (n:Entity) WHERE n.group_id = $graph_id "
        )
        if uuid_cursor:
            query += "AND n.uuid > $cursor "
        query += "RETURN n ORDER BY n.uuid LIMIT $limit"

        params = {"graph_id": graph_id, "limit": limit}
        if uuid_cursor:
            params["cursor"] = uuid_cursor

        ontology_types = self._get_ontology_types(graph_id)

        with self._driver.session() as session:
            result = session.run(query, **params)
            nodes = []
            for record in result:
                n = record["n"]
                # Use stored 'labels' property (Graphiti stores entity types there)
                # rather than Neo4j node labels (which are just ['Entity'])
                stored_labels = n.get("labels", [])
                if not stored_labels or not isinstance(stored_labels, list):
                    stored_labels = list(n.labels) if hasattr(n, "labels") else []

                # Enrich: if only default labels, infer type from ontology
                custom = [l for l in stored_labels if l not in ("Entity", "Node")]
                if not custom and ontology_types:
                    inferred = self._infer_entity_type(
                        n.get("name", ""), n.get("summary", ""), ontology_types
                    )
                    if inferred:
                        stored_labels = stored_labels + [inferred]

                nodes.append(NodeResult(
                    uuid_=n.get("uuid", n.get("name", "")),
                    name=n.get("name", ""),
                    labels=stored_labels,
                    summary=n.get("summary", ""),
                    attributes=_sanitize_neo4j_props(dict(n)) if n else {},
                    created_at=str(n.get("created_at")) if n.get("created_at") else None,
                ))
            return nodes

    def get(self, uuid_: str) -> Optional[NodeResult]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n LIMIT 1",
                uuid=uuid_,
            )
            record = result.single()
            if not record:
                return None
            n = record["n"]
            stored_labels = n.get("labels", [])
            if not stored_labels or not isinstance(stored_labels, list):
                stored_labels = list(n.labels) if hasattr(n, "labels") else []
            return NodeResult(
                uuid_=n.get("uuid", ""),
                name=n.get("name", ""),
                labels=stored_labels,
                summary=n.get("summary", ""),
                attributes=dict(n) if n else {},
            )

    def get_entity_edges(self, node_uuid: str) -> List[EdgeResult]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (a:Entity {uuid: $uuid})-[r]-(b:Entity) "
                "RETURN r, a.uuid AS source, b.uuid AS target",
                uuid=node_uuid,
            )
            edges = []
            for record in result:
                r = record["r"]
                edges.append(EdgeResult(
                    uuid_=r.get("uuid", ""),
                    name=r.get("name", "") or type(r).__name__,
                    fact=r.get("fact", ""),
                    source_node_uuid=record["source"],
                    target_node_uuid=record["target"],
                ))
            return edges


class _EdgeNamespace:
    """Mimics client.graph.edge.* from Zep Cloud SDK."""

    def __init__(self, driver):
        self._driver = driver

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None) -> List[EdgeResult]:
        query = (
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE a.group_id = $graph_id "
        )
        if uuid_cursor:
            query += "AND r.uuid > $cursor "
        query += "RETURN r, a.uuid AS source, b.uuid AS target ORDER BY r.uuid LIMIT $limit"

        params = {"graph_id": graph_id, "limit": limit}
        if uuid_cursor:
            params["cursor"] = uuid_cursor

        with self._driver.session() as session:
            result = session.run(query, **params)
            edges = []
            for record in result:
                r = record["r"]
                edges.append(EdgeResult(
                    uuid_=r.get("uuid", ""),
                    name=r.get("name", "") or type(r).__name__,
                    fact=r.get("fact", ""),
                    source_node_uuid=record["source"],
                    target_node_uuid=record["target"],
                    valid_at=str(r.get("valid_at")) if r.get("valid_at") else None,
                    invalid_at=str(r.get("invalid_at")) if r.get("invalid_at") else None,
                    expired_at=str(r.get("expired_at")) if r.get("expired_at") else None,
                    created_at=str(r.get("created_at")) if r.get("created_at") else None,
                ))
            return edges


class _EpisodeNamespace:
    """Mimics client.graph.episode.* from Zep Cloud SDK."""

    def __init__(self, driver):
        self._driver = driver

    def get(self, uuid_: str) -> EpisodeResult:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Episode {uuid: $uuid}) RETURN e LIMIT 1",
                uuid=uuid_,
            )
            record = result.single()
            if record:
                e = record["e"]
                return EpisodeResult(
                    uuid_=e.get("uuid", uuid_),
                    processed=e.get("processed", True),
                )
            # Graphiti processes synchronously, so if episode exists it's done
            return EpisodeResult(uuid_=uuid_, processed=True)


class GraphitiZepClient:
    """
    Drop-in replacement for `from zep_cloud.client import Zep`.

    Usage:
        from adapter import Zep
        client = Zep(api_key="unused")  # api_key ignored, uses local Neo4j
    """

    def __init__(
        self,
        api_key: str = "",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "prememora_local",
    ):
        # api_key is accepted but ignored — kept for compatibility
        self.graph = _GraphNamespace(neo4j_uri, neo4j_user, neo4j_password)
