# Zep Cloud to Graphiti API Mapping

MiroFish was built against the Zep Cloud SDK (`zep_cloud.client.Zep`). This document maps Zep Cloud API calls to their Graphiti + Neo4j equivalents, as implemented in `adapter/client.py`.

## Why

Zep deprecated its open-source tier in April 2025. Rather than rewriting all MiroFish service code, we built an adapter that presents the same interface backed by local infrastructure.

## Graph Operations

| Zep Cloud API | Adapter Implementation |
|---------------|----------------------|
| `client.graph.add(graph_id, type, data)` | Creates Graphiti instance + stores graph metadata in Neo4j `Graph` node |
| `client.graph.search(graph_id, query)` | `graphiti.search()` with fastembed embeddings, returns `SearchResults` |

## Node Operations

| Zep Cloud API | Adapter Implementation |
|---------------|----------------------|
| `client.graph.node.get(graph_id, node_id)` | Neo4j Cypher: `MATCH (n:Entity {uuid: $id})` |
| `client.graph.node.get_by_graph_id(graph_id, cursor, limit)` | Neo4j Cypher: `MATCH (n:Entity) WHERE n.group_id = $graph_id` with cursor pagination |
| `client.graph.node.get_entity_edges(graph_id, node_id)` | Neo4j Cypher: `MATCH (n)-[r:RELATES_TO]-(m) WHERE n.uuid = $id` |

## Edge Operations

| Zep Cloud API | Adapter Implementation |
|---------------|----------------------|
| `client.graph.edge.get(graph_id, edge_id)` | Neo4j Cypher: `MATCH ()-[r:RELATES_TO {uuid: $id}]->()` |
| `client.graph.edge.get_by_graph_id(graph_id, cursor, limit)` | Neo4j Cypher: `MATCH ()-[r:RELATES_TO]->() WHERE r.group_id = $graph_id` with cursor pagination |

## Episode Operations

| Zep Cloud API | Adapter Implementation |
|---------------|----------------------|
| `client.graph.episode.add(graph_id, episodes)` | `graphiti.add_episode()` for each episode sequentially |

## Type Mapping

| Zep Cloud Type | Adapter Type | Location |
|----------------|-------------|----------|
| `zep_cloud.NodeResult` | `NodeResult` dataclass | `adapter/zep_types.py` |
| `zep_cloud.EdgeResult` | `EdgeResult` dataclass | `adapter/zep_types.py` |
| `zep_cloud.SearchResults` | `SearchResults` dataclass | `adapter/zep_types.py` |
| `zep_cloud.EpisodeData` | `EpisodeData` dataclass | `adapter/zep_types.py` |
| `zep_cloud.EntityModel` | `EntityModel` Pydantic model | `adapter/ontology_stubs.py` |
| `zep_cloud.EdgeModel` | `EdgeModel` Pydantic model | `adapter/ontology_stubs.py` |
| `zep_cloud.EntityText` | `EntityText = str` | `adapter/ontology_stubs.py` |

## Neo4j Storage Gotchas

### Relationship Types

Graphiti stores ALL Neo4j relationships as type `RELATES_TO`. The actual relationship name is in the `r.name` property, not `type(r)`.

```cypher
-- WRONG: type(r) always returns "RELATES_TO"
MATCH ()-[r]->() RETURN type(r)

-- RIGHT: actual name is in the property
MATCH ()-[r:RELATES_TO]->() RETURN r.name
```

### Node Labels

Graphiti does not set custom Neo4j node labels. All entity nodes have label `Entity`. The "type" is stored as a `labels` property (list) on the node, which the adapter enriches via ontology-based heuristic matching.

### DateTime Fields

Neo4j returns `neo4j.time.DateTime` objects, not Python `datetime`. These must be converted to strings before JSON serialization. The adapter's `_sanitize_neo4j_props()` handles this.

### Embedding Vectors

Node properties include `name_embedding` (large float arrays). These are filtered out by `_sanitize_neo4j_props()` to keep API responses compact.

## Pagination

Zep Cloud uses cursor-based pagination. The adapter implements this by sorting nodes/edges by UUID and using `WHERE n.uuid > $cursor` for the next page. See `adapter/client.py` `_NodeNamespace.get_by_graph_id()`.

MiroFish's `zep_paging.py` utility handles the pagination loop, calling `get_by_graph_id()` repeatedly until no more results are returned.
