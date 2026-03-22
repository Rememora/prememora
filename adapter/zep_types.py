"""
Type stubs that match zep_cloud SDK types used by MiroFish.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeData:
    data: str
    type: str = "text"


@dataclass
class EntityEdgeSourceTarget:
    source: str = "Entity"
    target: str = "Entity"


@dataclass
class NodeResult:
    uuid_: str = ""
    name: str = ""
    labels: List[str] = field(default_factory=list)
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class EdgeResult:
    uuid_: str = ""
    name: str = ""
    fact: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)
    fact_type: str = ""

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class EpisodeResult:
    uuid_: str = ""
    processed: bool = False

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class SearchResults:
    edges: List[EdgeResult] = field(default_factory=list)
    nodes: List[NodeResult] = field(default_factory=list)


class InternalServerError(Exception):
    pass
