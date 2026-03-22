"""
Graphiti adapter layer — drop-in replacement for Zep Cloud in MiroFish.

Provides the same interface that MiroFish's services expect from zep_cloud,
but backed by local Graphiti + Neo4j.
"""

from .client import GraphitiZepClient as Zep
from .zep_types import EpisodeData, EntityEdgeSourceTarget

__all__ = ["Zep", "EpisodeData", "EntityEdgeSourceTarget"]
