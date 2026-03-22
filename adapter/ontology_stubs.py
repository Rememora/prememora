"""
Stubs for zep_cloud.external_clients.ontology types.

Graphiti learns ontology from data rather than requiring explicit schema.
These stubs let MiroFish's graph_builder.py code run without changes —
the ontology is stored as metadata but Graphiti does its own entity extraction.
"""

from pydantic import BaseModel


# Type alias — Pydantic v2 can't schema-ify a str subclass without extra work
EntityText = str


class EntityModel(BaseModel):
    """Stub for zep_cloud EntityModel."""
    class Config:
        extra = "allow"


class EdgeModel(BaseModel):
    """Stub for zep_cloud EdgeModel."""
    class Config:
        extra = "allow"
