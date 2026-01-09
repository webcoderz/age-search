from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Type
from sqlalchemy.orm import Session
from .query import CypherQuery

@dataclass(frozen=True)
class GraphRelationship:
    edge: str
    direction: str = "out"      # out | in | both
    target_label: Optional[str] = None
    source_label: Optional[str] = None
    source_key: str = "id"
    target_key: str = "id"

    def __set_name__(self, owner: Type[Any], _name: str):
        if self.source_label is None:
            object.__setattr__(self, "source_label", owner.__name__)

    def __get__(self, instance: Any, owner: Type[Any]):
        if instance is None:
            return self
        return _BoundRel(self, instance)

@dataclass
class _BoundRel:
    rel: GraphRelationship
    inst: Any

    def query(self, session: Session, *, graph_name: Optional[str] = None) -> CypherQuery:
        src = self.rel.source_label or self.inst.__class__.__name__
        tgt = f":{self.rel.target_label}" if self.rel.target_label else ""
        edge = self.rel.edge

        if self.rel.direction == "out":
            pat = f"(n:{src} {{{self.rel.source_key}: $id}})-[:{edge}]->(m{tgt})"
        elif self.rel.direction == "in":
            pat = f"(n:{src} {{{self.rel.source_key}: $id}})<-[:{edge}]-(m{tgt})"
        else:
            pat = f"(n:{src} {{{self.rel.source_key}: $id}})-[:{edge}]-(m{tgt})"

        q = CypherQuery(
            session=session,
            match=f"MATCH {pat}",
            return_expr="m",
            params={"id": getattr(self.inst, self.rel.source_key)},
            graph_name=graph_name,
        )
        return q

    def __call__(self, session: Session, *, graph_name: Optional[str] = None) -> CypherQuery:
        return self.query(session, graph_name=graph_name)

    def add(self, session: Session, other: Any, *, graph_name: Optional[str] = None):
        src = self.rel.source_label or self.inst.__class__.__name__
        tgt = self.rel.target_label or other.__class__.__name__
        edge = self.rel.edge
        cy = f"""
        MATCH (n:{src} {{{self.rel.source_key}: $src_id}})
        MATCH (m:{tgt} {{{self.rel.target_key}: $tgt_id}})
        MERGE (n)-[:{edge}]->(m)
        RETURN m
        """
        q = CypherQuery(session, cy, "m", {"src_id": getattr(self.inst, self.rel.source_key),
                                          "tgt_id": getattr(other, self.rel.target_key)}, graph_name=graph_name)
        return q.first()
