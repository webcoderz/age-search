from __future__ import annotations
from typing import Any, Optional
from sqlalchemy.orm import Session
from .cypher import cypher_json

class GraphNodeMixin:
    graph_label: str = ""         # default: class name
    graph_id_field: str = "id"
    vertex_property_key: str = "id"

    @classmethod
    def _label(cls) -> str:
        return cls.graph_label or cls.__name__

    def graph_id(self) -> Any:
        return getattr(self, self.graph_id_field)

    def graph_upsert(self, session: Session, *, graph_name: Optional[str] = None, props: Optional[dict[str, Any]] = None):
        label = self._label()
        _id = self.graph_id()
        props = props or {}
        props.setdefault(self.vertex_property_key, _id)

        cy = f"""
        MERGE (n:{label} {{{self.vertex_property_key}: $id}})
        SET n += $props
        RETURN n
        """
        rows = cypher_json(session, cy, params={"id": _id, "props": props}, graph_name=graph_name)
        return rows[0] if rows else None

    def graph_delete(self, session: Session, *, graph_name: Optional[str] = None, detach: bool = True):
        label = self._label()
        _id = self.graph_id()
        cy = f"""
        MATCH (n:{label} {{{self.vertex_property_key}: $id}})
        {"DETACH " if detach else ""}DELETE n
        RETURN n
        """
        return cypher_json(session, cy, params={"id": _id}, graph_name=graph_name)
