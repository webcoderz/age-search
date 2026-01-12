from __future__ import annotations

from typing import Iterable, Optional

from sqlalchemy import ForeignKey, Integer, String, Table, Text, Column, select
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session

from .base import Base
from .cypher import cypher_json
from .mixins_graph import GraphNodeMixin
from .relationships import GraphRelationship


class Label(Base, GraphNodeMixin):
    """
    Hierarchical label/taxonomy node.

    Relational source of truth:
      - adjacency list via parent_id

    Optional AGE mirror (if you upsert these nodes):
      (:Label {id, slug, name})-[:PARENT_OF]->(:Label)
    """

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))

    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("labels.id"), nullable=True, index=True)
    parent: Mapped[Optional["Label"]] = relationship(
        "Label",
        remote_side="Label.id",
        back_populates="children",
    )
    children: Mapped[list["Label"]] = relationship(
        "Label",
        back_populates="parent",
        cascade="all, delete-orphan",
    )

    # Optional precomputed path (e.g. "/root/child/grandchild") for prefix filtering.
    path: Mapped[Optional[str]] = mapped_column(Text, nullable=True, index=True)

    # Graph config (AGE)
    graph_label = "Label"
    graph_id_field = "id"
    vertex_property_key = "id"

    # Graph traversal helpers (AGE)
    children_graph = GraphRelationship("PARENT_OF", direction="out", target_label="Label", source_label="Label")
    parent_graph = GraphRelationship("PARENT_OF", direction="in", target_label="Label", source_label="Label")

    def __repr__(self) -> str:
        return f"<Label id={self.id} slug={self.slug!r}>"


def make_doc_labels_table(
    metadata,
    *,
    doc_table: str,
    label_table: str = "labels",
    table_name: str = "doc_labels",
    doc_id_col: str = "doc_id",
    label_id_col: str = "label_id",
    doc_pk: str = "id",
    label_pk: str = "id",
) -> Table:
    """
    Create (or return existing) many-to-many association table between a document table and labels.

    This is intentionally explicit so you can use any doc model/table name.
    """
    existing = metadata.tables.get(table_name)
    if existing is not None:
        return existing

    return Table(
        table_name,
        metadata,
        Column(doc_id_col, ForeignKey(f"{doc_table}.{doc_pk}", ondelete="CASCADE"), primary_key=True),
        Column(label_id_col, ForeignKey(f"{label_table}.{label_pk}", ondelete="CASCADE"), primary_key=True),
    )


def descendant_label_ids(
    session: Session,
    *,
    root_label_id: int,
    include_self: bool = True,
) -> list[int]:
    """
    Relational subtree expansion using a recursive CTE (works on Postgres and SQLite).
    """
    lbl = Label.__table__

    cte = select(lbl.c.id).where(lbl.c.id == int(root_label_id)).cte(recursive=True)
    cte = cte.union_all(select(lbl.c.id).where(lbl.c.parent_id == cte.c.id))

    ids = [int(r[0]) for r in session.execute(select(cte.c.id)).all()]
    if not include_self:
        ids = [i for i in ids if i != int(root_label_id)]
    return ids


def doc_ids_for_labels(
    session: Session,
    *,
    doc_labels: Table,
    label_ids: Iterable[int],
    label_id_col: str = "label_id",
    doc_id_col: str = "doc_id",
) -> list[int]:
    """
    Relational mapping from label ids to doc ids through the association table.
    """
    ids = [int(i) for i in label_ids]
    if not ids:
        return []

    stmt = select(getattr(doc_labels.c, doc_id_col)).distinct().where(getattr(doc_labels.c, label_id_col).in_(ids))
    return [int(r[0]) for r in session.execute(stmt).all()]


def graph_descendant_label_ids(
    session: Session,
    *,
    graph_name: str,
    root_label_id: int,
    max_hops: int = 25,
    include_self: bool = True,
    limit: int = 5000,
) -> list[int]:
    """
    AGE subtree expansion for Label nodes connected by :PARENT_OF edges.
    """
    cy = f"""
    MATCH (root:Label {{id: $root}})
    MATCH (root)-[:PARENT_OF*1..{int(max_hops)}]->(d:Label)
    RETURN DISTINCT d.id
    LIMIT {int(limit)}
    """
    rows = cypher_json(session, cy, params={"root": int(root_label_id)}, graph_name=graph_name)
    out = [int(x) for x in rows if x is not None]
    if include_self:
        out.append(int(root_label_id))
    # preserve order; dedupe
    seen: set[int] = set()
    deduped: list[int] = []
    for i in out:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


def graph_doc_ids_for_label_ids(
    session: Session,
    *,
    graph_name: str,
    label_ids: list[int],
    doc_label: str = "Doc",
    label_label: str = "Label",
    edge: str = "HAS_LABEL",
    limit: int = 50000,
) -> list[int]:
    """
    Return doc ids for docs connected to ANY of the provided label ids via (:Doc)-[:HAS_LABEL]->(:Label).
    """
    if not label_ids:
        return []

    cy = f"""
    MATCH (d:{doc_label})-[:{edge}]->(l:{label_label})
    WHERE l.id IN $ids
    RETURN DISTINCT d.id
    LIMIT {int(limit)}
    """
    rows = cypher_json(session, cy, params={"ids": [int(i) for i in label_ids]}, graph_name=graph_name)
    return [int(x) for x in rows if x is not None]


def graph_doc_ids_in_label_subtree(
    session: Session,
    *,
    graph_name: str,
    root_label_id: int,
    max_hops: int = 25,
    include_self: bool = True,
    doc_label: str = "Doc",
    edge: str = "HAS_LABEL",
    limit: int = 50000,
) -> list[int]:
    label_ids = graph_descendant_label_ids(
        session,
        graph_name=graph_name,
        root_label_id=root_label_id,
        max_hops=max_hops,
        include_self=include_self,
        limit=min(int(limit), 5000),
    )
    return graph_doc_ids_for_label_ids(
        session,
        graph_name=graph_name,
        label_ids=label_ids,
        doc_label=doc_label,
        edge=edge,
        limit=limit,
    )

