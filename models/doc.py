from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Text

from age_search import (
    Base,
    GraphNodeMixin,
    VectorMixin,
    FTSSearchMixin,
    BM25SearchMixin,
    GraphRelationship,
)

class Doc(
    Base,
    GraphNodeMixin,
    VectorMixin,
    FTSSearchMixin,
    BM25SearchMixin,
):
    """
    Canonical document model supporting:
      - Graph traversal (Apache AGE)
      - Vector similarity (pgvector, cosine)
      - Full-text search (Postgres FTS)
      - BM25 search (pg_search)
      - Hybrid ranking
    """

    __tablename__ = "docs"

    # ------------------------
    # Relational identity
    # ------------------------
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ------------------------
    # Text content (lexical search)
    # ------------------------
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # ------------------------
    # Graph config (AGE)
    # ------------------------
    graph_label = "Doc"
    graph_id_field = "id"
    vertex_property_key = "id"

    # ------------------------
    # Vector config (pgvector)
    # ------------------------
    vector_dim = 1536
    # embedding column provided by VectorMixin

    # ------------------------
    # FTS config (built-in Postgres)
    # ------------------------
    fts_config = "english"
    fts_source_field = "content"
    # content_tsv column provided by FTSSearchMixin

    # ------------------------
    # BM25 config (pg_search)
    # ------------------------
    bm25_key_field = "id"
    bm25_default_field = "content"

    # ------------------------
    # Graph relationships
    # ------------------------
    related = GraphRelationship(
        "RELATED_TO",
        direction="out",
        target_label="Doc",
    )

    mentions = GraphRelationship(
        "MENTIONS",
        direction="out",
        target_label="Doc",
    )

    # ------------------------
    # Convenience helpers
    # ------------------------
    def __repr__(self) -> str:
        return f"<Doc id={self.id}>"
