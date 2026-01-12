from __future__ import annotations
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from .exceptions import MisconfiguredModelError

class BM25SearchMixin:
    """
    Requires extension: CREATE EXTENSION pg_search;
    Requires a bm25 index, e.g.:
      CREATE INDEX my_idx ON my_table USING bm25 (id, content) WITH (key_field='id');
    pg_search docs show operator @@@ and scoring via paradedb.score(key_field). :contentReference[oaicite:4]{index=4}
    """

    bm25_key_field: str = "id"
    bm25_default_field: str = "content"    # the text column you search most often

    @classmethod
    def bm25_search(
        cls,
        session: Session,
        query: str,
        *,
        k: int = 20,
        field: Optional[str] = None,
        with_snippet: bool = False,
    ):
        field = field or cls.bm25_default_field
        key = cls.bm25_key_field

        if not hasattr(cls, "__tablename__"):
            raise MisconfiguredModelError("Model must be a mapped table with __tablename__")
        table = cls.__tablename__

        # We compute BM25 score and optionally a snippet.
        # Score is computed using paradedb.score(key_field). :contentReference[oaicite:5]{index=5}
        snippet_sql = f", paradedb.snippet({field}) AS snippet" if with_snippet else ""
        sql = text(f"""
            SELECT {key} AS id,
                   paradedb.score({key}) AS score
                   {snippet_sql}
            FROM {table}
            WHERE {field} @@@ :q
            ORDER BY paradedb.score({key}) DESC
            LIMIT :k
        """)
        return session.execute(sql, {"q": query, "k": int(k)}).all()

    @classmethod
    def bm25_search_objects(
        cls,
        session: Session,
        query: str,
        *,
        k: int = 20,
        field: Optional[str] = None,
    ):
        ids_scores = cls.bm25_search(session, query, k=k, field=field, with_snippet=False)
        ids = [int(r[0]) for r in ids_scores]
        if not ids:
            return []
        # preserve order
        rows = session.execute(text(f"SELECT * FROM {cls.__tablename__} WHERE {cls.bm25_key_field} = ANY(:ids)"), {"ids": ids}).all()
        by_id = {int(getattr(r, cls.bm25_key_field, r[0])): r for r in rows}  # crude fallback
        return [by_id[i] for i in ids if i in by_id]
