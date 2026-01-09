from __future__ import annotations
from sqlalchemy import Computed, Index, func, select
from sqlalchemy.orm import Mapped, mapped_column, Session
from typing import Any

class FTSSearchMixin:
    fts_config: str = "english"
    fts_source_field: str = "content"

    # default computed column name content_tsv
    content_tsv: Mapped[Any] = mapped_column(
        Computed("to_tsvector('english', coalesce(content, ''))", persisted=True)
    )

    @classmethod
    def fts_index(cls) -> Index:
        return Index(f"ix_{cls.__tablename__}_fts", cls.content_tsv, postgresql_using="gin")

    @classmethod
    def fts_search(cls, session: Session, query: str, *, k: int = 20):
        tsq = func.websearch_to_tsquery(cls.fts_config, query)
        rank = func.ts_rank_cd(cls.content_tsv, tsq)
        stmt = select(cls).where(cls.content_tsv.op("@@")(tsq)).order_by(rank.desc()).limit(int(k))
        return session.execute(stmt).scalars().all()
