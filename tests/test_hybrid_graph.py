from __future__ import annotations

from typing import Sequence

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from age_search.base import Base
from age_search.hybrid_graph import hybrid_search_results_constrained


def test_hybrid_search_results_constrained_filters_to_allowed_ids(session, engine):
    class Doc(Base):
        __tablename__ = "docs_constrained"

        id: Mapped[int] = mapped_column(primary_key=True)
        content: Mapped[str] = mapped_column(String, nullable=False)

        _bm25_ids: list[int] = [2, 1, 3]
        _vec_ids: list[int] = [3, 1, 2]

        @classmethod
        def bm25_search(cls, _session, _query_text: str, *, k: int = 50, **_kw):  # noqa: ANN001
            ids = cls._bm25_ids[:k]
            return [(i, 1.0) for i in ids]

        @classmethod
        def vector_search(
            cls,
            _session,
            _query_vec: Sequence[float],
            *,
            k: int = 50,
            **_kw,
        ):
            ids = cls._vec_ids[:k]
            objs = _session.query(cls).filter(cls.id.in_(ids)).all()  # noqa: SLF001
            by_id = {int(o.id): o for o in objs}
            return [by_id[i] for i in ids if i in by_id]

    Base.metadata.create_all(engine)
    session.add_all([Doc(id=1, content="a"), Doc(id=2, content="b"), Doc(id=3, content="c")])
    session.commit()

    results = hybrid_search_results_constrained(
        session,
        Doc,
        query_text="ignored",
        query_vec=[0.0, 1.0],
        allowed_ids=[1, 2],  # exclude 3
        limit=20,
        fetch_objects=True,
    )

    assert {r.id for r in results} <= {1, 2}
    assert all(r.obj is not None for r in results)

