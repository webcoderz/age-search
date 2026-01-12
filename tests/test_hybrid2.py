from __future__ import annotations

from typing import Sequence

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from age_search.base import Base
from age_search.hybrid2 import hybrid_search_results


def test_hybrid_search_results_scores_ranks_and_snippets(session, engine):
    class DocHybrid2(Base):
        __tablename__ = "docs_hybrid2"

        id: Mapped[int] = mapped_column(primary_key=True)
        content: Mapped[str] = mapped_column(String, nullable=False)

        _bm25_ids: list[int] = [2, 1]
        _vec_ids: list[int] = [3, 1]

        @classmethod
        def bm25_search(
            cls,
            _session,
            _query_text: str,
            *,
            k: int = 50,
            with_snippet: bool = False,
            **_kw,
        ):
            ids = cls._bm25_ids[:k]
            if with_snippet:
                return [(i, 10.0 - n, f"snippet:{i}") for n, i in enumerate(ids)]
            return [(i, 10.0 - n) for n, i in enumerate(ids)]

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

    session.add_all(
        [
            DocHybrid2(id=1, content="a"),
            DocHybrid2(id=2, content="b"),
            DocHybrid2(id=3, content="c"),
        ]
    )
    session.commit()

    results = hybrid_search_results(
        session,
        DocHybrid2,
        query_text="ignored",
        query_vec=[0.0, 1.0],
        prefer_bm25=True,
        limit=20,
        fetch_objects=True,
    )

    assert [r.id for r in results] == [1, 2, 3]

    r1 = next(r for r in results if r.id == 1)
    assert r1.obj is not None
    assert r1.lexical_rank == 2
    assert r1.semantic_rank == 2
    assert r1.rrf_score is not None

    r2 = next(r for r in results if r.id == 2)
    assert r2.lexical_rank == 1
    assert r2.semantic_rank is None
    assert r2.bm25_score is not None
    assert r2.snippet == "snippet:2"

    r3 = next(r for r in results if r.id == 3)
    assert r3.lexical_rank is None
    assert r3.semantic_rank == 1

