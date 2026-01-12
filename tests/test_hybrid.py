from __future__ import annotations

from typing import Sequence

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from age_search.base import Base
from age_search.hybrid import hybrid_search, rrf


def test_rrf_stability_and_limit():
    # ids in earlier lists get inserted first; ties should remain stable due to Python's stable sort
    out = rrf([[2, 1], [3, 1]], limit=2)
    assert out == [1, 2]


def test_hybrid_search_prefers_bm25_path(session, engine):
    class DocHybrid(Base):
        __tablename__ = "docs_hybrid"

        id: Mapped[int] = mapped_column(primary_key=True)
        content: Mapped[str] = mapped_column(String, nullable=False)

        _bm25_ids: list[int] = [2, 1]
        _vec_ids: list[int] = [3, 1]

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

    session.add_all(
        [
            DocHybrid(id=1, content="a"),
            DocHybrid(id=2, content="b"),
            DocHybrid(id=3, content="c"),
        ]
    )
    session.commit()

    out = hybrid_search(
        session,
        DocHybrid,
        query_text="ignored",
        query_vec=[0.0, 1.0],
        prefer_bm25=True,
        limit=20,
    )

    # With these rankings, id=1 should win (present in both lists).
    assert [o.id for o in out] == [1, 2, 3]


def test_hybrid_search_falls_back_to_fts_when_bm25_missing(session, engine):
    class DocFTS(Base):
        __tablename__ = "docs_fts"

        id: Mapped[int] = mapped_column(primary_key=True)
        content: Mapped[str] = mapped_column(String, nullable=False)

        _fts_ids: list[int] = [2, 1]
        _vec_ids: list[int] = [3, 1]

        @classmethod
        def fts_search(cls, _session, _query_text: str, *, k: int = 50, **_kw):  # noqa: ANN001
            ids = cls._fts_ids[:k]
            objs = _session.query(cls).filter(cls.id.in_(ids)).all()  # noqa: SLF001
            by_id = {int(o.id): o for o in objs}
            return [by_id[i] for i in ids if i in by_id]

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
            DocFTS(id=1, content="a"),
            DocFTS(id=2, content="b"),
            DocFTS(id=3, content="c"),
        ]
    )
    session.commit()

    out = hybrid_search(
        session,
        DocFTS,
        query_text="ignored",
        query_vec=[0.0, 1.0],
        prefer_bm25=True,  # should still fall back because no bm25_search attr
        limit=20,
    )

    assert [o.id for o in out] == [1, 2, 3]

