from __future__ import annotations

from typing import Sequence

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from age_search.base import Base
from age_search.taxonomy import Label, make_doc_labels_table
from age_search.hybrid_relational import hybrid_search_results_in_label_subtree_relational


def test_hybrid_search_results_in_label_subtree_relational(session, engine):
    class Doc(Base):
        __tablename__ = "docs_rel"

        id: Mapped[int] = mapped_column(primary_key=True)
        content: Mapped[str] = mapped_column(String, nullable=False)

        _bm25_ids: list[int] = [1, 2, 3]
        _vec_ids: list[int] = [3, 2, 1]

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

    doc_labels = make_doc_labels_table(Base.metadata, doc_table="docs_rel", table_name="doc_labels_rel")
    Base.metadata.create_all(engine)

    # labels: 10(root) -> 11(child)
    root = Label(id=10, slug="root", name="Root")
    child = Label(id=11, slug="child", name="Child", parent=root)
    session.add_all([root, child])

    session.add_all([Doc(id=1, content="a"), Doc(id=2, content="b"), Doc(id=3, content="c")])
    session.commit()

    # doc 1 labeled with child, doc 3 labeled with unrelated label 12
    other = Label(id=12, slug="other", name="Other")
    session.add(other)
    session.commit()

    session.execute(doc_labels.insert().values(doc_id=1, label_id=11))
    session.execute(doc_labels.insert().values(doc_id=3, label_id=12))
    session.commit()

    results = hybrid_search_results_in_label_subtree_relational(
        session,
        Doc,
        root_label_id=10,
        doc_labels=doc_labels,
        query_text="ignored",
        query_vec=[0.0, 1.0],
        prefer_bm25=True,
        limit=20,
        fetch_objects=True,
    )

    # Only doc 1 is under root subtree (via child); doc 3 is excluded.
    assert [r.id for r in results] == [1]
    assert results[0].obj is not None

