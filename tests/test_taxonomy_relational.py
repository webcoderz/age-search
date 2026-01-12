from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from age_search.base import Base
from age_search.taxonomy import Label, descendant_label_ids, make_doc_labels_table, doc_ids_for_labels


def test_descendant_label_ids_recursive_cte(session, engine):
    Base.metadata.create_all(engine)

    root = Label(id=1, slug="root", name="Root")
    child = Label(id=2, slug="child", name="Child", parent=root)
    grand = Label(id=3, slug="grand", name="Grand", parent=child)
    session.add_all([root, child, grand])
    session.commit()

    assert descendant_label_ids(session, root_label_id=1, include_self=True) == [1, 2, 3]
    assert descendant_label_ids(session, root_label_id=2, include_self=True) == [2, 3]
    assert descendant_label_ids(session, root_label_id=2, include_self=False) == [3]


def test_doc_ids_for_labels_via_association_table(session, engine):
    class Doc(Base):
        __tablename__ = "docs"

        id: Mapped[int] = mapped_column(primary_key=True)
        content: Mapped[str] = mapped_column(String, nullable=False)

    doc_labels = make_doc_labels_table(Base.metadata, doc_table="docs")
    Base.metadata.create_all(engine)

    session.add_all(
        [
            Doc(id=1, content="a"),
            Doc(id=2, content="b"),
            Label(id=10, slug="l1", name="L1"),
            Label(id=11, slug="l2", name="L2"),
        ]
    )
    session.commit()

    session.execute(doc_labels.insert().values(doc_id=1, label_id=10))
    session.execute(doc_labels.insert().values(doc_id=2, label_id=11))
    session.commit()

    assert sorted(doc_ids_for_labels(session, doc_labels=doc_labels, label_ids=[10, 11])) == [1, 2]
    assert doc_ids_for_labels(session, doc_labels=doc_labels, label_ids=[10]) == [1]

