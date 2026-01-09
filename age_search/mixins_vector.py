from __future__ import annotations
from typing import Any, Sequence, Literal
from sqlalchemy.orm import Mapped, mapped_column, Session
from sqlalchemy import select
from pgvector.sqlalchemy import VECTOR

Distance = Literal["cosine", "l2", "ip"]

class VectorMixin:
    vector_dim: int = 1536
    embedding: Mapped[Any] = mapped_column(VECTOR(vector_dim), nullable=True)

    @classmethod
    def vector_search(cls, session: Session, qvec: Sequence[float], *, k: int = 20, distance: Distance = "cosine", where=None):
        col = cls.embedding
        if distance == "cosine":
            order = col.cosine_distance(qvec)
        elif distance == "l2":
            order = col.l2_distance(qvec)
        else:
            order = -col.inner_product(qvec)

        stmt = select(cls)
        if where is not None:
            stmt = stmt.where(where)
        stmt = stmt.order_by(order).limit(int(k))
        return session.execute(stmt).scalars().all()
