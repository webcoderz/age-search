from __future__ import annotations
from typing import Sequence, Type, TypeVar
from sqlalchemy import select
from sqlalchemy.orm import Session
from .cypher import cypher_json

T = TypeVar("T")

def rrf(ids_lists: list[list[int]], *, k: int = 60, limit: int = 20) -> list[int]:
    scores: dict[int, float] = {}
    for ids in ids_lists:
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]]

def hybrid_search(
    session: Session,
    model: Type[T],
    *,
    query_text: str,
    query_vec: Sequence[float],
    k_lex: int = 50,
    k_vec: int = 50,
    limit: int = 20,
    prefer_bm25: bool = True,
) -> list[T]:
    # lexical candidates
    lex_ids: list[int] = []
    if prefer_bm25 and hasattr(model, "bm25_search"):
        rows = model.bm25_search(session, query_text, k=k_lex)  # type: ignore
        lex_ids = [int(r[0]) for r in rows]
    elif hasattr(model, "fts_search"):
        lex_ids = [int(o.id) for o in model.fts_search(session, query_text, k=k_lex)]  # type: ignore

    # vector candidates
    vec_objs = model.vector_search(session, query_vec, k=k_vec, distance="cosine")  # type: ignore
    vec_ids = [int(o.id) for o in vec_objs]

    fused = rrf([lex_ids, vec_ids], limit=limit)
    if not fused:
        return []
    rows = session.execute(select(model).where(model.id.in_(fused))).scalars().all()  # type: ignore
    by_id = {int(r.id): r for r in rows}  # type: ignore
    return [by_id[i] for i in fused if i in by_id]

def graph_expand_ids(
    session: Session,
    *,
    graph_name: str,
    label: str,
    seed_ids: list[int],
    edge: str,
    hops: int = 1,
    limit: int = 500,
) -> list[int]:
    """
    Expand from seed vertex ids (stored as property id) and return neighbor ids.
    """
    cy = f"""
    MATCH (n:{label})
    WHERE n.id IN $ids
    MATCH (n)-[:{edge}*1..{int(hops)}]->(m:{label})
    RETURN DISTINCT m.id
    LIMIT {int(limit)}
    """
    rows = cypher_json(session, cy, params={"ids": seed_ids}, graph_name=graph_name)
    # rows are json scalars
    return [int(x) for x in rows if x is not None]
