from __future__ import annotations

from typing import Any, Sequence, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from .results import SearchResult

T = TypeVar("T")


def _rrf_scores(ranked_ids: list[list[int]], *, k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ids in ranked_ids:
        for r, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + r)
    return scores


def hybrid_search_results(
    session: Session,
    model: Type[T],
    *,
    query_text: str,
    query_vec: Sequence[float],
    k_lex: int = 50,
    k_vec: int = 50,
    limit: int = 20,
    prefer_bm25: bool = True,
    rrf_k: int = 60,
    fetch_objects: bool = True,
) -> list[SearchResult[T]]:
    # ---------- lexical ----------
    lex_ids: list[int] = []
    bm25_scores: dict[int, float] = {}
    snippets: dict[int, str] = {}
    fts_ranks: dict[int, float] = {}

    if prefer_bm25 and hasattr(model, "bm25_search"):
        rows = model.bm25_search(session, query_text, k=k_lex, with_snippet=True)  # type: ignore
        # rows: (id, score, snippet?)
        for i, row in enumerate(rows, start=1):
            _id = int(row[0])
            lex_ids.append(_id)
            bm25_scores[_id] = float(row[1]) if row[1] is not None else None  # type: ignore[assignment]
            if len(row) >= 3 and row[2] is not None:
                snippets[_id] = str(row[2])
    elif hasattr(model, "fts_search"):
        # fts_search returns objects; rank not exposed in our earlier mixin
        objs = model.fts_search(session, query_text, k=k_lex)  # type: ignore
        lex_ids = [int(o.id) for o in objs]

    # ---------- semantic ----------
    vec_objs = model.vector_search(session, query_vec, k=k_vec, distance="cosine")  # type: ignore
    vec_ids = [int(o.id) for o in vec_objs]

    # ---------- fuse ----------
    rrf = _rrf_scores([lex_ids, vec_ids], k=rrf_k)
    fused = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:limit]

    # ---------- hydrate ----------
    obj_map: dict[int, T] = {}
    if fetch_objects and fused:
        objs = session.execute(select(model).where(model.id.in_(fused))).scalars().all()  # type: ignore
        obj_map = {int(o.id): o for o in objs}  # type: ignore

    # ranks
    lex_rank = {i: r for r, i in enumerate(lex_ids, start=1)}
    sem_rank = {i: r for r, i in enumerate(vec_ids, start=1)}

    out: list[SearchResult[T]] = []
    for _id in fused:
        out.append(
            SearchResult(
                id=_id,
                obj=obj_map.get(_id),
                bm25_score=bm25_scores.get(_id),
                fts_rank=fts_ranks.get(_id),
                snippet=snippets.get(_id),
                lexical_rank=lex_rank.get(_id),
                semantic_rank=sem_rank.get(_id),
                rrf_score=rrf.get(_id),
            )
        )
    return out
