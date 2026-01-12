from __future__ import annotations

from typing import Sequence, Type, TypeVar

from sqlalchemy import Table
from sqlalchemy.orm import Session

from .hybrid_graph import hybrid_search_results_constrained
from .results import SearchResult
from .taxonomy import descendant_label_ids, doc_ids_for_labels

T = TypeVar("T")


def hybrid_search_results_in_label_subtree_relational(
    session: Session,
    model: Type[T],
    *,
    root_label_id: int,
    doc_labels: Table,
    query_text: str,
    query_vec: Sequence[float],
    include_self: bool = True,
    k_lex: int = 50,
    k_vec: int = 50,
    limit: int = 20,
    prefer_bm25: bool = True,
    rrf_k: int = 60,
    fetch_objects: bool = True,
) -> list[SearchResult[T]]:
    """
    Relational-only label-subtree constrained hybrid search:
      1) expand label subtree via recursive CTE (Label.parent_id)
      2) fetch allowed doc ids via the association table
      3) run constrained hybrid search
    """
    label_ids = descendant_label_ids(session, root_label_id=root_label_id, include_self=include_self)
    allowed_doc_ids = doc_ids_for_labels(session, doc_labels=doc_labels, label_ids=label_ids)
    return hybrid_search_results_constrained(
        session,
        model,
        query_text=query_text,
        query_vec=query_vec,
        allowed_ids=allowed_doc_ids,
        k_lex=k_lex,
        k_vec=k_vec,
        limit=limit,
        prefer_bm25=prefer_bm25,
        rrf_k=rrf_k,
        fetch_objects=fetch_objects,
    )

