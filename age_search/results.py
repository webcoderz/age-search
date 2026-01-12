from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class SearchResult(Generic[T]):
    id: int
    obj: Optional[T] = None

    # Lexical
    bm25_score: Optional[float] = None
    fts_rank: Optional[float] = None
    snippet: Optional[str] = None
    lexical_rank: Optional[int] = None

    # Semantic
    vector_distance: Optional[float] = None
    semantic_rank: Optional[int] = None

    # Fusion
    rrf_score: Optional[float] = None
