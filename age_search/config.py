from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AGEGraphConfig:
    graph_name: str = "knowledge_graph"
    # Search path must include ag_catalog + schema containing pgvector + pg_search (often public)
    search_path: str = 'ag_catalog, public'
