from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from sqlalchemy.orm import Session
from .cypher import cypher_json

@dataclass
class CypherQuery:
    session: Session
    match: str
    return_expr: str
    params: dict[str, Any] = field(default_factory=dict)
    where_clauses: list[str] = field(default_factory=list)
    limit_n: Optional[int] = None
    graph_name: Optional[str] = None

    def where(self, clause: str, **params: Any) -> "CypherQuery":
        self.where_clauses.append(clause)
        self.params.update(params)
        return self

    def limit(self, n: int) -> "CypherQuery":
        self.limit_n = int(n)
        return self

    def _compile(self) -> str:
        parts = [self.match.strip()]
        if self.where_clauses:
            parts.append("WHERE " + " AND ".join(f"({c})" for c in self.where_clauses))
        parts.append(f"RETURN {self.return_expr} AS row")
        if self.limit_n is not None:
            parts.append(f"LIMIT {self.limit_n}")
        return "\n".join(parts)

    def all(self) -> list[Any]:
        return cypher_json(self.session, self._compile(), params=self.params, graph_name=self.graph_name)

    def first(self) -> Optional[Any]:
        self.limit(1)
        rows = self.all()
        return rows[0] if rows else None
