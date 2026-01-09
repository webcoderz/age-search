from __future__ import annotations
from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.orm import Session

@dataclass(frozen=True)
class Capabilities:
    has_age: bool
    has_vector: bool
    has_pg_search: bool

def _has_ext(session: Session, name: str) -> bool:
    return session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = :n"),
        {"n": name},
    ).first() is not None

def detect_capabilities(session: Session) -> Capabilities:
    return Capabilities(
        has_age=_has_ext(session, "age"),
        has_vector=_has_ext(session, "vector"),
        has_pg_search=_has_ext(session, "pg_search"),
    )
