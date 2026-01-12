from __future__ import annotations
import json
from typing import Any, Optional
import re
from sqlalchemy import text
from sqlalchemy.orm import Session
from .config import AGEGraphConfig

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_safe_ident(value: str, *, what: str) -> str:
    """
    Validate identifiers that will be interpolated into SQL text (not bound params).
    """
    if not _SAFE_IDENT.fullmatch(value):
        raise ValueError(f"Unsafe {what}: {value!r}")
    return value


def _require_safe_graph_name(graph: str) -> str:
    # AGE graph names are identifiers; keep this strict for SQL safety.
    return _require_safe_ident(graph, what="graph name")

def _dollar_quote(value: str, *, tag_base: str = "age") -> str:
    """
    Return a PostgreSQL dollar-quoted literal for `value`, choosing a delimiter tag
    that doesn't occur in the value.
    """
    # Deterministic tag selection: try a small set, then fall back to numbered tags.
    candidates = [tag_base, "cypher", "q", "p", f"{tag_base}{len(value)}"]
    for tag in candidates:
        delim = f"${tag}$"
        if delim not in value:
            return f"{delim}{value}{delim}"
    i = 0
    while True:
        tag = f"{tag_base}{i}"
        delim = f"${tag}$"
        if delim not in value:
            return f"{delim}{value}{delim}"
        i += 1


def _cfg(session: Session, graph_name: Optional[str] = None) -> AGEGraphConfig:
    bind = session.get_bind()
    cfg = None
    if bind is not None:
        # Connection objects have `.info`; Engine often does not.
        info = getattr(bind, "info", None)
        if isinstance(info, dict):
            cfg = info.get("agegraph_cfg")
        if cfg is None:
            cfg = getattr(bind, "agegraph_cfg", None)
    if cfg is None:
        cfg = AGEGraphConfig(graph_name=graph_name or "knowledge_graph")
    if graph_name:
        cfg = AGEGraphConfig(graph_name=graph_name, search_path=cfg.search_path)
    return cfg

def cypher_json(
    session: Session,
    cypher: str,
    *,
    params: Optional[dict[str, Any]] = None,
    graph_name: Optional[str] = None,
    returns_alias: str = "row",
) -> list[Any]:
    """
    Executes cypher() and returns JSON-friendly Python objects using agtype_to_json().
    """
    cfg = _cfg(session, graph_name)
    params = params or {}

    # Apache AGE cypher() is picky:
    # - graph name must be a literal constant (not a bind param)
    # - query must be a dollar-quoted string constant (not a bind param)
    # We'll inline graph/query/params safely:
    # - graph name + alias are strict identifiers
    # - query + params are dollar-quoted string literals (delimiter chosen to avoid collisions)
    graph_lit = _require_safe_graph_name(cfg.graph_name)
    alias = _require_safe_ident(returns_alias, what="returns alias")

    cypher_lit = _dollar_quote(cypher, tag_base="cy")
    params_lit = _dollar_quote(json.dumps(params), tag_base="jp")

    sql = text(
        f"""
        SELECT agtype_to_json({alias}) AS {alias}
        FROM cypher(
          '{graph_lit}'::name,
          {cypher_lit},
          {params_lit}
        ) AS ({alias} agtype);
        """
    )

    rows = session.execute(
        sql,
        {},
    ).all()

    out: list[Any] = []
    for (val,) in rows:
        # psycopg may return json already decoded; if not, keep as-is.
        out.append(val)
    return out
