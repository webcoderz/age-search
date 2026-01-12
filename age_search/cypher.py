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

    # Apache AGE cypher() often requires the graph name to be a *literal constant*,
    # not a bound parameter. To keep this safe, we strictly validate identifiers
    # and inline the graph name while still binding cypher text + params.
    graph_lit = _require_safe_graph_name(cfg.graph_name)
    alias = _require_safe_ident(returns_alias, what="returns alias")

    # AGE expects params as agtype. agtype accepts JSON-ish text, so we pass a JSON string
    # and cast to agtype.
    sql = text(
        f"""
        SELECT agtype_to_json({alias}) AS {alias}
        FROM cypher(
          '{graph_lit}'::name,
          CAST(:cypher AS cstring),
          CAST(:params AS agtype)
        ) AS ({alias} agtype);
        """
    )

    rows = session.execute(
        sql,
        {"cypher": cypher, "params": json.dumps(params)},
    ).all()

    out: list[Any] = []
    for (val,) in rows:
        # psycopg may return json already decoded; if not, keep as-is.
        out.append(val)
    return out
