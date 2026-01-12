from __future__ import annotations
import json
from typing import Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from .config import AGEGraphConfig

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

    # NOTE: avoid `:params::json` here; SQLAlchemy's text() parser may not
    # recognize bindparams when followed by `::type`. Use CAST(...) instead.
    sql = text(
        f"""
        SELECT agtype_to_json({returns_alias}) AS {returns_alias}
        FROM cypher(:graph, :cypher, CAST(:params AS json)) AS ({returns_alias} agtype);
        """
    )

    rows = session.execute(
        sql,
        {"graph": cfg.graph_name, "cypher": cypher, "params": json.dumps(params)},
    ).all()

    out: list[Any] = []
    for (val,) in rows:
        # psycopg may return json already decoded; if not, keep as-is.
        out.append(val)
    return out
