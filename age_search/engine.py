from __future__ import annotations

from typing import Any, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from pgvector.psycopg import register_vector

from .config import AGEGraphConfig

def create_engine_all_in_one(
    url: str,
    *,
    graph_name: str = "knowledge_graph",
    search_path: str = "ag_catalog, public",
    echo: bool = False,
    pool_pre_ping: bool = True,
    connect_args: Optional[dict[str, Any]] = None,
) -> Engine:
    """
    Engine that is safe with pooling:
      - registers pgvector adapters on connect
      - runs LOAD 'age' + SET search_path on every checkout
    """
    cfg = AGEGraphConfig(graph_name=graph_name, search_path=search_path)

    engine = create_engine(
        url,
        echo=echo,
        pool_pre_ping=pool_pre_ping,
        connect_args=connect_args or {},
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # noqa: ANN001
        try:
            register_vector(dbapi_conn)
        except Exception:
            # Adapter registration shouldn't break startup; server-side errors will surface on query.
            pass

    @event.listens_for(engine, "checkout")
    def _on_checkout(dbapi_conn, _record, _proxy):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        try:
            cur.execute("LOAD 'age';")
            cur.execute(f"SET search_path = {cfg.search_path};")
        finally:
            cur.close()

    engine.info["agegraph_cfg"] = cfg
    return engine
