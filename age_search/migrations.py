from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Type

from sqlalchemy import Engine, text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.engine import Connection

VectorIndexKind = Literal["hnsw", "ivfflat", "none"]


@dataclass(frozen=True)
class InstallSpec:
    graph_name: str = "knowledge_graph"
    search_path: str = "ag_catalog, public"
    vector_metric_opclass: str = "vector_cosine_ops"   # cosine preset
    vector_index: VectorIndexKind = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    ivfflat_lists: int = 100
    enable_bm25: bool = False
    enable_fts: bool = True
    analyze_after: bool = True

def ensure_extensions(conn: Connection, *, age: bool = True, vector: bool = True, pg_search: bool = False):
    if age:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
    if vector:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    if pg_search:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search;"))

def ensure_graph(conn: Connection, graph_name: str):
    # Create graph if missing
    conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = :g) THEN
            PERFORM create_graph(:g);
          END IF;
        END$$;
    """), {"g": graph_name})

def ensure_fts_index(conn, table: str, tsv_column: str, index_name: str):
    conn.execute(text(f"""
      CREATE INDEX IF NOT EXISTS {index_name}
      ON {table} USING GIN ({tsv_column});
    """))

def ensure_bm25_index(conn, table: str, index_name: str, *, key_field: str, fields: list[str]):
    cols = ", ".join([key_field] + fields)
    conn.execute(text(f"""
      CREATE INDEX IF NOT EXISTS {index_name}
      ON {table}
      USING bm25 ({cols})
      WITH (key_field = '{key_field}');
    """))


def ensure_hnsw_index(conn, table: str, column: str, index_name: str, *, opclass: str, m: int, ef_construction: int):
    conn.execute(text(f"""
      CREATE INDEX IF NOT EXISTS {index_name}
      ON {table}
      USING hnsw ({column} {opclass})
      WITH (m = {int(m)}, ef_construction = {int(ef_construction)});
    """))


def ensure_ivfflat_index(conn, table: str, column: str, index_name: str, *, opclass: str, lists: int):
    conn.execute(text(f"""
      CREATE INDEX IF NOT EXISTS {index_name}
      ON {table}
      USING ivfflat ({column} {opclass})
      WITH (lists = {int(lists)});
    """))


def analyze_table(conn, table: str):
    conn.execute(text(f"ANALYZE {table};"))


def set_session_tuning(conn, *, ivfflat_probes: Optional[int] = None, hnsw_ef_search: Optional[int] = None):
    # call per-session (or per transaction) as desired
    if ivfflat_probes is not None:
        conn.execute(text("SET ivfflat.probes = :p"), {"p": int(ivfflat_probes)})
    if hnsw_ef_search is not None:
        conn.execute(text("SET hnsw.ef_search = :e"), {"e": int(hnsw_ef_search)})


def install_all(
    engine: Engine,
    *,
    models: Iterable[Type[DeclarativeBase]],
    spec: InstallSpec = InstallSpec(),
):
    """
    One-shot installer:
      - creates extensions
      - creates AGE graph
      - creates indexes for each model if it has the expected mixin attributes
    """
    with engine.begin() as conn:
        ensure_extensions(conn, age=True, vector=True, pg_search=spec.enable_bm25)
        ensure_graph(conn, spec.graph_name)

        for model in models:
            table = model.__tablename__

            # FTS: if model has content_tsv attr
            if spec.enable_fts and hasattr(model, "content_tsv"):
                ensure_fts_index(conn, table, "content_tsv", f"ix_{table}_fts")

            # BM25: if enabled and model has bm25 fields
            if spec.enable_bm25 and hasattr(model, "bm25_key_field"):
                key_field = getattr(model, "bm25_key_field", "id")
                default_field = getattr(model, "bm25_default_field", "content")
                ensure_bm25_index(
                    conn,
                    table,
                    f"ix_{table}_bm25",
                    key_field=key_field,
                    fields=[default_field],
                )

            # Vector
            if spec.vector_index != "none" and hasattr(model, "embedding"):
                col = "embedding"
                if spec.vector_index == "hnsw":
                    ensure_hnsw_index(
                        conn, table, col, f"ix_{table}_emb_hnsw",
                        opclass=spec.vector_metric_opclass,
                        m=spec.hnsw_m,
                        ef_construction=spec.hnsw_ef_construction,
                    )
                elif spec.vector_index == "ivfflat":
                    ensure_ivfflat_index(
                        conn, table, col, f"ix_{table}_emb_ivf",
                        opclass=spec.vector_metric_opclass,
                        lists=spec.ivfflat_lists,
                    )

            if spec.analyze_after:
                analyze_table(conn, table)

def ensure_age_label_property_index(conn, graph: str, label: str, prop: str, index_name: str):
    conn.execute(text(f"SELECT create_vlabel('{graph}', '{label}');"))  # if needed
    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {label} ({prop});"))
