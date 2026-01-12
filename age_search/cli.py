from __future__ import annotations

import os
import argparse

from sqlalchemy import create_engine, text

from .migrations import install_all, InstallSpec


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise SystemExit(f"Missing env var: {name}")
    return v


def cmd_doctor(args: argparse.Namespace) -> int:
    url = args.url or _env("DATABASE_URL")
    engine = create_engine(url)
    with engine.begin() as conn:
        exts = conn.execute(text("SELECT extname FROM pg_extension")).all()
        extset = {r[0] for r in exts}
    print("Extensions:", ", ".join(sorted(extset)))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    url = args.url or _env("DATABASE_URL")
    # This uses plain create_engine because init should not require LOAD 'age'
    engine = create_engine(url)

    # user imports their models in init_db script typically; here we just do extensions + graph
    with engine.begin() as conn:
        # minimal init for extensions/graph
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        if args.bm25:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search;"))
        conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = :g) THEN
            PERFORM create_graph(:g);
          END IF;
        END$$;
        """), {"g": args.graph_name})

    print("Initialized extensions + graph. Next run your app-specific init_db to create tables + indexes.")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """
    App-specific index creation still needs model imports.
    We support a simple pattern: user provides python import path to a list named MODELS.
    """
    url = args.url or _env("DATABASE_URL")
    module_path = args.models_module

    # dynamic import
    mod = __import__(module_path, fromlist=["MODELS"])
    models = getattr(mod, "MODELS", None)
    if not models:
        raise SystemExit(f"{module_path} must define MODELS = [Model1, Model2, ...]")

    engine = create_engine(url)
    install_all(
        engine,
        models=models,
        spec=InstallSpec(
            graph_name=args.graph_name,
            enable_bm25=args.bm25,
            enable_fts=not args.no_fts,
            vector_index=args.vector_index,
            analyze_after=not args.no_analyze,
        ),
    )
    print("Indexes installed.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="agegraph")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("--url", help="DATABASE_URL")
    p_doc.set_defaults(func=cmd_doctor)

    p_init = sub.add_parser("init")
    p_init.add_argument("--url", help="DATABASE_URL")
    p_init.add_argument("--graph-name", default="knowledge_graph")
    p_init.add_argument("--bm25", action="store_true")
    p_init.add_argument("--vector-index", choices=["hnsw", "ivfflat", "none"], default="hnsw")
    p_init.add_argument("--no-fts", action="store_true")
    p_init.add_argument("--no-analyze", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_idx = sub.add_parser("index")
    p_idx.add_argument("--url", help="DATABASE_URL")
    p_idx.add_argument("--graph-name", default="knowledge_graph")
    p_idx.add_argument("--bm25", action="store_true")
    p_idx.add_argument("--vector-index", choices=["hnsw", "ivfflat", "none"], default="hnsw")
    p_idx.add_argument("--no-fts", action="store_true")
    p_idx.add_argument("--no-analyze", action="store_true")
    p_idx.add_argument("--models-module", required=True, help="Python module path exporting MODELS=[...]")
    p_idx.set_defaults(func=cmd_index)

    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)
