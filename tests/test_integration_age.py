from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from age_search.cypher import cypher_json
from age_search.engine import create_engine_all_in_one

pytestmark = pytest.mark.integration


def test_age_cypher_json_smoke():
    """
    Smoke test against a real Postgres+Apache AGE instance.

    CI should provide DATABASE_URL, e.g.:
      postgresql+psycopg://postgres:postgres@localhost:5432/postgres
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set (integration test).")

    graph = os.getenv("AGE_GRAPH_NAME", "ci_graph")
    engine = create_engine_all_in_one(url, graph_name=graph)

    # Ensure extension + graph exist.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
        exists = conn.execute(
            text("SELECT 1 FROM ag_catalog.ag_graph WHERE name = :g"),
            {"g": graph},
        ).first()
        if not exists:
            # Avoid DO-block bind typing issues with psycopg; use plain SQL.
            conn.execute(
                text("SELECT create_graph(CAST(:g AS name))"),
                {"g": graph},
            )

    with Session(engine) as session:
        rows = cypher_json(session, "RETURN 1", graph_name=graph)
        assert rows == [1]

