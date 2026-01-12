from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import age_search.hybrid as hybrid_mod


def test_graph_expand_ids_builds_query_and_coerces_ints(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with Session(engine) as session:
        called = {}

        def fake_cypher_json(s, cy, *, params, graph_name):  # noqa: ANN001
            called["session"] = s
            called["cy"] = cy
            called["params"] = params
            called["graph_name"] = graph_name
            return [3, "4", None]

        monkeypatch.setattr(hybrid_mod, "cypher_json", fake_cypher_json)

        out = hybrid_mod.graph_expand_ids(
            session,
            graph_name="knowledge_graph",
            label="Doc",
            seed_ids=[1, 2],
            edge="RELATED_TO",
            hops=2,
            limit=10,
        )

        assert out == [3, 4]
        assert called["graph_name"] == "knowledge_graph"
        assert called["params"] == {"ids": [1, 2]}
        assert "MATCH (n:Doc)" in called["cy"]
        assert "RELATED_TO*1..2" in called["cy"]
        assert "LIMIT 10" in called["cy"]

