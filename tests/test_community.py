from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import age_search.community as comm


def test_connected_components_union_find():
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (4, 5)]
    out = comm.connected_components(nodes, edges)
    assert out == [[1, 2, 3], [4, 5]]


def test_graph_edge_list_ids_parses_rows(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with Session(engine) as session:
        called = {}

        def fake_cypher_json(s, cy, *, graph_name, params=None, returns_alias="row"):  # noqa: ANN001
            called["cy"] = cy
            called["graph_name"] = graph_name
            return [[1, "2"], None, ["x"], [3, 4]]

        monkeypatch.setattr(comm, "cypher_json", fake_cypher_json)
        edges = comm.graph_edge_list_ids(
            session,
            graph_name="g",
            label="Doc",
            edge="RELATED_TO",
            direction="both",
            limit=123,
        )

        assert edges == [(1, 2), (3, 4)]
        assert called["graph_name"] == "g"
        assert "RELATED_TO" in called["cy"]
        assert "LIMIT 123" in called["cy"]

