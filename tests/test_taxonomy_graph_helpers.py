from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import age_search.taxonomy as tax


def test_graph_descendant_label_ids_builds_query_and_dedupes(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with Session(engine) as session:
        called = {}

        def fake_cypher_json(s, cy, *, params, graph_name):  # noqa: ANN001
            called["cy"] = cy
            called["params"] = params
            called["graph_name"] = graph_name
            return [2, "3", None, 2]

        monkeypatch.setattr(tax, "cypher_json", fake_cypher_json)

        out = tax.graph_descendant_label_ids(
            session,
            graph_name="g",
            root_label_id=1,
            max_hops=7,
            include_self=True,
            limit=999,
        )

        assert out == [2, 3, 1]
        assert called["graph_name"] == "g"
        assert called["params"] == {"root": 1}
        assert "PARENT_OF*1..7" in called["cy"]
        assert "LIMIT 999" in called["cy"]

