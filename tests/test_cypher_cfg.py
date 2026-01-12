from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from age_search.config import AGEGraphConfig
from age_search.cypher import _cfg


def test_cfg_pulls_from_engine_info_and_allows_override_graph_name():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    setattr(engine, "agegraph_cfg", AGEGraphConfig(graph_name="g1", search_path="ag_catalog, public"))

    with Session(engine) as session:
        cfg = _cfg(session)
        assert cfg.graph_name == "g1"

        cfg2 = _cfg(session, graph_name="g2")
        assert cfg2.graph_name == "g2"
        assert cfg2.search_path == "ag_catalog, public"

