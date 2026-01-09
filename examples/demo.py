from sqlalchemy.orm import sessionmaker, Mapped, mapped_column
from sqlalchemy import Integer, Text, text

from agegraph_search import (
    Base, create_engine_all_in_one,
    GraphNodeMixin, VectorMixin, FTSSearchMixin, BM25SearchMixin,
    GraphRelationship, hybrid_search, graph_expand_ids
)

DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"

class Doc(Base, GraphNodeMixin, VectorMixin, FTSSearchMixin, BM25SearchMixin):
    __tablename__ = "docs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text)

    related = GraphRelationship("RELATED_TO", target_label="Doc")

def main():
    engine = create_engine_all_in_one(DB_URL, graph_name="knowledge_graph")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as s:
        # extensions + graph (normally migrations)
        s.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
        s.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # optional:
        # s.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search;"))
        s.execute(text("""
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'knowledge_graph') THEN
                PERFORM create_graph('knowledge_graph');
              END IF;
            END$$;
        """))
        s.commit()

        d1 = Doc(content="hello world shoes", embedding=[0.1] * Doc.vector_dim)
        d2 = Doc(content="running shoes and graph search", embedding=[0.11] * Doc.vector_dim)
        s.add_all([d1, d2])
        s.commit()

        d1.graph_upsert(s)
        d2.graph_upsert(s)
        d1.related.add(s, d2)
        s.commit()

        # hybrid (BM25 if installed, else FTS fallback if you flip prefer_bm25=False)
        hits = hybrid_search(s, Doc, query_text="running shoes", query_vec=d1.embedding, prefer_bm25=True)
        print("hybrid ids:", [h.id for h in hits])

        # expand in graph
        expanded_ids = graph_expand_ids(s, graph_name="knowledge_graph", label="Doc",
                                       seed_ids=[h.id for h in hits], edge="RELATED_TO", hops=2)
        print("expanded ids:", expanded_ids)

if __name__ == "__main__":
    main()
