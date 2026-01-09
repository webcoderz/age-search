from sqlalchemy import create_engine
from age_search.migrations import install_all, InstallSpec
from models.doc import Doc

DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"

engine = create_engine(DB_URL)

install_all(
    engine,
    models=[Doc],
    spec=InstallSpec(
        graph_name="knowledge_graph",
        enable_bm25=True,        # requires CREATE EXTENSION pg_search
        enable_fts=True,
        vector_index="hnsw",     # or "ivfflat" for huge corpora
        analyze_after=True,
    ),
)

print("Database initialized successfully.")
