
# `age_search`
[![CI](https://github.com/webcoderz/age-search/actions/workflows/ci.yml/badge.svg)](https://github.com/webcoderz/age-search/actions/workflows/ci.yml)

A **unified SQLAlchemy extension** that combines:

* **Apache AGE** → graph traversal (Cypher)
* **pgvector** → semantic vector search (cosine, HNSW / IVFFLAT)
* **Postgres FTS** → built-in full-text search
* **BM25 (pg_search / ParadeDB)** → high-quality lexical ranking
* **Hybrid search** → lexical + semantic fusion
* **Graph-constrained search** → search + expand via graph topology

All inside **one Postgres database**, **one SQLAlchemy session**, **one transaction model**.

This package does **not** try to pretend graphs are tables.
Instead, it gives you **clean primitives that compose**.

---

## Why this exists

Most systems need all of the following at once:

* semantic similarity (embeddings)
* keyword relevance (BM25 / FTS)
* graph structure (relationships, hops, communities)
* transactional consistency
* deployability (migrations, pooling, ORM)

Postgres already supports all of this — but the integration story is painful.

This package provides:

* safe engine/session setup
* sane defaults
* index + migration helpers
* ORM-friendly APIs
* zero magic that fights SQLAlchemy internals

---

## Core design principles

1. **Relational tables own the data**
2. **AGE owns topology**
3. **Vectors stay in tables**
4. **Graph nodes reference table primary keys**
5. **Hybrid search is explicit and debuggable**
6. **Everything works under normal SQLAlchemy pooling**

---

## Installation

```bash
pip install -e .
```

Dependencies:

* Python ≥ 3.10
* SQLAlchemy ≥ 2.0
* psycopg3
* pgvector
* Apache AGE installed server-side
* Optional: `pg_search` (BM25)

---

## Engine setup (IMPORTANT)

AGE requires **per-connection initialization**.

Always create your engine using:

```python
from age_search import create_engine_all_in_one

engine = create_engine_all_in_one(
    DATABASE_URL,
    graph_name="knowledge_graph",
)
```

This automatically:

* registers pgvector adapters
* runs `LOAD 'age'`
* sets `search_path = ag_catalog, public`
* is safe under connection pooling

---

## Canonical `Doc` model

This is the reference model used throughout the examples.

```python
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Text

from age_search import (
    Base,
    GraphNodeMixin,
    VectorMixin,
    FTSSearchMixin,
    BM25SearchMixin,
    GraphRelationship,
)

class Doc(
    Base,
    GraphNodeMixin,
    VectorMixin,
    FTSSearchMixin,
    BM25SearchMixin,
):
    __tablename__ = "docs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Graph configuration
    graph_label = "Doc"
    graph_id_field = "id"
    vertex_property_key = "id"

    # Vector configuration
    vector_dim = 1536   # cosine by default

    # FTS
    fts_config = "english"

    # BM25
    bm25_key_field = "id"
    bm25_default_field = "content"

    # Graph relationships
    related = GraphRelationship("RELATED_TO", target_label="Doc")
```

---

## Database initialization (one-time)

Create **extensions, graph, and indexes** with a single call.

### `init_db.py`

```python
from sqlalchemy import create_engine
from age_search.migrations import install_all, InstallSpec
from models.doc import Doc

engine = create_engine(DATABASE_URL)

install_all(
    engine,
    models=[Doc],
    spec=InstallSpec(
        graph_name="knowledge_graph",
        enable_fts=True,
        enable_bm25=True,        # requires pg_search extension
        vector_index="hnsw",     # or "ivfflat"
        analyze_after=True,
    ),
)

print("Database initialized.")
```

This creates:

* `age`, `vector`, (optional) `pg_search` extensions
* AGE graph
* FTS GIN index
* BM25 index
* pgvector cosine index (HNSW or IVFFLAT)
* runs `ANALYZE`

---

## Optional: auto-sync graph vertices

Keep AGE graph nodes in sync with ORM rows automatically.

```python
from age_search.hooks import install_graph_sync
from models.doc import Doc

install_graph_sync(Doc)
```

Behavior:

* insert/update → `MERGE (Doc {id})`
* delete → `DETACH DELETE`

Safe under normal ORM usage.

---

## Writing data

```python
doc1 = Doc(content="Graph neural networks for fraud detection", embedding=vec1)
doc2 = Doc(content="Vector databases and hybrid search", embedding=vec2)

session.add_all([doc1, doc2])
session.commit()
```

If graph sync is enabled, vertices are created automatically.

---

## Graph operations (AGE)

### Create relationships

```python
doc1.related.add(session, doc2)
session.commit()
```

### Traverse neighbors

```python
neighbors = doc1.related(session).limit(10).all()
```

Returns **JSON-decoded AGE nodes**, not ORM objects (by design).

---

## Vector search (pgvector)

Cosine similarity is the default.

```python
hits = Doc.vector_search(
    session,
    query_vec,
    k=20,
    distance="cosine",
)
```

Uses:

* `embedding <-> query_vec`
* HNSW or IVFFLAT index automatically

---

## Full-text search (Postgres FTS)

```python
hits = Doc.fts_search(
    session,
    "graph neural networks",
    k=20,
)
```

Uses:

* `tsvector`
* `websearch_to_tsquery`
* GIN index

---

## BM25 search (pg_search / ParadeDB)

```python
rows = Doc.bm25_search(
    session,
    "graph neural networks",
    k=20,
    with_snippet=True,
)
```

Each row contains:

* `id`
* BM25 score
* optional snippet

To return ORM objects:

```python
docs = Doc.bm25_search_objects(session, "graph neural networks")
```

---

## Hybrid search (lexical + semantic)

### Simple hybrid (RRF)

```python
from age_search import hybrid_search

results = hybrid_search(
    session,
    Doc,
    query_text="graph neural networks",
    query_vec=query_embedding,
    prefer_bm25=True,
)
```

This:

1. runs BM25 (or FTS fallback)
2. runs vector search
3. fuses ranks via **Reciprocal Rank Fusion**
4. returns ORM objects in fused order

---

## Typed hybrid results (scores + metadata)

```python
from age_search.hybrid2 import hybrid_search_results

results = hybrid_search_results(
    session,
    Doc,
    query_text="graph neural networks",
    query_vec=query_embedding,
)

for r in results:
    print(
        r.id,
        r.rrf_score,
        r.bm25_score,
        r.semantic_rank,
        r.snippet,
    )
```

This is what you want for:

* debugging
* evals
* ranking analysis
* explainability

---

## Graph-constrained search

### Expand after search

```python
from age_search import graph_expand_ids

seed_ids = [r.id for r in results]

expanded_ids = graph_expand_ids(
    session,
    graph_name="knowledge_graph",
    label="Doc",
    seed_ids=seed_ids,
    edge="RELATED_TO",
    hops=2,
)
```

You can then:

* re-rank
* fetch objects
* or run another hybrid search inside this subset

---

## Hierarchical labels (taxonomy)

You typically want **two layers**:

- **Relational taxonomy tables (source of truth)**: fast filtering, constraints, auditing
- **AGE mirror** (optional): traversal/reasoning (`PARENT_OF`, `HAS_LABEL`)

### Relational taxonomy

Use the built-in `Label` model (adjacency list via `parent_id`):

```python
from age_search import Base
from age_search.taxonomy import Label
```

For a document↔label join table, create it explicitly so your doc table name can be anything:

```python
from age_search.taxonomy import make_doc_labels_table

doc_labels = make_doc_labels_table(Base.metadata, doc_table="docs")
```

To expand a subtree in pure SQL (recursive CTE):

```python
from age_search.taxonomy import descendant_label_ids

ids = descendant_label_ids(session, root_label_id=42)
```

### AGE mirror (optional)

Mirror taxonomy into AGE:

- `(:Label {id, slug, name})`
- `(:Label)-[:PARENT_OF]->(:Label)`
- `(:Doc)-[:HAS_LABEL]->(:Label)`

Then you can do **graph-constrained hybrid search in one call**:

```python
from age_search import hybrid_search_results_in_label_subtree

results = hybrid_search_results_in_label_subtree(
    session,
    Doc,
    graph_name="knowledge_graph",
    root_label_id=42,
    query_text="graph neural networks",
    query_vec=query_embedding,
)
```

---

## Weighted edges

You can attach properties (including a numeric `weight`) when creating a relationship:

```python
# adds/updates relationship properties on the AGE edge
doc1.related.add(session, doc2, weight=0.8, props={"source": "cooccur"})
session.commit()
```

---

## Community detection helpers (connected components)

For a simple baseline "community" definition, you can compute **connected components**
from an AGE edge list:

```python
from age_search.community import graph_connected_components

communities = graph_connected_components(
    session,
    graph_name="knowledge_graph",
    label="Doc",
    edge="RELATED_TO",
)
```

---

## Benchmark + eval harness

There’s a lightweight, dependency-free eval module (`age_search.eval`) with common IR metrics.
You provide `EvalCase` objects and a `search(case) -> ranked_ids` function:

```python
from age_search.eval import EvalCase, evaluate

cases = [
    EvalCase(name="q1", relevant_ids={1, 2, 3}),
]

report = evaluate(cases, search=lambda c: [1, 9, 2, 8], benchmark=True)
print(report)
```

---

## Development + release notes

- **CI** runs `ruff` + `pytest` on PRs.
- **Releases**: use GitHub Actions → workflow **Publish Python distribution to PyPI** (manual dispatch) to bump version, tag, build, and publish.

---

## Index strategies (cosine)

### HNSW (default)

* best recall/latency
* heavier build
* good general default

### IVFFLAT

* faster build
* smaller
* requires `ANALYZE`
* tune with:

```sql
SET ivfflat.probes = 10;
```

Switch via:

```python
InstallSpec(vector_index="ivfflat")
```

---

## CLI (optional)

```bash
agegraph doctor
agegraph init --bm25 --vector-index hnsw
agegraph index --models-module your_app.models
```

Useful for:

* ops
* CI
* smoke tests

---

## Mental model summary

| Layer        | Technology | Role                |
| ------------ | ---------- | ------------------- |
| Tables       | SQLAlchemy | source of truth     |
| Vectors      | pgvector   | semantic similarity |
| Lexical      | FTS / BM25 | keyword relevance   |
| Graph        | Apache AGE | topology            |
| Fusion       | RRF        | hybrid ranking      |
| Transactions | Postgres   | consistency         |

Nothing is hidden. Everything composes.

---

## What this is good for

* RAG systems
* knowledge graphs
* recommendation engines
* fraud / AML
* search + reasoning
* graph-aware retrieval
* eval pipelines

---

## What this deliberately does NOT do

* pretend graphs are tables
* auto-load graph neighbors via lazy ORM relationships
* hide Cypher behind magic joins
* force a specific embedding model
* lock you into one search strategy

---
