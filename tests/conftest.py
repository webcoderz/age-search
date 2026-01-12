from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture()
def engine():
    # Pure unit tests: keep it local and dependency-free (no Postgres required).
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    return eng


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        yield s

