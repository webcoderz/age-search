from __future__ import annotations

from age_search.eval import (
    EvalCase,
    dcg_at_k,
    evaluate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_basic_metrics():
    pred = [1, 2, 3, 4]
    rel = {2, 4}

    assert precision_at_k(pred, rel, 2) == 0.5
    assert recall_at_k(pred, rel, 2) == 0.5
    assert mrr(pred, rel) == 0.5  # first relevant is at rank 2
    assert dcg_at_k(pred, rel, 4) > 0.0
    assert 0.0 <= ndcg_at_k(pred, rel, 4) <= 1.0


def test_evaluate_aggregates():
    cases = [
        EvalCase(name="a", relevant_ids={1}),
        EvalCase(name="b", relevant_ids={2}),
    ]

    def search(c: EvalCase):
        return [1, 2, 3] if c.name == "a" else [2, 1, 3]

    rep = evaluate(cases, search=search, k=2, benchmark=False)
    assert rep.n == 2
    assert rep.precision_at_10 >= 0.0
    assert rep.recall_at_10 >= 0.0
    assert rep.mrr >= 0.0
    assert rep.ndcg_at_10 >= 0.0
    assert rep.p50_ms is None

