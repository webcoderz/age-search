from __future__ import annotations

from dataclasses import dataclass
from math import log2
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable, Optional, Sequence


def precision_at_k(pred: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = [int(x) for x in pred[: int(k)]]
    if not top:
        return 0.0
    hits = sum(1 for x in top if int(x) in relevant)
    return hits / len(top)


def recall_at_k(pred: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = [int(x) for x in pred[: int(k)]]
    hits = sum(1 for x in top if int(x) in relevant)
    return hits / len(relevant)


def mrr(pred: Sequence[int], relevant: set[int]) -> float:
    for i, x in enumerate(pred, start=1):
        if int(x) in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(pred: Sequence[int], relevant: set[int], k: int) -> float:
    top = [int(x) for x in pred[: int(k)]]
    score = 0.0
    for i, x in enumerate(top, start=1):
        rel = 1.0 if x in relevant else 0.0
        score += (2.0**rel - 1.0) / log2(i + 1.0)
    return score


def ndcg_at_k(pred: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    ideal = [1] * min(len(relevant), int(k))
    ideal_dcg = sum((2.0**1 - 1.0) / log2(i + 1.0) for i in range(1, len(ideal) + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(pred, relevant, k) / ideal_dcg


@dataclass(frozen=True)
class EvalCase:
    name: str
    relevant_ids: set[int]


@dataclass(frozen=True)
class EvalReport:
    n: int
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None


def _percentile_ms(samples: list[float], pct: float) -> Optional[float]:
    if not samples:
        return None
    xs = sorted(samples)
    i = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[max(0, min(i, len(xs) - 1))]


def evaluate(
    cases: Iterable[EvalCase],
    *,
    search: Callable[[EvalCase], Sequence[int]],
    k: int = 10,
    benchmark: bool = False,
) -> EvalReport:
    """
    Minimal eval harness:
      - user provides cases with relevant_ids
      - user provides `search(case)` returning ranked ids

    If benchmark=True, also records latency per case.
    """
    p_list: list[float] = []
    r_list: list[float] = []
    mrr_list: list[float] = []
    ndcg_list: list[float] = []
    lat_ms: list[float] = []

    n = 0
    for c in cases:
        n += 1
        t0 = perf_counter()
        pred = [int(x) for x in search(c)]
        if benchmark:
            lat_ms.append((perf_counter() - t0) * 1000.0)

        rel = {int(x) for x in c.relevant_ids}
        p_list.append(precision_at_k(pred, rel, k))
        r_list.append(recall_at_k(pred, rel, k))
        mrr_list.append(mrr(pred, rel))
        ndcg_list.append(ndcg_at_k(pred, rel, k))

    return EvalReport(
        n=n,
        precision_at_10=mean(p_list) if p_list else 0.0,
        recall_at_10=mean(r_list) if r_list else 0.0,
        mrr=mean(mrr_list) if mrr_list else 0.0,
        ndcg_at_10=mean(ndcg_list) if ndcg_list else 0.0,
        p50_ms=_percentile_ms(lat_ms, 50) if benchmark else None,
        p95_ms=_percentile_ms(lat_ms, 95) if benchmark else None,
    )

