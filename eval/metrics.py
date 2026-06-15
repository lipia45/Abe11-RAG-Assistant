"""Information-retrieval metrics for evaluating the retriever.

All functions take:
    ranked: an ordered sequence of retrieved ids (best first), and
    relevant: the set/collection of ground-truth relevant ids.

They are pure (no I/O, no global state) and operate on hashable ids of any type.
Run ``python eval/metrics.py`` for a self-test on a tiny worked example.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence


def _top_k(ranked: Sequence[Hashable], k: int) -> Sequence[Hashable]:
    if k <= 0:
        raise ValueError("k must be > 0")
    return ranked[:k]


def recall_at_k(
    ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Fraction of relevant ids retrieved within the top-k.

    Returns 0.0 when there are no relevant ids.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    found = sum(1 for rid in _top_k(ranked, k) if rid in rel)
    return found / len(rel)


def precision_at_k(
    ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Fraction of the top-k retrieved ids that are relevant.

    Denominator is ``k`` (the IR convention), not ``len(ranked)``.
    """
    rel = set(relevant)
    topk = _top_k(ranked, k)
    found = sum(1 for rid in topk for _ in [0] if rid in rel)
    return found / k


def hit_rate_at_k(
    ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """1.0 if at least one relevant id appears in the top-k, else 0.0."""
    rel = set(relevant)
    return 1.0 if any(rid in rel for rid in _top_k(ranked, k)) else 0.0


def mrr(ranked: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    """Reciprocal rank of the first relevant id (1-indexed); 0.0 if none found.

    This is the per-query reciprocal rank; average it over queries to get MRR.
    """
    rel = set(relevant)
    for rank, rid in enumerate(ranked, start=1):
        if rid in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Discounted Cumulative Gain at k with binary relevance.

    Uses the standard ``rel_i / log2(i + 1)`` discount (1-indexed positions).
    """
    rel = set(relevant)
    dcg = 0.0
    for i, rid in enumerate(_top_k(ranked, k), start=1):
        if rid in rel:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    ranked: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """Normalized DCG at k with binary relevance.

    Normalizes by the ideal DCG (all relevant items ranked first). Returns 0.0
    when there are no relevant ids.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = dcg_at_k(ranked, rel, k)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _selftest() -> None:
    """Self-test on a tiny worked example with hand-computed expected values."""
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "x"}  # "x" is relevant but never retrieved

    # recall@3: retrieved relevant among top-3 = {b} -> 1 of 3 relevant.
    assert math.isclose(recall_at_k(ranked, relevant, 3), 1 / 3), recall_at_k(ranked, relevant, 3)
    # recall@5: {b, d} -> 2 of 3.
    assert math.isclose(recall_at_k(ranked, relevant, 5), 2 / 3)

    # precision@3: 1 relevant in top-3 / 3.
    assert math.isclose(precision_at_k(ranked, relevant, 3), 1 / 3)
    # precision@5: 2 relevant / 5.
    assert math.isclose(precision_at_k(ranked, relevant, 5), 2 / 5)

    # hit_rate@1: top-1 is "a" (not relevant) -> 0.
    assert hit_rate_at_k(ranked, relevant, 1) == 0.0
    # hit_rate@2: "b" relevant -> 1.
    assert hit_rate_at_k(ranked, relevant, 2) == 1.0

    # mrr: first relevant "b" at rank 2 -> 1/2.
    assert math.isclose(mrr(ranked, relevant), 0.5)

    # dcg@5: hits at positions 2 and 4 -> 1/log2(3) + 1/log2(5).
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert math.isclose(dcg_at_k(ranked, relevant, 5), expected_dcg)

    # ndcg@5: idcg = ideal ranks 1,2,3 (min(3 relevant, 5)) -> 1/log2(2)+1/log2(3)+1/log2(4).
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ndcg_at_k(ranked, relevant, 5), expected_dcg / idcg)

    # Perfect ranking gives ndcg == 1.0.
    perfect = ["b", "d", "x", "a", "c"]
    assert math.isclose(ndcg_at_k(perfect, relevant, 5), 1.0)

    # Empty relevant set is handled gracefully.
    assert recall_at_k(ranked, set(), 3) == 0.0
    assert ndcg_at_k(ranked, set(), 3) == 0.0

    print("metrics.py self-test passed.")


if __name__ == "__main__":
    _selftest()
