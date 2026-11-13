"""Ranking metrics for labeled image-retrieval evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence


def metrics_at_k(
    ranked_paths: Sequence[str], relevant_paths: Iterable[str], k: int
) -> dict[str, float]:
    """Calculate Precision@K, Recall@K, AP@K, and reciprocal rank at K.

    Precision uses K as its denominator even when fewer than K results are
    available. AP@K is normalized by ``min(number_of_relevant_items, K)``.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    relevant = set(relevant_paths)
    top_paths = list(ranked_paths[:k])
    hits = [path in relevant for path in top_paths]
    hit_count = sum(hits)
    relevant_count = len(relevant)

    precision = hit_count / k
    recall = hit_count / relevant_count if relevant_count else 0.0

    precision_sum = 0.0
    cumulative_hits = 0
    first_relevant_rank = None
    for rank, is_relevant in enumerate(hits, start=1):
        if is_relevant:
            cumulative_hits += 1
            precision_sum += cumulative_hits / rank
            if first_relevant_rank is None:
                first_relevant_rank = rank

    ap_denominator = min(relevant_count, k)
    average_precision = precision_sum / ap_denominator if ap_denominator else 0.0
    reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "reciprocal_rank": reciprocal_rank,
    }
