"""Tests for evaluation metrics; fixtures are synthetic and not a benchmark."""

import pytest

from retrieval_metrics import metrics_at_k


def test_metrics_at_k_for_known_ranking():
    result = metrics_at_k(
        ["relevant-a.jpg", "unrelated.jpg", "relevant-b.jpg", "relevant-c.jpg"],
        {"relevant-a.jpg", "relevant-b.jpg", "relevant-c.jpg"},
        3,
    )

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["average_precision"] == pytest.approx(5 / 9)
    assert result["reciprocal_rank"] == 1.0


def test_metrics_at_k_handles_no_hits_and_short_rankings():
    result = metrics_at_k(["unrelated.jpg"], {"relevant.jpg"}, 5)

    assert result == {
        "precision": 0.0,
        "recall": 0.0,
        "average_precision": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_metrics_at_k_rejects_invalid_k():
    with pytest.raises(ValueError, match="positive integer"):
        metrics_at_k([], {"relevant.jpg"}, 0)
