"""Shared, dependency-free evaluation helpers.

Average precision (AP) instead of a single precision/recall at 0.5 is
the right metric here: with real conjunctions this rare (~2% positive),
one threshold tells you almost nothing -- AP summarizes performance
across all thresholds, which is also what the project's own metrics
table (forecasting: AP/AUROC) specifies.
"""

from __future__ import annotations


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Standard AP: sort by predicted score descending, walk down the
    ranking, average precision at each point a true positive appears."""
    if sum(labels) == 0:
        return 0.0

    ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    precisions = []
    for i, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return sum(precisions) / sum(labels)


def precision_recall_accuracy(preds: list[int], labels: list[int]) -> dict:
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    correct = sum(p == l for p, l in zip(preds, labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = correct / len(labels) if labels else 0.0
    return {"precision": precision, "recall": recall, "accuracy": accuracy}
def expected_calibration_error(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Bins predictions by confidence, compares average predicted
    probability to observed positive frequency within each bin --
    a large gap means the model's stated confidence doesn't match
    reality, regardless of how accurate it is on average.

    This is the actual test of whether an uncertainty head is doing
    something real: a model can have perfect AP while being badly
    miscalibrated (e.g. always predicting 0.9 when it should say 0.3),
    and a plain accuracy/AP number would never reveal that.

    Returns
    -------
    float
        Weighted average |predicted_confidence - observed_frequency|
        across bins, weighted by bin population. 0.0 = perfectly
        calibrated; empty input returns 0.0 (nothing to miscalibrate).
    """
    if not scores:
        return 0.0

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total_error = 0.0
    n = len(scores)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = [(s, l) for s, l in zip(scores, labels) if lo <= s < hi or (hi == 1.0 and s == 1.0)]
        if not in_bin:
            continue
        bin_scores = [s for s, _ in in_bin]
        bin_labels = [l for _, l in in_bin]
        avg_confidence = sum(bin_scores) / len(bin_scores)
        observed_frequency = sum(bin_labels) / len(bin_labels)
        total_error += (len(in_bin) / n) * abs(avg_confidence - observed_frequency)

    return total_error