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