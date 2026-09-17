"""Train on one scenario, evaluate forecasting performance on a held-out
scenario, and compare against a naive persistence baseline.

Uses crossing orbital planes (mirrored inclinations) so pairs genuinely
converge/diverge over time -- same-plane radial stacking makes future
separation trivially predictable from current separation, which made
earlier versions of this script meaningless (both models hit 1.0/1.0).

Note: real orbital geometry makes conjunctions RARE -- expect a small,
imbalanced positive count. That's physically honest, not a scenario bug.

Run: python scripts/evaluate_lstm_baseline.py
"""

import math
import random
from datetime import datetime, timedelta

import torch

from constellai.models.dataset import build_forecast_examples
from constellai.models.lstm_baseline import PairRiskLSTM, train_one_epoch
from constellai.orbital_mechanics.synthetic import make_circular_satellite

STEP = timedelta(minutes=2)
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
HORIZON_END = OBS_END + timedelta(hours=1, minutes=30)
THRESHOLD_KM = 300.0


def make_scenario(seed: int, id_offset: int, n: int = 30):
    """Mirrored inclinations -> orbital planes actually cross, giving
    real converge/diverge dynamics instead of static radial offsets."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        sign = 1 if i % 2 == 0 else -1
        records.append(make_circular_satellite(
            satellite_id=id_offset + i,
            altitude_km=500.0 + rng.uniform(-2, 2),
            inclination_rad=sign * math.radians(45 + rng.uniform(-3, 3)),
            raan_rad=rng.uniform(0, 0.3),
            mean_anomaly_rad=rng.uniform(0, 2 * math.pi),
        ))
    return records


def naive_persistence_metrics(examples):
    preds = [int(ex.features[-1, -1] < THRESHOLD_KM) for ex in examples]
    return precision_recall_accuracy(preds, [ex.label for ex in examples])


def precision_recall_accuracy(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    correct = sum(p == l for p, l in zip(preds, labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = correct / len(labels) if labels else 0.0
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


def main():
    train_examples = build_forecast_examples(
        make_scenario(seed=1, id_offset=0), OBS_START, OBS_END, HORIZON_END, STEP, THRESHOLD_KM
    )
    test_examples = build_forecast_examples(
        make_scenario(seed=2, id_offset=1000), OBS_START, OBS_END, HORIZON_END, STEP, THRESHOLD_KM
    )
    print(f"train: {len(train_examples)} examples, {sum(e.label for e in train_examples)} positive")
    print(f"test:  {len(test_examples)} examples, {sum(e.label for e in test_examples)} positive")

    model = PairRiskLSTM()
    # Imbalanced positives -> weight the loss so rare conjunctions aren't
    # just ignored by the model.
    n_pos = max(sum(e.label for e in train_examples), 1)
    n_neg = len(train_examples) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(15):
        total = 0.0
        for ex in train_examples:
            x = torch.from_numpy(ex.features).unsqueeze(0)
            y = torch.tensor([float(ex.label)])
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch}: loss={total / len(train_examples):.4f}")

    model.eval()
    with torch.no_grad():
        preds = [
            int(torch.sigmoid(model(torch.from_numpy(ex.features).unsqueeze(0))) > 0.5)
            for ex in test_examples
        ]
    lstm_metrics = precision_recall_accuracy(preds, [ex.label for ex in test_examples])
    naive_metrics = naive_persistence_metrics(test_examples)

    print("\n--- held-out scenario results ---")
    print("naive persistence baseline:", naive_metrics)
    print("LSTM forecaster:           ", lstm_metrics)


if __name__ == "__main__":
    main()