"""Train + evaluate the static-GNN baseline, same scenario design as the
LSTM baseline so results are directly comparable.

Run: python scripts/train_gnn_baseline.py
"""

import math
import random
from datetime import datetime, timedelta

import torch

from constellai.models.graph_dataset import build_graph_snapshot
from constellai.models.gnn_baseline import EdgeRiskGNN, train_one_epoch
from constellai.orbital_mechanics.synthetic import make_circular_satellite

STEP = timedelta(minutes=2)
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
HORIZON_END = OBS_END + timedelta(hours=1, minutes=30)
THRESHOLD_KM = 300.0
MARGIN_KM = 50.0


def make_scenario(seed: int, id_offset: int, n: int = 30):
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
    train_snap = build_graph_snapshot(
        make_scenario(seed=1, id_offset=0), OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM
    )
    test_snap = build_graph_snapshot(
        make_scenario(seed=2, id_offset=1000), OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM
    )
    print(f"train graph: {len(train_snap.node_ids)} nodes, {len(train_snap.edge_labels)} edges, "
          f"{train_snap.edge_labels.sum()} positive")
    print(f"test graph:  {len(test_snap.node_ids)} nodes, {len(test_snap.edge_labels)} edges, "
          f"{test_snap.edge_labels.sum()} positive")

    n_pos = max(int(train_snap.edge_labels.sum()), 1)
    n_neg = len(train_snap.edge_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    model = EdgeRiskGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        loss = train_one_epoch(model, train_snap, optimizer, pos_weight=pos_weight)
        if epoch % 10 == 0 or epoch == 49:
            print(f"epoch {epoch}: loss={loss:.4f}")

    model.eval()
    with torch.no_grad():
        node_features = torch.from_numpy(test_snap.node_features)
        edge_index = torch.from_numpy(test_snap.edge_index)
        edge_features = torch.from_numpy(test_snap.edge_features)
        logits = model(node_features, edge_index, edge_features)
        preds = (torch.sigmoid(logits) > 0.5).int().tolist()

    metrics = precision_recall_accuracy(preds, test_snap.edge_labels.tolist())
    print("\n--- held-out scenario results ---")
    print("static GNN:", metrics)


if __name__ == "__main__":
    main()