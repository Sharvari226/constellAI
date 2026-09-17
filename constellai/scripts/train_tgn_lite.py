"""Train + evaluate the TGN-lite model on a single scenario.

Slower than the LSTM/static-GNN baselines by design: it processes a
genuine chronological event stream (sequential GRU updates), not a
single batched tensor op. Runtime is dominated by event count, which
scales with (num_candidate_pairs x observation_timesteps) -- STEP below
is deliberately coarser than the other scripts to keep this runnable in
a couple minutes; tighten it once you're past prototyping.

Run: python scripts/train_tgn_lite.py
"""

import math
import random
import time
from datetime import datetime, timedelta

import torch

from constellai.models.tgnn.dynamic_graph import build_dynamic_graph
from constellai.models.tgnn.evaluation import average_precision, precision_recall_accuracy
from constellai.models.tgnn.tgn_lite import TGNLite, forward_and_score
from constellai.orbital_mechanics.synthetic import make_circular_satellite

STEP = timedelta(minutes=5)  # coarser than train_gnn_baseline.py's 2 min -- fewer events, faster
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
HORIZON_END = OBS_END + timedelta(hours=1, minutes=30)
THRESHOLD_KM = 300.0
MARGIN_KM = 50.0
N_EPOCHS = 20


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


def main():
    train_graph = build_dynamic_graph(
        make_scenario(seed=1, id_offset=0), OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM
    )
    test_graph = build_dynamic_graph(
        make_scenario(seed=2, id_offset=1000), OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM
    )
    print(f"train: {len(train_graph.events)} events, {len(train_graph.pair_labels)} pairs, "
          f"{sum(train_graph.pair_labels.values())} positive")
    print(f"test:  {len(test_graph.events)} events, {len(test_graph.pair_labels)} pairs, "
          f"{sum(test_graph.pair_labels.values())} positive")

    n_pos = max(sum(train_graph.pair_labels.values()), 1)
    n_neg = len(train_graph.pair_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        logits, labels = forward_and_score(model, train_graph)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor(labels, dtype=torch.float), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch}: loss={loss.item():.4f} ({time.time() - start:.0f}s elapsed)")

    model.eval()
    with torch.no_grad():
        logits, labels = forward_and_score(model, test_graph)
        scores = torch.sigmoid(logits).tolist()
    preds = [int(s > 0.5) for s in scores]

    print("\n--- held-out scenario results ---")
    print("TGN-lite:", {"ap": average_precision(scores, labels), **precision_recall_accuracy(preds, labels)})


if __name__ == "__main__":
    main()