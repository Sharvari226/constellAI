"""Multi-scenario, multi-seed comparison: naive persistence vs LSTM vs
static GNN vs TGN-lite.

TGN-lite is deliberately run at smaller scale (fewer scenarios, fewer
seeds) than the other two models -- its sequential event-stream
processing makes it far slower (minutes per scenario vs seconds), so
matching LSTM/GNN's scenario count here isn't practical. This is a
real, acknowledged limitation: TGN-lite's numbers below are less
statistically solid than the other two. Report that caveat alongside
any number you take from this, don't just report the number.

Run: python scripts/compare_baselines.py
"""

import math
import random
from datetime import datetime, timedelta

import torch

from constellai.models.tgnn.dataset import build_forecast_examples
from constellai.models.tgnn.dynamic_graph import build_dynamic_graph
from constellai.models.tgnn.evaluation import average_precision, precision_recall_accuracy
from constellai.models.tgnn.gnn_baseline import EdgeRiskGNN, train_one_epoch as train_gnn_epoch
from constellai.models.tgnn.graph_dataset import build_graph_snapshot
from constellai.models.tgnn.lstm_baseline import PairRiskLSTM
from constellai.models.tgnn.tgn_lite import TGNLite, forward_and_score
from constellai.orbital_mechanics.synthetic import make_circular_satellite

STEP = timedelta(minutes=2)
STEP_TGN = timedelta(minutes=5)  # coarser -- fewer events, TGN-lite stays runnable
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
HORIZON_END = OBS_END + timedelta(hours=1, minutes=30)
THRESHOLD_KM = 300.0
MARGIN_KM = 50.0
SATS_PER_SCENARIO = 30

N_TRAIN_SCENARIOS = 6
N_TEST_SCENARIOS = 6

# TGN-lite: smaller scale, multiple seeds instead -- runtime-bounded, not
# a claim these are equally rigorous comparisons.
N_TGN_SEEDS = 3
N_TGN_TEST_SCENARIOS = 3
TGN_EPOCHS = 40


def make_scenario(seed: int, id_offset: int, n: int = SATS_PER_SCENARIO):
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


def run_lstm(train_scenarios, test_scenarios):
    train_ex = [ex for s in train_scenarios for ex in
                build_forecast_examples(s, OBS_START, OBS_END, HORIZON_END, STEP, THRESHOLD_KM)]
    test_ex = [ex for s in test_scenarios for ex in
               build_forecast_examples(s, OBS_START, OBS_END, HORIZON_END, STEP, THRESHOLD_KM)]

    n_pos = max(sum(e.label for e in train_ex), 1)
    n_neg = len(train_ex) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = PairRiskLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(15):
        for ex in train_ex:
            x = torch.from_numpy(ex.features).unsqueeze(0)
            y = torch.tensor([float(ex.label)])
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = [torch.sigmoid(model(torch.from_numpy(ex.features).unsqueeze(0))).item() for ex in test_ex]
    labels = [ex.label for ex in test_ex]
    naive_preds = [int(ex.features[-1, -1] < THRESHOLD_KM) for ex in test_ex]

    return {
        "lstm": {"ap": average_precision(scores, labels), **precision_recall_accuracy([int(s > 0.5) for s in scores], labels)},
        "naive": {"ap": average_precision([1 - p for p in naive_preds], labels), **precision_recall_accuracy(naive_preds, labels)},
        "n_examples": len(test_ex), "n_positive": sum(labels),
    }


def run_gnn(train_scenarios, test_scenarios):
    train_snaps = [build_graph_snapshot(s, OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM) for s in train_scenarios]
    test_snaps = [build_graph_snapshot(s, OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM) for s in test_scenarios]

    all_train_labels = [l for snap in train_snaps for l in snap.edge_labels.tolist()]
    n_pos = max(sum(all_train_labels), 1)
    n_neg = len(all_train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    model = EdgeRiskGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        for snap in train_snaps:
            train_gnn_epoch(model, snap, optimizer, pos_weight=pos_weight)

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for snap in test_snaps:
            node_features = torch.from_numpy(snap.node_features)
            edge_index = torch.from_numpy(snap.edge_index)
            edge_features = torch.from_numpy(snap.edge_features)
            logits = model(node_features, edge_index, edge_features)
            scores.extend(torch.sigmoid(logits).tolist())
            labels.extend(snap.edge_labels.tolist())

    return {
        "gnn": {"ap": average_precision(scores, labels), **precision_recall_accuracy([int(s > 0.5) for s in scores], labels)},
        "n_examples": len(labels), "n_positive": sum(labels),
    }


def run_tgn_one_seed(seed, test_scenarios):
    torch.manual_seed(seed)
    train_graph = build_dynamic_graph(
        make_scenario(seed=seed, id_offset=seed * 100), OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM
    )
    n_pos = max(sum(train_graph.pair_labels.values()), 1)
    n_neg = len(train_graph.pair_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(TGN_EPOCHS):
        optimizer.zero_grad()
        logits, labels = forward_and_score(model, train_graph)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor(labels, dtype=torch.float), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for test_scenario in test_scenarios:
            test_graph = build_dynamic_graph(test_scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
            logits, lab = forward_and_score(model, test_graph)
            scores.extend(torch.sigmoid(logits).tolist())
            labels.extend(lab)
    return scores, labels


def run_tgn(test_scenarios):
    """Averages AP across N_TGN_SEEDS independently trained models --
    this is the honest way to report a model this noisy: the spread
    across seeds, not one lucky (or unlucky) run."""
    aps = []
    pooled_scores, pooled_labels = [], []
    for seed in range(N_TGN_SEEDS):
        scores, labels = run_tgn_one_seed(seed, test_scenarios)
        aps.append(average_precision(scores, labels))
        pooled_scores.extend(scores)
        pooled_labels.extend(labels)

    preds = [int(s > 0.5) for s in pooled_scores]
    return {
        "tgn": {
            "ap_mean": sum(aps) / len(aps),
            "ap_min": min(aps),
            "ap_max": max(aps),
            **precision_recall_accuracy(preds, pooled_labels),
        },
        "n_examples": len(pooled_labels), "n_positive": sum(pooled_labels),
    }


def main():
    train_scenarios = [make_scenario(seed=i, id_offset=i * 100) for i in range(N_TRAIN_SCENARIOS)]
    test_scenarios = [make_scenario(seed=1000 + i, id_offset=1000 + i * 100) for i in range(N_TEST_SCENARIOS)]
    tgn_test_scenarios = test_scenarios[:N_TGN_TEST_SCENARIOS]

    lstm_results = run_lstm(train_scenarios, test_scenarios)
    gnn_results = run_gnn(train_scenarios, test_scenarios)
    print("training TGN-lite across seeds -- this is the slow part, expect several minutes...")
    tgn_results = run_tgn(tgn_test_scenarios)

    print(f"\nLSTM/GNN pool: {lstm_results['n_examples']} pairs, {lstm_results['n_positive']} positive "
          f"(6 scenarios, STEP={STEP})")
    print(f"TGN-lite pool: {tgn_results['n_examples']} pairs, {tgn_results['n_positive']} positive "
          f"({N_TGN_SEEDS} seeds x {N_TGN_TEST_SCENARIOS} scenarios, STEP={STEP_TGN} -- SMALLER SCALE, less rigorous)")

    print(f"\n{'model':<20}{'AP':>18}{'precision':>12}{'recall':>10}{'accuracy':>10}")
    for name, r in [("naive persistence", {**lstm_results["naive"], "ap_str": f"{lstm_results['naive']['ap']:.3f}"}),
                    ("LSTM", {**lstm_results["lstm"], "ap_str": f"{lstm_results['lstm']['ap']:.3f}"}),
                    ("static GNN", {**gnn_results["gnn"], "ap_str": f"{gnn_results['gnn']['ap']:.3f}"}),
                    ("TGN-lite", {**tgn_results["tgn"], "ap_str": f"{tgn_results['tgn']['ap_mean']:.3f} "
                                                                    f"[{tgn_results['tgn']['ap_min']:.2f}-{tgn_results['tgn']['ap_max']:.2f}]"})]:
        print(f"{name:<20}{r['ap_str']:>18}{r['precision']:>12.3f}{r['recall']:>10.3f}{r['accuracy']:>10.3f}")


if __name__ == "__main__":
    main()