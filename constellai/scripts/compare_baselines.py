"""Multi-scenario, multi-seed comparison: naive persistence vs LSTM vs
static GNN vs TGN-lite (point-estimate and uncertainty-aware).

All three learned models train on the SAME train_scenarios and the SAME
candidate-pair population (M2's candidate_pairs_by_regime). LSTM, GNN,
and TGN-lite all run across multiple seeds and report mean + [min-max]
AP -- LSTM/GNN were previously unseeded, meaning a single run was one
random draw from training stochasticity, not a reproducible result.

TGN-lite runs TWICE: once with the original point-estimate risk head
(BCE loss), once with the Beta-distribution uncertainty head (Beta-NLL
loss) -- the second run also reports Expected Calibration Error (ECE),
answering this project's secondary hypothesis (H2): does calibrated
uncertainty actually improve calibration, not just exist as a feature.

TGN-lite still runs at a coarser time step (STEP_TGN) and a smaller test
set than LSTM/GNN -- genuinely acknowledged limitations, not fixed here.

Run: python -m constellai.scripts.compare_baselines
"""

import math
import random
from datetime import datetime, timedelta

import torch

from constellai.models.tgnn.dataset import build_forecast_examples
from constellai.models.tgnn.dynamic_graph import build_dynamic_graph
from constellai.models.tgnn.evaluation import (
    average_precision,
    expected_calibration_error,
    precision_recall_accuracy,
)
from constellai.models.tgnn.gnn_baseline import EdgeRiskGNN, train_one_epoch as train_gnn_epoch
from constellai.models.tgnn.graph_dataset import build_graph_snapshot
from constellai.models.tgnn.lstm_baseline import PairRiskLSTM
from constellai.models.tgnn.tgn_lite import (
    TGNLite,
    beta_nll_loss,
    forward_and_score,
    forward_and_score_with_uncertainty,
)
from constellai.orbital_mechanics.synthetic import make_circular_satellite

STEP = timedelta(minutes=2)
STEP_TGN = timedelta(minutes=5)
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
HORIZON_END = OBS_END + timedelta(hours=1, minutes=30)
THRESHOLD_KM = 300.0
MARGIN_KM = 50.0
SATS_PER_SCENARIO = 30

N_TRAIN_SCENARIOS = 6
N_TEST_SCENARIOS = 6

N_SEEDS = 3
N_TGN_TEST_SCENARIOS = 3
TGN_EPOCHS = 8
LSTM_EPOCHS = 15
GNN_EPOCHS = 50


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


def run_lstm_one_seed(seed, train_ex, test_ex):
    torch.manual_seed(seed)

    n_pos = max(sum(e.label for e in train_ex), 1)
    n_neg = len(train_ex) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = PairRiskLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(LSTM_EPOCHS):
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
    return scores


def run_lstm(train_scenarios, test_scenarios, margin_km):
    train_ex = [ex for s in train_scenarios for ex in
                build_forecast_examples(s, OBS_START, OBS_END, HORIZON_END, STEP, THRESHOLD_KM, margin_km)]
    test_ex = [ex for s in test_scenarios for ex in
               build_forecast_examples(s, OBS_START, OBS_END, HORIZON_END, STEP, THRESHOLD_KM, margin_km)]
    labels = [ex.label for ex in test_ex]
    naive_preds = [int(ex.features[-1, -1] < THRESHOLD_KM) for ex in test_ex]

    aps, pooled_scores = [], []
    for seed in range(N_SEEDS):
        scores = run_lstm_one_seed(seed, train_ex, test_ex)
        aps.append(average_precision(scores, labels))
        pooled_scores.append(scores)

    avg_scores = [sum(s[i] for s in pooled_scores) / N_SEEDS for i in range(len(test_ex))]
    preds = [int(s > 0.5) for s in avg_scores]

    return {
        "lstm": {"ap_mean": sum(aps) / len(aps), "ap_min": min(aps), "ap_max": max(aps),
                 **precision_recall_accuracy(preds, labels)},
        "naive": {"ap": average_precision([1 - p for p in naive_preds], labels), **precision_recall_accuracy(naive_preds, labels)},
        "n_examples": len(test_ex), "n_positive": sum(labels),
    }


def run_gnn_one_seed(seed, train_snaps, test_snaps):
    torch.manual_seed(seed)

    all_train_labels = [l for snap in train_snaps for l in snap.edge_labels.tolist()]
    n_pos = max(sum(all_train_labels), 1)
    n_neg = len(all_train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    model = EdgeRiskGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(GNN_EPOCHS):
        for snap in train_snaps:
            train_gnn_epoch(model, snap, optimizer, pos_weight=pos_weight)

    model.eval()
    scores = []
    with torch.no_grad():
        for snap in test_snaps:
            node_features = torch.from_numpy(snap.node_features)
            edge_index = torch.from_numpy(snap.edge_index)
            edge_features = torch.from_numpy(snap.edge_features)
            logits = model(node_features, edge_index, edge_features)
            scores.extend(torch.sigmoid(logits).tolist())
    return scores


def run_gnn(train_scenarios, test_scenarios):
    train_snaps = [build_graph_snapshot(s, OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM) for s in train_scenarios]
    test_snaps = [build_graph_snapshot(s, OBS_START, OBS_END, HORIZON_END, STEP, MARGIN_KM, THRESHOLD_KM) for s in test_scenarios]
    labels = [l for snap in test_snaps for l in snap.edge_labels.tolist()]

    aps, pooled_scores = [], []
    for seed in range(N_SEEDS):
        scores = run_gnn_one_seed(seed, train_snaps, test_snaps)
        aps.append(average_precision(scores, labels))
        pooled_scores.append(scores)

    avg_scores = [sum(s[i] for s in pooled_scores) / N_SEEDS for i in range(len(labels))]
    preds = [int(s > 0.5) for s in avg_scores]

    return {
        "gnn": {"ap_mean": sum(aps) / len(aps), "ap_min": min(aps), "ap_max": max(aps),
                **precision_recall_accuracy(preds, labels)},
        "n_examples": len(labels), "n_positive": sum(labels),
    }


def train_tgn_one_epoch(model, train_graphs, pos_weight, optimizer):
    total_loss = 0.0
    for graph in train_graphs:
        optimizer.zero_grad()
        logits, labels = forward_and_score(model, graph)
        if logits.numel() == 0:
            continue
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor(labels, dtype=torch.float), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_graphs)


def run_tgn_one_seed(seed, train_scenarios, test_scenarios):
    torch.manual_seed(seed)

    train_graphs = [
        build_dynamic_graph(s, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
        for s in train_scenarios
    ]

    all_train_labels = [l for g in train_graphs for l in g.pair_labels.values()]
    n_pos = max(sum(all_train_labels), 1)
    n_neg = len(all_train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])

    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(TGN_EPOCHS):
        loss = train_tgn_one_epoch(model, train_graphs, pos_weight, optimizer)
        print(f"    [seed {seed}] epoch {epoch+1}/{TGN_EPOCHS}, loss={loss:.4f}", flush=True)

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for test_scenario in test_scenarios:
            test_graph = build_dynamic_graph(test_scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
            logits, lab = forward_and_score(model, test_graph)
            if logits.numel() == 0:
                continue
            scores.extend(torch.sigmoid(logits).tolist())
            labels.extend(lab)
    return scores, labels


def run_tgn(train_scenarios, test_scenarios):
    aps = []
    pooled_scores, pooled_labels = [], []
    for seed in range(N_SEEDS):
        scores, labels = run_tgn_one_seed(seed, train_scenarios, test_scenarios)
        aps.append(average_precision(scores, labels))
        pooled_scores.extend(scores)
        pooled_labels.extend(labels)

    preds = [int(s > 0.5) for s in pooled_scores]
    return {
        "tgn": {
            "ap_mean": sum(aps) / len(aps), "ap_min": min(aps), "ap_max": max(aps),
            **precision_recall_accuracy(preds, pooled_labels),
        },
        "n_examples": len(pooled_labels), "n_positive": sum(pooled_labels),
    }


def train_tgn_one_epoch_uncertainty(model, train_graphs, optimizer):
    total_loss = 0.0
    for graph in train_graphs:
        optimizer.zero_grad()
        alphas, betas, labels = forward_and_score_with_uncertainty(model, graph)
        if alphas.numel() == 0:
            continue
        loss = beta_nll_loss(alphas, betas, torch.tensor(labels, dtype=torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_graphs)


def run_tgn_one_seed_uncertainty(seed, train_scenarios, test_scenarios):
    torch.manual_seed(seed)
    train_graphs = [
        build_dynamic_graph(s, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
        for s in train_scenarios
    ]
    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(TGN_EPOCHS):
        loss = train_tgn_one_epoch_uncertainty(model, train_graphs, optimizer)
        print(f"    [seed {seed}, uncertainty] epoch {epoch+1}/{TGN_EPOCHS}, loss={loss:.4f}", flush=True)

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for test_scenario in test_scenarios:
            test_graph = build_dynamic_graph(test_scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
            alphas, betas, lab = forward_and_score_with_uncertainty(model, test_graph)
            if alphas.numel() == 0:
                continue
            means = (alphas / (alphas + betas)).tolist()
            scores.extend(means)
            labels.extend(lab)
    return scores, labels


def run_tgn_uncertainty(train_scenarios, test_scenarios):
    aps, eces = [], []
    for seed in range(N_SEEDS):
        scores, labels = run_tgn_one_seed_uncertainty(seed, train_scenarios, test_scenarios)
        aps.append(average_precision(scores, labels))
        eces.append(expected_calibration_error(scores, labels))
    return {
        "ap_mean": sum(aps) / len(aps), "ap_min": min(aps), "ap_max": max(aps),
        "ece_mean": sum(eces) / len(eces),
    }


def main():
    train_scenarios = [make_scenario(seed=i, id_offset=i * 100) for i in range(N_TRAIN_SCENARIOS)]
    test_scenarios = [make_scenario(seed=1000 + i, id_offset=1000 + i * 100) for i in range(N_TEST_SCENARIOS)]
    tgn_test_scenarios = test_scenarios[:N_TGN_TEST_SCENARIOS]

    print(f"training LSTM across {N_SEEDS} seeds...")
    lstm_results = run_lstm(train_scenarios, test_scenarios, MARGIN_KM)
    print(f"training static GNN across {N_SEEDS} seeds...")
    gnn_results = run_gnn(train_scenarios, test_scenarios)
    print(f"training TGN-lite (point estimate) across {N_SEEDS} seeds...")
    tgn_results = run_tgn(train_scenarios, tgn_test_scenarios)

    print(f"\nLSTM pool: {lstm_results['n_examples']} pairs, {lstm_results['n_positive']} positive "
          f"({N_TRAIN_SCENARIOS}+{N_TEST_SCENARIOS} scenarios, STEP={STEP})")
    print(f"GNN pool:  {gnn_results['n_examples']} pairs, {gnn_results['n_positive']} positive (same pair population as LSTM)")
    print(f"TGN-lite pool: {tgn_results['n_examples']} pairs, {tgn_results['n_positive']} positive "
          f"({N_SEEDS} seeds x {N_TGN_TEST_SCENARIOS} test scenarios, trained on all {N_TRAIN_SCENARIOS} "
          f"train scenarios, STEP={STEP_TGN} -- coarser step AND smaller test set than LSTM/GNN)")

    print(f"\n{'model':<20}{'AP (mean [min-max])':>28}{'precision':>12}{'recall':>10}{'accuracy':>10}")
    rows = [
        ("naive persistence", f"{lstm_results['naive']['ap']:.3f}", lstm_results["naive"]),
        ("LSTM", f"{lstm_results['lstm']['ap_mean']:.3f} [{lstm_results['lstm']['ap_min']:.2f}-{lstm_results['lstm']['ap_max']:.2f}]", lstm_results["lstm"]),
        ("static GNN", f"{gnn_results['gnn']['ap_mean']:.3f} [{gnn_results['gnn']['ap_min']:.2f}-{gnn_results['gnn']['ap_max']:.2f}]", gnn_results["gnn"]),
        ("TGN-lite", f"{tgn_results['tgn']['ap_mean']:.3f} [{tgn_results['tgn']['ap_min']:.2f}-{tgn_results['tgn']['ap_max']:.2f}]", tgn_results["tgn"]),
    ]
    for name, ap_str, r in rows:
        print(f"{name:<20}{ap_str:>28}{r['precision']:>12.3f}{r['recall']:>10.3f}{r['accuracy']:>10.3f}")

    print(f"\ntraining TGN-lite WITH uncertainty head (Beta-NLL) across {N_SEEDS} seeds...")
    tgn_uncertainty_results = run_tgn_uncertainty(train_scenarios, tgn_test_scenarios)
    print(f"TGN-lite (point estimate):    AP {tgn_results['tgn']['ap_mean']:.3f} "
          f"[{tgn_results['tgn']['ap_min']:.2f}-{tgn_results['tgn']['ap_max']:.2f}]")
    print(f"TGN-lite (uncertainty-aware): AP {tgn_uncertainty_results['ap_mean']:.3f} "
          f"[{tgn_uncertainty_results['ap_min']:.2f}-{tgn_uncertainty_results['ap_max']:.2f}], "
          f"ECE {tgn_uncertainty_results['ece_mean']:.4f}")


if __name__ == "__main__":
    main()