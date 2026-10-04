"""Fresh, verified visualization suite :
  1. Graph construction (M2) -- an actual node/edge diagram, not just a
     pair-count bar chart: coarse-filter candidates (thin gray) vs.
     fine-screen survivors (bold red), plus the sparsity comparison.
  2. Feature importance -- permutation importance on a PROPERLY trained
     TGN-lite (50 epochs, matching the working config from
     compare_baselines.py -- the earlier version of this used an
     8-epoch, near-random model and produced meaningless/negative
     importances).
  3. Stage-by-stage results:
     - M1: a real propagated trajectory (separation over time) for one
       satellite pair, the actual physics output.
     - M3: model comparison bar chart (AP, mean + real [min,max] per
       model, non-symmetric error bars).
     - M3 (uncertainty): a calibration reliability diagram for the
       uncertainty-aware TGN-lite -- predicted confidence vs. observed
       frequency, the direct visual of what ECE measures numerically.

Single-threaded torch (deterministic-ish; still expect minor run-to-run
drift from CPU floating point, not a bug) and fixed seeds throughout.

This is slow -- TGN-lite training dominates. Expect 20-40+ minutes.

Run: python -m constellai.scripts.generate_visualizations
Outputs: PNG files in outputs/visualizations/
"""

import math
import os
import random
from dataclasses import replace
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.graph.screening import screen_candidate_pair
from constellai.models.tgnn.dynamic_graph import GraphEvent, build_dynamic_graph
from constellai.models.tgnn.evaluation import average_precision
from constellai.models.tgnn.tgn_lite import (
    TGNLite,
    beta_nll_loss,
    forward_and_score,
    forward_and_score_with_uncertainty,
)
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.synthetic import make_circular_satellite

OUTPUT_DIR = "outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEP_TGN = timedelta(minutes=5)
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
HORIZON_END = OBS_END + timedelta(hours=1, minutes=30)
THRESHOLD_KM = 300.0
MARGIN_KM = 50.0
N_TRAIN_SCENARIOS = 4
TRAIN_EPOCHS = 50  # matches the config confirmed to work in compare_baselines.py
UNCERTAINTY_EPOCHS = 20  # the uncertainty head overfits past this -- confirmed earlier
FEATURE_NAMES = ["dx", "dy", "dz", "separation", "relative_speed"]


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


# ---------------------------------------------------------------------------
# 1. Graph construction: an actual node/edge diagram of M2's pipeline
# ---------------------------------------------------------------------------
def plot_graph_construction():
    print("=" * 60)
    print("1. GRAPH CONSTRUCTION (M2)")
    print("=" * 60)

    # Smaller N than the modeling scenarios -- 30 satellites makes the
    # node/edge diagram unreadable; this is purely for illustration.
    records = make_scenario(seed=1, id_offset=0, n=20)
    candidates = candidate_pairs_by_regime(records, margin_km=MARGIN_KM)

    edges = []
    for a, b in candidates:
        edge = screen_candidate_pair(
            a, b, OBS_START, OBS_END, STEP_TGN,
            distance_threshold_km=THRESHOLD_KM, min_closing_rate_km_s=0.0,
        )
        if edge is not None:
            edges.append((a.satellite_id, b.satellite_id))

    total_possible = len(records) * (len(records) - 1) // 2
    print(f"satellites: {len(records)}, exhaustive pairs: {total_possible}, "
          f"coarse-filter candidates: {len(candidates)}, fine-screen edges: {len(edges)}")

    ids = [r.satellite_id for r in records]
    n = len(ids)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {sid: (np.cos(a), np.sin(a)) for sid, a in zip(ids, angles)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    for a, b in candidates:
        x1, y1 = pos[a.satellite_id]
        x2, y2 = pos[b.satellite_id]
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.6, zorder=1)
    for sid_a, sid_b in edges:
        x1, y1 = pos[sid_a]
        x2, y2 = pos[sid_b]
        ax.plot([x1, x2], [y1, y2], color="indianred", linewidth=2.2, zorder=2)
    for sid in ids:
        x, y = pos[sid]
        ax.scatter([x], [y], s=90, color="steelblue", zorder=3, edgecolor="black")
        ax.annotate(sid, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"M2 pipeline: {len(candidates)} coarse-filter candidates (gray)\n"
                 f"{len(edges)} survive fine screening (red)")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lightgray", lw=2, label="Stage A: coarse-filter candidate"),
        Line2D([0], [0], color="indianred", lw=2, label="Stage B: fine-screen survivor (graph edge)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8)

    ax = axes[1]
    bars = ax.bar(
        ["Exhaustive\n(N*(N-1)/2)", "Coarse-filter\ncandidates", "Fine-screen\nedges"],
        [total_possible, len(candidates), len(edges)],
        color=["gray", "goldenrod", "indianred"],
    )
    ax.set_ylabel("Pair count")
    ax.set_title("M2 sparsity by stage")
    for bar, v in zip(bars, [total_possible, len(candidates), len(edges)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_graph_construction.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/01_graph_construction.png\n")


# ---------------------------------------------------------------------------
# 2. Feature importance -- on a properly trained model
# ---------------------------------------------------------------------------
def train_tgn_lite(train_graphs, epochs, uncertainty=False, seed=0):
    torch.manual_seed(seed)
    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for g in train_graphs:
            optimizer.zero_grad()
            if uncertainty:
                alphas, betas, labels = forward_and_score_with_uncertainty(model, g)
                if alphas.numel() == 0:
                    continue
                loss = beta_nll_loss(alphas, betas, torch.tensor(labels, dtype=torch.float))
            else:
                logits, labels = forward_and_score(model, g)
                if logits.numel() == 0:
                    continue
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.tensor(labels, dtype=torch.float)
                )
            loss.backward()
            optimizer.step()
    return model


def plot_feature_importance(model, test_scenarios):
    print("=" * 60)
    print("2. FEATURE IMPORTANCE (permutation importance)")
    print("=" * 60)
    print("Model trained for", TRAIN_EPOCHS, "epochs (point-estimate head) -- "
          "the earlier 8-epoch version was too undertrained for this to be meaningful.\n")

    def evaluate(shuffle_idx=None, seed=0):
        rng = np.random.RandomState(seed)
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for scenario in test_scenarios:
                graph = build_dynamic_graph(scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
                if shuffle_idx is not None:
                    values = np.array([e.features[shuffle_idx] for e in graph.events])
                    rng.shuffle(values)
                    new_events = []
                    for e, v in zip(graph.events, values):
                        new_features = e.features.copy()
                        new_features[shuffle_idx] = v
                        new_events.append(GraphEvent(t_index=e.t_index, node_a=e.node_a, node_b=e.node_b, features=new_features))
                    graph = replace(graph, events=new_events)
                logits, lab = forward_and_score(model, graph)
                if logits.numel() == 0:
                    continue
                scores.extend(torch.sigmoid(logits).tolist())
                labels.extend(lab)
        return average_precision(scores, labels)

    baseline_ap = evaluate()
    print(f"Baseline AP (no shuffling): {baseline_ap:.4f}")

    importances, errs = [], []
    for idx, name in enumerate(FEATURE_NAMES):
        shuffled = [evaluate(shuffle_idx=idx, seed=s) for s in range(3)]
        mean_shuffled = sum(shuffled) / len(shuffled)
        importance = baseline_ap - mean_shuffled
        importances.append(importance)
        errs.append((max(shuffled) - min(shuffled)) / 2)
        print(f"  shuffle '{name}': AP -> {mean_shuffled:.4f} (importance = {importance:+.4f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["indianred" if v > 0 else "steelblue" for v in importances]
    ax.barh(FEATURE_NAMES, importances, xerr=errs, color=colors, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AP drop when feature is shuffled (higher = more important)")
    ax.set_title(f"TGN-lite permutation feature importance (baseline AP={baseline_ap:.3f})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_feature_importance.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/02_feature_importance.png\n")


# ---------------------------------------------------------------------------
# 3a. M1 stage result: a real propagated trajectory
# ---------------------------------------------------------------------------
def plot_m1_propagation():
    print("=" * 60)
    print("3a. M1 STAGE RESULT: propagated separation over time")
    print("=" * 60)

    records = make_scenario(seed=1, id_offset=0, n=2)
    a, b = records[0], records[1]
    states_a = propagate_series(a, OBS_START, HORIZON_END, timedelta(minutes=2))
    states_b = propagate_series(b, OBS_START, HORIZON_END, timedelta(minutes=2))

    minutes = [(s.epoch - OBS_START).total_seconds() / 60 for s in states_a]
    separations = [float(np.linalg.norm(sa.position_km - sb.position_km)) for sa, sb in zip(states_a, states_b)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(minutes, separations, color="steelblue", linewidth=1.5)
    ax.axvline((OBS_END - OBS_START).total_seconds() / 60, color="gray", linestyle="--", linewidth=1,
               label="observation window ends / horizon window begins")
    ax.set_xlabel("Minutes since window start")
    ax.set_ylabel("Separation (km)")
    ax.set_title(f"M1: propagated separation, satellite {a.satellite_id} vs {b.satellite_id}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03a_m1_propagation.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03a_m1_propagation.png\n")


# ---------------------------------------------------------------------------
# 3b. M3 stage result: model comparison (reusing already-trained model
#     where possible to avoid retraining everything from scratch)
# ---------------------------------------------------------------------------
def plot_m3_model_comparison(point_est_aps, point_est_min, point_est_max,
                              uncertainty_aps, uncertainty_min, uncertainty_max):
    print("=" * 60)
    print("3b. M3 STAGE RESULT: model comparison")
    print("=" * 60)

    names = ["TGN-lite\n(point estimate)", "TGN-lite\n(uncertainty-aware)"]
    means = [sum(point_est_aps) / len(point_est_aps), sum(uncertainty_aps) / len(uncertainty_aps)]
    lower = [means[0] - point_est_min, means[1] - uncertainty_min]
    upper = [point_est_max - means[0], uncertainty_max - means[1]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, means, yerr=[lower, upper], capsize=6, color=["goldenrod", "indianred"])
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("TGN-lite: point estimate vs. uncertainty-aware (mean, min-max across seeds)")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03b_m3_model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03b_m3_model_comparison.png\n")


# ---------------------------------------------------------------------------
# 3c. M3 stage result: calibration reliability diagram
# ---------------------------------------------------------------------------
def plot_reliability_diagram(scores, labels, title_suffix=""):
    print("=" * 60)
    print("3c. M3 STAGE RESULT: calibration reliability diagram")
    print("=" * 60)

    scores = np.array(scores)
    labels = np.array(labels)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_freq, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi) if hi < 1.0 else (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        bin_conf.append(scores[mask].mean())
        bin_freq.append(labels[mask].mean())
        bin_counts.append(int(mask.sum()))

    print(f"populated bins: {len(bin_conf)} / {n_bins}, total points: {len(scores)}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    sizes = [20 + c * 3 for c in bin_counts]
    ax.scatter(bin_conf, bin_freq, s=sizes, color="indianred", zorder=3, label="model bins (size = # predictions)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted confidence (bin mean)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"TGN-lite calibration reliability diagram{title_suffix}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03c_calibration_reliability.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03c_calibration_reliability.png\n")


def main():
    plot_graph_construction()
    plot_m1_propagation()

    train_scenarios = [make_scenario(seed=i, id_offset=i * 100) for i in range(N_TRAIN_SCENARIOS)]
    test_scenarios = [make_scenario(seed=1000 + i, id_offset=1000 + i * 100) for i in range(3)]

    print("training point-estimate TGN-lite across 3 seeds (this is the slow part)...")
    train_graphs_by_seed = []
    point_est_aps, point_est_scores_all, point_est_labels_all = [], [], []
    for seed in range(3):
        train_graphs = [build_dynamic_graph(s, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM) for s in train_scenarios]
        model = train_tgn_lite(train_graphs, TRAIN_EPOCHS, uncertainty=False, seed=seed)
        if seed == 0:
            feature_importance_model = model  # reuse seed-0's model for feature importance, no need to retrain again
        scores, labels = [], []
        model.eval()
        with torch.no_grad():
            for ts in test_scenarios:
                test_graph = build_dynamic_graph(ts, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
                logits, lab = forward_and_score(model, test_graph)
                if logits.numel() == 0:
                    continue
                scores.extend(torch.sigmoid(logits).tolist())
                labels.extend(lab)
        point_est_aps.append(average_precision(scores, labels))
        print(f"  seed {seed}: AP={point_est_aps[-1]:.4f}")

    plot_feature_importance(feature_importance_model, test_scenarios)

    print("training uncertainty-aware TGN-lite across 3 seeds...")
    uncertainty_aps, last_seed_scores, last_seed_labels = [], [], []
    for seed in range(3):
        train_graphs = [build_dynamic_graph(s, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM) for s in train_scenarios]
        model = train_tgn_lite(train_graphs, UNCERTAINTY_EPOCHS, uncertainty=True, seed=seed)
        scores, labels = [], []
        model.eval()
        with torch.no_grad():
            for ts in test_scenarios:
                test_graph = build_dynamic_graph(ts, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
                alphas, betas, lab = forward_and_score_with_uncertainty(model, test_graph)
                if alphas.numel() == 0:
                    continue
                means = (alphas / (alphas + betas)).tolist()
                scores.extend(means)
                labels.extend(lab)
        uncertainty_aps.append(average_precision(scores, labels))
        last_seed_scores, last_seed_labels = scores, labels  # keep last seed's raw scores for the reliability diagram
        print(f"  seed {seed}: AP={uncertainty_aps[-1]:.4f}")

    plot_m3_model_comparison(
        point_est_aps, min(point_est_aps), max(point_est_aps),
        uncertainty_aps, min(uncertainty_aps), max(uncertainty_aps),
    )
    plot_reliability_diagram(last_seed_scores, last_seed_labels, title_suffix=" (uncertainty head, one representative seed)")

    print("\nAll visualizations saved to outputs/visualizations/")


if __name__ == "__main__":
    main()