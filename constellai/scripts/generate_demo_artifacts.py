"""Demonstration artifacts for presentation/paper: dataset statistics,
feature importance, model performance comparison, and an example of
what the system's actual output looks like.

Run: python -m constellai.scripts.generate_demo_artifacts
Outputs: PNG files in outputs/demo/, plus printed statistics.
"""

import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # no display needed, just saves files
import matplotlib.pyplot as plt
import numpy as np
import torch

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.models.tgnn.dynamic_graph import build_dynamic_graph
from constellai.models.tgnn.tgn_lite import TGNLite, forward_and_score_with_uncertainty, beta_nll_loss
from constellai.orbital_mechanics.regime import altitude_band
from constellai.orbital_mechanics.synthetic import make_circular_satellite
from constellai.scripts.compare_baselines import (
    make_scenario, run_lstm, run_gnn, run_tgn, run_tgn_uncertainty,
    OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM,
    N_TRAIN_SCENARIOS, N_TEST_SCENARIOS, N_TGN_TEST_SCENARIOS,
)

OUTPUT_DIR = "outputs/demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Dataset demonstration: what does the synthetic constellation look like?
# ---------------------------------------------------------------------------
def demo_dataset():
    print("=" * 60)
    print("1. DATASET DEMONSTRATION")
    print("=" * 60)

    scenario = make_scenario(seed=0, id_offset=0)
    altitudes = [altitude_band(r).perigee_alt_km for r in scenario]
    inclinations_deg = [np.degrees(r.satrec.inclo) for r in scenario]

    print(f"Satellites in one scenario: {len(scenario)}")
    print(f"Altitude range: {min(altitudes):.2f} - {max(altitudes):.2f} km")
    print(f"Inclination range: {min(inclinations_deg):.2f} - {max(inclinations_deg):.2f} deg")

    candidates = candidate_pairs_by_regime(scenario, margin_km=MARGIN_KM)
    total_possible = len(scenario) * (len(scenario) - 1) // 2
    print(f"Total possible pairs: {total_possible}")
    print(f"Coarse-filter candidate pairs: {len(candidates)} "
          f"({100 * len(candidates) / total_possible:.1f}% of exhaustive)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(altitudes, bins=15, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Perigee altitude (km)")
    axes[0].set_ylabel("Satellite count")
    axes[0].set_title("Synthetic constellation: altitude distribution")

    axes[1].bar(["Exhaustive\n(N*(N-1)/2)", "Coarse-filter\ncandidates"],
                [total_possible, len(candidates)], color=["indianred", "seagreen"])
    axes[1].set_ylabel("Pair count")
    axes[1].set_title("M2 sparsity: pairs actually screened")
    for i, v in enumerate([total_possible, len(candidates)]):
        axes[1].text(i, v + 1, str(v), ha="center")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_dataset_overview.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/01_dataset_overview.png\n")


# ---------------------------------------------------------------------------
# 2. Feature importance (permutation importance on a trained TGN-lite)
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["dx", "dy", "dz", "separation", "relative_speed"]


def demo_feature_importance():
    print("=" * 60)
    print("2. FEATURE IMPORTANCE (permutation importance)")
    print("=" * 60)
    print("Method: train once, then shuffle ONE feature dimension across")
    print("all events, remeasure AP, repeat per feature. Larger AP drop =")
    print("more important feature. This is model-agnostic and doesn't")
    print("require gradient access, unlike attention-weight inspection.\n")

    train_scenarios = [make_scenario(seed=i, id_offset=i * 100) for i in range(N_TRAIN_SCENARIOS)]
    test_scenarios = [make_scenario(seed=1000 + i, id_offset=1000 + i * 100) for i in range(N_TGN_TEST_SCENARIOS)]

    torch.manual_seed(0)
    train_graphs = [
        build_dynamic_graph(s, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
        for s in train_scenarios
    ]
    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(15):
        for graph in train_graphs:
            optimizer.zero_grad()
            alphas, betas, labels = forward_and_score_with_uncertainty(model, graph)
            if alphas.numel() == 0:
                continue
            loss = beta_nll_loss(alphas, betas, torch.tensor(labels, dtype=torch.float))
            loss.backward()
            optimizer.step()

    def evaluate(shuffle_feature_idx=None, seed=0):
        rng = np.random.RandomState(seed)
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for scenario in test_scenarios:
                graph = build_dynamic_graph(scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)

                if shuffle_feature_idx is not None:
                    values = np.array([e.features[shuffle_feature_idx] for e in graph.events])
                    rng.shuffle(values)
                    from constellai.models.tgnn.dynamic_graph import GraphEvent
                    new_events = []
                    for e, shuffled_val in zip(graph.events, values):
                        new_features = e.features.copy()
                        new_features[shuffle_feature_idx] = shuffled_val
                        new_events.append(GraphEvent(
                            t_index=e.t_index, node_a=e.node_a, node_b=e.node_b, features=new_features,
                        ))
                    from dataclasses import replace
                    graph = replace(graph, events=new_events)

                alphas, betas, lab = forward_and_score_with_uncertainty(model, graph)
                if alphas.numel() == 0:
                    continue
                scores.extend((alphas / (alphas + betas)).tolist())
                labels.extend(lab)
        from constellai.models.tgnn.evaluation import average_precision
        return average_precision(scores, labels)

    # Baseline AP (no shuffling), then AP with each feature individually
    # shuffled -- averaged over 3 shuffle seeds per feature, since a
    # single shuffle is a noisy point estimate.
    baseline_ap = evaluate(shuffle_feature_idx=None)
    print(f"Baseline AP (no shuffling): {baseline_ap:.4f}\n")

    importances = []
    for idx, feature_name in enumerate(FEATURE_NAMES):
        shuffled_aps = [evaluate(shuffle_feature_idx=idx, seed=s) for s in range(3)]
        avg_shuffled_ap = sum(shuffled_aps) / len(shuffled_aps)
        importance = baseline_ap - avg_shuffled_ap
        importances.append(importance)
        print(f"  shuffle '{feature_name}': AP drops to {avg_shuffled_ap:.4f} "
              f"(importance = {importance:+.4f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["indianred" if imp > 0 else "steelblue" for imp in importances]
    ax.barh(FEATURE_NAMES, importances, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AP drop when feature is shuffled (higher = more important)")
    ax.set_title("TGN-lite permutation feature importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_feature_importance.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/02_feature_importance.png\n")


# ---------------------------------------------------------------------------
# 3. Model performance comparison (reuses the already-validated pipeline)
# ---------------------------------------------------------------------------
def demo_model_performance():
    print("=" * 60)
    print("3. MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    train_scenarios = [make_scenario(seed=i, id_offset=i * 100) for i in range(N_TRAIN_SCENARIOS)]
    test_scenarios = [make_scenario(seed=1000 + i, id_offset=1000 + i * 100) for i in range(N_TEST_SCENARIOS)]
    tgn_test_scenarios = test_scenarios[:N_TGN_TEST_SCENARIOS]

    lstm_results = run_lstm(train_scenarios, test_scenarios, MARGIN_KM)
    gnn_results = run_gnn(train_scenarios, test_scenarios)
    tgn_results = run_tgn(train_scenarios, tgn_test_scenarios)
    tgn_unc_results = run_tgn_uncertainty(train_scenarios, tgn_test_scenarios)

    names = ["Naive", "LSTM", "Static GNN", "TGN-lite\n(point est.)", "TGN-lite\n(uncertainty)"]
    aps = [
        lstm_results["naive"]["ap"],
        lstm_results["lstm"]["ap_mean"],
        gnn_results["gnn"]["ap_mean"],
        tgn_results["tgn"]["ap_mean"],
        tgn_unc_results["ap_mean"],
    ]

    # Separate lower/upper error distances -- the naive baseline has no
    # seed spread (ap_errs entries of 0 for both), the rest use their
    # real [min, max] across seeds, each measured independently from
    # the mean rather than assuming symmetry.
    lower_errs = [
        0,
        lstm_results["lstm"]["ap_mean"] - lstm_results["lstm"]["ap_min"],
        gnn_results["gnn"]["ap_mean"] - gnn_results["gnn"]["ap_min"],
        tgn_results["tgn"]["ap_mean"] - tgn_results["tgn"]["ap_min"],
        tgn_unc_results["ap_mean"] - tgn_unc_results["ap_min"],
    ]
    upper_errs = [
        0,
        lstm_results["lstm"]["ap_max"] - lstm_results["lstm"]["ap_mean"],
        gnn_results["gnn"]["ap_max"] - gnn_results["gnn"]["ap_mean"],
        tgn_results["tgn"]["ap_max"] - tgn_results["tgn"]["ap_mean"],
        tgn_unc_results["ap_max"] - tgn_unc_results["ap_mean"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, aps, yerr=[lower_errs, upper_errs], capsize=5,
                   color=["gray", "steelblue", "seagreen", "goldenrod", "indianred"])
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Conjunction forecasting: model comparison (mean, min-max across seeds)")
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{ap:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_model_performance.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03_model_performance.png")
    print(f"ECE (uncertainty-aware model): {tgn_unc_results['ece_mean']:.4f}\n")


# ---------------------------------------------------------------------------
# 4. Output demonstration: what does a real prediction look like, and how
#    would it actually be consumed by an operator?
# ---------------------------------------------------------------------------
def demo_output_visualization():
    print("=" * 60)
    print("4. OUTPUT DEMONSTRATION")
    print("=" * 60)
    print("Simulates the actual operational output: a ranked list of")
    print("flagged conjunctions with calibrated risk + uncertainty --")
    print("this is what an operator dashboard would actually show,")
    print("sorted by risk, not a raw model score dump.\n")

    scenario = make_scenario(seed=42, id_offset=0)
    torch.manual_seed(0)

    train_scenarios = [make_scenario(seed=i, id_offset=i * 100) for i in range(3)]
    train_graphs = [
        build_dynamic_graph(s, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
        for s in train_scenarios
    ]
    model = TGNLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(15):
        for graph in train_graphs:
            optimizer.zero_grad()
            alphas, betas, labels = forward_and_score_with_uncertainty(model, graph)
            if alphas.numel() == 0:
                continue
            loss = beta_nll_loss(alphas, betas, torch.tensor(labels, dtype=torch.float))
            loss.backward()
            optimizer.step()

    model.eval()
    test_graph = build_dynamic_graph(scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
    with torch.no_grad():
        alphas, betas, labels = forward_and_score_with_uncertainty(model, test_graph)

    id_to_node = test_graph.node_ids
    pairs = list(test_graph.pair_labels.keys())[:len(alphas)]

    rows = []
    for (a_idx, b_idx), alpha, beta, label in zip(pairs, alphas, betas, labels):
        mean_risk = (alpha / (alpha + beta)).item()
        variance = (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))).item()
        rows.append((id_to_node[a_idx], id_to_node[b_idx], mean_risk, variance ** 0.5, label))

    rows.sort(key=lambda r: -r[2])

    print(f"{'Sat A':<8}{'Sat B':<8}{'Risk (mean)':<14}{'Uncertainty (std)':<20}{'Actual outcome'}")
    print("-" * 65)
    for a, b, risk, std, label in rows[:10]:
        outcome = "CONJUNCTION" if label == 1 else "no event"
        print(f"{a:<8}{b:<8}{risk:<14.4f}{std:<20.4f}{outcome}")

    print(f"\nThis IS the actual operator-facing output: a ranked alert")
    print(f"list, not a raw score dump -- an operator would review the")
    print(f"top N rows, prioritized by risk, with the uncertainty column")
    print(f"indicating which predictions warrant closer manual review")
    print(f"(high risk + high uncertainty = 'look at this one personally',")
    print(f"high risk + low uncertainty = 'model is confident, act on it').")

    with open(f"{OUTPUT_DIR}/04_sample_output.txt", "w") as f:
        f.write(f"{'Sat A':<8}{'Sat B':<8}{'Risk (mean)':<14}{'Uncertainty (std)':<20}{'Actual outcome'}\n")
        f.write("-" * 65 + "\n")
        for a, b, risk, std, label in rows:
            outcome = "CONJUNCTION" if label == 1 else "no event"
            f.write(f"{a:<8}{b:<8}{risk:<14.4f}{std:<20.4f}{outcome}\n")
    print(f"\nSaved full ranked list: {OUTPUT_DIR}/04_sample_output.txt")


if __name__ == "__main__":
    demo_dataset()
    demo_feature_importance()
    demo_model_performance()
    demo_output_visualization()
    print("\nAll demo artifacts saved to outputs/demo/")