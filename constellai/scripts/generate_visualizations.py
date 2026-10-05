"""
Visualization suite, built WITHOUT retraining anything:

1. Graph construction (M2) -- a real node/edge diagram with nodes
   colored by altitude shell and edges typed by pipeline stage
   (coarse-filter candidate vs. confirmed fine-screen edge, labeled
   with real miss distances), plus the two decision mechanisms:
   Stage A (altitude-band overlap) and Stage B (miss distance +
   closing rate).

2. Model performance -- plotted from the already-completed training
   run's results (hardcoded below), not retrained.

Deliberately NOT included here:
- Feature importance
- Calibration reliability diagram

Both require live per-example model predictions, which means either
retraining or loading a saved checkpoint. Add them once a TGN-lite
checkpoint is available.

Run:
    python -m constellai.scripts.generate_visualizations

Outputs:
    PNG files in outputs/visualizations/
"""

import math
import os
import random
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.orbital_mechanics.conjunction import screen_pair
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.regime import altitude_band
from constellai.orbital_mechanics.synthetic import make_circular_satellite


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=1, minutes=30)
STEP = timedelta(minutes=5)

MARGIN_KM = 50.0
THRESHOLD_KM = 300.0


# ---------------------------------------------------------------------------
# Model performance results
# ---------------------------------------------------------------------------
# These values come from the already-completed training run.
# Nothing is retrained by this visualization script.
#
# precision/recall/accuracy are None where the run did not report them.
#
# If you retrain later and obtain different numbers, update RESULTS.
# ---------------------------------------------------------------------------

RESULTS = {
    "naive persistence": {
        "ap_mean": 0.017,
        "ap_min": 0.017,
        "ap_max": 0.017,
        "precision": 1.000,
        "recall": 0.205,
        "accuracy": 0.987,
    },
    "LSTM": {
        "ap_mean": 0.791,
        "ap_min": 0.74,
        "ap_max": 0.82,
        "precision": 0.556,
        "recall": 0.909,
        "accuracy": 0.986,
    },
    "static GNN": {
        "ap_mean": 0.183,
        "ap_min": 0.17,
        "ap_max": 0.19,
        "precision": 0.163,
        "recall": 0.773,
        "accuracy": 0.929,
    },
    "TGN-lite\n(point est.)": {
        "ap_mean": 0.236,
        "ap_min": 0.18,
        "ap_max": 0.28,
        "precision": 0.048,
        "recall": 0.711,
        "accuracy": 0.835,
    },
    "TGN-lite\n(uncertainty)": {
        "ap_mean": 0.014,
        "ap_min": 0.01,
        "ap_max": 0.02,
        "precision": None,
        "recall": None,
        "accuracy": None,
    },
}

TGN_UNCERTAINTY_ECE = 0.3343


# ---------------------------------------------------------------------------
# Synthetic multi-shell scenario
# ---------------------------------------------------------------------------

def make_multishell_scenario(seed: int, n_per_shell: int = 7):
    """
    Create three well-separated altitude shells.

    This deliberately differs from the single tight altitude band used
    for TGN-lite training data.

    The separated shells allow the visualization to actually exercise
    Stage A altitude-band filtering.
    """

    rng = random.Random(seed)

    shells = [500.0, 1000.0, 1500.0]

    records = []
    sid = 0

    for altitude in shells:
        for i in range(n_per_shell):
            sign = 1 if i % 2 == 0 else -1

            records.append(
                make_circular_satellite(
                    satellite_id=sid,
                    altitude_km=altitude + rng.uniform(-2, 2),
                    inclination_rad=sign * math.radians(
                        45 + rng.uniform(-3, 3)
                    ),
                    raan_rad=rng.uniform(0, 0.3),
                    mean_anomaly_rad=rng.uniform(0, 2 * math.pi),
                )
            )

            sid += 1

    return records


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

def plot_graph_construction():
    print("=" * 60)
    print("1. GRAPH CONSTRUCTION (M2)")
    print("   Nodes, edges, and relationship types")
    print("=" * 60)

    records = make_multishell_scenario(seed=3, n_per_shell=7)

    candidates = candidate_pairs_by_regime(
        records,
        margin_km=MARGIN_KM,
    )

    edges_with_dist = []

    for a, b in candidates:
        sa = propagate_series(
            a,
            OBS_START,
            OBS_END,
            STEP,
        )

        sb = propagate_series(
            b,
            OBS_START,
            OBS_END,
            STEP,
        )

        event = screen_pair(sa, sb)

        if event.miss_distance_km < THRESHOLD_KM:
            edges_with_dist.append(
                (
                    a.satellite_id,
                    b.satellite_id,
                    event.miss_distance_km,
                )
            )

    total_possible = len(records) * (len(records) - 1) // 2

    print(
        f"satellites: {len(records)} "
        f"(3 shells x 7)"
    )

    print(
        f"exhaustive pairs: {total_possible}"
    )

    print(
        f"coarse-filter candidates "
        f"(Stage A survivors): {len(candidates)}"
    )

    print(
        f"fine-screen edges "
        f"(Stage B survivors): {len(edges_with_dist)}"
    )

    for a, b, distance in edges_with_dist:
        print(
            f"  edge {a}-{b}: "
            f"miss distance {distance:.1f} km"
        )

    print()

    # -----------------------------------------------------------------------
    # Node positions
    # -----------------------------------------------------------------------

    n = len(records)

    angles = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False,
    )

    pos = {
        record.satellite_id: (
            np.cos(angle),
            np.sin(angle),
        )
        for record, angle in zip(records, angles)
    }

    shell_colors = [
        "steelblue",
        "seagreen",
        "goldenrod",
    ]

    shell_names = [
        "Shell 1 (~500 km)",
        "Shell 2 (~1000 km)",
        "Shell 3 (~1500 km)",
    ]

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7.5),
    )

    ax = axes[0]

    # -----------------------------------------------------------------------
    # Relationship type 1:
    # Coarse-filter candidate
    # -----------------------------------------------------------------------

    for a, b in candidates:
        x1, y1 = pos[a.satellite_id]
        x2, y2 = pos[b.satellite_id]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="lightgray",
            linewidth=0.5,
            zorder=1,
        )

    # -----------------------------------------------------------------------
    # Relationship type 2:
    # Confirmed graph edge
    # -----------------------------------------------------------------------

    for sid_a, sid_b, distance in edges_with_dist:
        x1, y1 = pos[sid_a]
        x2, y2 = pos[sid_b]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="indianred",
            linewidth=2.5,
            zorder=2,
        )

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        ax.annotate(
            f"{distance:.0f} km",
            (mx, my),
            fontsize=6.5,
            color="indianred",
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.8,
            ),
        )

    # -----------------------------------------------------------------------
    # Nodes colored by altitude shell
    # -----------------------------------------------------------------------

    for i, record in enumerate(records):
        shell_idx = i // 7

        x, y = pos[record.satellite_id]

        ax.scatter(
            [x],
            [y],
            s=110,
            color=shell_colors[shell_idx],
            zorder=3,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.annotate(
            record.satellite_id,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
        )

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"M2 graph: {len(records)} nodes "
        f"(colored by altitude shell)\n"
        f"{len(candidates)} candidate pairs (gray), "
        f"{len(edges_with_dist)} confirmed edges "
        f"(red, labeled by miss distance)"
    )

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------

    from matplotlib.lines import Line2D

    legend_elements = (
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=9,
                label=name,
            )
            for color, name in zip(
                shell_colors,
                shell_names,
            )
        ]
        + [
            Line2D(
                [0],
                [0],
                color="lightgray",
                lw=1.5,
                label="Candidate pair (Stage A only)",
            ),
            Line2D(
                [0],
                [0],
                color="indianred",
                lw=2.5,
                label="Graph edge (passed Stage B)",
            ),
        ]
    )

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        fontsize=7.5,
        ncol=2,
    )

    # -----------------------------------------------------------------------
    # Pair count plot
    # -----------------------------------------------------------------------

    ax = axes[1]

    bars = ax.bar(
        [
            "Exhaustive\n(N*(N-1)/2)",
            "Coarse-filter\ncandidates\n(Stage A)",
            "Fine-screen\nedges\n(Stage B)",
        ],
        [
            total_possible,
            len(candidates),
            len(edges_with_dist),
        ],
        color=[
            "gray",
            "goldenrod",
            "indianred",
        ],
    )

    ax.set_ylabel("Pair count")
    ax.set_title("M2 pipeline: pair count at each stage")

    values = [
        total_possible,
        len(candidates),
        len(edges_with_dist),
    ]

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(value),
            ha="center",
        )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )

    output_path = (
        f"{OUTPUT_DIR}/01_graph_construction.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}\n")


# ---------------------------------------------------------------------------
# 1b. Stage A mechanism: altitude-band overlap
# ---------------------------------------------------------------------------

def plot_stage_a_mechanism():
    print("=" * 60)
    print("1b. STAGE A MECHANISM: altitude-band overlap")
    print("=" * 60)

    print(
        "The actual decision rule: a candidate pair exists only if "
        "the two satellites' [perigee, apogee] bands -- inflated "
        "by the margin -- overlap."
    )

    print(
        "Necessary, not sufficient: non-overlap guarantees safety; "
        "overlap only earns a pair the expensive Stage B check.\n"
    )

    rng = random.Random(3)

    records = []

    sid = 0

    for altitude in [
        500.0,
        500.0,
        510.0,
        1000.0,
        1000.0,
    ]:
        records.append(
            make_circular_satellite(
                satellite_id=sid,
                altitude_km=altitude + rng.uniform(-1, 1),
                inclination_rad=math.radians(45),
                raan_rad=rng.uniform(0, 0.3),
                mean_anomaly_rad=rng.uniform(
                    0,
                    2 * math.pi,
                ),
            )
        )

        sid += 1

    bands = {
        record.satellite_id: altitude_band(record)
        for record in records
    }

    candidates = candidate_pairs_by_regime(
        records,
        margin_km=MARGIN_KM,
    )

    print(
        f"{len(candidates)} candidate pairs found "
        f"among {len(records)} satellites\n"
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    # -----------------------------------------------------------------------
    # Plot altitude bands
    # -----------------------------------------------------------------------

    for i, record in enumerate(records):
        band = bands[record.satellite_id]

        # Margin-inflated band
        ax.plot(
            [
                band.perigee_alt_km - MARGIN_KM,
                band.apogee_alt_km + MARGIN_KM,
            ],
            [i, i],
            linewidth=2,
            color="lightsteelblue",
            solid_capstyle="butt",
            zorder=1,
        )

        # True band
        ax.plot(
            [
                band.perigee_alt_km,
                band.apogee_alt_km,
            ],
            [i, i],
            linewidth=8,
            color="steelblue",
            solid_capstyle="butt",
            zorder=2,
        )

        ax.text(
            band.apogee_alt_km + MARGIN_KM + 15,
            i,
            record.satellite_id,
            va="center",
            fontsize=9,
        )

    # -----------------------------------------------------------------------
    # Candidate-pair indicators
    # -----------------------------------------------------------------------

    for a, b in candidates:
        indices = [
            idx
            for idx, record in enumerate(records)
            if record.satellite_id
            in (
                a.satellite_id,
                b.satellite_id,
            )
        ]

        i, j = indices

        mid_alt = (
            bands[a.satellite_id].perigee_alt_km
            + bands[b.satellite_id].perigee_alt_km
        ) / 2

        ax.annotate(
            "",
            xy=(mid_alt, min(i, j) - 0.3),
            xytext=(mid_alt, max(i, j) + 0.3),
            arrowprops=dict(
                arrowstyle="-",
                color="indianred",
                linewidth=1.3,
                linestyle=":",
            ),
        )

    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([])

    ax.set_xlabel("Altitude (km)")

    ax.set_title(
        f"Stage A: altitude-band overlap decides candidacy\n"
        f"(dark = true band, light = margin-inflated, "
        f"dotted red = {len(candidates)} candidate pairs)"
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="steelblue",
            lw=6,
            label="True altitude band",
        ),
        Line2D(
            [0],
            [0],
            color="lightsteelblue",
            lw=3,
            label=f"+/- {MARGIN_KM:.0f} km margin",
        ),
        Line2D(
            [0],
            [0],
            color="indianred",
            lw=1.3,
            linestyle=":",
            label="Overlap -> candidate pair",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )

    output_path = (
        f"{OUTPUT_DIR}/01b_stage_a_mechanism.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}\n")


# ---------------------------------------------------------------------------
# 1c. Stage B mechanism: miss distance decision
# ---------------------------------------------------------------------------

def plot_stage_b_mechanism():
    print("=" * 60)
    print("1c. STAGE B MECHANISM: miss distance decides edge inclusion")
    print("=" * 60)

    records = make_multishell_scenario(
        seed=3,
        n_per_shell=7,
    )

    candidates = candidate_pairs_by_regime(
        records,
        margin_km=MARGIN_KM,
    )

    miss_distances = []
    closing_rates = []
    is_edge = []

    for a, b in candidates:
        sa = propagate_series(
            a,
            OBS_START,
            OBS_END,
            STEP,
        )

        sb = propagate_series(
            b,
            OBS_START,
            OBS_END,
            STEP,
        )

        event = screen_pair(
            sa,
            sb,
        )

        miss_distances.append(
            event.miss_distance_km
        )

        closing_rates.append(
            event.relative_speed_km_s
        )

        is_edge.append(
            event.miss_distance_km < THRESHOLD_KM
        )

    print(
        f"candidates evaluated: {len(candidates)}, "
        f"edges found: {sum(is_edge)}\n"
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    colors = [
        "indianred" if edge else "lightgray"
        for edge in is_edge
    ]

    ax.scatter(
        miss_distances,
        closing_rates,
        c=colors,
        s=30,
        edgecolor="black",
        linewidth=0.3,
        zorder=3,
    )

    ax.axvline(
        THRESHOLD_KM,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.axvspan(
        0,
        THRESHOLD_KM,
        color="indianred",
        alpha=0.05,
        zorder=1,
    )

    ax.set_xlabel(
        "Miss distance (km)"
    )

    ax.set_ylabel(
        "Relative speed at TCA (km/s)"
    )

    ax.set_title(
        f"Stage B: which coarse-filter candidates become graph edges\n"
        f"({sum(is_edge)}/{len(candidates)} candidates qualify)"
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="indianred",
            markersize=8,
            label="Graph edge (qualifies)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="lightgray",
            markersize=8,
            label="Rejected (too far)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1,
            label=(
                f"distance threshold = "
                f"{THRESHOLD_KM:.0f} km"
            ),
        ),
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=8,
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )

    output_path = (
        f"{OUTPUT_DIR}/01c_stage_b_mechanism.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}\n")


# ---------------------------------------------------------------------------
# 2. M1 stage result: real propagated trajectory
# ---------------------------------------------------------------------------

def plot_m1_propagation():
    print("=" * 60)
    print("2. M1 STAGE RESULT: propagated separation over time")
    print("=" * 60)

    records = make_multishell_scenario(
        seed=1,
        n_per_shell=2,
    )

    a, b = records[0], records[1]

    horizon_end = (
        OBS_END
        + (OBS_END - OBS_START)
    )

    states_a = propagate_series(
        a,
        OBS_START,
        horizon_end,
        timedelta(minutes=2),
    )

    states_b = propagate_series(
        b,
        OBS_START,
        horizon_end,
        timedelta(minutes=2),
    )

    minutes = [
        (
            state.epoch - OBS_START
        ).total_seconds() / 60
        for state in states_a
    ]

    separations = [
        float(
            np.linalg.norm(
                state_a.position_km
                - state_b.position_km
            )
        )
        for state_a, state_b
        in zip(states_a, states_b)
    ]

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    ax.plot(
        minutes,
        separations,
        color="steelblue",
        linewidth=1.5,
    )

    ax.axvline(
        (
            OBS_END - OBS_START
        ).total_seconds() / 60,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=(
            "observation window ends / "
            "horizon window begins"
        ),
    )

    ax.set_xlabel(
        "Minutes since window start"
    )

    ax.set_ylabel(
        "Separation (km)"
    )

    ax.set_title(
        f"M1: propagated separation, "
        f"satellite {a.satellite_id} "
        f"vs {b.satellite_id}"
    )

    ax.legend(
        fontsize=8
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )

    output_path = (
        f"{OUTPUT_DIR}/02_m1_propagation.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}\n")


# ---------------------------------------------------------------------------
# 3. Model performance
# ---------------------------------------------------------------------------

def plot_model_comparison():
    print("=" * 60)
    print("3. MODEL PERFORMANCE")
    print("   From completed training run -- no retraining")
    print("=" * 60)

    names = list(RESULTS.keys())

    means = [
        RESULTS[name]["ap_mean"]
        for name in names
    ]

    lower = [
        RESULTS[name]["ap_mean"]
        - RESULTS[name]["ap_min"]
        for name in names
    ]

    upper = [
        RESULTS[name]["ap_max"]
        - RESULTS[name]["ap_mean"]
        for name in names
    ]

    colors = [
        "gray",
        "steelblue",
        "seagreen",
        "goldenrod",
        "indianred",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5.5),
    )

    # -----------------------------------------------------------------------
    # AP plot
    # -----------------------------------------------------------------------

    ax = axes[0]

    bars = ax.bar(
        names,
        means,
        yerr=[lower, upper],
        capsize=5,
        color=colors,
    )

    ax.set_ylabel(
        "Average Precision (AP)"
    )

    ax.set_title(
        "AP: mean, with real [min, max] across seeds"
    )

    for bar, value in zip(
        bars,
        means,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    ax.tick_params(
        axis="x",
        labelsize=8,
    )

    # -----------------------------------------------------------------------
    # Precision / Recall / Accuracy
    # -----------------------------------------------------------------------

    ax = axes[1]

    metrics = [
        "precision",
        "recall",
        "accuracy",
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.15

    plotted_names = [
        name
        for name in names
        if RESULTS[name]["precision"] is not None
    ]

    for i, name in enumerate(
        plotted_names
    ):
        values = [
            RESULTS[name][metric]
            for metric in metrics
        ]

        offset = (
            (
                i
                - len(plotted_names) / 2
            )
            * width
            + width / 2
        )

        ax.bar(
            x + offset,
            values,
            width,
            label=name.replace(
                "\n",
                " ",
            ),
            color=colors[names.index(name)],
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            metric.capitalize()
            for metric in metrics
        ]
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Precision / Recall / Accuracy by model\n"
        "(TGN-lite uncertainty-aware omitted -- "
        "not logged in that run)"
    )

    ax.legend(
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )

    output_path = (
        f"{OUTPUT_DIR}/03_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")

    print(
        f"(TGN-lite uncertainty-aware ECE: "
        f"{TGN_UNCERTAINTY_ECE} -- reported in text; "
        f"no raw scores available to plot a "
        f"reliability diagram)\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plot_graph_construction()
    plot_stage_a_mechanism()
    plot_stage_b_mechanism()
    plot_m1_propagation()
    plot_model_comparison()

    print(
        "All visualizations saved to "
        "outputs/visualizations/"
    )


if __name__ == "__main__":
    main()
