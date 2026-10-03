"""Create figures from current graph functions and saved demo output only."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

from constellai.graph.build import build_graph
from constellai.graph.filters import candidate_pairs_by_regime
from constellai.graph.validation import run_false_negative_gate
from constellai.scripts.run_false_negative_gate import (
    ALTITUDE_KM, ALTITUDE_SPREAD_KM, DISTANCE_THRESHOLD_KM, OBS_END, OBS_START,
    REGIME_MARGIN_KM, STEP, build_scenario,
)


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
NAVY = "#263746"
BLUE = "#2f6f9f"
GRAY = "#9eabb5"
AMBER = "#e09f3e"
RED = "#b23a48"
GREEN = "#39805b"


def svg_start(width: int, height: int, title: str, description: str = "") -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<title>{escape(title)}</title>', f'<desc>{escape(description)}</desc>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263746}.title{font-size:24px;font-weight:700}.subtitle{font-size:13px}.label{font-size:13px}.small{font-size:11px}.node-label{font-size:10px}</style>']


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "middle", fill: str | None = None) -> str:
    color = f' fill="{fill}"' if fill else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}"{color}>{escape(value)}</text>'


def save_svg(name: str, parts: list[str]) -> None:
    parts.append("</svg>")
    (FIGURES / name).write_text("\n".join(parts), encoding="utf-8")


def graph_panel(parts: list[str], records, edges, x0: int, title: str, edge_color: str) -> None:
    cx, cy, radius = x0 + 180, 260, 145
    n = len(records)
    positions = {r.satellite_id: (cx + radius * math.cos(2 * math.pi * i / n - math.pi / 2),
                                  cy + radius * math.sin(2 * math.pi * i / n - math.pi / 2))
                 for i, r in enumerate(records)}
    parts.append(text(cx, 65, title, "label"))
    parts.append(text(cx, 88, f"{n} satellites · {len(edges)} edges", "small"))
    opacity = 0.28 if len(edges) > 100 else 0.75
    for a, b in edges:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{edge_color}" stroke-width="{0.7 if len(edges)>100 else 1.5}" opacity="{opacity}"/>')
    for sat_id, (x, y) in positions.items():
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{BLUE}" stroke="white" stroke-width="1.4"/>')
        angle = math.atan2(y - cy, x - cx)
        parts.append(text(x + 16 * math.cos(angle), y + 16 * math.sin(angle) + 3, str(sat_id), "node-label"))
    parts.append(f'<circle cx="{cx-46}" cy="466" r="6" fill="{BLUE}"/>')
    parts.append(text(cx-34, 470, "Satellite", "small", "start"))
    parts.append(f'<line x1="{cx+22}" y1="466" x2="{cx+49}" y2="466" stroke="{edge_color}" stroke-width="2"/>')
    parts.append(text(cx+56, 470, "Candidate connection", "small", "start"))


def make_m2_graph_figures() -> None:
    records = build_scenario(seed=0)
    possible = [(records[i].satellite_id, records[j].satellite_id)
                for i in range(len(records)) for j in range(i + 1, len(records))]
    candidates = candidate_pairs_by_regime(records, margin_km=REGIME_MARGIN_KM)
    coarse = [(a.satellite_id, b.satellite_id) for a, b in candidates]
    graph = build_graph(records, OBS_START, OBS_END, STEP,
                        regime_margin_km=REGIME_MARGIN_KM,
                        distance_threshold_km=DISTANCE_THRESHOLD_KM)
    fine = [edge.satellite_ids for edge in graph.edges]

    title = "M2 candidate graph stages · existing synthetic gate scenario, seed 0"
    parts = svg_start(1120, 545, title, "Topology diagrams for exhaustive pairs, coarse altitude candidates, and fine-screen edges. Circular node positions do not represent orbital position.")
    parts.append(text(560, 35, title, "title"))
    graph_panel(parts, records, possible, 0, "Before coarse filtering", GRAY)
    graph_panel(parts, records, coarse, 375, "After altitude-regime filtering", AMBER)
    graph_panel(parts, records, fine, 750, "After fine relative-dynamics screen", RED)
    parts.append(text(560, 520,
                      f"{len(records)} satellites · {ALTITUDE_KM:.0f} ± {ALTITUDE_SPREAD_KM:.0f} km · {REGIME_MARGIN_KM:g} km regime margin · {DISTANCE_THRESHOLD_KM:g} km screen distance · topology only, not orbital geometry",
                      "small"))
    save_svg("m2_graph_stages.svg", parts)

    counts = [len(possible), len(coarse), len(fine)]
    labels = ["Exhaustive pairs", "Coarse candidates", "Fine-screen edges"]
    colors = ["#738290", AMBER, RED]
    max_count = max(counts, default=1)
    parts = svg_start(900, 520, "M2 graph sparsification · current pipeline")
    parts.append(text(450, 42, "M2 graph sparsification · current pipeline", "title"))
    parts.append(text(450, 474, "Pair count", "label"))
    left, gap, bar_w = 110, 165, 145
    for i, (value, label, color) in enumerate(zip(counts, labels, colors)):
        height = 330 * value / max_count if max_count else 0
        x = left + i * (bar_w + gap)
        y = 420 - height
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{height:.1f}" fill="{color}"/>')
        parts.append(text(x + bar_w / 2, y - 12, str(value), "label"))
        parts.append(text(x + bar_w / 2, 443, label, "small"))
    parts.append(text(450, 495,
                      f"Coarse retained: {len(coarse)/len(possible):.1%} · Fine retained: {len(fine)/len(possible):.1%}" if possible else "No pairs",
                      "small"))
    save_svg("m2_sparsification.svg", parts)

    gate = run_false_negative_gate(records, OBS_START, OBS_END, STEP,
                                   distance_threshold_km=DISTANCE_THRESHOLD_KM,
                                   regime_margin_km=REGIME_MARGIN_KM)
    missed = len(gate.missed_pairs)
    retained = len(gate.baseline_flagged_ids) - missed
    values = [missed, retained]
    labels = ["False negatives", "Retained baseline pairs"]
    colors = [RED, GREEN]
    parts = svg_start(900, 360, "M2 false-negative gate · existing validation procedure")
    parts.append(text(450, 42, "M2 false-negative gate · existing validation procedure", "title"))
    parts.append(text(450, 110, f"Exhaustive baseline flagged {len(gate.baseline_flagged_ids)} relevant pairs", "label"))
    x, y, total_width = 90, 155, 720
    total = max(len(gate.baseline_flagged_ids), 1)
    for value, label, color in zip(values, labels, colors):
        width = total_width * value / total
        if width:
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="62" fill="{color}"/>')
            parts.append(text(x + width / 2, y + 39, str(value), "label", fill="white"))
        x += width
    for i, (label, color) in enumerate(zip(labels, colors)):
        lx = 140 + i * 260
        parts.append(f'<rect x="{lx}" y="260" width="14" height="14" fill="{color}"/>')
        parts.append(text(lx + 22, 272, label, "small", "start"))
    parts.append(text(450, 315, f"Existing gate false-negative rate: {gate.false_negative_rate:.3f} · single scenario, seed 0", "small"))
    save_svg("m2_false_negative_validation.svg", parts)


def make_architecture_diagram() -> None:
    stages = [
        ("TLE / orbital data", "Input records"),
        ("TLE validation", "Parse and validate elements"),
        ("SGP4 propagation", "State vectors over time"),
        ("Dynamic graph construction", "M2 candidate screening"),
        ("TGNN conjunction forecasting", "M3 temporal risk model"),
        ("Risk + uncertainty", "Forecast outputs"),
        ("Constrained MARL", "M4 maneuver proposal"),
        ("Physics-based safety filter", "M5 action check"),
        ("Maneuver / monitoring", "Execute or continue monitoring"),
    ]
    colors = ["#e8f1f8", "#e8f1f8", "#e8f1f8", "#e9f3ec", "#f7f0df", "#f7f0df", "#f3eaf4", "#f8e9e9", "#e8f1f8"]
    parts = svg_start(760, 980, "ConstellAI end-to-end architecture", "Explanatory system diagram, not experimental result.")
    parts.append(text(380, 38, "ConstellAI end-to-end architecture", "title"))
    for i, ((name, desc), color) in enumerate(zip(stages, colors)):
        y = 76 + i * 94
        parts.append(f'<rect x="160" y="{y}" width="440" height="66" rx="12" fill="{color}" stroke="#52616b" stroke-width="1.3"/>')
        parts.append(text(380, y + 27, name, "label"))
        parts.append(text(380, y + 48, desc, "small"))
        if i < len(stages) - 1:
            parts.append(f'<line x1="380" y1="{y+67}" x2="380" y2="{y+91}" stroke="#52616b" stroke-width="2" marker-end="url(#arrow)"/>')
    # Arrowhead marker definition.
    parts.insert(5, '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#52616b"/></marker></defs>')
    parts.append(text(380, 944, "Explanatory architecture figure; not an experimental result.", "small"))
    save_svg("constellai_architecture.svg", parts)


if __name__ == "__main__":
    make_m2_graph_figures()
    make_architecture_diagram()
    print(f"Saved SVG visualizations to {FIGURES}")
