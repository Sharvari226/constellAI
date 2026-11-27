from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

DEMO_ROOT=Path(__file__).resolve().parents[1]
REPO_ROOT=DEMO_ROOT.parent
sys.path.insert(0,str(DEMO_ROOT))
from dashboard.data_loader import load_snapshot, load_final_results

st.set_page_config(page_title="ConstellAI | Mission Control",page_icon="🛰️",layout="wide")
st.markdown("""
<style>
:root {
  --ink: #172b45;
  --muted: #52657b;
  --line: #d8e1eb;
  --surface: #ffffff;
  --canvas: #f3f6fa;
  --accent: #1769aa;
}
.stApp { background: var(--canvas); color: var(--ink); }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1540px; }
h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.02em; }
p, label, .stMarkdown, .stCaption { color: var(--muted); }
.hero {
  padding: 1.8rem 2rem;
  margin-bottom: 1.2rem;
  border: 1px solid #cbd9e7;
  border-left: 5px solid var(--accent);
  border-radius: 14px;
  background: linear-gradient(115deg, #ffffff 0%, #eaf3fb 100%);
  box-shadow: 0 5px 18px rgba(28, 54, 82, .06);
}
.hero h1 { margin: 0 0 .45rem 0; font-size: 2rem; }
.hero p { margin: 0 0 .55rem 0; color: #3f5872; font-size: 1rem; }
.small-muted { color: #526b83; font-size: .86rem; }
[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  box-shadow: 0 3px 12px rgba(28, 54, 82, .045);
}
[data-testid="stMetricLabel"] { color: #52657b !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #142f4b !important; font-weight: 750; }
[data-testid="stMetricDelta"] { color: #1769aa !important; }
div[data-testid="stTabs"] button {
  color: #425a73;
  font-weight: 600;
  padding-top: .75rem;
  padding-bottom: .75rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: #145d98;
  border-bottom-color: #1769aa;
}
div[data-testid="stAlert"] { border-radius: 10px; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
}
.stPlotlyChart { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: .35rem; }
hr { border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<h1>🛰️ ConstellAI <span style="font-weight:500">/ Mission Control</span></h1>'
    '<p>Satellite conjunction screening · uncertainty-aware risk signals · fuel-aware maneuver evaluation</p>'
    '<span class="small-muted">Research demonstration — not an operational collision-avoidance service</span>'
    '</div>',
    unsafe_allow_html=True
)
meta,sat,candidates,edges=load_snapshot(); results=load_final_results()
tabs=st.tabs(["Overview","Live Starlink snapshot","Sparse conjunction graph","Risk & maneuver evaluation","Project status"])
with tabs[0]:
    st.subheader("System overview")
    st.write("The dashboard separates the real CelesTrak snapshot demonstration from the project's controlled synthetic evaluation. They are different evidence streams and are not combined into one performance claim.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Starlink snapshot",f"{len(sat):,}" if not sat.empty else "Not prepared")
    c2.metric("Coarse candidates",f"{len(candidates):,}" if not candidates.empty else "—")
    c3.metric("Screened edges",f"{len(edges):,}" if not edges.empty else "—")
    c4.metric("Frozen policies",f"{len(results):,}" if not results.empty else "—")
    st.info("Start by preparing the CelesTrak snapshot using the command in demo/README.md. The dashboard never fabricates missing snapshot data.")
    if not results.empty:
        eligible=results[results["meets_target"]==True].sort_values("final_dv")
        st.markdown("**Frozen Stream-7 comparison — synthetic evaluation**")
        st.dataframe(results[["family","policy","n","final_free","final_dv","meets_target"]],use_container_width=True,hide_index=True)
        if not eligible.empty: st.success(f"Lowest-Δv policy among policies meeting the 0.98 target: {eligible.iloc[0]['family']} ({eligible.iloc[0]['policy']}) — {eligible.iloc[0]['final_dv']:.3f} m/s mean Δv per agent.")
with tabs[1]:
    st.subheader("CelesTrak Starlink · real TLE snapshot")
    if meta:
        st.caption(f"Retrieved: {meta['retrieved_at_utc']} · propagated epoch: {meta['snapshot_epoch_utc']} · frame: TEME")
        a,b,c=st.columns(3);a.metric("Satellites",meta["satellites_parsed"]);b.metric("Inclination median",f"{sat.inclination_deg.median():.2f}°" if not sat.empty else "—");c.metric("Altitude median",f"{sat.perigee_alt_km.median():.1f} km" if not sat.empty else "—")
        if not sat.empty:
            fig=px.scatter(
                sat, x="raan_deg", y="inclination_deg", color="perigee_alt_km",
                hover_name="name", hover_data=["satellite_id", "apogee_alt_km"],
                title="Orbital element snapshot · RAAN vs inclination",
                labels={"raan_deg": "RAAN (°)", "inclination_deg": "Inclination (°)",
                        "perigee_alt_km": "Perigee altitude (km)"}
            )
            fig.update_layout(
                template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                font_color="#263e57", title_font_color="#172b45",
                margin=dict(l=24, r=24, t=60, b=24)
            )
            fig.update_xaxes(gridcolor="#e6edf4", zerolinecolor="#d8e1eb")
            fig.update_yaxes(gridcolor="#e6edf4", zerolinecolor="#d8e1eb")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sat, use_container_width=True, hide_index=True)
    else: st.warning("No snapshot found. Run the preparation command first.")
with tabs[2]:
    st.subheader("M2 · two-stage sparse conjunction screening")
    if meta:
        n=meta["satellites_parsed"]; total=meta["exhaustive_pairs"]; kept=meta["coarse_candidates"]
        x,y,z=st.columns(3);x.metric("Exhaustive pairs",f"{total:,}");y.metric("Altitude-regime candidates",f"{kept:,}");z.metric("Fine-screen edges",f"{meta['screened_edges']:,}")
        st.write(f"Coarse filtering retained {100*kept/total:.2f}% of all possible pairs." if total else "No pairs available.")
        pair_counts = pd.DataFrame({
            "Stage": ["All unordered pairs", "Altitude candidates", "Fine-screen edges"],
            "Pairs": [total, kept, meta["screened_edges"]]
        })
        fig = px.bar(
            pair_counts, x="Stage", y="Pairs",
            title="Pair-count reduction through the M2 pipeline",
            labels={"Pairs": "Number of pairs", "Stage": ""}
        )
        fig.update_layout(
            template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
            font_color="#263e57", title_font_color="#172b45",
            margin=dict(l=24, r=24, t=60, b=24), showlegend=False
        )
        fig.update_yaxes(gridcolor="#e6edf4")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The coarse filter and fine screen call the repository's existing M2 functions. This snapshot is a demonstration; the research validation gate uses its documented synthetic scenario and must not be conflated with these live-catalog counts.")
        if not edges.empty: st.dataframe(edges.sort_values("miss_distance_km"),use_container_width=True,hide_index=True)
    else: st.warning("Prepare a Starlink snapshot to populate M2 results.")
with tabs[3]:
    st.subheader("Frozen trigger-ladder results · Stream 7")
    st.caption("These are committed synthetic evaluation results, not results from the live Starlink snapshot.")
    if not results.empty:
        target=st.slider("Collision-free target",0.90,1.00,0.98,0.005)
        view=results.copy();view["meets_selected_target"]=view["final_free"]>=target
        filtered_view=view[view["meets_selected_target"]].copy()
        if filtered_view.empty:
            st.info(f"No policies meet the selected target of {target:.3f}.")
        fig = px.scatter(
            filtered_view, x="final_free", y="final_dv", color="family", text="family",
            hover_data=["policy", "n", "free_lo", "free_hi", "dv_lo", "dv_hi"],
            title="Collision-free rate vs mean Δv · lower fuel use is better",
            labels={"final_free": "Collision-free rate", "final_dv": "Mean Δv (m/s)",
                    "family": "Policy family"}
        )
        fig.update_layout(
            template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
            font_color="#263e57", title_font_color="#172b45",
            margin=dict(l=24, r=24, t=60, b=24)
        )
        fig.update_xaxes(tickformat=".0%", gridcolor="#e6edf4")
        fig.update_yaxes(gridcolor="#e6edf4")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            filtered_view[["family", "policy", "n", "final_free", "free_lo", "free_hi",
                  "final_dv", "dv_lo", "dv_hi", "meets_selected_target"]],
            use_container_width=True, hide_index=True
        )
        st.caption("The frozen report's primary target is 0.98. Changing the slider only filters the displayed results; it does not rerun threshold selection or alter the frozen experiment.")
    else: st.warning("Committed artifacts/04_final_results/final_stream7_report.csv was not found.")
with tabs[4]:
    st.subheader("Implementation boundaries")
    st.markdown("""
| Area | What this demo shows | Evidence boundary |
|---|---|---|
| M1 orbital mechanics | CelesTrak TLE parsing, SGP4 propagation, orbital elements | Real snapshot at one UTC epoch |
| M2 graph screening | Altitude-regime candidate filter and conjunction fine screen | Real snapshot; separate from synthetic false-negative gate |
| Tracking / forecasting | Existing project modules | Not presented as live Starlink tracking unless a matching end-to-end pipeline is explicitly run |
| M3 learned forecasting | GNN/LSTM/TGN code exists | Stable visualization README reports no committed M3 result files/predictions |
| M4 MARL | RL implementations | No committed episode-return/fuel/trajectory result plots in stable baseline |
| M5 safety filter | CBF-style safety filter implementation | No committed proposed/filtered action intervention dataset |
| Trigger ladder | Frozen policy evaluation | Stream-7 synthetic evaluation artifacts |
""")
    st.warning("ConstellAI is a research prototype. It does not provide operational conjunction warnings, certified maneuver commands, or formal safety guarantees.")
