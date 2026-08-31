<div align="center">

# ConstellAI

**Anticipatory, physics-informed orbital intelligence for space traffic management**

[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()
[![Tests](https://img.shields.io/badge/tests-46%2F46%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

</div>

---

## Table of Contents

- [What This Is](#what-this-is)
- [The Novelty Claim, Precisely](#the-novelty-claim-precisely)
- [Build Status](#build-status)
- [Architecture](#architecture)
- [How Each Stage Works, In Detail](#how-each-stage-works-in-detail)
- [Why Each Design Choice Is Defensible](#why-each-design-choice-is-defensible)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Baseline & Ablation Ladders](#baseline--ablation-ladders)
- [What This Project Never Claims](#what-this-project-never-claims)
- [Team](#team)
- [License](#license)

---

## What This Is

Growing satellite constellations are increasing close-approach ("conjunction") events
faster than current space traffic management practice can absorb. This is not
fundamentally a lack-of-prediction problem — operational literature on the subject
consistently identifies the actual bottleneck as **operator alert workload**: as
constellations scale, the number of "these two satellites might collide" alerts grows
faster than humans can triage them.

ConstellAI addresses this with a three-part pipeline:

1. A **continuous-time temporal graph neural network** forecasts conjunctions multiple
   days ahead, with **calibrated uncertainty** rather than a flat point estimate.
2. A **constrained multi-agent reinforcement learning** policy plans coordinated,
   fuel-optimal avoidance maneuvers ahead of time — treating "don't collide" as a
   **hard safety constraint**, not a soft reward penalty.
3. A **physics-based safety filter** provides a final, real-time backstop before any
   maneuver actually executes.

Every module is built on top of authoritative orbital mechanics — the machine learning
components augment prediction, prioritization, and coordination; they never override
physics.

## The Novelty Claim, Precisely

It is important to state this narrowly and honestly, because a broader claim doesn't
survive contact with the existing literature.

**Not the claim:** "TGNN + MARL for space traffic management" — this combination, and
even *scalable graph-based safety control* specifically, is already active, published
work. **GCBF+** (Zhang et al., IEEE Transactions on Robotics, 2025) provides a
GNN-parameterized control barrier function with a *formal safety proof*, scaling to
1000+ agents — a stronger, more general version of what an early draft of this project
proposed as its safety-layer contribution. GS-MARL (2026) and a 2025 paper on
communication-constrained MARL for satellite collision avoidance occupy adjacent
ground as well.

**The actual claim:** existing scalable multi-agent safety systems (GCBF+, GS-MARL) are
**reactive** — they act on current relative state, in real time. ConstellAI's
contribution is **anticipatory, multi-day, uncertainty-calibrated forecasting** driving
safety-constrained coordination, so maneuvers get planned fuel-optimally *ahead of
time*, rather than forced reactively once risk is already imminent. This distinction is
argued explicitly, with citations, in the paper's related-work section — not assumed or
asserted around.

## Build Status

| Layer | Status | What "done" means here |
|---|---|---|
| **M1 — Orbital Mechanics Core** | ✅ Complete | TLE parsing, SGP4 propagation, validated against Vallado's published reference state vectors to floating-point tolerance — not an internal consistency check, an external ground-truth check |
| **Step 3 — Non-ML Baseline** | ✅ Complete | Exhaustive O(N²) conjunction screener; the number every model below has to beat, recorded before any ML exists to be tempted to tune toward it |
| **M2 — Sparse Dynamic Graph** | ✅ Complete | Coarse altitude-band filter + fine relative-dynamics screen, combined into one canonical `build_graph()` pipeline; false-negative gate *mechanism* verified correct (a real-constellation run to record the actual false-negative rate is the one remaining open item) |
| **M3 — Temporal Forecasting (TGNN)** | 🔶 Near complete | Continuous-time graph network + uncertainty head — in progress by the modeling team |
| **M4 — Constrained MARL** | 🔶 Near complete | CMDP-formulated coordination policy — in progress by the modeling team |
| **M5 — Safety Filter** | ⬜ Not started | Planned as an explicit, cited comparison against GCBF+, not an independently "novel" mechanism |
| **M6 — Simulation & Evaluation** | 🔶 Partial | Baseline scenario tooling exists; full Monte Carlo evaluation harness (≥500 trials) pending |
| **M7 — Infra / MLOps** | ⬜ Not started | CI/CD, experiment tracking (MLflow/W&B), config-driven reproducibility |

**Test suite: 46/46 passing.** Run `python -m pytest tests/ -v` to verify locally — see
[Testing](#testing) below.

## Architecture

```
TLE data (public orbital element sets)
   │
   ▼
M1  Orbit determination + SGP4 propagation
    Perturbations: J2 (Earth oblateness), atmospheric drag, solar radiation pressure
   │
   ▼
M2  Sparse dynamic graph construction
    Stage A — coarse filter:  altitude-band overlap (sweep-line, O(N log N))
    Stage B — fine screen:    TCA + miss distance + closing rate on survivors
   │
   ▼
M3  Continuous-time temporal graph network (TGNN)
    Per-satellite memory, updated on each graph event (not fixed time-slices)
    Point-process head → time-to-closest-approach
    Uncertainty head    → calibrated distribution, not a point estimate
    Output: P(collision), predicted TCA, calibrated uncertainty
   │
   ├──────────────────────────────────┐
   ▼                                  ▼
M4  Constrained Multi-Agent RL     (flags high-risk subgraph
    CMDP formulation:                 for M5, below)
      maximize  mission utility − fuel cost        │
      subject to  P(collision) ≤ ε                 │
    (a hard constraint, not a reward penalty)       │
   │                                                │
   ▼                                                ▼
Proposed maneuver ─────────────► M5  Physics-based safety filter (CBF/HOCBF-style)
                                      Real-time QP-based check
                                      Scoped ONLY to M3's flagged subgraph
                                      Explicitly compared against GCBF+
   │
   ▼
Execute, or apply minimal-deviation correction

──────────────────────────────────────────────────────────────
Running throughout every stage:
  M6 — independent ground-truth simulator + Monte Carlo evaluation
  M7 — experiment tracking, CI/CD, reproducibility infrastructure
```

## How Each Stage Works, In Detail

### M1 — Orbital Mechanics Core

Satellites publish orbital state as **TLEs** (Two-Line Element sets) — six Keplerian
parameters plus drag terms, at a reference epoch. **SGP4** propagates this forward in
time, accounting for:

- **J2 perturbation** — Earth's oblateness causes orbital precession; the single
  largest perturbation for most orbits.
- **Atmospheric drag** — significant in LEO, decays orbits over time, and is itself
  uncertain (density varies with solar activity).
- **Solar radiation pressure** — smaller effect, more relevant for high
  area-to-mass-ratio objects.

This module is deliberately physics-only — no learning happens here, and nothing
downstream is ever allowed to override its output. If a machine-learned component ever
disagreed with SGP4 about where a satellite physically is, SGP4 wins. This is validated
against Vallado's published reference state vectors (`tests/physics_validation/`), not
merely checked for internal consistency.

### M2 — Sparse Dynamic Graph

Checking every satellite pair for risk is `O(N²)` — 45 pairs for 10 satellites, but
~500,000 for 1,000. This module reduces that cost in two stages, without silently
dropping real risks:

**Stage A (coarse filter):** two satellites can only conjunct if their orbital altitude
ranges overlap — a satellite confined to 400–420 km can never approach one confined to
1000–1020 km, regardless of relative phase. This is a *necessary*, not sufficient,
condition: it can never produce a false negative, and it's fine if some false positives
pass through. Implemented as a sweep over satellites sorted by perigee altitude, so
each satellite only needs comparing against the *currently active* set, not every other
satellite.

**Stage B (fine screen):** on coarse-filter survivors, the actual relative dynamics are
checked — projected miss distance *and* closing rate, both required together. A pair
can be geometrically close but separating fast (not a risk); a pair can be far apart
but closing fast (a real risk). Raw distance alone can't distinguish these; this is
what "relative dynamics, not raw distance" concretely means in code.

**The false-negative gate:** because Stage B's thresholds could, in principle, be set
wrong, this module includes a dedicated validation gate that runs the combined sparse
graph against the exhaustive Step-3 baseline on the same scenario and reports exactly
what — if anything — the sparse graph missed, as a number, not an assumption. The gate
mechanism itself is verified (it correctly catches a deliberately introduced miss);
running it against a real, larger synthetic constellation to record the actual project
result is the next step.

### M3 — Temporal Forecasting (TGNN)

Rather than treating the constellation as fixed-interval snapshots (which loses
precision on exactly *when* a close approach happens — operationally meaningful down to
seconds/minutes), this module maintains a per-satellite memory vector that updates
continuously as new graph events occur, following the continuous-time temporal graph
network paradigm (in the style of TGN). Two prediction heads sit on top of that memory:

- A **time-to-event head** (point-process style, following DyRep) predicting *when*
  closest approach will occur, not just whether it will.
- An **uncertainty head** (following Pinto et al.'s Bayesian approach to CDM sequence
  prediction) outputting a *calibrated distribution*, not a flat probability — because
  a collision-probability estimate from a multi-day-ahead propagation carries real,
  compounding uncertainty from TLE error, propagation error, and atmospheric drag
  uncertainty, and pretending otherwise misleads whatever consumes the forecast
  downstream.

### M4 — Constrained Multi-Agent RL

Each satellite is an RL agent balancing mission objectives against fuel cost — but
never allowed to trade safety away for either. Safety is formulated as a **Constrained
Markov Decision Process (CMDP)**:

```
maximize    mission utility − fuel cost
subject to  P(collision) ≤ ε   for every flagged risky pair
```

This is deliberate, not incidental: reward-penalty formulations for safety are a
documented failure mode in the constrained-RL literature this project is built on — a
fixed penalty coefficient causes policies to either ignore safety (small penalty) or
abandon the mission entirely (large penalty), with no stable middle ground, because a
fixed penalty can't adapt to how close the current policy is to violating the
constraint. The baseline ladder (below) keeps the reward-penalty approach as an
explicit comparison point rather than discarding it silently — showing the constrained
formulation is better, not just asserting it.

### M5 — Physics Safety Filter

A final, real-time check — in the style of a **Control Barrier Function (CBF)** —
verifies a proposed maneuver keeps the satellite within a mathematically defined safe
set, applying the minimal necessary correction if it doesn't (solved as a small
quadratic program). This is explicitly **not** claimed as independently novel: GCBF+
already provides a stronger, formally-proven version of graph-based scalable safety
filtering. This module's actual job is to adapt that approach to real orbital dynamics
and benchmark directly against it, scoped only to the satellites M3 flags as high-risk
— which is what keeps it computationally tractable at constellation scale, distinct
from applying an equivalent check exhaustively.

### M6 — Simulation & Evaluation / M7 — Infrastructure

An independently built ground-truth simulator — deliberately owned separately from the
modeling pipeline it evaluates, so the two can never quietly drift apart — generates
test scenarios and runs large-scale Monte Carlo evaluation. Every experiment result is
tied to a specific configuration and code version, so reported numbers are
reproducible, not anecdotal.

## Why Each Design Choice Is Defensible

| Choice | Why, specifically |
|---|---|
| Physics never overridden by learning | Non-negotiable for aerospace-reviewer credibility |
| Two-stage sparse graph, not exhaustive | Exhaustive pairwise checking is `O(N²)` — a documented scaling wall in the closest comparable published safety system (Parikh et al.'s CBF-based satellite servicing paper) |
| Coarse filter uses altitude bands specifically | A *necessary* (not sufficient) condition — guarantees no false negatives at the coarse stage by construction |
| Continuous-time TGNN, not snapshot-based | Conjunction timing precision matters operationally; fixed-interval binning loses it |
| Calibrated uncertainty, not a point estimate | Point-estimate forecasting is already behind the accepted standard in this literature (Pinto et al.) |
| Safety as a CMDP constraint, not a reward penalty | Reward-penalty safety is a documented, citable anti-pattern, not a stylistic choice |
| Safety filter scoped to the flagged subgraph | Keeps M5 tractable at scale, avoiding the `O(N²)` wall even GCBF+-style approaches face without this scoping |
| Safety filter explicitly benchmarked against GCBF+ | Intellectual honesty — this module adapts and compares, it does not originate the mechanism |

## Repository Structure

```
constellai/
├── pyproject.toml
├── constellai/
│   ├── common/
│   │   └── constants.py            # every physical constant, cited, single source of truth
│   ├── orbital_mechanics/
│   │   ├── tle.py                  # TLE parsing (2-line and 3-line formats)
│   │   ├── propagation.py          # SGP4 wrapper; inclusive-start/exclusive-end sampling
│   │   ├── conjunction.py          # TCA + miss distance + closing rate from a state pair
│   │   ├── regime.py               # altitude-band computation
│   │   └── synthetic.py            # controlled-altitude satellite generation via sgp4init
│   ├── graph/
│   │   ├── filters.py              # coarse altitude-band filter (sweep-line)
│   │   ├── screening.py            # fine relative-dynamics screen
│   │   ├── build.py                # combined M2 pipeline (single canonical graph builder)
│   │   └── validation.py           # false-negative gate
│   ├── simulation/
│   │   └── baseline.py             # Step-3 exhaustive non-ML baseline
│   ├── models/
│   │   ├── tgnn/                   # M3 — continuous-time forecasting (in progress)
│   │   └── marl/                   # M4 — constrained multi-agent RL (in progress)
│   └── safety/                     # M5 — not yet started
└── tests/
    ├── unit/                       # one file per module above
    ├── physics_validation/         # the M1 hard gate — Vallado reference vectors
    └── integration/                # not yet populated
```

## Getting Started

**Prerequisites:** Python 3.11+, Git.

```bash
git clone https://github.com/Sharvari226/constellAI.git
cd constellAI/constellai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

Verify the install:

```bash
python -m pytest tests/ -v
```

Expected: **46 passed**.

## Testing

| Suite | Location | What it checks |
|---|---|---|
| Physics validation | `tests/physics_validation/` | Propagator output matches Vallado's published reference vectors exactly (to floating-point tolerance) — the M1 hard gate |
| Unit — orbital mechanics | `tests/unit/test_tle.py`, `test_propagation.py`*, `test_conjunction.py`, `test_regime.py`, `test_synthetic.py` | Parsing, propagation contracts, conjunction geometry, altitude bands |
| Unit — graph | `tests/unit/test_filters.py`, `test_screening.py`, `test_build.py`, `test_validation.py` | Coarse filter correctness and pruning behavior, fine-screen edge inclusion, combined pipeline, false-negative gate mechanism |
| Unit — baseline | `tests/unit/test_baseline.py` | Exhaustive O(N²) screening correctness |

<sub>*If `test_propagation.py` doesn't exist yet as a standalone file in your checkout,
propagation is covered indirectly through `test_physics_validation` and every other
module that depends on it — worth adding a dedicated one.</sub>

Run everything:

```bash
python -m pytest tests/ -v
```

Run one layer at a time during development:

```bash
python -m pytest tests/physics_validation/ -v   # the M1 gate specifically
python -m pytest tests/unit/test_build.py -v     # the combined M2 pipeline specifically
```

## Roadmap

- [x] **Step 0** — Environment & repo setup
- [x] **Step 1** — Literature review: STM motivation papers, MARL literature (MADDPG,
      QMIX, COMA, MAPPO), TGNN architectures (TGN, EvolveGCN, TGAT, JODIE, DyRep, DySAT,
      CAW), Safe RL / constrained-optimization literature (CPO, a SafeRL/SafeMARL
      survey, a CBF-based satellite-servicing paper) — plus competitive positioning
      against GCBF+, GS-MARL, and a 2025 satellite-MARL paper
- [x] **Step 2** — M1: propagator validated against Vallado test vectors; independent
      simulator baseline
- [x] **Step 3** — Non-ML baseline
- [x] **Step 4** — M2: sparse graph construction; false-negative gate mechanism
      verified (real-constellation run to record the actual rate is the remaining
      open item)
- [ ] **Step 5** — M3: temporal forecasting (near complete)
- [ ] **Step 6** — M4: constrained MARL baseline ladder (near complete)
- [ ] **Step 7** — M5: safety filter, explicit comparison against GCBF+
- [ ] **Step 8** — Integration: M3 → M4 → M5, end-to-end on a multi-day scenario
- [ ] **Step 9** — Scalability analysis (N = 10 → 1000+), full ablations, Monte Carlo
      evaluation (≥500 trials)
- [ ] **Step 10** — Paper draft, mentor red-team review, capstone demonstration

## Baseline & Ablation Ladders

**M3 (forecasting) — three-way comparison:**
1. Non-ML distance-threshold baseline (no learning)
2. Plain per-satellite LSTM (learning, no graph structure)
3. Full continuous-time TGNN + uncertainty head (proposed)

**M4 (coordination) — five-way comparison:**
1. Single-agent RL, no coordination
2. IPPO (independent learning, no shared information)
3. MAPPO + reward penalty (the original project plan — kept as the baseline it turned out to be)
4. MADDPG (continuous-action comparison)
5. Constrained/Lagrangian formulation (the proposed method)

**Key ablations:**
- Uncertainty head on/off — does calibrated uncertainty measurably reduce the downstream policy's constraint-violation rate?
- Safety filter on/off — does M5 add anything beyond constrained RL alone?
- Sparse graph vs. exhaustive pairwise — the false-negative rate; load-bearing for the entire efficiency claim
- Reward-penalty vs. constrained MARL — reproduces the constrained-optimization literature's own fixed-penalty comparison, in this domain
- ConstellAI's safety filter vs. GCBF+ directly, on identical scenarios

## What This Project Never Claims

- **Not** "guaranteed safety" — report empirical Monte Carlo success rate, the same way the closest published comparable systems report theirs.
- **Not** "a novel scalable safety mechanism" — GCBF+ already published a stronger, formally-proven version; ConstellAI adapts and compares.
- **Not** "scales to thousands of satellites" without the Step 9 scaling curve to back it, benchmarked against GCBF+'s own published numbers.
- **Not** "no information loss from graph sparsity" without the false-negative gate's actual measured rate attached.

## Team

| Role | Focus |
|---|---|
| Dev A + Dev B | M1 (orbital mechanics) + M6 (simulation & evaluation) |
| Dev C | M2 (sparse graph) + M3 (temporal forecasting) |
| Dev D | M4 (constrained MARL), solo |
| Mentor + Dev D | M5 (safety filter), paired |
