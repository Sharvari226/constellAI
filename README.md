# ConstellAI

**Forecast-conditioned, safety-constrained, fuel-aware maneuver planning for satellite constellations.**

> **Status (6 Nov 2026): research prototype, work in progress.**
> The infrastructure for the central experiment (noisy tracking, ground-truth event labels, an analytic
> forecaster, a constrained objective) is built and tested. **The central experiment itself has not been run.**
> No result in this repository supports a claim that forecasting improves planning. Sections 5 and 6 list
> exactly what is verified, what is broken, and what is missing.

---

## Table of contents

1. [Aims](#1-aims)
2. [Problem setting and modelling choices](#2-problem-setting-and-modelling-choices)
3. [System overview](#3-system-overview)
4. [Formulation](#4-formulation)
5. [What is built so far](#5-what-is-built-so-far)
6. [Known issues and open problems](#6-known-issues-and-open-problems)
7. [What is left (roadmap)](#7-what-is-left-roadmap)
8. [Planned experimental protocol](#8-planned-experimental-protocol)
9. [Reproducing the current state](#9-reproducing-the-current-state)
10. [Repository layout](#10-repository-layout)
11. [Related work](#11-related-work)
12. [Established methods vs. our contributions](#12-established-methods-vs-our-contributions)
13. [Working conventions](#13-working-conventions)

---

## 1. Aims

Low Earth orbit is congested. Conjunction management is not only a prediction problem or only a control problem:
it requires anticipating risk, deciding under uncertainty, coordinating actions between satellites, and conserving
fuel. ConstellAI connects these pieces into one validated, reproducible pipeline and tests whether the connection
actually helps.

### 1.1 Research question (the scientific north star)

> **Does calibrated, multi-horizon conjunction-risk forecasting improve the safety–fuel trade-off of
> forecast-conditioned satellite maneuver planning, compared with reactive planning under the same dynamics,
> constraints, and evaluation scenarios?**

This is a question, not a conclusion. Every design choice, metric and experiment is judged against it.

### 1.2 Working hypotheses and what would falsify them

| # | Hypothesis | Supported if | Challenged if |
|---|---|---|---|
| H1 | With uncertain tracks, a planner that receives calibrated risk forecasts achieves a better Δv-vs-collision frontier than a reactive planner. | The oracle (B), analytic (F) and learned (C) forecast arms beat the reactive arm (A) with confidence intervals that exclude zero. | The oracle arm does **not** beat the reactive arm. Then forecasting cannot help in this environment, which is itself a valid finding. |
| H2 | Calibration matters for planning, not just for forecast metrics. | The calibrated learned forecaster (C) beats the same model without calibration (D) on the frontier. | C ≈ D. |
| H3 | A learned forecaster adds value over physics-based propagation. | The learned model beats the analytic Monte Carlo forecast (F), especially under noise-model mismatch. | F ≈ C, or F > C. Then the right answer to the research question is "use the analytic forecast". |

### 1.3 Secondary aims

- A mathematically justified constrained objective for multi-agent maneuver planning (Section 4.4), tested against
  the original formulation under identical conditions.
- An honest evaluation of whether a temporal graph model (TGN-lite) is warranted compared with simpler models.
- Safety treated as a scientific claim: empirical avoidance, constraint satisfaction in a tested environment, a
  filter that works under stated assumptions, and a formal guarantee are kept strictly separate.

### 1.4 Explicit non-claims

- **No formal safety guarantee** is claimed anywhere. Any future guarantee requires its assumptions to be stated and
  proven.
- **No novelty claim** is made yet. A literature review is in progress (Section 11).
- The tracking noise is a **synthetic sensitivity setting**, not realistic orbit-determination uncertainty.
- Forecast calibration is **relative to the scenario distribution**, not to real-world conjunction prevalence.
- Results from a single training seed or the small M3 pilot are **pilot evidence only**.

---

## 2. Problem setting and modelling choices

| Item | Choice | Note |
|---|---|---|
| Dynamics | Linear Hill–Clohessy–Wiltshire (HCW) relative motion in the LVLH frame | Reference circular orbit at 500 km altitude; mean motion from WGS-72 constants |
| Primary | A static object at the origin of the frame | Treated as a pseudo-threat with relative state `−x_ego` |
| Agents | N controlled satellites (default 3) sharing one policy | Thrust-limited: 0.01 m/s² per axis |
| Step | dt = 10 s, up to 200 steps per episode | |
| Collision | Separation < r_c = 50 m at a grid step, strict inequality | r_c is a **proxy sphere**, not a hard-body radius |
| Fuel | Σ ‖thrust‖·dt, i.e. Δv in m/s | The 100-unit budget never binds in practice (max Δv ≈ 34.6 m/s) |
| Tracks | Kalman filter on position-only measurements | Synthetic estimator; see Section 4.2 |
| Scenarios | Built backwards from a chosen time of closest approach (TCA) | 15% of agents on a zero-thrust collision course; see Section 4.6 |

---

## 3. System overview

```
 orbital state  ──►  conjunction candidates  ──►  risk forecasting  ──►  maneuver planning  ──►  safety filter
 (propagation,        (sparse dynamic graph,       (M3: TGN-lite /        (M4: constrained        (M5: CBF / HOCBF)
  TLE, relative        screening)                   LSTM / analytic MC)    multi-agent PPO)
  motion)
                                         ▲                                      │
                          tracking (Kalman filter, noisy tracks)                ▼
                                                                     evaluation harness (M6)
```

The modules were developed in stages M1–M6. The mapping below follows file naming and was not re-audited in the
current phase unless marked.

| Stage | Package | Purpose | Audited in current phase |
|---|---|---|---|
| Orbital mechanics | `orbital_mechanics/` | TLE parsing, SGP4-style propagation, HCW relative motion, conjunction geometry, synthetic data, Space-Track fetch | Only `relative_motion.py` |
| Graph construction | `graph/` | Sparse conjunction-candidate graphs, screening, filters, validation | No |
| Risk forecasting (M3) | `models/tgnn/` | TGN-lite, LSTM and static-GNN baselines, risk adapter, evaluation | Yes (see Section 5.4) |
| Planning (M4) | `models/marl/` | Environments, REINFORCE/IPPO/MAPPO/MADDPG/Constrained MAPPO, checkpoints | Yes |
| Safety (M5) | `safety/` | CBF filter, multi-threat filter, filtered environments | **No** |
| Evaluation (M6) | `evaluation/` | Scenario sampler, trial runner, scaled environment, Monte Carlo, gain sweep, nonlinear environment | Partly (`scenarios.py`, `trials.py`, `scaled_env.py`) |
| Tracking (new) | `tracking/` | Kalman filter and tracked multi-agent environment | New |
| Forecasting (new) | `forecasting/` | Ground-truth event labels and analytic Monte Carlo forecaster | New |

---

## 4. Formulation

### 4.1 Event and counterfactual

For an ego agent and a threat, with relative state `x = threat − ego` (for the primary, `x = −ego`), zero thrust
from the forecast time on gives `x(s) = Φ^s x₀`. The event for horizon `H` is

```
E_H = { ∃ s ∈ {1, …, H/dt} :  ‖position(x(s))‖ < r_c }
```

evaluated on the environment's own grid with the environment's strict inequality. This is **risk if no action is
taken**, not the probability of collision after the policy maneuvers. Horizons are `{100, 200, 400, 600}` s.
Events are nested, so a valid forecast is monotone in `H`.

### 4.2 Tracks

Each ego keeps a linear Kalman filter per threat (the primary plus each neighbour), on the 6-D relative state with
HCW dynamics. Measurements are positions with isotropic noise `σ_p ∈ {0.1, 0.5, 1.0}·r_c = {5, 25, 50} m`.
The filter's known input is the ego's applied thrust. Because the neighbour's thrust is unknown, a
**cooperative assumption** is used for the main experiment: neighbours broadcast their applied thrust, so the
neighbour filter is exact apart from measurement noise. Without that assumption the filter is statistically
inconsistent under correlated thrust (Section 5.5).

### 4.3 Forecast

- **Analytic Monte Carlo (arm F):** sample the filter posterior `N(μ, P)`, propagate each sample with `Φ^s`, and
  report first-passage frequencies. Common random numbers make it deterministic and monotone in `H` by construction.
- **Instantaneous-Gaussian bound:** the maximum over the grid of `P(‖r(t_s)‖ < r_c)`. It is a lower bound on
  first-passage risk and is kept as a documented, weaker baseline.
- **Learned forecaster v2 (planned):** discrete-time survival likelihood over interval hazards
  `F_k = 1 − Π_{j≤k}(1 − h_j)`, no `pos_weight`, with a shared monotone calibration map fitted on a held-out stream.

### 4.4 Constrained objective (current implementation)

Each agent solves a constrained problem with a vector of separate signals:

```
maximise   E[ Σ γ^t r_t ]       subject to       E[ Σ γ^t c_t ] ≤ d
```

**Mission reward** (before the fuel term added by the scaling wrapper):

```
r_t = − ( w_track · min( max(0, |dist_to_primary − 1000 m| − 50 m) / 1000 m , 3 )  +  w_smooth · ‖Δv_t − Δv_{t−1}‖ / dv_max )
```

with `w_track = 1`, `w_smooth = 0.05`. The fuel penalty is applied once, by `ScaledMultiAgentEnv(fuel_weight)`;
the objective's own fuel weight is 0 to avoid double counting.
**Known problem:** the tracking term is a placeholder inherited from the first environment (Section 6, item 2).

**Shaped conjunction cost** (the training constraint, a *proxy*):

```
c_conj = max over threats of  clip( (d_warn − d_pred) / (d_warn − d_safe) , 0 , 1 )²
```

where `d_pred` is the closest separation within the next 10 steps under linear relative motion, `d_safe = r_c`, and
`d_warn = 5 r_c`. The threat list is the primary plus all live neighbours, computed from true states (privileged).
The multiplier `λ` is updated either by dual ascent (current training script) or by a PID Lagrangian
(`PIDLagrange`, implemented and tested, not yet used by the training script). The actor maximises
`A_r − λ·A_c` with separate advantage estimators.

**Important:** the constrained quantity is the shaped proximity cost. It is **not** a bound on collision
probability, and the paper must not state that `λ` tracks the real collision constraint. True collision rate,
collision-free rate, minimum miss distance and Δv are always reported separately.

Computed and logged but **not yet constrained**: `c_sec` (secondary-conjunction risk created by a burn), and
`c_filter` (policy-versus-filter override, zero in the unfiltered environment).

### 4.5 Observation contract

| Block | Dim | Content |
|---|---|---|
| fuel | 1 | own fuel / budget (exact) |
| slot 0 | 13 | primary track: valid flag, mean position/1000, mean velocity, position std/1000, velocity std |
| slot 1 | 13 | neighbour track with the smallest predicted miss, chosen **from track means only** so selection is identical in every arm |
| forecast | 8 | four cumulative risks per slot (forecast arms only) |

Actor input is 27-dimensional (35 with a forecaster). The centralised critic receives the **privileged true
state** (7 numbers per agent, the original scaled layout), identically in every arm, so arms differ only in the
actor's input. Measurement noise is drawn for every pair at every step regardless of liveness or actions, so noise
sequences cannot depend on the policy.

### 4.6 Scenarios and RNG streams

Scenarios are built backwards from a chosen TCA (miss vector perpendicular to the approach velocity, HCW integrated
backwards under zero thrust), so the collision-course fraction is controlled by construction. Streams are keyed by
`(stream, seed, index)` and are disjoint:

| Stream | Use |
|---|---|
| 1 | Policy training |
| 2 | **Final test, all arms. Untouched until the end** |
| 3 | Forecaster training (never the policy-training scenarios) |
| 4 | Validation: model selection, alarm threshold, filter and calibration checks |
| 5 | Forecaster calibration |
| 6 | Pilot and reference-baseline scripts (defined inline in `reference_baselines.py`, not yet a constant) |

Warning-time presets: **short** (150–600 s, the main experiment) and **long** (500–1500 s, sensitivity run).
Earlier action is cheaper by roughly `Δv ≈ 2d/T` (a back-of-envelope estimate ignoring HCW coupling), which is the
mechanism by which forecasts could help.

---

## 5. What is built so far

All code below is on branch `feat/noisy-tracks`. The tag `baseline-pre-objective` pins the original code
(commit `edcaf04`), which is the "original formulation" any new formulation must be compared against.

### 5.1 Components

| Area | Files | Verification |
|---|---|---|
| Constrained objective | `models/marl/objective.py` | Unit tests (`test_objective.py`, `test_objective_env.py`) |
| Environment fixes | `models/marl/multi_agent_environment.py` | Two-phase `step()` so collisions and costs are independent of agent order; fuel double-count removed; applied thrust and true-state accessors exposed. Symmetry and exposure tests, including a test that the env's relative dynamics equal the filter's model |
| Constrained MAPPO additions | `models/marl/constrained_mappo.py`, `joint_obs.py`, `checkpoint.py` | Optional `pid` multiplier (default path unchanged); privileged-critic hook; `actor_obs_dim` in checkpoints (backward compatible); explained variance logged. `test_network_hooks.py` |
| Kalman filter | `tracking/kalman.py` | Transition matrices derived by probing the env's propagator; 8 tests including NEES consistency and a test with power to detect an inconsistent filter |
| Tracked environment | `tracking/tracker.py` | 8 contract tests: shapes, prefix property between arms, determinism, noise independent of actions, done-threat handling, privileged state, reward/cost pass-through |
| Event labels and analytic forecaster | `forecasting/events.py` | 9 tests, including exact agreement with a brute-force propagator and with the environment's first collision step |
| Scenario streams and presets | `evaluation/scenarios.py` | `test_env_exposure.py` |
| Diagnostic scripts | `scripts/check_track_consistency.py`, `check_forecast_calibration.py`, `reference_baselines.py` | The first two have been run; `reference_baselines.py` is written but its output has not been reviewed |

The unit suite last ran in full at 269 passed, 1 skipped; the forecasting tests (9) and the later objective and
logging additions were run individually and passed.

### 5.2 Verified: reference policies on the fixed environment

Default (long-warning) scenarios, 200 trials, evaluation stream, seed 0:

| Policy | Collision-free | Mean min separation | Fuel per agent |
|---|---|---|---|
| zero thrust | 109/200 (0.545) | 72.8 m | 0 |
| constant full burn | 167/200 (0.835) | 114.7 m | 32.8 |

The same policies on the order-dependent environment gave 106/200 and 159/200, so the ordering bug shifted
reference rows by 1.5 to 4 points. All results generated before the fix are **superseded** (including everything
under `outputs/m4` and `outputs/m4_scaled`).

### 5.3 Verified: filter consistency on real environment rollouts

Mean per-episode NEES (target 6; 30 episodes, short-warning scenarios, validation stream):

| Configuration | Primary pair | Neighbour pair |
|---|---|---|
| Thrust **not** broadcast, zero or random policy | 5.95 to 6.09 | 3.3 to 3.6 (zero), about 6.0 (random) |
| Thrust **not** broadcast, constant burn | 5.95 to 6.09 | **35.6 / 75.6 / 101.3** at σ_p = 5 / 25 / 50 m |
| Thrust **broadcast** (cooperative) | 5.95 to 6.09 | **5.69 to 5.86** in all nine cells |

The non-cooperative filter is biased when all agents burn the same thrust and cannot be fixed by tuning process
noise. The cooperative assumption must be stated in the paper.

### 5.4 Verified: analytic forecast on the validation stream

Analytic Monte Carlo forecast (2000 samples), thrust broadcast, zero policy, 20 episodes (5,497 rows; heavily
correlated). Event rates at H = 100/200/400/600 s: 0.5%, 1.1%, 2.0%, 2.5%.

| σ_p | Brier skill vs. base rate at H = 100 / 200 / 400 / 600 s |
|---|---|
| 5 m | 0.996 / 0.970 / 0.896 / 0.790 |
| 25 m | 0.844 / 0.733 / 0.538 / 0.440 |
| 50 m | 0.584 / 0.483 / 0.319 / 0.257 |

All skill confidence intervals exclude zero, so the scenarios carry forecastable signal and the gate in the
forecaster specification passes. ECE is small in absolute terms (≤ 0.015) but proportionally larger at long
horizons and large σ_p; with 20 episodes this is suggestive, not established.

### 5.5 M3 pilot (older task, different event)

M3 predicts "separation below 300 km within 90 minutes", not the M4 event. Rerun on the current code
(3 seeds; LSTM and GNN use 6 test scenarios at a 2-minute step, TGN-lite 3 test scenarios at a 5-minute step):

| Model | AP (seed range) | Precision | Recall |
|---|---|---|---|
| LSTM | 0.756 (0.66 to 0.84) | 0.760 | 0.864 |
| Static GNN | 0.183 | 0.162 | 0.773 |
| TGN-lite (point) | 0.246 (0.18 to 0.31) | 0.047 | 0.711 |
| TGN-lite (Beta uncertainty head) | 0.014, ECE 0.334 | | |

The effective sample is 3 to 6 scenarios and the comparison is not matched, so this supports no confident ranking.
It does show that the simple LSTM clearly outperforms every TGN variant on this task. TGN-lite is treated as a
documented negative pilot, not as the forecaster for the central experiment.

---

## 6. Known issues and open problems

Listed in rough order of how much each threatens the credibility of the research.

1. **The reward critic is not learning.** In a 100-episode, one-seed Constrained MAPPO run, reward-critic explained
   variance was 0.01 (first 10 episodes) and 0.02 (last 10); cost-critic explained variance fell from 0.25 to −0.03.
   Capping the tracking term cut the final critic loss from about 516,000 to about 343 (the uncapped quadratic term
   was the main driver of the blow-up), but that is a scale effect, not evidence of a working critic.
2. **The mission term is a placeholder that conflicts with the scenarios.** It rewards sitting on a 1000 m shell
   around the primary, whereas scenarios send agents past the primary at 40 to 200 m. A zero-thrust policy therefore
   earns a large start-dependent penalty. A replacement (deviation from the unmaneuvered nominal path, with unchanged
   δ, L and cap) is **proposed and awaiting approval, not implemented**.
3. **The constraint budget `d` is uncalibrated.** The old threshold (0.05) was set for a binary cost; with the shaped
   cost, mean `mean_cost_return` is about 0.67 and λ ratchets upward (1.0 to 1.6 in 100 episodes).
4. **`mean_cost_return` is the wrong multiplier input.** It comes from GAE returns (critic-dependent) and goes
   negative although true costs are non-negative. The PID update should use the observed discounted cost. The
   cost-advantage normaliser can also amplify noise when costs are nearly constant.
5. **Scope of the tracked-observation path.** Only MAPPO and Constrained MAPPO work with tracked observations.
   REINFORCE, IPPO and MADDPG still use the old 7-dimensional layout, and REINFORCE the old single-agent objective.
   The forecast-augmented environment exists only for the single-agent case.
6. **Fixed-penalty baselines are mis-tuned.** `DEFAULT_PENALTY_COEFF` was tuned against a binary cost and now
   multiplies a shaped one.
7. **The M3 forecaster does not match the M4 task.** One horizon, a 300 km / 90-minute target, no velocity
   direction in its features, no use of event timestamps (so the continuous-time claim is unsupported), a Beta head
   that can collapse, and probabilities of unverified calibration (`pos_weight` on one path, Beta-NLL on the other).
   A trained forecaster has **never** been used inside an M4 experiment, and no TGN checkpoint is tracked.
8. **Scenario prior.** The sampler is not flat: collision-course misses are at most 40 m and the others 75 to
   200 m. A Gaussian posterior ignores this structure, so a learned forecaster might beat the analytic one by
   exploiting the sampler, not physics. Calibration claims must be scoped to the scenario distribution, and a
   differently shaped sampler is needed as a robustness check.
9. **The evaluation harness stores summaries only.** No raw per-trial results (so no paired tests from saved
   artifacts), no origin versus agent–agent split, no first-burn lead time.
10. **M5 is unaudited.** The safety filter has not been reviewed or stress-tested in this work (late warnings,
    multiple threats, infeasible constraints, secondary conjunctions). No safety claim is supportable yet.
11. **Cooperative-thrust assumption.** See Section 4.2. A non-cooperative variant (an augmented filter with an
    unknown-acceleration state) is a possible robustness experiment.
12. **Repository hygiene.** About 50 `.pyc` files are tracked in git and show as modified after every run;
    `test_mappo_learns_to_reduce_collisions` starts the agents inside the collision radius and is likely
    degenerate; the `risk_adapter.py` docstring still describes a ground-truth binary safety cost.

---

## 7. What is left (roadmap)

Order matters: later phases are meaningless if earlier ones fail. Each item states its acceptance criterion.

### Phase A: make the planner trainable and the baselines meaningful

- [ ] **A1. Replace the mission term** (Section 6, item 2). *Accept:* zero-thrust policy earns exactly 0 reward;
      deviation is 0 at t = 0; deviation under constant thrust equals the difference of two independently
      propagated states. Then rerun the control and compare critic explained variance.
- [ ] **A2. Fix critic learning if explained variance stays near 0** (value or return normalisation, documented as a
      separate change). *Accept:* reward-critic explained variance well above 0 on the training batch and stable.
- [ ] **A3. Review `reference_baselines.py` output** for both warning presets (zero, random, full-burn policies;
      discounted shaped-cost scale).
- [ ] **A4. Calibrate the budget `d` from data**, wire `PIDLagrange` into the training loop with the observed
      discounted cost, clamp the cost-advantage normaliser.
- [ ] **A5. Evaluation harness:** deterministic evaluation wrapper for tracked actors; save raw per-trial results;
      record origin versus agent–agent collisions and first-burn lead time.
- [ ] **A6. Regenerate all baselines** under the fixed environment on the short-warning scenarios, applied
      identically to every method.

### Phase B: the falsification test (before any learned forecaster)

- [ ] **B1. Oracle forecast hook** in the tracked environment (true pair state, nested 0/1 risk per horizon).
- [ ] **B2. Arm runner** for A (reactive), B (oracle), E (threshold heuristic) and F (analytic forecast), with
      matched compute budgets.
- [ ] **B3. Pilot frontier:** Δv-versus-collision sweep over the cost budget, at least 5 training seeds, short and
      long warning presets. *Decision gate:* if the oracle arm does not beat the reactive arm, stop; forecasting
      cannot help here and the paper reports that.

### Phase C: forecaster v2 (only if the gate in B3 passes)

- [ ] **C1.** Training data on streams 3 and 5 from a mixture of zero-thrust and random-thrust rollouts.
- [ ] **C2.** MLP on track features with the discrete-time survival loss; TGN-lite only if it beats the MLP and the
      analytic baseline.
- [ ] **C3.** Shared monotone calibration (one temperature and bias on the logit of `F_k`, `a > 0`); report any
      violations introduced by per-horizon maps.
- [ ] **C4.** Validate against the base rate, analytic Monte Carlo and the MLP, with scenario-clustered bootstrap
      intervals. Losses are reported.
- [ ] **C5.** Noise-mismatch experiment (train at one σ, test at another, give the analytic forecaster the wrong σ).
- [ ] **C6.** Arms C (learned, calibrated) and D (learned, uncalibrated) on the frontier.

### Phase D: breadth and safety

- [ ] **D1.** Define the threshold-heuristic arm precisely (burn rule: direction, magnitude, trigger).
- [ ] **D2.** Ablations: forecast on/off, adaptive versus fixed multiplier, uncertainty input on/off.
- [ ] **D3.** Scaling with constellation size, and compute overhead per forecast query.
- [ ] **D4.** M5 integration and audit: assumptions of the barrier formulation, optimizer versus formulation, late
      warnings, multiple threats, infeasibility, actuator limits, secondary conjunctions. No formal claim without a
      proof.
- [ ] **D5.** Extend the tracked-observation path to the remaining baselines, or state in the paper that they are
      excluded.

### Phase E: finish

- [ ] **E1.** Literature review completed before any novelty claim.
- [ ] **E2.** Cleanup: untrack `.pyc` files (`.gitignore`), repair the degenerate test, update stale docstrings.
- [ ] **E3.** Paper: every claim traceable to code, derivation or experiment; negative results reported.

---

## 8. Planned experimental protocol

### 8.1 Arms

| Arm | Actor receives | Role |
|---|---|---|
| A. Reactive | Noisy tracks of the primary and one neighbour, current only | The stated baseline |
| B. Oracle forecast | A + exact risk from the true state, several horizons | Upper bound; falsification test |
| C. Learned, calibrated | A + forecaster output, recalibrated on a held-out stream | The real system |
| D. Learned, uncalibrated | A + the same forecaster without recalibration | Isolates the value of calibration |
| E. Threshold heuristic | A, with a fixed burn rule | The simple operational baseline |
| F. Analytic forecast | A + analytic Monte Carlo forecast | Physics-based reference |

**Controlled across arms:** dynamics, the fixed environment, the frozen objective, the network architecture (except
input width), PPO budget, noise realisations, and the evaluation scenarios.

### 8.2 Metrics

| Metric | Definition | Limitation |
|---|---|---|
| Collision-free rate (episode level) | Fraction of trials with no agent collision (any agent) | Hides how many agents; also report per agent |
| Min miss distance | Smallest primary or agent–agent separation in the trial (m) | Mean over trials hides tails |
| Δv per agent | Σ ‖thrust‖·dt (m/s) | Depends on warning time |
| Δv–collision frontier | Sweep of the cost budget; both axes with intervals | Needs ≥ 5 seeds to mean anything |
| First-burn lead time | Time from first alarm to first burn (s) | Needs per-trial logging |
| Forecast Brier, log-loss | Per horizon, against ground-truth labels | Relative to the scenario distribution |
| Forecast ECE | Equal-mass bins, with the floor expected for a perfectly calibrated forecaster | Biased upward in finite samples |
| Filter NEES | Mean per scenario, against the chi-square interval | Tests consistency, not accuracy |

### 8.3 Statistics

- Unit of analysis: the **training seed** (at least 5), with a scenario-level paired bootstrap. Timesteps and pairs
  inside one scenario are never treated as independent.
- Evaluation scenarios (stream 2) are fixed and shared by all arms; no tuning on stream 2.
- Alarm thresholds and calibration maps are chosen on streams 4 and 5 only.
- Results are written to JSON with the git commit hash.

---

## 9. Reproducing the current state

Developed on Windows with Python 3.13 (PowerShell). Dependencies are declared in `pyproject.toml` (not reviewed
here); the code uses NumPy and PyTorch.

```powershell
git clone https://github.com/Sharvari226/constellAI
cd constellAI
git switch feat/noisy-tracks
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .           # or install the dependencies listed in pyproject.toml, plus pytest
$env:PYTHONPATH = (Get-Location).Path    # needed when running scripts by path
```

```powershell
# unit tests (about 3 minutes for the full suite)
python -m pytest tests/unit -q

# filter consistency on real environment rollouts (cooperative thrust broadcast)
python constellai\scripts\check_track_consistency.py --coop --accel-std 0.0

# calibration of the analytic forecast on the validation stream
python constellai\scripts\check_forecast_calibration.py

# reference policies and shaped-cost scale, both warning presets
python constellai\scripts\reference_baselines.py

# short Constrained MAPPO control run (single seed, 100 episodes)
python constellai\scripts\train_m4_ladder.py --rung constrained_mappo --seeds 0 --episodes 100 --out-dir outputs\my_run
```

The M3 comparison script (`scripts/compare_baselines.py`) prints results but does not save them; redirect its
output to a file and record the commit hash alongside it.

---

## 10. Repository layout

```
constellai/
  common/                 constants
  orbital_mechanics/      tle.py, propagation.py, relative_motion.py, conjunction.py, regime.py, synthetic.py, fetch.py, spacetrack.py
  graph/                  build.py, screening.py, filters.py, validation.py
  models/
    tgnn/                 tgn_lite.py, lstm_baseline.py, gnn_baseline.py, dataset.py, dynamic_graph.py,
                          graph_dataset.py, evaluation.py, risk_adapter.py
    marl/                 environment.py, multi_agent_environment.py, forecast_augmented_environment.py,
                          single_agent_baseline.py, ippo.py, mappo.py, maddpg.py, constrained_mappo.py,
                          checkpoint.py,
                          objective.py        [new]  constrained objective, PID Lagrangian
                          joint_obs.py        [new]  privileged-critic hook
  safety/                 cbf_filter.py, multi_threat_filter.py, filtered_environment.py
  simulation/             baseline.py
  tracking/               [new] kalman.py, tracker.py
  forecasting/            [new] events.py
  evaluation/             scenarios.py, trials.py, scaled_env.py, policies.py, stats.py, envs.py,
                          monte_carlo.py, gain_sweep.py, nonlinear_env.py, tiers.py
  scripts/                train_m4_ladder.py, compare_baselines.py, ... plus
                          [new] check_track_consistency.py, check_forecast_calibration.py, reference_baselines.py
tests/
  unit/                   unit and contract tests
  physics_validation/     Vallado reference checks
outputs/                  tracked artifacts from earlier phases; those under m4/ and m4_scaled/ predate the
                          environment fix and are superseded
```

---

## 11. Related work

Status: **preliminary; not a completed review.** The following were located and read at abstract level only.

- *Markov Decision Processes for Satellite Maneuver Planning and Collision Avoidance* (arXiv:2501.02667) studies how
  far in advance an operator should maneuver to trade fuel against collision probability under incoming
  information updates. It is the closest work found to our safety–fuel question.
- Greco et al., *A robust Bayesian agent for optimal collision avoidance manoeuvre planning* (University of
  Strathclyde, 2021) combines a particle filter for estimation with an agent that classifies risk events and
  allocates manoeuvres under uncertainty.
- *ConstrainedZero* (IJCAI 2024) is a belief-state planner that uses adaptive conformal inference to adjust a
  failure-probability threshold. It is not satellite-specific but is relevant to calibrated constraint handling.
- Stooke et al., *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods* (ICML 2020), is the basis of
  `PIDLagrange` (cited from memory; not re-verified in this phase).

The sources the team's reviewer cited (a NASA conjunction-assessment AI/ML compendium, a 2025 high-covariance
conjunction-prediction paper, and a 2026 belief-space planning paper) could **not** be confirmed in the current
search and should not be cited until located. Broad claims such as "first uncertainty-aware satellite collision
avoidance" are not supportable. Any novelty claim will be limited to what the controlled experiment shows.

---

## 12. Established methods vs. our contributions

| Component | Status |
|---|---|
| HCW relative dynamics; Kalman filtering; CBF/HOCBF filters | Established |
| PPO / MAPPO with a centralised critic; Lagrangian constrained RL; PID Lagrangian multiplier | Established |
| TGN, LSTM and GNN risk classifiers | Established |
| Shaped proximity cost, tracked-observation contract, cooperative thrust broadcast | Our engineering choices; no novelty claimed |
| Interval-hazard parametrisation with a shared monotone calibration map | Standard ingredients (discrete-time survival analysis, temperature scaling); the combination for this task is a design choice |
| Controlled comparison of reactive, oracle, analytic and learned forecasts under uncertain tracks, with a Δv-versus-collision frontier | **The intended contribution.** Novelty not established; pending the literature review and the experiment itself |

The paper must keep this separation explicit.

---

## 13. Working conventions

- Work on feature branches; **do not push to `main` directly** (`main` and `origin/main` still point at the original
  code).
- Add files by explicit path. **Never `git add .` or `git add -A`**; tracked `.pyc` files change on every run.
- Every experiment states what is tested, why it matters to the research question, the controlled variables, the
  baselines, the metrics, what would support or falsify the hypothesis, and what remains unjustified.
- Results that contradict the hypothesis are reported as such. Results from before the environment fix are not mixed
  into any table.
- Any status claim in this README must be re-verified against code, tests and logs before it is reused in the paper.

*Licence and citation information: to be added.*
