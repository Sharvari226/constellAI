"""Ground-truth event labels and an analytic Monte Carlo forecaster.

Event E_H(ego, threat): the ego-threat separation falls below the collision radius at
some env step s = 1..H/dt when BOTH bodies apply zero thrust from the forecast time on.
Relative state x = threat - ego (the primary is a static pseudo-threat, x = -ego), so
x(s) = Phi^s x0 and the test is |position(s)| < r_c on the env's own dt grid with the
env's strict inequality. This is risk-if-no-action, not the probability of collision
after the policy manoeuvres.
"""

from __future__ import annotations

import numpy as np

from constellai.tracking.kalman import hcw_transition

HORIZONS_S = (100.0, 200.0, 400.0, 600.0)


def horizon_steps(dt, horizons_s=HORIZONS_S):
    steps = []
    for h in horizons_s:
        k = int(round(h / dt))
        if k < 1 or abs(k * dt - h) > 1e-9:
            raise ValueError(f"horizon {h} s is not a positive multiple of dt={dt} s")
        steps.append(k)
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("horizons must be strictly increasing")
    return steps


class FirstEntryEngine:
    """First step s in 1..k_max with |position(s)| < r_c under zero thrust (0 = never)."""

    def __init__(self, mean_motion, dt, collision_radius_m, k_max):
        phi, _ = hcw_transition(mean_motion, dt)
        pows = np.empty((k_max, 3, 6))
        m = np.eye(6)
        for s in range(k_max):
            m = phi @ m
            pows[s] = m[:3, :]
        self.pos_pows = pows  # row s-1 maps x0 to the position at step s
        self.rc = float(collision_radius_m)
        self.k_max = k_max

    def distances(self, x0):
        x0 = np.atleast_2d(np.asarray(x0, dtype=float))
        pos = np.einsum("ksi,bi->bks", self.pos_pows, x0)
        return np.linalg.norm(pos, axis=2)

    def first_entry(self, x0):
        entered = self.distances(x0) < self.rc
        first = entered.argmax(axis=1) + 1
        return np.where(entered.any(axis=1), first, 0)


def cumulative_labels(first, steps):
    first = np.asarray(first)
    return np.stack([(first > 0) & (first <= k) for k in steps], axis=1).astype(float)


def interval_index(first, steps):
    """0 = no event within the last horizon, else the 1-based interval of the first entry."""
    first = np.asarray(first)
    out = np.zeros(len(first), dtype=int)
    for k in reversed(range(len(steps))):
        out = np.where((first > 0) & (first <= steps[k]), k + 1, out)
    return out


def pair_states(truth):
    """truth: 6-vectors by agent id. Returns [(ego, threat, x0)]; threat = -1 is the primary."""
    out = []
    for i, xi in enumerate(truth):
        xi = np.asarray(xi, dtype=float)
        out.append((i, -1, -xi))
        for j, xj in enumerate(truth):
            if j != i:
                out.append((i, j, np.asarray(xj, dtype=float) - xi))
    return out


class AnalyticForecaster:
    """Monte Carlo first-passage probabilities from a Gaussian track N(mean, cov).

    Common random numbers (one fixed standard-normal draw) make calls deterministic and
    the cumulative output monotone in the horizon by construction. Callable as
    forecaster(mean, cov) -> (n_horizons,), which is the TrackedMultiAgentEnv hook.
    """

    def __init__(self, mean_motion, dt, collision_radius_m, horizons_s=HORIZONS_S,
                 n_samples=2000, seed=0, gate_sigmas=None):
        self.steps = horizon_steps(dt, horizons_s)
        self.engine = FirstEntryEngine(mean_motion, dt, collision_radius_m, self.steps[-1])
        self._z = np.random.default_rng(seed).standard_normal((n_samples, 6))
        self.gate_sigmas = gate_sigmas

    def _samples(self, mean, cov):
        cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(cov + 1e-9 * np.eye(6))
        return np.asarray(mean, dtype=float) + self._z @ chol.T

    def _gated(self, mean, cov):
        """True when the track cannot reach the collision radius within the horizon: every propagated
        mean position is more than gate_sigmas posterior standard deviations (sqrt of the position
        covariance trace) beyond r_c. The Monte Carlo estimate is zero or below its resolution there,
        so skipping it does not change a trigger decision for thresholds above that resolution."""
        pows = self.engine.pos_pows
        d = np.linalg.norm(np.einsum("ksi,i->ks", pows, np.asarray(mean, dtype=float)), axis=1)
        tr = np.einsum("ksi,ij,ksj->k", pows, np.asarray(cov, dtype=float), pows)
        return bool(np.all(d > self.engine.rc + self.gate_sigmas * np.sqrt(np.maximum(tr, 0.0))))

    def __call__(self, mean, cov):
        if self.gate_sigmas is not None and self._gated(mean, cov):
            return np.zeros(len(self.steps))
        first = self.engine.first_entry(self._samples(mean, cov))
        return cumulative_labels(first, self.steps).mean(axis=0)

    def instantaneous(self, mean, cov):
        """Max over the grid of P(|r(t_s)| < r_c): a lower bound on first-passage risk."""
        inside = self.engine.distances(self._samples(mean, cov)) < self.engine.rc
        run = np.maximum.accumulate(inside.mean(axis=0))
        return np.array([run[k - 1] for k in self.steps])
