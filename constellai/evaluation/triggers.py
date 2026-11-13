"""Trigger-ladder policies: same burn mechanics, different trigger signals.

A0 DistanceTriggerPolicy: trigger on CURRENT separation (no extrapolation); burns directly away.
F/B ProbabilityTriggerPolicy: trigger on a forecast probability read from the forecast block of
the 41-dim tracked observation (analytic Monte Carlo or oracle); burns along -m, m = predicted
closest-approach vector from the track means (the same direction rule as the threshold heuristic).
The point-estimate trigger (E) is ThresholdHeuristicPolicy in heuristic.py.
"""

import numpy as np

from constellai.evaluation.heuristic import SLOT0, SLOT1, closest_approach

FC0, FC1 = 33, 37  # forecast blocks (primary, neighbour) in the 41-dim observation


def _sideways(vel):
    perp = np.cross(vel, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(perp) < 1e-9:
        perp = np.cross(vel, np.array([0.0, 1.0, 0.0]))
    n = np.linalg.norm(perp)
    return perp / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def away_direction(m, vel):
    n = np.linalg.norm(m)
    return -m / n if n > 1e-9 else _sideways(vel)


class DistanceTriggerPolicy:
    def __init__(self, trigger, collision_radius_m, burn_fraction=1.0, position_scale_m=1000.0):
        self.trigger = float(trigger)
        self.rc = float(collision_radius_m)
        self.burn_fraction = burn_fraction
        self.scale = position_scale_m
        self.name = f"dist_D{trigger:g}"

    def _action(self, obs):
        obs = np.asarray(obs, dtype=float)
        best = None
        for start in (SLOT0, SLOT1):
            if obs[start] < 0.5:
                continue
            pos = obs[start + 1:start + 4] * self.scale
            vel = obs[start + 4:start + 7]
            d = float(np.linalg.norm(pos))
            if best is None or d < best[0]:
                best = (d, pos, vel)
        if best is None or best[0] >= self.trigger * self.rc:
            return np.zeros(3)
        return self.burn_fraction * away_direction(best[1], best[2])

    def act(self, observations):
        return [self._action(o) for o in observations]


class ProbabilityTriggerPolicy:
    def __init__(self, p_star, horizon_index, kind="analytic", burn_fraction=1.0,
                 position_scale_m=1000.0, horizon_s=600.0):
        self.p_star = float(p_star)
        self.k = int(horizon_index)
        self.burn_fraction = burn_fraction
        self.scale = position_scale_m
        self.horizon_s = horizon_s
        self.name = f"prob_{kind}_p{p_star:g}_h{horizon_index}"

    def _action(self, obs):
        obs = np.asarray(obs, dtype=float)
        if obs.shape[0] < 41:
            raise ValueError("ProbabilityTriggerPolicy needs the 41-dim observation with a forecast block")
        best = None
        for start, fc in ((SLOT0, FC0), (SLOT1, FC1)):
            if obs[start] < 0.5:
                continue
            p = float(obs[fc + self.k])
            if p < self.p_star:
                continue
            pos = obs[start + 1:start + 4] * self.scale
            vel = obs[start + 4:start + 7]
            _, m = closest_approach(pos, vel, self.horizon_s)
            if best is None or p > best[0]:
                best = (p, m, vel)
        if best is None:
            return np.zeros(3)
        return self.burn_fraction * away_direction(best[1], best[2])

    def act(self, observations):
        return [self._action(o) for o in observations]


class HcwMissTriggerPolicy:
    """E-HCW: a point-estimate trigger with exact dynamics (the control for E versus F).

    Like ThresholdHeuristicPolicy, but the predicted miss is the minimum separation of the
    HCW-propagated track mean over the grid, including the current separation. The burn
    direction is the shared straight-line rule, so only the TRIGGER differs from every other arm.
    """

    def __init__(self, tau, engine, burn_fraction=1.0, position_scale_m=1000.0, horizon_s=600.0):
        self.tau = float(tau)
        self.engine = engine
        self.burn_fraction = burn_fraction
        self.scale = position_scale_m
        self.horizon_s = horizon_s
        self.name = f"hcw_tau{tau:g}"

    def _action(self, obs):
        obs = np.asarray(obs, dtype=float)
        best = None
        for start in (SLOT0, SLOT1):
            if obs[start] < 0.5:
                continue
            pos = obs[start + 1:start + 4] * self.scale
            vel = obs[start + 4:start + 7]
            miss = min(float(np.linalg.norm(pos)), float(self.engine.distances(np.concatenate([pos, vel])).min()))
            if best is None or miss < best[0]:
                best = (miss, pos, vel)
        if best is None or best[0] >= self.tau * self.engine.rc:
            return np.zeros(3)
        _, pos, vel = best
        _, m = closest_approach(pos, vel, self.horizon_s)
        return self.burn_fraction * away_direction(m, vel)

    def act(self, observations):
        return [self._action(o) for o in observations]


class HcwMarginTriggerPolicy:
    """k-sigma margin rule: the uncertainty-aware point-estimate baseline.

    Trigger iff min(current separation, minimum HCW-propagated separation of the track mean over the
    grid) < r_c + k * sigma_eff, with sigma_eff the RMS of the three position standard deviations in
    the observation. The burn direction is the shared straight-line rule, so only the TRIGGER differs.
    """

    def __init__(self, k, engine, burn_fraction=1.0, position_scale_m=1000.0, horizon_s=600.0):
        self.k = float(k)
        self.engine = engine
        self.burn_fraction = burn_fraction
        self.scale = position_scale_m
        self.horizon_s = horizon_s
        self.name = f"hcwm_k{k:g}"

    def _action(self, obs):
        obs = np.asarray(obs, dtype=float)
        best = None
        for start in (SLOT0, SLOT1):
            if obs[start] < 0.5:
                continue
            pos = obs[start + 1:start + 4] * self.scale
            vel = obs[start + 4:start + 7]
            std = obs[start + 7:start + 10] * self.scale
            sigma_eff = float(np.sqrt(np.mean(std ** 2)))
            miss = min(float(np.linalg.norm(pos)), float(self.engine.distances(np.concatenate([pos, vel])).min()))
            slack = miss - self.engine.rc - self.k * sigma_eff
            if best is None or slack < best[0]:
                best = (slack, pos, vel)
        if best is None or best[0] >= 0.0:
            return np.zeros(3)
        _, pos, vel = best
        _, m = closest_approach(pos, vel, self.horizon_s)
        return self.burn_fraction * away_direction(m, vel)

    def act(self, observations):
        return [self._action(o) for o in observations]
