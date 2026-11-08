"""Threshold-heuristic avoidance (arm E): a simple operational baseline.

Reads only the filtered track means in the tracked observation. For each valid track it
extrapolates straight-line relative motion up to horizon_s, finds the closest approach, and
takes the most threatening track. If that predicted miss is below trigger * collision_radius
the agent burns along -m (m = threat relative position at closest approach), which increases
the miss distance; otherwise it coasts. No hysteresis; HCW curvature is ignored in the
extrapolation (a documented limitation).
"""

import numpy as np

SLOT0 = 7      # fuel (1) + own slot error (6)
SLOT1 = 20     # + primary slot (13)


def closest_approach(rel_pos, rel_vel, horizon_s):
    vv = float(rel_vel @ rel_vel)
    t = 0.0 if vv < 1e-12 else float(np.clip(-(rel_pos @ rel_vel) / vv, 0.0, horizon_s))
    m = rel_pos + rel_vel * t
    return float(np.linalg.norm(m)), m


class ThresholdHeuristicPolicy:
    def __init__(self, trigger, collision_radius_m, horizon_s=600.0, burn_fraction=1.0,
                 position_scale_m=1000.0):
        self.trigger = float(trigger)
        self.rc = float(collision_radius_m)
        self.horizon_s = horizon_s
        self.burn_fraction = burn_fraction
        self.scale = position_scale_m
        self.name = f"heuristic_tau{trigger:g}"

    def _action(self, obs):
        obs = np.asarray(obs, dtype=float)
        best = None
        for start in (SLOT0, SLOT1):
            if obs[start] < 0.5:
                continue
            pos = obs[start + 1:start + 4] * self.scale
            vel = obs[start + 4:start + 7]
            miss, m = closest_approach(pos, vel, self.horizon_s)
            if best is None or miss < best[0]:
                best = (miss, m, vel)
        if best is None or best[0] >= self.trigger * self.rc:
            return np.zeros(3)
        miss, m, vel = best
        if np.linalg.norm(m) > 1e-9:
            direction = -m / np.linalg.norm(m)
        else:  # exactly head-on: step sideways, perpendicular to the relative velocity
            perp = np.cross(vel, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp) < 1e-9:
                perp = np.cross(vel, np.array([0.0, 1.0, 0.0]))
            direction = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-9 else np.array([0.0, 0.0, 1.0])
        return self.burn_fraction * direction

    def act(self, observations):
        return [self._action(o) for o in observations]
