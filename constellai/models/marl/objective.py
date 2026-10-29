# constellai/models/marl/objective.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ObjectiveConfig:
    d_safe: float = 1.0      # km, keep-out radius (hard collision)
    d_warn: float = 5.0      # km, soft-cost onset
    horizon: float = 600.0   # s, look-ahead for predicted miss
    dv_max: float = 1.0      # action scale, normalises fuel to [0,1]
    deadband: float = 0.05   # km, no station-keeping penalty inside this
    track_scale: float = 10.0
    w_fuel: float = 1.0
    w_track: float = 0.1
    w_smooth: float = 0.05
    burn_eps: float = 1e-6

def predicted_miss(r, v, horizon):
    """Min separation over [0, horizon] under linear relative motion."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    vv = float(v @ v)
    t = 0.0 if vv < 1e-12 else float(np.clip(-(r @ v) / vv, 0.0, horizon))
    return float(np.linalg.norm(r + v * t))

def shaped_risk(d, cfg):
    """0 beyond d_warn, quadratic ramp to 1 at d_safe, 1 inside."""
    x = (cfg.d_warn - d) / (cfg.d_warn - cfg.d_safe)
    return float(np.clip(x, 0.0, 1.0) ** 2)

def reward(dv, dv_prev, slot_err, cfg):
    fuel = np.linalg.norm(dv) / cfg.dv_max
    track = max(0.0, np.linalg.norm(slot_err) - cfg.deadband) / cfg.track_scale
    smooth = np.linalg.norm(np.asarray(dv) - np.asarray(dv_prev)) / cfg.dv_max
    return -(cfg.w_fuel * fuel + cfg.w_track * track**2 + cfg.w_smooth * smooth)

def costs(dv, a_policy, a_filtered, threats, others, cfg):
    """threats/others: lists of (r_rel, v_rel). Returns dict of per-step costs."""
    c_conj = max((shaped_risk(predicted_miss(r, v, cfg.horizon), cfg)
                  for r, v in threats), default=0.0)
    burned = np.linalg.norm(dv) > cfg.burn_eps
    c_sec = 0.0
    if burned:
        c_sec = max((shaped_risk(predicted_miss(r, v, cfg.horizon), cfg)
                     for r, v in others), default=0.0)
    c_filter = (np.linalg.norm(np.asarray(a_policy) - np.asarray(a_filtered))
                / cfg.dv_max)
    collided = any(np.linalg.norm(r) < cfg.d_safe for r, _ in threats + others)
    return {"conj": c_conj, "sec": c_sec, "filter": float(c_filter),
            "collision": float(collided)}   # collision: eval only, not a λ cost

class PIDLagrange:
    """Per-constraint multiplier (Stooke et al. 2020); damps λ oscillation."""
    def __init__(self, budget, kp=0.5, ki=0.05, kd=0.2, lam_max=50.0):
        self.d, self.kp, self.ki, self.kd, self.lam_max = budget, kp, ki, kd, lam_max
        self.I, self.prev_J, self.lam = 0.0, None, 0.0
    def update(self, mean_cost):
        err = mean_cost - self.d
        self.I = max(0.0, self.I + self.ki * err)
        D = 0.0 if self.prev_J is None else max(0.0, mean_cost - self.prev_J)
        self.prev_J = mean_cost
        self.lam = float(np.clip(self.kp * max(err, 0) + self.I + self.kd * D,
                                 0.0, self.lam_max))
        return self.lam