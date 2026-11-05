"""TrackedMultiAgentEnv: replaces exact-state actor observations with filtered tracks.

Wraps a ScaledMultiAgentEnv (so rewards, costs, fuel penalty and action scaling are
passed through unchanged). Ego i keeps a filter for the primary and one per live
neighbour. Slot 1 is the neighbour with the smallest predicted miss from the TRACK
MEANS only, so slot selection is identical in every arm.

Measurement noise is drawn for every pair at every step regardless of liveness or
actions, so noise sequences do not depend on the policy. Finished egos return their
last observation (frozen). The privileged state used by the critic is exposed via
privileged_state() / joint_observation().
"""

from __future__ import annotations

import numpy as np

from constellai.evaluation.scaled_env import POSITION_SCALE_M
from constellai.models.marl.objective import predicted_miss
from constellai.tracking.kalman import PairFilter

SLOT_DIM = 13
N_HORIZONS = 4


class TrackedMultiAgentEnv:
    def __init__(self, scaled_env, sigma_p, process_accel_std, sigma_v0=1.0,
                 noise_seed=0, forecaster=None, selection_horizon_s=600.0):
        self.scaled = scaled_env
        self.raw = scaled_env.env
        self.n_agents = scaled_env.n_agents
        self.dt = self.raw.dt
        self.max_thrust_mps2 = scaled_env.max_thrust_mps2
        self.action_limit = scaled_env.action_limit
        self.fuel_budget = scaled_env.fuel_budget
        self.sigma_p = sigma_p
        self.process_accel_std = process_accel_std
        self.sigma_v0 = sigma_v0
        self.noise_seed = noise_seed
        self.forecaster = forecaster
        self.selection_horizon_s = selection_horizon_s
        self.obs_dim = 1 + 2 * SLOT_DIM + (2 * N_HORIZONS if forecaster is not None else 0)
        self._episode = 0
        self._primary, self._neighbour = [], {}
        self._last_obs, self._slot1 = [], []
        self._rng_step = None
        self.last_noise = None

    # ---- episode control -------------------------------------------------
    def reset(self, initial_positions_m, initial_velocities_mps, episode_index=None):
        idx = self._episode if episode_index is None else int(episode_index)
        self._episode = idx + 1
        self.scaled.reset(initial_positions_m, initial_velocities_mps)
        rng_init = np.random.default_rng([self.noise_seed, idx, 0])
        self._rng_step = np.random.default_rng([self.noise_seed, idx, 1])
        truth = self.raw.agent_state_vectors()
        n, mm, dt = self.n_agents, self.raw.mean_motion, self.dt
        self._primary, self._neighbour = [], {}
        for i in range(n):
            f = PairFilter(mm, dt, self.sigma_p, 0.0, self.sigma_v0)  # primary: known, no process noise
            f.init(-truth[i], rng_init)
            self._primary.append(f)
            for j in range(n):
                if j != i:
                    g = PairFilter(mm, dt, self.sigma_p, self.process_accel_std, self.sigma_v0)
                    g.init(truth[j] - truth[i], rng_init)
                    self._neighbour[(i, j)] = g
        live = self.raw.live_mask()
        self._slot1 = [None] * n
        self._last_obs = [self._build_obs(i, live) for i in range(n)]
        return [o.copy() for o in self._last_obs]

    def step(self, actions):
        live_before = self.raw.live_mask()
        _, result = self.scaled.step(actions)
        n = self.n_agents
        noise_pri = self._rng_step.standard_normal((n, 3))
        noise_nb = self._rng_step.standard_normal((n, n, 3))
        self.last_noise = (noise_pri, noise_nb)
        truth = self.raw.agent_state_vectors()
        live_after = self.raw.live_mask()
        for i in range(n):
            if not live_before[i]:
                continue
            u_ego = np.asarray(result.infos[i]["thrust_mps2"], dtype=float)
            known = -u_ego  # (known threat accel) - (ego applied accel)
            fp = self._primary[i]
            fp.predict(known)
            fp.update(-truth[i][:3] + self.sigma_p * noise_pri[i])
            for j in range(n):
                if j == i or not live_before[j]:
                    continue
                g = self._neighbour[(i, j)]
                g.predict(known)
                g.update(truth[j][:3] - truth[i][:3] + self.sigma_p * noise_nb[i, j])
        obs = []
        for i in range(n):
            if live_before[i]:
                self._last_obs[i] = self._build_obs(i, live_after)
            obs.append(self._last_obs[i].copy())
        return obs, result

    # ---- observation building -------------------------------------------
    @staticmethod
    def _slot(f):
        std = np.sqrt(np.diag(f.cov))
        return np.concatenate([[1.0], f.mean[:3] / POSITION_SCALE_M, f.mean[3:],
                               std[:3] / POSITION_SCALE_M, std[3:]])

    def _select_neighbour(self, i, live):
        best, best_miss = None, np.inf
        for j in range(self.n_agents):
            if j == i or not live[j]:
                continue
            m = self._neighbour[(i, j)].mean
            miss = predicted_miss(m[:3], m[3:], self.selection_horizon_s)
            if miss < best_miss:
                best, best_miss = j, miss
        return best

    def _forecast(self, f):
        out = np.asarray(self.forecaster(f.mean, f.cov), dtype=float)
        assert out.shape == (N_HORIZONS,), out.shape
        return out

    def _build_obs(self, i, live):
        fuel = self.raw.fuel_remaining()[i] / self.fuel_budget
        primary = self._primary[i]
        j = self._select_neighbour(i, live)
        self._slot1[i] = j
        nb = self._neighbour[(i, j)] if j is not None else None
        parts = [[fuel], self._slot(primary), self._slot(nb) if nb is not None else np.zeros(SLOT_DIM)]
        if self.forecaster is not None:
            parts.append(self._forecast(primary))
            parts.append(self._forecast(nb) if nb is not None else np.zeros(N_HORIZONS))
        return np.concatenate(parts)

    # ---- privileged / diagnostic accessors ------------------------------
    def slot1_index(self, i):
        return self._slot1[i]

    def privileged_state(self):
        truth, fuel = self.raw.agent_state_vectors(), self.raw.fuel_remaining()
        return np.concatenate([
            np.concatenate([x[:3] / POSITION_SCALE_M, x[3:], [f / self.fuel_budget]])
            for x, f in zip(truth, fuel)
        ])

    joint_observation = privileged_state

    def pair_track(self, i, j=None):
        truth = self.raw.agent_state_vectors()
        if j is None:
            f, x = self._primary[i], -truth[i]
        else:
            f, x = self._neighbour[(i, j)], truth[j] - truth[i]
        return f.mean.copy(), f.cov.copy(), x
