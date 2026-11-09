"""Does the learned actor's mean action exceed the thrust limit (which the env clips)?"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from constellai.evaluation.scenarios import make_rng, sample_scenario, warning_config

_spec = importlib.util.spec_from_file_location("evaluate_tracked", Path(__file__).with_name("evaluate_tracked.py"))
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_tracked"] = ev
_spec.loader.exec_module(ev)

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--episodes", type=int, default=10)
args = p.parse_args()

cfg = warning_config("short")
actor, meta, obs_dim = ev.load_actor(args.ckpt)
env = ev.make_env(cfg, sigma_p=meta["sigma_p"], forecast=meta["forecast"], mission=meta["mission"], noise_seed=12345)
means = []
for ep in range(args.episodes):
    pos, vel = sample_scenario(make_rng(ev.PILOT_STREAM, 0, ep), cfg)
    obs = env.reset(pos, vel, episode_index=ep)
    done = False
    while not done:
        live = env.raw.live_mask()
        with torch.no_grad():
            acts = [actor(torch.from_numpy(np.asarray(o)).float()).mean.numpy().astype(float) for o in obs]
        means.extend(a for a, l in zip(acts, live) if l)
        obs, res = env.step(acts)
        done = all(res.dones)

m = np.array(means)
mx = np.abs(m).max(axis=1)
print(f"{args.ckpt}: {len(m)} live agent-steps over {args.episodes} pilot episodes")
print(f"  fraction with any |mean| > 1 (clipped by the env): {np.mean(mx > 1.0):.3f}")
print(f"  fraction with any |mean| > 3: {np.mean(mx > 3.0):.3f}")
print(f"  max |mean| percentiles 10/50/90: {np.percentile(mx, [10, 50, 90]).round(2).tolist()}")
print(f"  mean signed action per axis: {m.mean(axis=0).round(2).tolist()}")
if hasattr(actor, "log_std"):
    print(f"  policy std per axis (training exploration): {actor.log_std.exp().detach().numpy().round(3).tolist()}")
