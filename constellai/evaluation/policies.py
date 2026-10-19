# Policy adapters: turn a trained (or untrained) M4 rung into the one
# interface the trial runner needs, act(observations) -> actions.
#
# Every rung's ACTOR sees only its own local observation, so deployment is
# the same for all rungs: each agent runs the actor independently. The
# centralized critics are training-only and are never used here. This is
# also how rung 1 (trained single-agent) is deployed on N agents.
#
# Evaluation is deterministic by default (the policy mean), the standard
# choice for reporting safety rates. MADDPG's actor is deterministic already.

from __future__ import annotations

import numpy as np
import torch

from constellai.models.marl.checkpoint import LoadedCheckpoint, build_networks


class TorchPolicy:
    def __init__(self, name: str, action_fn):
        self.name = name
        self.action_fn = action_fn

    def act(self, observations):
        actions = []
        with torch.no_grad():
            for obs in observations:
                tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32))
                actions.append(self.action_fn(tensor).numpy().astype(np.float64))
        return actions


def _make_action_fn(rung: str, networks: dict, deterministic: bool):
    if rung == "maddpg":
        actor = networks["actor"]
        return lambda obs: actor(obs)
    if rung == "reinforce":
        get_dist = networks["policy"]
    elif rung == "ippo":
        policy = networks["policy"]
        get_dist = lambda obs: policy(obs)[0]  # ActorCritic returns (dist, value)
    else:  # mappo, constrained_mappo
        get_dist = networks["actor"]

    def act(obs):
        dist = get_dist(obs)
        return dist.mean if deterministic else dist.sample()

    return act


def policy_from_networks(name: str, rung: str, networks: dict, deterministic: bool = True) -> TorchPolicy:
    for net in networks.values():
        net.eval()
    return TorchPolicy(name, _make_action_fn(rung, networks, deterministic))


def policy_from_checkpoint(loaded: LoadedCheckpoint, deterministic: bool = True) -> TorchPolicy:
    return policy_from_networks(loaded.rung, loaded.rung, loaded.networks, deterministic)


def untrained_policy(rung: str, n_agents: int, max_thrust_mps2: float, seed: int) -> TorchPolicy:
    # Exactly the initial weights train_rung() starts from for this seed
    # (it seeds torch, then builds the networks first), so this is the paired
    # "before training" reference for the trained checkpoint of the same seed.
    torch.manual_seed(seed)
    networks = build_networks(rung, n_agents, max_thrust_mps2)
    return policy_from_networks(f"{rung}_untrained", rung, networks, deterministic=True)