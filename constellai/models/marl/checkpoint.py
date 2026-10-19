# Save/load for the five M4 rungs.
#
# A checkpoint stores network weights plus the metadata needed to rebuild
# the exact architecture (rung, n_agents, thrust limit). It is meant for
# EVALUATION: MADDPG target networks and all optimizer state are not saved,
# so training cannot be resumed from a checkpoint.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from constellai.models.marl.ippo import ActorCritic
from constellai.models.marl.maddpg import CentralizedQCritic, DeterministicActor
from constellai.models.marl.mappo import Actor, CentralizedCritic
from constellai.models.marl.single_agent_baseline import GaussianPolicy, ValueNet

RUNG_NAMES = ("reinforce", "ippo", "mappo", "maddpg", "constrained_mappo")
FORMAT_VERSION = 1


@dataclass
class LoadedCheckpoint:
    rung: str
    n_agents: int
    max_thrust_mps2: float
    networks: dict
    meta: dict


def build_networks(rung: str, n_agents: int, max_thrust_mps2: float = 0.01) -> dict:
    # fresh, untrained networks with the same architecture training uses
    if rung == "reinforce":
        return {"policy": GaussianPolicy(), "value": ValueNet()}
    if rung == "ippo":
        return {"policy": ActorCritic()}
    if rung == "mappo":
        return {"actor": Actor(), "critic": CentralizedCritic(n_agents=n_agents)}
    if rung == "maddpg":
        return {
            "actor": DeterministicActor(max_action=max_thrust_mps2),
            "critic": CentralizedQCritic(n_agents=n_agents),
        }
    if rung == "constrained_mappo":
        return {
            "actor": Actor(),
            "reward_critic": CentralizedCritic(n_agents=n_agents),
            "cost_critic": CentralizedCritic(n_agents=n_agents),
        }
    raise ValueError(f"unknown rung {rung!r}; expected one of {RUNG_NAMES}")


def save_checkpoint(path, rung: str, n_agents: int, max_thrust_mps2: float, networks: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "rung": rung,
        "n_agents": n_agents,
        "max_thrust_mps2": max_thrust_mps2,
        "state_dicts": {name: net.state_dict() for name, net in networks.items()},
        # JSON round trip keeps meta to plain types so weights_only loading works
        "meta": json.loads(json.dumps(meta)),
    }
    torch.save(payload, path)


def load_checkpoint(path) -> LoadedCheckpoint:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {payload['format_version']}")
    networks = build_networks(payload["rung"], payload["n_agents"], payload["max_thrust_mps2"])
    for name, net in networks.items():
        net.load_state_dict(payload["state_dicts"][name])
        net.eval()
    return LoadedCheckpoint(
        rung=payload["rung"],
        n_agents=payload["n_agents"],
        max_thrust_mps2=payload["max_thrust_mps2"],
        networks=networks,
        meta=payload["meta"],
    )