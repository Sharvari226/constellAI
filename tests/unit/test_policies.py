# Tests for the M6 policy adapters.

import numpy as np
import pytest
import torch

from constellai.evaluation.policies import policy_from_checkpoint, policy_from_networks, untrained_policy
from constellai.models.marl.checkpoint import RUNG_NAMES, build_networks, load_checkpoint, save_checkpoint

OBS = [np.array([500.0, -200.0, 30.0, 0.4, 0.1, -0.05, 100.0]) for _ in range(3)]


@pytest.mark.parametrize("rung", RUNG_NAMES)
def test_policy_returns_one_action_per_agent(rung):
    policy = untrained_policy(rung, n_agents=3, max_thrust_mps2=0.01, seed=0)
    actions = policy.act(OBS)
    assert len(actions) == 3
    assert all(a.shape == (3,) for a in actions)


@pytest.mark.parametrize("rung", RUNG_NAMES)
def test_untrained_policy_is_reproducible_for_a_seed(rung):
    a = untrained_policy(rung, 3, 0.01, seed=2).act(OBS)
    b = untrained_policy(rung, 3, 0.01, seed=2).act(OBS)
    c = untrained_policy(rung, 3, 0.01, seed=3).act(OBS)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.allclose(a[0], c[0])


@pytest.mark.parametrize("rung", RUNG_NAMES)
def test_checkpoint_policy_matches_the_saved_networks(rung, tmp_path):
    torch.manual_seed(1)
    networks = build_networks(rung, 3, 0.01)
    path = tmp_path / f"{rung}.pt"
    save_checkpoint(path, rung, 3, 0.01, networks, meta={"seed": 1})

    direct = policy_from_networks("direct", rung, networks).act(OBS)
    restored = policy_from_checkpoint(load_checkpoint(path)).act(OBS)

    for a, b in zip(direct, restored):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_maddpg_actions_respect_the_thrust_limit():
    policy = untrained_policy("maddpg", 3, 0.01, seed=0)
    for action in policy.act(OBS):
        assert np.max(np.abs(action)) <= 0.01 + 1e-9