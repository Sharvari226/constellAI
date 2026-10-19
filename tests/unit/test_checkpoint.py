# Tests for M4 checkpoint save/load.

import pytest
import torch

from constellai.models.marl.checkpoint import RUNG_NAMES, build_networks, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("rung", RUNG_NAMES)
def test_checkpoint_roundtrip_preserves_weights(rung, tmp_path):
    torch.manual_seed(0)
    n_agents = 1 if rung == "reinforce" else 3
    networks = build_networks(rung, n_agents, max_thrust_mps2=0.01)
    path = tmp_path / f"{rung}.pt"
    save_checkpoint(path, rung, n_agents, 0.01, networks, meta={"seed": 7, "ranges": (0.3, 1.0)})

    loaded = load_checkpoint(path)

    assert loaded.rung == rung
    assert loaded.n_agents == n_agents
    assert set(loaded.networks) == set(networks)
    for name, net in networks.items():
        original = net.state_dict()
        restored = loaded.networks[name].state_dict()
        assert original.keys() == restored.keys()
        for key in original:
            assert torch.equal(original[key], restored[key])
    assert loaded.meta["seed"] == 7
    assert loaded.meta["ranges"] == [0.3, 1.0]  # tuples come back as lists
    assert all(not net.training for net in loaded.networks.values())


def test_maddpg_actor_keeps_its_thrust_limit(tmp_path):
    networks = build_networks("maddpg", 3, max_thrust_mps2=0.02)
    path = tmp_path / "maddpg.pt"
    save_checkpoint(path, "maddpg", 3, 0.02, networks, meta={})
    loaded = load_checkpoint(path)
    assert loaded.networks["actor"].max_action == 0.02


def test_unknown_rung_is_rejected():
    with pytest.raises(ValueError):
        build_networks("not_a_rung", 3)