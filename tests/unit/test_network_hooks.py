import numpy as np
import torch

from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import reference_mean_motion
from constellai.models.marl.checkpoint import build_networks, load_checkpoint, save_checkpoint
from constellai.models.marl.constrained_mappo import collect_rollout, constrained_update
from constellai.models.marl.joint_obs import joint_observation
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv

N = reference_mean_motion()
Z = np.zeros(3)


class _PlainEnv:
    pass


class _HookEnv:
    def joint_observation(self):
        return np.arange(4.0)


def test_joint_observation_falls_back_to_concatenation():
    obs = [np.ones(3), 2 * np.ones(3)]
    np.testing.assert_array_equal(joint_observation(_PlainEnv(), obs), np.concatenate(obs))


def test_joint_observation_uses_env_hook():
    out = joint_observation(_HookEnv(), [np.ones(27), np.ones(27)])
    np.testing.assert_array_equal(out, np.arange(4.0))


def test_builder_sizes_actor_but_not_critic():
    nets = build_networks("constrained_mappo", 3, 1.0, actor_obs_dim=27)
    assert nets["actor"].net[0].in_features == 27
    for key in ("reward_critic", "cost_critic"):
        assert list(nets[key].parameters())[0].shape[1] == 3 * 7 + 3
    assert build_networks("mappo", 3, 1.0)["actor"].net[0].in_features == 7


def test_checkpoint_roundtrip_and_backward_compat(tmp_path):
    nets = build_networks("constrained_mappo", 2, 1.0, actor_obs_dim=27)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, "constrained_mappo", 2, 1.0, nets, {"seed": 0}, actor_obs_dim=27)
    assert torch.load(path, weights_only=True)["actor_obs_dim"] == 27
    load_checkpoint(path)  # raises a size mismatch if the actor were rebuilt at width 7
    old = build_networks("constrained_mappo", 2, 1.0)
    old_path = tmp_path / "old.pt"
    save_checkpoint(old_path, "constrained_mappo", 2, 1.0, old, {"seed": 0})
    payload = torch.load(old_path, weights_only=True)
    payload.pop("actor_obs_dim")
    torch.save(payload, old_path)
    load_checkpoint(old_path)


def test_constrained_mappo_trains_on_tracked_observations():
    raw = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N, max_steps=8)
    scaled = ScaledMultiAgentEnv(raw, 2, raw.max_thrust_mps2, raw.dt)
    env = TrackedMultiAgentEnv(scaled, sigma_p=25.0, process_accel_std=0.0, neighbour_thrust_known=True)
    nets = build_networks("constrained_mappo", 2, 1.0, actor_obs_dim=env.obs_dim, critic_obs_dim=13)
    actor, rc, cc = nets["actor"], nets["reward_critic"], nets["cost_critic"]
    pos = [np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0])]
    trajs = collect_rollout(env, actor, rc, cc, pos, [Z.copy(), Z.copy()])
    assert trajs[0].observations[0].shape == (33,)
    assert trajs[0].joint_observations[0].shape == (2 * 13,)
    out = constrained_update(
        trajs, actor, rc, cc, lagrange_lambda=1.0,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        reward_critic_optimizer=torch.optim.Adam(rc.parameters()),
        cost_critic_optimizer=torch.optim.Adam(cc.parameters()),
    )
    assert all(np.isfinite(v) for v in out.values())


def test_critic_obs_dim_is_configurable_and_roundtrips(tmp_path):
    nets = build_networks("constrained_mappo", 2, 1.0, actor_obs_dim=33, critic_obs_dim=13)
    assert list(nets["reward_critic"].parameters())[0].shape[1] == 2 * 13 + 2
    path = tmp_path / "ck2.pt"
    save_checkpoint(path, "constrained_mappo", 2, 1.0, nets, {"seed": 0}, actor_obs_dim=33, critic_obs_dim=13)
    load_checkpoint(path)  # raises on a size mismatch
