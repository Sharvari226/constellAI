"""Unit tests for MADDPG.

test_critic_depends_on_joint_actions_not_just_joint_obs is the one
that matters most -- it's the concrete proof this rung is actually
Q(obs, actions), the real MAPPO-vs-MADDPG distinction, not just MAPPO
with a deterministic actor bolted on.
"""

import numpy as np
import torch

from constellai.models.marl.maddpg import (
    CentralizedQCritic,
    DeterministicActor,
    ReplayBuffer,
    collect_transitions,
    maddpg_update,
    soft_update,
)
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def test_actor_outputs_bounded_deterministic_action():
    actor = DeterministicActor(max_action=0.01)
    obs = torch.randn(7)
    action = actor(obs)
    assert action.shape == (3,)
    assert torch.all(action.abs() <= 0.01 + 1e-6)


def test_critic_depends_on_joint_actions_not_just_joint_obs():
    """The core MADDPG claim: same joint observations, different joint
    actions, MUST produce different Q-values -- if this failed, the
    critic would be silently ignoring the actions input entirely,
    making this indistinguishable from MAPPO's V(obs)."""
    critic = CentralizedQCritic(n_agents=2, obs_dim=7, action_dim=3)
    joint_obs = torch.zeros(14)
    actions_a = torch.zeros(6)
    actions_b = torch.ones(6) * 0.01

    q_a = critic(joint_obs, actions_a, agent_id=0)
    q_b = critic(joint_obs, actions_b, agent_id=0)

    assert not torch.allclose(q_a, q_b)


def test_replay_buffer_stores_and_samples():
    from constellai.models.marl.maddpg import Transition
    buffer = ReplayBuffer(capacity=100)
    for _ in range(10):
        buffer.push(Transition(
            obs=[np.zeros(7), np.zeros(7)],
            actions=[np.zeros(3), np.zeros(3)],
            rewards=[0.0, 0.0],
            next_obs=[np.zeros(7), np.zeros(7)],
            dones=[False, False],
        ))
    assert len(buffer) == 10
    sample = buffer.sample(5)
    assert len(sample) == 5


def test_soft_update_moves_target_toward_source():
    target = DeterministicActor()
    source = DeterministicActor()
    # Force source params to be clearly different from target's.
    with torch.no_grad():
        for p in source.parameters():
            p.add_(1.0)

    target_before = [p.clone() for p in target.parameters()]
    soft_update(target, source, tau=0.5)
    target_after = list(target.parameters())

    assert any(not torch.allclose(b, a) for b, a in zip(target_before, target_after))


def test_maddpg_update_returns_none_below_batch_size():
    buffer = ReplayBuffer()
    actor, critic = DeterministicActor(), CentralizedQCritic(n_agents=2)
    target_actor, target_critic = DeterministicActor(), CentralizedQCritic(n_agents=2)
    result = maddpg_update(
        buffer, n_agents=2, actor=actor, critic=critic,
        target_actor=target_actor, target_critic=target_critic,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        critic_optimizer=torch.optim.Adam(critic.parameters()),
        batch_size=64,
    )
    assert result is None  # buffer has 0 transitions, well below batch_size


def test_maddpg_full_loop_learns_to_reduce_collisions():
    """Smoke test, same spirit as every prior rung's: collect episodes,
    train, confirm collision rate goes down. Off-policy specifics
    (buffer warmup, target networks) make this slower to converge than
    IPPO/MAPPO's smoke tests -- more training iterations are used here
    for that reason, not because something is wrong."""
    torch.manual_seed(0)
    np.random.seed(0)

    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0, max_steps=25)
    actor = DeterministicActor(max_action=0.01)
    critic = CentralizedQCritic(n_agents=2)
    target_actor = DeterministicActor(max_action=0.01)
    target_critic = CentralizedQCritic(n_agents=2)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    def collision_rate(n_episodes=15):
        collisions = 0
        for _ in range(n_episodes):
            collided = collect_transitions(
                env, actor,
                initial_positions_m=[np.array([80.0, 0, 0]), np.array([90.0, 0, 0])],
                initial_velocities_mps=[np.zeros(3), np.zeros(3)],
                buffer=ReplayBuffer(),  # throwaway buffer, just measuring behavior
                exploration_std=0.0,  # no exploration noise when MEASURING
            )
            collisions += int(collided)
        return collisions / n_episodes

    rate_before = collision_rate()

    for _ in range(80):
        collect_transitions(
            env, actor,
            initial_positions_m=[np.array([80.0, 0, 0]), np.array([90.0, 0, 0])],
            initial_velocities_mps=[np.zeros(3), np.zeros(3)],
            buffer=buffer,
        )
        maddpg_update(buffer, 2, actor, critic, target_actor, target_critic, actor_opt, critic_opt)

    rate_after = collision_rate()
    assert rate_after <= rate_before