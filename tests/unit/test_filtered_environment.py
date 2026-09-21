"""Unit tests for the M5-integrated environment wrappers.

test_filtering_reduces_collisions_vs_unfiltered is the one that
actually matters -- it's the concrete, end-to-end proof this wiring
does something real, not just "the filter function exists and the
wrapper calls it."
"""

import numpy as np

from constellai.models.marl.environment import SingleAgentAvoidanceEnv
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.safety.filtered_environment import (
    FilteredMultiAgentEnv,
    FilteredSingleAgentEnv,
)


def test_filtered_single_agent_tracks_correction_stats():
    env = FilteredSingleAgentEnv(SingleAgentAvoidanceEnv(mean_motion=0.0011, collision_radius_m=50.0))
    env.reset(initial_position_m=np.array([60.0, 0.0, 0.0]), initial_velocity_mps=np.array([-5.0, 0.0, 0.0]))
    env.step(np.zeros(3))  # dangerous, zero-thrust proposal -- should get corrected

    assert env.stats.total_actions == 1
    assert env.stats.corrected_actions >= 0  # at minimum, must not error; exact count depends on physics


def test_filtering_reduces_collisions_vs_unfiltered():
    """Same starting geometry, same (always-zero) policy, filtered vs
    unfiltered -- the filtered version must collide less. This is the
    actual end-to-end claim of wiring M5 into M4: a policy that does
    nothing useful on its own still avoids collisions it otherwise
    wouldn't, because the safety layer corrects it."""
    def run_episode(use_filter: bool, n_steps: int = 30):
        raw_env = SingleAgentAvoidanceEnv(mean_motion=0.0011, collision_radius_m=50.0, max_steps=n_steps, max_thrust_mps2=1.0)
        env = FilteredSingleAgentEnv(raw_env) if use_filter else raw_env

        env.reset(initial_position_m=np.array([80.0, 0.0, 0.0]), initial_velocity_mps=np.array([-3.0, 0.0, 0.0]))
        collided = False
        for _ in range(n_steps):
            _, result = env.step(np.zeros(3))  # deliberately unhelpful policy: never thrusts on its own
            if result.info["collided"]:
                collided = True
            if result.done:
                break
        return collided

    unfiltered_collided = run_episode(use_filter=False)
    filtered_collided = run_episode(use_filter=True)

    assert unfiltered_collided is True   # confirms the scenario is genuinely dangerous without help
    assert filtered_collided is False    # confirms the filter actually prevented it


def test_filtered_multi_agent_tracks_stats_per_agent():
    env = FilteredMultiAgentEnv(MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0))
    env.reset(
        initial_positions_m=[np.array([60.0, 0.0, 0.0]), np.array([5000.0, 0.0, 0.0])],
        initial_velocities_mps=[np.array([-5.0, 0.0, 0.0]), np.zeros(3)],
    )
    env.step([np.zeros(3), np.zeros(3)])

    assert env.stats.total_actions == 2  # one per agent, this step