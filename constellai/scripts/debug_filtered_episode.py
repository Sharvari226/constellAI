"""Diagnostic: print the filter's actual decision at every step of the
failing test scenario, rather than inferring from pass/fail alone."""

import numpy as np

from constellai.models.marl.environment import SingleAgentAvoidanceEnv
from constellai.safety.multi_threat_filter import Threat, multi_threat_safe_action

env = SingleAgentAvoidanceEnv(mean_motion=0.0011, collision_radius_m=50.0, max_steps=30)
obs = env.reset(initial_position_m=np.array([80.0, 0.0, 0.0]), initial_velocity_mps=np.array([-3.0, 0.0, 0.0]))

last_position = obs[:3]
last_velocity = obs[3:6]

for step in range(30):
    threat = Threat(relative_position_m=last_position, relative_velocity_mps=last_velocity, safety_radius_m=50.0)
    filter_result = multi_threat_safe_action(
        threats=[threat], proposed_action_mps2=np.zeros(3),
        mean_motion=0.0011, max_thrust_mps2=env.max_thrust_mps2,
    )
    separation = float(np.linalg.norm(last_position))
    print(f"step {step}: sep={separation:.2f}m, corrected={filter_result.was_corrected}, "
          f"solver_success={filter_result.solver_success}, safe_action={filter_result.safe_action}, "
          f"barrier_h={filter_result.barrier_values}")

    obs, result = env.step(filter_result.safe_action)
    last_position, last_velocity = obs[:3], obs[3:6]

    if result.info["collided"]:
        print(f"  *** COLLIDED at step {step} ***")
        break
    if result.done:
        print(f"  episode done at step {step}")
        break