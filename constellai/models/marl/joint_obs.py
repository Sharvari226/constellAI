"""Critic input for centralized-critic methods.

Environments that distinguish the actor's observation from the privileged joint
state expose joint_observation(); otherwise the concatenated actor observations
are used, which is the original behaviour.
"""

import numpy as np


def joint_observation(env, obs_list):
    fn = getattr(env, "joint_observation", None)
    return np.asarray(fn() if callable(fn) else np.concatenate(obs_list), dtype=np.float64)
