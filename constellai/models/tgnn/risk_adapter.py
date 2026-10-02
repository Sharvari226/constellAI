"""M3->M4 bridge: wraps a trained TGNLite model so M4's environments can
query "predicted collision risk" for their current relative state.

Design choice, stated explicitly: this augments the RL OBSERVATION
only -- it never touches safety_cost. safety_cost remains ground-truth
collision, exactly as validated in every M4 test tonight (including
the Lagrangian rung's lambda-adaptation tests). Feeding a forecast into
the safety cost itself (CPO's own "cost shaping" idea) was considered
and rejected here: it would mean rung 5's Lagrange multiplier trains
against a mix of ground truth and a fallible forecast, undermining the
exact property ("lambda provably tracks real constraint violation")
already verified. Observation augmentation carries none of that risk --
the policy can learn to use the forecast or ignore it, but the
underlying CMDP's correctness is untouched.

Honest limitation: M4's state/M3's GraphEvent.features share the same
5 relative-dynamics quantities [dx,dy,dz,separation,rel_speed], which
is what makes this bridge genuine rather than forced. But feeding ONE
instantaneous state through TGNLite.run_events gives the model
per-query cold-start memory (no accumulated history) -- real,
useful risk information, but not the multi-day temporal context
TGN-lite is designed to exploit. This is a disclosed simplification of
the integration, not the model's full intended operating mode.
"""

from __future__ import annotations

import numpy as np
import torch

from constellai.models.tgnn.dynamic_graph import GraphEvent
from constellai.models.tgnn.tgn_lite import TGNLite


class TGNNRiskAdapter:
    """Wraps a trained TGNLite model for single-query risk prediction
    from a raw relative state, bypassing the multi-scenario graph
    pipeline (dynamic_graph.py) that TGN-lite normally consumes."""

    def __init__(self, model: TGNLite):
        self.model = model
        self.model.eval()

    def predict_risk(
        self, relative_position_m: np.ndarray, relative_velocity_mps: np.ndarray,
    ) -> tuple[float, float]:
        """Returns (mean_risk, std_risk) for the current relative state.

        Parameters
        ----------
        relative_position_m, relative_velocity_mps : np.ndarray, shape (3,)
            Same convention as M4's environments -- meters, m/s.

        Returns
        -------
        (mean_risk, std_risk) : tuple[float, float]
            mean_risk in [0, 1] (Beta distribution mean); std_risk is
            the Beta distribution's standard deviation, the actual
            calibrated uncertainty signal -- not a placeholder.
        """
        separation_m = float(np.linalg.norm(relative_position_m)) / 1000.0
        rel_speed = float(np.linalg.norm(relative_velocity_mps))
        features = np.concatenate([
            relative_position_m / 1000.0,
            [separation_m, rel_speed],
        ]).astype(np.float32)

        event = GraphEvent(t_index=0, node_a=0, node_b=1, features=features)

        with torch.no_grad():
            _, pair_memory_snapshots = self.model.run_events(num_nodes=2, events=[event])
            feat_tensor = torch.from_numpy(features)
            mem_a, mem_b = pair_memory_snapshots[(0, 1)]
            alpha, beta = self.model.predict_pair_distribution(mem_a, mem_b, feat_tensor)

        mean = (alpha / (alpha + beta)).item()
        variance = (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))).item()
        return mean, variance ** 0.5