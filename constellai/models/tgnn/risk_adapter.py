"""M3->M4 bridge: wraps a trained TGNLite model so M4's environments can
query "predicted collision risk" for their current relative state.

Design choice, stated explicitly: this augments the RL OBSERVATION
only -- it never touches safety_cost. safety_cost remains ground-truth
collision, exactly as validated in every M4 test. Feeding a forecast
into the safety cost itself (CPO's own "cost shaping" idea) was
considered and rejected here: it would mean the Lagrangian rung's
multiplier trains against a mix of ground truth and a fallible
forecast, undermining the already-verified "lambda provably tracks real
constraint violation" property. Observation augmentation carries none
of that risk -- the policy can learn to use the forecast or ignore it,
the underlying CMDP's correctness is untouched.

UNIT FIX (previously a real bug): TGN-lite trains exclusively on
km/km-s features (from propagation.py's propagate_series). Position was
already converted m -> km here, but relative SPEED was passed through
unconverted (still m/s) -- every risk query was feeding the model a
velocity value ~1000x outside its training scale. Both are now
converted consistently.

Disclosed, NOT fixed by the above: M4's HCW frame operates at
meter-to-kilometer scale (50m collision radius, 1000m mission target),
while TGN-lite trained on whole-orbit separations at hundreds-to-
thousands-of-km scale (THRESHOLD_KM=300 in the training scenarios).
Even with correct units, M4's actual separations are far outside
anything TGN-lite saw in training -- this bridge's predictions should
be treated as unvalidated at M4's operating scale until TGN-lite is
retrained on data at that scale, or inputs are deliberately rescaled
into its training distribution. Do not present this bridge's output as
reliable without that caveat.
"""

from __future__ import annotations

import numpy as np
import torch

from constellai.models.tgnn.dynamic_graph import GraphEvent
from constellai.models.tgnn.tgn_lite import TGNLite

M_TO_KM = 1.0 / 1000.0


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
            the Beta distribution's standard deviation.
        """
        position_km = relative_position_m * M_TO_KM
        separation_km = float(np.linalg.norm(position_km))
        rel_speed_km_s = float(np.linalg.norm(relative_velocity_mps)) * M_TO_KM

        features = np.concatenate([
            position_km,
            [separation_km, rel_speed_km_s],
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