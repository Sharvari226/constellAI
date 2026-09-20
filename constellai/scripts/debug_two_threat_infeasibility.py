"""Diagnostic: why does the two-threat scenario report solver_success=False
even at a generous thrust bound? Prints the actual scipy result object
(message, constraint values at the returned point) instead of just the
boolean success flag, so we can see the real reason rather than guess.
"""

import numpy as np
from scipy.optimize import minimize

from constellai.safety.multi_threat_filter import Threat, _hocbf_constraint_terms

threats = [
    Threat(relative_position_m=np.array([60.0, 0.0, 0.0]), relative_velocity_mps=np.array([-5.0, 0.0, 0.0])),
    Threat(relative_position_m=np.array([-60.0, 0.0, 0.0]), relative_velocity_mps=np.array([5.0, 0.0, 0.0])),
]
mean_motion = 0.0011
max_thrust = 1.0

constraints = []
for i, threat in enumerate(threats):
    p, v, r = threat.relative_position_m, threat.relative_velocity_mps, threat.safety_radius_m
    grad, b, h = _hocbf_constraint_terms(p, v, r, mean_motion, 1.0, 1.0)
    print(f"Threat {i}: gradient={grad}, b={b:.4f}, h={h:.4f}")
    constraints.append({"type": "ineq", "fun": (lambda u, g=grad, bb=b: np.dot(g, u) - bb)})

x0 = np.zeros(3)
bounds = [(-max_thrust, max_thrust)] * 3

result = minimize(
    lambda u: float(np.dot(u - x0, u - x0)),
    x0=x0, method="SLSQP", bounds=bounds, constraints=constraints,
)

print(f"\nsuccess={result.success}")
print(f"message={result.message}")
print(f"x={result.x}")
for i, c in enumerate(constraints):
    print(f"constraint {i} value at solution: {c['fun'](result.x):.4f} (must be >= 0)")