# Small statistics helpers for M6 reporting.

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # Wilson score interval for a binomial proportion (default 95%). Preferred
    # over the normal approximation because it stays inside [0, 1] and behaves
    # sensibly near 0% and 100%, which is exactly where safety rates live.
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    # at the boundaries the exact bound is 0 or 1; floating point can land a hair inside
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high