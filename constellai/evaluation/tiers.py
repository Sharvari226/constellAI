# Scenario difficulty tiers for the M6 Monte Carlo evaluation.
#
# nominal: the distribution used for training and the Phase 2 diagnostics. Collision
#   courses develop over 500-1500 s, so even a tiny steady drift defeats most of them
#   (a drift of 1e-4 m/s^2 over 1000 s is already 50 m, the whole collision radius).
# short_notice: collision courses develop over 200-450 s at higher closing speed
#   (0.8-1.5 m/s; at 10 s per step that is still <= 15 m per step, so the 50 m
#   collision sphere cannot be tunnelled through). Avoiding a hit now needs roughly
#   30% of the thrust limit applied early, so a small drift is no longer enough and
#   the safety filter's contribution becomes measurable.
#
# Both tiers have a zero-thrust TRIAL collision rate of about 45-50% (checked by
# tests/unit/test_tiers.py), so the baselines are comparable across tiers.

from __future__ import annotations

from constellai.evaluation.scenarios import ScenarioConfig

TIERS = {
    "nominal": ScenarioConfig(),
    "short_notice": ScenarioConfig(
        time_to_tca_s=(200.0, 450.0),
        approach_speed_mps=(0.8, 1.5),
    ),
}