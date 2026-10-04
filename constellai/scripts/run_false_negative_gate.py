"""M2's final open item: run the false-negative gate against a real,
larger synthetic constellation and report the actual rate -- not the
gate mechanism's correctness (already verified in tests/unit/
test_validation.py), but the real project result this whole design
depends on.

Deliberately uses ONE tight altitude shell (not several well-separated
ones) -- that's the hard case for the coarse filter, since every
satellite becomes a coarse-filter candidate against every other one at
this scale. A multi-shell scenario would pass trivially and tell us
much less about whether the sparsity claim actually holds.

Run: python -m constellai.scripts.run_false_negative_gate
"""

import math
import random
from datetime import datetime, timedelta

from constellai.graph.validation import run_false_negative_gate
from constellai.orbital_mechanics.synthetic import make_circular_satellite

N_SATELLITES = 18
ALTITUDE_KM = 500.0
ALTITUDE_SPREAD_KM = 3.0  # tight -- deliberately makes the coarse filter's job hard
OBS_START = datetime(2026, 1, 1)
OBS_END = OBS_START + timedelta(hours=2)
STEP = timedelta(minutes=2)
DISTANCE_THRESHOLD_KM = 300.0
REGIME_MARGIN_KM = 50.0


def build_scenario(seed: int, n: int = N_SATELLITES) -> list:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        sign = 1 if i % 2 == 0 else -1
        records.append(make_circular_satellite(
            satellite_id=i,
            altitude_km=ALTITUDE_KM + rng.uniform(-ALTITUDE_SPREAD_KM, ALTITUDE_SPREAD_KM),
            inclination_rad=sign * math.radians(45 + rng.uniform(-5, 5)),
            raan_rad=rng.uniform(0, 2 * math.pi),
            mean_anomaly_rad=rng.uniform(0, 2 * math.pi),
        ))
    return records


def main():
    print(f"Building a {N_SATELLITES}-satellite single-shell scenario "
          f"(altitude {ALTITUDE_KM}km +/- {ALTITUDE_SPREAD_KM}km)...")

    all_reports = []
    for seed in range(5):
        records = build_scenario(seed)
        report = run_false_negative_gate(
            records, OBS_START, OBS_END, STEP,
            threshold_km=DISTANCE_THRESHOLD_KM,
            margin_km=REGIME_MARGIN_KM,
        )
        all_reports.append(report)

        total_possible = N_SATELLITES * (N_SATELLITES - 1) // 2
        print(f"\n--- seed {seed} ---")
        print(f"  total possible pairs (exhaustive): {total_possible}")
        print(f"  baseline flagged: {report.baseline_flagged_count}")
        print(f"  coarse filter MISSED (false negatives): {len(report.coarse_filter_missed)}")
        print(f"  fine screen rejected (different risk definition, not a failure): {len(report.fine_screen_missed)}")
        print(f"  coarse-filter false-negative rate: {report.coarse_filter_false_negative_rate:.4f}")
        print(f"  gate passed (zero coarse-filter misses): {report.passed}")
        if report.coarse_filter_missed:
            print(f"  missed events: {report.coarse_filter_missed}")

    total_baseline_flags = sum(r.baseline_flagged_count for r in all_reports)
    total_missed = sum(len(r.coarse_filter_missed) for r in all_reports)
    overall_rate = total_missed / total_baseline_flags if total_baseline_flags else 0.0

    print(f"\n=== Overall across 5 seeds ===")
    print(f"Total baseline-flagged conjunctions: {total_baseline_flags}")
    print(f"Total missed by coarse filter: {total_missed}")
    print(f"Overall false-negative rate: {overall_rate:.4f}")
    if overall_rate == 0.0:
        print("Zero misses across all seeds -- the sparsity claim holds at this scale/margin.")
    else:
        print("Non-zero misses -- report this rate honestly; consider whether "
              "REGIME_MARGIN_KM needs loosening before treating the sparse "
              "graph design as validated.")


if __name__ == "__main__":
    main()