"""Report the one-time stream 7 evaluation of frozen thresholds (no selection happens here)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import select_and_compare as sc
from constellai.evaluation.stats import wilson_interval

PAIRS = [
    ("A0", "E"),
    ("E", "E_hcw"),
    ("E_hcw", "Ehcw_k"),
    ("Ehcw_k", "F600"),
    ("E_hcw", "F600"),
    ("F600", "B"),
    ("F600", "F400"),
    ("F400", "F200")
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="frozen_thresholds.json")
    ap.add_argument("--final-dir", default="outputs/final_stream7")
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--boot", type=int, default=2000)

    a = ap.parse_args()

    frozen = json.loads(Path(a.frozen).read_text())["chosen"]

    rng = np.random.default_rng(0)
    res, rows = {}, []

    for fam, info in frozen.items():
        f = (
            Path(a.final_dir)
            / fam
            / f"eval_{info['policy']}_stream7.json"
        )

        if not f.exists():
            print(f"{fam}: missing {f}")
            continue

        j = json.loads(f.read_text())

        assert j["stream"] == 7, (
            f"{f}: stream {j['stream']} is not 7"
        )

        t = sorted(j["trials"], key=lambda x: x["index"])

        free = np.array([
            0.0 if x["collided"] else 1.0
            for x in t
        ])

        dv = np.array([
            x["mean_fuel_used"]
            for x in t
        ])

        res[fam] = (free, dv)

        lo, hi = wilson_interval(
            int(free.sum()),
            len(free)
        )

        idx = rng.integers(
            0, len(dv), (a.boot, len(dv))
        )

        m = dv[idx].mean(axis=1)
        dlo, dhi = np.percentile(m, [2.5, 97.5])

        rows.append(dict(
            family=fam,
            policy=info["policy"],
            n=len(free),
            final_free=round(free.mean(), 3),
            free_lo=round(lo, 3),
            free_hi=round(hi, 3),
            meets_target=bool(free.mean() >= a.target),
            final_dv=round(dv.mean(), 3),
            dv_lo=round(dlo, 3),
            dv_hi=round(dhi, 3),
            val_free=round(info["val_free"], 3),
            val_dv=round(info["val_dv"], 3),
            dv_change_vs_val=round(
                dv.mean() - info["val_dv"], 3
            )
        ))

    out = Path(a.final_dir)
    cols = list(rows[0]) if rows else []

    with open(out / "report.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    print(f"FINAL stream 7, target {a.target}")

    for r in rows:
        print(r)

    print("paired differences on stream 7 (first minus second):")

    for f1, f2 in PAIRS:
        if f1 in res and f2 in res:
            assert len(res[f1][0]) == len(res[f2][0]), (
                "unequal n; cannot pair"
            )

            d, (lo, hi) = sc.boot(
                res[f1][1],
                res[f2][1],
                rng,
                a.boot
            )

            s, (slo, shi) = sc.boot(
                res[f1][0],
                res[f2][0],
                rng,
                a.boot
            )

            print(
                f"  {f1} - {f2}: "
                f"dv {d:+.4f} [{lo:+.4f},{hi:+.4f}]   "
                f"collision-free {s:+.3f} "
                f"[{slo:+.3f},{shi:+.3f}]"
            )

    print("wrote", out / "report.csv")


if __name__ == "__main__":
    main()
