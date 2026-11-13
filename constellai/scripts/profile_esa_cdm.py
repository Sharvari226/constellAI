import argparse
import csv
from collections import defaultdict
import numpy as np

KEEP = ["event_id", "time_to_tca", "mission_id", "risk", "max_risk_estimate", "miss_distance", "relative_speed",
        "mahalanobis_distance", "t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t", "t_sigma_n", "c_sigma_n",
        "t_position_covariance_det", "c_position_covariance_det"]

SIGMAS = ["t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t", "t_sigma_n", "c_sigma_n"]
HIGH = -6.0  # log10 risk threshold for the high-risk label (confirm against the dataset description)


def read_events(path):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: header.index(c) for c in KEEP if c in header}
        absent = [c for c in KEEP if c not in header]
        for row in reader:
            rec = {}
            for c, i in idx.items():
                try:
                    rec[c] = float(row[i])
                except (ValueError, IndexError):
                    rec[c] = float("nan")
            rows[int(rec["event_id"])].append(rec)
    return rows, absent


def pct(values, q=(5, 50, 95)):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return [round(float(x), 3) for x in np.percentile(a, q)] if a.size else []


def profile(path, label):
    rows, absent = read_events(path)
    events = sorted(rows)
    print(f"\n== {label}: {path}")
    if absent:
        print("  columns not found:", absent)
    counts = np.array([len(rows[e]) for e in events])
    print(f"  events {len(events)}, CDMs {int(counts.sum())}; CDMs per event min/median/max "
          f"{counts.min()}/{int(np.median(counts))}/{counts.max()}")
    tt = [r["time_to_tca"] for e in events for r in rows[e]]
    print(f"  time_to_tca (days), percentiles 5/50/95 over all CDMs: {pct(tt)}")
    last = {e: min(rows[e], key=lambda r: r["time_to_tca"]) for e in events}
    final = np.array([last[e]["risk"] for e in events])
    for thr in (-6.0, -5.0, -4.0):
        k = int(np.nansum(final > thr))
        print(f"  final risk (log10) > {thr:g}: {k} events ({k / len(events):.2%})")
    for t0 in (2.0, 3.0, 5.0):
        have = [e for e in events if any(r["time_to_tca"] >= t0 for r in rows[e])]
        if not have:
            continue
        latest = {e: min((r for r in rows[e] if r["time_to_tca"] >= t0), key=lambda r: r["time_to_tca"]) for e in have}
        y = np.array([last[e]["risk"] > HIGH for e in have])
        s = np.array([latest[e]["risk"] > HIGH for e in have])
        tp = int((y & s).sum())
        print(f"  cut at {t0:g} d to TCA: {len(have)} events, {int(y.sum())} end high-risk; naive 'latest risk > {HIGH:g}' "
              f"flags {int(s.sum())} (precision {tp / max(int(s.sum()), 1):.2f}, recall {tp / max(int(y.sum()), 1):.2f})")
        if t0 == 2.0:
            for c in SIGMAS:
                print(f"    {c}: 5/50/95 percentiles {pct([latest[e][c] for e in have])}")
            for c in ("risk", "miss_distance", "mahalanobis_distance", "max_risk_estimate"):
                print(f"    missing {c}: {float(np.mean([not np.isfinite(latest[e][c]) for e in have])):.1%}")


def profile_private(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("true_risk")
        a = np.array([float(row[i]) for row in reader])
    print(f"\n== test labels: {path}: {a.size} events")
    for thr in (-6.0, -5.0, -4.0):
        print(f"  true_risk > {thr:g}: {int((a > thr).sum())} ({(a > thr).mean():.2%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--private")
    a = ap.parse_args()
    if a.train:
        profile(a.train, "train")
    if a.test:
        profile(a.test, "test (public CDMs)")
    if a.private:
        profile_private(a.private)


if __name__ == "__main__":
    main()
