"""ESA Kelvins CDM study: signal ladder, tuned vs untuned cost-ratio decisions, calibration by covariance size.

Cost model (no fuel claim): a flag costs r, a missed high-risk event costs 1, r = c_b / L.
Labels: train = final CDM risk > -6 (assumed definition, confirm in the dataset text); test = true_risk > -6.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

RATIOS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SIG = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n"]
BASE = ["time_to_tca", "risk", "miss_distance", "mahalanobis_distance", "max_risk_estimate"]
THR = -6.0
FEATS = ["time_to_tca", "risk", "max_risk_estimate", "mahalanobis_distance", "log_miss", "n_cdm", "risk_delta"] \
    + ["l_" + c for c in SIG] + ["cov_size"]


def read(path):
    want = ["event_id"] + BASE + SIG
    have = pd.read_csv(path, nrows=0).columns
    miss = [c for c in want if c not in have]
    assert not miss, f"{path}: missing {miss}; columns start {list(have)[:30]}"
    return pd.read_csv(path, usecols=want)


def at_cut(df, cut):
    d = df[df.time_to_tca >= cut].sort_values(["event_id", "time_to_tca"])
    g = d.groupby("event_id")
    X = g.head(1).set_index("event_id")
    X["n_cdm"] = g.size()
    X["risk_delta"] = g["risk"].first() - g["risk"].last()
    X["log_miss"] = np.log10(X["miss_distance"].clip(lower=1e-3))
    for c in SIG:
        X["l_" + c] = np.log10(X[c].clip(lower=1e-6))
    X["cov_size"] = X[["l_c_sigma_r", "l_c_sigma_t", "l_c_sigma_n"]].sum(axis=1)
    return X


def final_labels(df):
    f = df.sort_values(["event_id", "time_to_tca"]).groupby("event_id").head(1).set_index("event_id")["risk"]
    return f > THR


def tune(score, y, r):
    o = np.argsort(-score, kind="stable")
    s = score[o]
    yy = y[o].astype(float)
    tp = np.concatenate([[0.0], np.cumsum(yy)])
    k = np.arange(len(s) + 1)
    cost = r * k + (yy.sum() - tp)
    ok = np.ones(len(s) + 1, dtype=bool)
    ok[1:-1] = s[:-1] > s[1:]
    cost = np.where(ok, cost, np.inf)
    j = int(np.argmin(cost))
    return (np.inf if j == 0 else s[j - 1])


def regret(flag, y, r, idx):
    reg = r * flag + (y & ~flag) - r * y
    m = reg[idx].mean(axis=1)
    return reg.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def ece(p, y, nb=10):
    q = np.quantile(p, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb - 1)
    out = 0.0
    for i in range(nb):
        m = b == i
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--cuts", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--ctx-ratio", type=float, default=0.05)
    ap.add_argument("--out-dir", default="outputs/esa_study")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr, te = read(a.train), read(a.test)
    pr = pd.read_csv(a.private)
    assert {"event_id", "true_risk"} <= set(pr.columns), f"private columns: {list(pr.columns)}"
    yt_all = pr.set_index("event_id")["true_risk"] > THR
    ytr = final_labels(tr)
    ids = np.array(sorted(tr.event_id.unique()))
    perm = np.random.default_rng(0).permutation(len(ids))
    fit_ids = set(ids[perm[: int(0.7 * len(ids))]])

    rows, ctx = [], []
    for cut in a.cuts:
        Xtr, Xte = at_cut(tr, cut), at_cut(te, cut)
        Xte = Xte[Xte.index.isin(yt_all.index)]
        y_tr = ytr.reindex(Xtr.index).values.astype(bool)
        y_te = yt_all.reindex(Xte.index).values.astype(bool)
        is_fit = Xtr.index.isin(fit_ids)
        Xf, Xv, yf, yv = Xtr[is_fit], Xtr[~is_fit], y_tr[is_fit], y_tr[~is_fit]
        print(f"\n=== cut {cut:g} d: train fit {len(Xf)} ({yf.sum()} pos), val {len(Xv)} ({yv.sum()} pos), "
              f"test {len(Xte)} ({y_te.sum()} pos); prevalence train {y_tr.mean():.4f} vs test {y_te.mean():.4f}")
        if yf.sum() < 30:
            print("WARNING: fewer than 30 positives in the fit split")
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
        model.fit(Xf[FEATS], yf)
        p4v, p4t = model.predict_proba(Xv[FEATS])[:, 1], model.predict_proba(Xte[FEATS])[:, 1]
        sig = {
            "S1_miss": (-Xv["log_miss"].values, -Xte["log_miss"].values, None, None),
            "S2_maha": (-Xv["mahalanobis_distance"].values, -Xte["mahalanobis_distance"].values, None, None),
            "S3_risk": (Xv["risk"].values, Xte["risk"].values, 10.0 ** Xv["risk"].values, 10.0 ** Xte["risk"].values),
            "S4_learned": (p4v, p4t, p4v, p4t),
        }
        idx = np.random.default_rng(1).integers(0, len(y_te), (a.n_boot, len(y_te)))
        edges = np.quantile(Xf["cov_size"], [1 / 3, 2 / 3])
        tert = np.digitize(Xte["cov_size"].values, edges)
        for r in RATIOS:
            for name, (sv, st, pv, pt) in sig.items():
                t = tune(sv, yv, r)
                m, lo, hi = regret(st >= t, y_te, r, idx)
                row = dict(cut=cut, signal=name, r=r, tuned_regret=m, tuned_lo=lo, tuned_hi=hi)
                if pt is not None:
                    mu, lou, hiu = regret(pt >= r, y_te, r, idx)
                    row.update(untuned_regret=mu, untuned_lo=lou, untuned_hi=hiu)
                rows.append(row)
            rows.append(dict(cut=cut, signal="flag_none", r=r, tuned_regret=((1 - r) * y_te).mean()))
            rows.append(dict(cut=cut, signal="flag_all", r=r, tuned_regret=(r * (1 - y_te)).mean()))
        for name, (sv, st, pv, pt) in sig.items():
            if pt is None:
                continue
            t = tune(sv, yv, a.ctx_ratio)
            for k in range(3):
                mk = tert == k
                if mk.sum() < 20:
                    continue
                yk = y_te[mk]
                ctx.append(dict(cut=cut, signal=name, tertile=k, n=int(mk.sum()), pos=int(yk.sum()),
                                prevalence=yk.mean(), mean_p=pt[mk].mean(), ece=ece(pt[mk], yk),
                                tuned_regret=regret(st[mk] >= t, yk, a.ctx_ratio, np.arange(mk.sum())[None, :])[0],
                                untuned_regret=regret(pt[mk] >= a.ctx_ratio, yk, a.ctx_ratio, np.arange(mk.sum())[None, :])[0]))
        print(f"ECE overall: S3_risk {ece(sig['S3_risk'][3], y_te):.4f}, S4_learned {ece(p4t, y_te):.4f}")

    R = pd.DataFrame(rows)
    C = pd.DataFrame(ctx)
    R.to_csv(out / "regret_by_cut_signal_ratio.csv", index=False)
    C.to_csv(out / "context_by_covariance_tertile.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n== regret per event vs oracle (lower is better); tuned on val, untuned = flag iff p >= r ==")
    print(R.round(4).to_string(index=False))
    print(f"\n== context test at r = {a.ctx_ratio:g}: tertile 0 = smallest chaser covariance, 2 = largest ==")
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
