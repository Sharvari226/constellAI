"""Follow-up to esa_study.py: paired differences, normalized tertile regret, prior-shift control."""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import esa_study as es


def rv(flag, y, r):
    return r * flag + (y & ~flag) - r * y


def shift(p, pi_tr, pi_te):
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    z = z + np.log(pi_te / (1 - pi_te)) - np.log(pi_tr / (1 - pi_tr))
    return 1 / (1 + np.exp(-z))


def tert_rows(tag, X, y, p, edges, r):
    t = np.digitize(X["cov_size"].values, edges)
    rows = []

    for k in range(3):
        m = t == k

        if m.sum() < 20 or y[m].sum() == 0:
            continue

        yk = y[m]
        none = (1 - r) * yk.mean()

        rows.append(dict(
            set=tag,
            tertile=k,
            n=int(m.sum()),
            pos=int(yk.sum()),
            prevalence=yk.mean(),
            mean_p=p[m].mean(),
            ratio=p[m].mean() / yk.mean(),
            ece=es.ece(p[m], yk),
            untuned_rel_regret=rv(p[m] >= r, yk, r).mean() / none
        ))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--cuts", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--n-boot", type=int, default=2000)

    a = ap.parse_args()

    tr, te = es.read(a.train), es.read(a.test)
    pr = pd.read_csv(a.private)

    yt_all = pr.set_index("event_id")["true_risk"] > es.THR
    ytr = es.final_labels(tr)

    ids = np.array(sorted(tr.event_id.unique()))
    perm = np.random.default_rng(0).permutation(len(ids))
    fit_ids = set(ids[perm[:int(0.7 * len(ids))]])

    pd.set_option("display.width", 220)

    for cut in a.cuts:
        Xtr, Xte = es.at_cut(tr, cut), es.at_cut(te, cut)
        Xte = Xte[Xte.index.isin(yt_all.index)]

        y_tr = ytr.reindex(Xtr.index).values.astype(bool)
        y_te = yt_all.reindex(Xte.index).values.astype(bool)

        fit = Xtr.index.isin(fit_ids)

        Xf, Xv = Xtr[fit], Xtr[~fit]
        yf, yv = y_tr[fit], y_tr[~fit]

        model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            random_state=0
        )

        model.fit(Xf[es.FEATS], yf)

        p4v = model.predict_proba(Xv[es.FEATS])[:, 1]
        p4t = model.predict_proba(Xte[es.FEATS])[:, 1]

        idx = np.random.default_rng(1).integers(
            0, len(y_te), (a.n_boot, len(y_te))
        )

        print(
            f"\n=== cut {cut:g} d "
            f"(test n={len(y_te)}, pos={int(y_te.sum())}) ==="
        )

        out = []

        for r in a.ratios:
            f3 = Xte["risk"].values >= es.tune(Xv["risk"].values, yv, r)
            f4 = p4t >= es.tune(p4v, yv, r)
            fu = p4t >= r

            comparisons = [
                ("tunedS3 - tunedS4",
                 rv(f3, y_te, r) - rv(f4, y_te, r)),
                ("untunedS4 - tunedS4",
                 rv(fu, y_te, r) - rv(f4, y_te, r))
            ]

            for name, d in comparisons:
                m = d[idx].mean(axis=1)

                out.append(dict(
                    r=r,
                    diff=name,
                    mean=d.mean(),
                    lo=np.percentile(m, 2.5),
                    hi=np.percentile(m, 97.5)
                ))

        print(pd.DataFrame(out).round(4).to_string(index=False))

        edges = np.quantile(Xf["cov_size"], [1 / 3, 2 / 3])

        r0 = 0.05
        pi_tr, pi_te = yf.mean(), y_te.mean()

        rows = tert_rows("val (no shift)", Xv, yv, p4v, edges, r0)

        rows += tert_rows(
            "test raw", Xte, y_te, p4t, edges, r0
        )

        rows += tert_rows(
            "test prior-shift-corrected (diagnostic)",
            Xte,
            y_te,
            shift(p4t, pi_tr, pi_te),
            edges,
            r0
        )

        print(
            f"\nS4 tertile calibration at r={r0:g}; "
            f"fit prevalence {pi_tr:.4f}, "
            f"test prevalence {pi_te:.4f}; "
            "ratio = mean_p / prevalence"
        )

        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
