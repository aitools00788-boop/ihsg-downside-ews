"""Downside-event early warning experiments.

Event: E_t = 1 if the next-day return of the IHSG is <= -1.5 % (robustness: -1.0 %, -2.0 %).
Compares the rival representations (continuous, binary) with the scale-invariant ablation
and the proposed hybrid representation under expanding-window walk-forward evaluation.

Usage:  python run_downside.py data/ihsg_daily.csv results figures
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import run_experiments as R
from features import BIN_COLS, CONT_COLS, HYB_COLS, SI_COLS, build

warnings.filterwarnings("ignore")
DATA, RES, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = R.SEEDS
TEST_YEARS = R.TEST_YEARS
REPS = R.REPS
ALERT_Q = 0.80  # alert when P(event) is above the 80th percentile of validation-period scores


def wf(F, cols, seed, ev, model="xgb"):
    rows = []
    yr = F["Date"].dt.year
    for Y in TEST_YEARS:
        tr, te = F[yr < Y], F[yr == Y]
        va = tr["Date"].dt.year >= Y - 2
        Xtr, ytr = tr[cols].values, tr[ev].values.astype(int)
        Xte, yte = te[cols].values, te[ev].values.astype(int)
        Xa, ya = tr.loc[~va, cols].values, tr.loc[~va, ev].values.astype(int)
        Xv, yv = tr.loc[va, cols].values, tr.loc[va, ev].values.astype(int)
        if cols is CONT_COLS:
            sc = MinMaxScaler().fit(Xtr)
            Xtr, Xte, Xa, Xv = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xa), sc.transform(Xv)
        if model == "xgb":
            m, _, mv = R.fit_xgb(Xa, ya, Xv, yv, seed, Xtr, ytr)
            p, pv = m.predict_proba(Xte)[:, 1], mv.predict_proba(Xv)[:, 1]
        elif model in ("lr", "rf"):
            mk = (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))) if model == "lr" else \
                 (lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=8, random_state=seed))
            pv = mk().fit(Xa, ya).predict_proba(Xv)[:, 1]
            p = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        elif model == "volrule":  # naive volatility-clustering rule: score = 20-day volatility
            p, pv = te["vol_20"].values, tr.loc[va, "vol_20"].values
        elif model == "climatology":
            p, pv = np.full(len(te), ytr.mean()), np.full(int(va.sum()), ytr.mean())
        rows.append(pd.DataFrame(dict(Date=te["Date"].values, year=Y, y=yte, p=p,
                                      ret_next=te["ret_next"].values, tau=np.quantile(pv, ALERT_Q))))
    return pd.concat(rows, ignore_index=True)


def alert_stats(P):
    S = P.p.values >= P.tau.values
    E = P.y.values.astype(bool)
    A, B, C, D = (S & E).sum(), (S & ~E).sum(), (~S & E).sum(), (~S & ~E).sum()
    prec = A / (A + B) if A + B else np.nan
    rec = A / (A + C) if A + C else np.nan
    nsr = (B / (B + D)) / (A / (A + C)) if A else np.nan
    return dict(alert_share=float(S.mean()), precision=float(prec), recall=float(rec),
                lift=float(prec / E.mean()) if A + B else np.nan, nsr=float(nsr),
                A=int(A), B=int(B), C=int(C), D=int(D))


def score(P):
    y, p = P.y.values, P.p.values
    auc = roc_auc_score(y, p) if P.p.nunique() > 1 else 0.5
    ap = average_precision_score(y, p) if P.p.nunique() > 1 else float(y.mean())
    return dict(auc=float(auc), pr_auc=float(ap), base_rate=float(y.mean()), **alert_stats(P))


def paired_block_bootstrap(y, pa, pb, B=2000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / block))
    d = []
    for _ in range(B):
        idx = (rng.integers(0, n - block, nb)[:, None] + np.arange(block)).ravel()[:n]
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.array(d)
    return dict(diff=float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), ci95=[float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5))], p_one_sided=float((d <= 0).mean()))


def main():
    F = build(pd.read_csv(DATA, parse_dates=["Date"]))
    last = F[F["target"].isna()].copy()
    F = F[F["target"].notna()].reset_index(drop=True)
    out = {}
    for thr in (-0.015, -0.010, -0.020):
        ev = f"ev_{abs(thr)*1000:.0f}"
        F[ev] = (F["ret_next"] <= thr).astype(int)
        res, preds = {}, {}
        for name, cols in REPS.items():
            seeds = SEEDS if thr == -0.015 else [0]
            sc = []
            for s in seeds:
                P = wf(F, cols, s, ev)
                sc.append(score(P))
                if s == 0:
                    preds[name] = P
            sc = pd.DataFrame(sc)
            res[name] = dict(mean=sc.mean().to_dict(), std=sc.std().to_dict())
            print(thr, name, sc.mean()[["auc", "pr_auc", "precision", "recall", "lift", "nsr"]].round(4).to_dict(), flush=True)
        for name, mdl in [("Climatology", "climatology"), ("Vol-20 rule", "volrule"),
                          ("LogReg (HYB)", "lr"), ("RandomForest (HYB)", "rf")]:
            P = wf(F, HYB_COLS, 0, ev, model=mdl)
            preds[name] = P
            res[name] = dict(mean=score(P))
            print(thr, name, {k: round(v, 4) for k, v in res[name]["mean"].items() if k in ("auc", "pr_auc", "precision", "recall", "lift", "nsr")}, flush=True)
        if thr == -0.015:
            y = preds["HYB (proposed)"].y.values
            res["bootstrap_auc_diff"] = {k: paired_block_bootstrap(y, preds["HYB (proposed)"].p.values, preds[k].p.values)
                                         for k in ["BIN (rival)", "CONT (rival)", "SI (ablation)", "Vol-20 rule",
                                                   "LogReg (HYB)", "RandomForest (HYB)"]}
            # random-split contrast (rival protocol) for the downside event
            rs = {}
            for name, cols in REPS.items():
                vals = []
                for s in SEEDS:
                    X, yy = F[cols].values, F[ev].values
                    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.3, shuffle=True, random_state=s)
                    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, shuffle=True, random_state=s)
                    m, _, _ = R.fit_xgb(Xa, ya, Xv, yv, s, Xtr, ytr)
                    vals.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
                rs[name] = dict(auc_mean=float(np.mean(vals)), auc_std=float(np.std(vals, ddof=1)))
            res["random_split_auc"] = rs
            # yearly AUC for the figure
            yearly = {}
            for name in ["CONT (rival)", "BIN (rival)", "HYB (proposed)", "Vol-20 rule"]:
                P = preds[name]
                yearly[name] = {int(Y): float(roc_auc_score(g.y, g.p)) for Y, g in P.groupby("year") if g.y.nunique() > 1}
            pd.DataFrame(yearly).to_csv(f"{RES}/downside_yearly_auc.csv")
            for name, P in preds.items():
                P.to_csv(f"{RES}/downside_pred_{name.split()[0]}.csv", index=False)
            # SHAP on the proposed model, refitted per fold (seed 0)
            import xgboost as xgbm
            contrib = []
            yrs = F["Date"].dt.year
            for Y in TEST_YEARS:
                tr, te = F[yrs < Y], F[yrs == Y]
                va = tr["Date"].dt.year >= Y - 2
                m, _, _ = R.fit_xgb(tr.loc[~va, HYB_COLS].values, tr.loc[~va, ev].values, tr.loc[va, HYB_COLS].values,
                                    tr.loc[va, ev].values, 0, tr[HYB_COLS].values, tr[ev].values)
                contrib.append(m.get_booster().predict(xgbm.DMatrix(te[HYB_COLS].values), pred_contribs=True)[:, :-1])
            imp = pd.Series(np.abs(np.vstack(contrib)).mean(axis=0), index=HYB_COLS).sort_values(ascending=False)
            imp.to_csv(f"{RES}/downside_shap_HYB.csv", header=["mean_abs_shap"])
            res["shap_top10"] = imp.head(10).to_dict()
            # latest warning for the last (unlabelled) trading day
            va = yrs >= yrs.max() - 1
            m, g, mv = R.fit_xgb(F.loc[~va, HYB_COLS].values, F.loc[~va, ev].values, F.loc[va, HYB_COLS].values,
                                 F.loc[va, ev].values, 0, F[HYB_COLS].values, F[ev].values)
            tau = float(np.quantile(mv.predict_proba(F.loc[va, HYB_COLS].values)[:, 1], ALERT_Q))
            p_last = float(m.predict_proba(last[HYB_COLS].values)[:, 1][-1]) if len(last) else None
            res["latest"] = dict(date=str(last.Date.iloc[-1].date()) if len(last) else None, p_event=p_last, tau=tau,
                                 alert=bool(p_last is not None and p_last >= tau), params=g)
        out[ev] = res
    with open(f"{RES}/results_downside.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("done")


if __name__ == "__main__":
    main()
