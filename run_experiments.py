"""Experiments for the paper:
Leakage-free walk-forward evaluation of continuous, binary and hybrid technical-indicator
representations for XGBoost-based next-day IHSG direction warning.

Usage:  python run_experiments.py data/ihsg_daily.csv results figures
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

from features import BIN_COLS, CONT_COLS, HYB_COLS, SI_COLS, build

warnings.filterwarnings("ignore")
DATA, RES, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [0, 1, 2, 3, 4]
TEST_YEARS = list(range(2010, 2027))
REPS = {"CONT (rival)": CONT_COLS, "BIN (rival)": BIN_COLS, "SI (ablation)": SI_COLS, "HYB (proposed)": HYB_COLS}
GRID = [dict(max_depth=d, min_child_weight=m) for d in (2, 3, 4) for m in (1, 10)]


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5
    return dict(acc=accuracy_score(y, yh), bacc=balanced_accuracy_score(y, yh),
                f1=f1_score(y, yh, zero_division=0), mcc=matthews_corrcoef(y, yh), auc=auc)


def xgb(seed, n_estimators=1000, **kw):
    return XGBClassifier(n_estimators=n_estimators, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                         tree_method="hist", n_jobs=8, random_state=seed, **kw)


def fit_xgb(Xtr, ytr, Xva, yva, seed, Xfull=None, yfull=None):
    """Grid search on the validation block with early stopping, then refit on train+val."""
    best = None
    for g in GRID:
        m = xgb(seed, early_stopping_rounds=50, **g)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ll = m.best_score
        if best is None or ll < best[0]:
            best = (ll, g, m.best_iteration + 1, m)
    _, g, nt, mval = best
    if Xfull is None:
        return mval, dict(g, n_estimators=nt), mval
    final = xgb(seed, n_estimators=max(nt, 10), **g)
    final.fit(Xfull, yfull, verbose=False)
    return final, dict(g, n_estimators=nt), mval


def scaler_for(cols):
    return MinMaxScaler() if cols is CONT_COLS else None


def prep(Xtr, Xte, cols, extra=None):
    sc = scaler_for(cols)
    if sc is None:
        return Xtr, Xte, extra
    sc.fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), (sc.transform(extra) if extra is not None else None)


# ------------------------------------------------------------------ protocol A: random split (rival)
def random_split(F, cols, seed, target="target"):
    X, y = F[cols].values, F[target].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=seed)
    Xtr, Xte, _ = prep(Xtr, Xte, cols)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, shuffle=True, random_state=seed)
    m, _, _ = fit_xgb(Xa, ya, Xv, yv, seed, Xtr, ytr)
    return metrics(yte, m.predict_proba(Xte)[:, 1])


# ------------------------------------------------------------------ protocol B: walk-forward
def walk_forward(F, cols, seed, model="xgb", target="target", keep_models=False):
    rows, models, chosen = [], {}, []
    yr = F["Date"].dt.year
    for Y in TEST_YEARS:
        tr, te = F[yr < Y], F[yr == Y]
        va_mask = tr["Date"].dt.year >= Y - 2
        Xtr, ytr = tr[cols].values, tr[target].values.astype(int)
        Xte, yte = te[cols].values, te[target].values.astype(int)
        Xa, ya = tr.loc[~va_mask, cols].values, tr.loc[~va_mask, target].values.astype(int)
        Xv, yv = tr.loc[va_mask, cols].values, tr.loc[va_mask, target].values.astype(int)
        if model == "xgb":
            sc = scaler_for(cols)
            if sc is not None:
                sc.fit(Xtr)
                Xtr, Xte, Xa, Xv = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xa), sc.transform(Xv)
            m, g, mv = fit_xgb(Xa, ya, Xv, yv, seed, Xtr, ytr)
            chosen.append(dict(year=Y, **g))
            p = m.predict_proba(Xte)[:, 1]
            # validation-period probabilities from the model fitted without the validation block
            # (best_iteration is used by predict_proba), used to set early-warning thresholds
            pv = mv.predict_proba(Xv)[:, 1]
            if keep_models:
                models[Y] = (m, Xte)
        elif model == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
            p, pv = m.predict_proba(Xte)[:, 1], None
        elif model == "rf":
            m = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt",
                                       n_jobs=8, random_state=seed).fit(Xtr, ytr)
            p, pv = m.predict_proba(Xte)[:, 1], None
        elif model == "majority":
            p, pv = np.full(len(te), ytr.mean()), None
        elif model == "persistence":
            p, pv = (te["ret_1"].values > 0).astype(float), None
        rows.append(pd.DataFrame(dict(Date=te["Date"].values, year=Y, y=yte, p=p, ret_next=te["ret_next"].values)))
        if pv is not None:
            rows[-1]["tau_hi"] = np.quantile(1 - pv, 0.80)
            rows[-1]["tau_lo"] = np.quantile(1 - pv, 0.20)
    return pd.concat(rows, ignore_index=True), models, chosen


def per_year(P):
    out = []
    for Y, g in P.groupby("year"):
        out.append(dict(year=int(Y), **metrics(g.y.values, g.p.values)))
    return pd.DataFrame(out)


def block_bootstrap_auc(y, p, B=1000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(B):
        starts = rng.integers(0, n - block, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pa, pb):
    a, b = (pa >= 0.5).astype(int) == y, (pb >= 0.5).astype(int) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    pval = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return dict(a_right_b_wrong=n01, a_wrong_b_right=n10, p_value=float(pval))


def ews_eval(P, big=-0.015):
    """Three-level signal: WASPADA (alert) if P(down) >= tau_hi, POSITIF if P(down) <= tau_lo."""
    pdn = 1 - P["p"].values
    sig = np.where(pdn >= P["tau_hi"].values, "WASPADA", np.where(pdn <= P["tau_lo"].values, "POSITIF", "NETRAL"))
    P = P.assign(signal=sig, down=(P["y"] == 0).astype(int), bigdrop=(P["ret_next"] <= big).astype(int))
    res = {}
    for s in ["WASPADA", "NETRAL", "POSITIF"]:
        g = P[P.signal == s]
        res[s] = dict(days=int(len(g)), share=float(len(g) / len(P)), p_down=float(g.down.mean()),
                      p_bigdrop=float(g.bigdrop.mean()), mean_ret_next_pct=float(g.ret_next.mean() * 100))
    for ev in ["down", "bigdrop"]:
        S, E = (P.signal == "WASPADA").values, P[ev].values.astype(bool)
        A, Bf, C, D = (S & E).sum(), (S & ~E).sum(), (~S & E).sum(), (~S & ~E).sum()
        res[f"alert_vs_{ev}"] = dict(precision=float(A / (A + Bf)), recall=float(A / (A + C)),
                                     base_rate=float(E.mean()),
                                     nsr=float((Bf / (Bf + D)) / (A / (A + C))) if A > 0 else None,
                                     A=int(A), B=int(Bf), C=int(C), D=int(D))
    return res, P


def main():
    t0 = time.time()
    raw = pd.read_csv(DATA, parse_dates=["Date"])
    F_all = build(raw)
    last = F_all[F_all["target"].isna()].copy()
    F = F_all[F_all["target"].notna()].reset_index(drop=True)
    info = dict(n_rows=int(len(F)), start=str(F.Date.min().date()), end=str(F.Date.max().date()),
                up_share=float(F.target.mean()), last_unlabeled=str(last.Date.max().date()) if len(last) else None)
    print(info, flush=True)
    out = dict(data=info)

    # A. random 70/30 split, 5 seeds
    A = {}
    for name, cols in REPS.items():
        ms = pd.DataFrame([random_split(F, cols, s) for s in SEEDS])
        A[name] = dict(mean=ms.mean().to_dict(), std=ms.std().to_dict())
        print("random", name, ms.mean().round(4).to_dict(), flush=True)
    out["random_split"] = A

    # B. walk-forward, 5 seeds for XGBoost representations
    B, preds, chosen_all = {}, {}, {}
    for name, cols in REPS.items():
        pooled, yearly = [], []
        for s in SEEDS:
            P, models, chosen = walk_forward(F, cols, s, keep_models=(name == "HYB (proposed)" and s == 0))
            pooled.append(metrics(P.y.values, P.p.values))
            yearly.append(per_year(P).assign(seed=s))
            if s == 0:
                preds[name] = P
                chosen_all[name] = chosen
                if models:
                    hyb_models = models
        pooled = pd.DataFrame(pooled)
        yr = pd.concat(yearly).groupby("year").mean().drop(columns="seed")
        B[name] = dict(pooled_mean=pooled.mean().to_dict(), pooled_std=pooled.std().to_dict(),
                       yearly_mean_auc=float(yr.auc.mean()), yearly_std_auc=float(yr.auc.std()),
                       yearly_mean_mcc=float(yr.mcc.mean()),
                       years_auc_gt_05=int((yr.auc > 0.5).sum()),
                       auc_ci95_seed0=block_bootstrap_auc(preds[name].y.values, preds[name].p.values))
        yr.to_csv(f"{RES}/yearly_{name.split()[0]}.csv")
        print("walkfwd", name, pooled.mean().round(4).to_dict(), flush=True)
    # baselines on the hybrid feature set (same folds)
    for name, mdl in [("Majority", "majority"), ("Persistence", "persistence"),
                      ("LogReg (HYB)", "lr"), ("RandomForest (HYB)", "rf")]:
        P, _, _ = walk_forward(F, HYB_COLS, 0, model=mdl)
        preds[name] = P
        B[name] = dict(pooled_mean=metrics(P.y.values, P.p.values),
                       auc_ci95_seed0=block_bootstrap_auc(P.y.values, P.p.values) if mdl != "majority" else [0.5, 0.5])
        print("walkfwd", name, {k: round(v, 4) for k, v in B[name]["pooled_mean"].items()}, flush=True)
    out["walk_forward"] = B
    out["hyperparams_seed0"] = chosen_all

    # significance (seed 0 predictions)
    y = preds["HYB (proposed)"].y.values
    out["mcnemar"] = {k: mcnemar(y, preds["HYB (proposed)"].p.values, preds[k].p.values)
                      for k in ["CONT (rival)", "BIN (rival)", "SI (ablation)", "LogReg (HYB)", "RandomForest (HYB)", "Persistence"]}
    # one-sided test: is directional accuracy above the majority rate?
    acc_n = int(((preds["HYB (proposed)"].p.values >= 0.5).astype(int) == y).sum())
    out["binom_vs_majority"] = dict(correct=acc_n, n=int(len(y)), p0=float(preds["Majority"].y.mean()),
                                    p_value=float(binomtest(acc_n, len(y), float(max(y.mean(), 1 - y.mean())),
                                                            alternative="greater").pvalue))

    # C. horizon-alignment diagnostic: same-day label with rival binary features
    diag = {}
    for name in ["BIN (rival)", "CONT (rival)"]:
        ms = pd.DataFrame([random_split(F, REPS[name], s, target="target_sameday") for s in SEEDS])
        P, _, _ = walk_forward(F, REPS[name], 0, target="target_sameday")
        diag[name] = dict(random_sameday=ms.mean().to_dict(), walkfwd_sameday=metrics(P.y.values, P.p.values))
        print("sameday", name, ms.mean().round(4).to_dict(), flush=True)
    out["sameday_diagnostic"] = diag

    # D. early-warning evaluation on proposed model (seed 0)
    for name in ["HYB (proposed)", "BIN (rival)", "CONT (rival)"]:
        res, P = ews_eval(preds[name])
        out.setdefault("ews", {})[name] = res
        if name == "HYB (proposed)":
            P.to_csv(f"{RES}/predictions_walkforward_HYB.csv", index=False)

    # E. SHAP (TreeSHAP via XGBoost pred_contribs) on all walk-forward test years, proposed model
    import xgboost as xgbm
    contrib = []
    for Y, (m, Xte) in hyb_models.items():
        c = m.get_booster().predict(xgbm.DMatrix(Xte, feature_names=None), pred_contribs=True)[:, :-1]
        contrib.append(c)
    contrib = np.vstack(contrib)
    shap_imp = pd.Series(np.abs(contrib).mean(axis=0), index=HYB_COLS).sort_values(ascending=False)
    shap_imp.to_csv(f"{RES}/shap_importance_HYB.csv", header=["mean_abs_shap"])
    out["shap_top10"] = shap_imp.head(10).to_dict()

    # F. latest signal: model trained on all labelled data, applied to the last unlabeled day
    if len(last):
        yrs = F.Date.dt.year
        va = yrs >= yrs.max() - 1
        m, g, mv = fit_xgb(F.loc[~va, HYB_COLS].values, F.loc[~va, "target"].values.astype(int),
                           F.loc[va, HYB_COLS].values, F.loc[va, "target"].values.astype(int), 0,
                           F[HYB_COLS].values, F["target"].values.astype(int))
        pv = 1 - mv.predict_proba(F.loc[va, HYB_COLS].values)[:, 1]
        hi, lo = np.quantile(pv, 0.8), np.quantile(pv, 0.2)
        pdown = float(1 - m.predict_proba(last[HYB_COLS].values)[:, 1][-1])
        out["latest_signal"] = dict(date=str(last.Date.iloc[-1].date()), p_down=pdown, tau_hi=float(hi), tau_lo=float(lo),
                                    signal="WASPADA" if pdown >= hi else ("POSITIF" if pdown <= lo else "NETRAL"),
                                    consensus=int(last.consensus.iloc[-1]))
    out["runtime_sec"] = round(time.time() - t0, 1)

    for name, P in preds.items():
        P.to_csv(f"{RES}/pred_{name.split()[0]}.csv", index=False)
    with open(f"{RES}/results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("done", out["runtime_sec"], "s")


if __name__ == "__main__":
    main()
