# IHSG Downside Early Warning — Experiment Repository

Companion code for the paper **"Hybrid Scale-Invariant Technical Indicators for Leakage-Free XGBoost Early Warning of IHSG Downside Days"** (Murti Wisnu Ragil Sastyawan, Universitas Dian Nuswantoro).

Rival paper: M. Nabipour, P. Nayyeri, H. Jabani, S. Shahab, A. Mosavi, "Predicting stock market trends using machine learning and deep learning algorithms via continuous and binary data; a comparative analysis," *IEEE Access*, vol. 8, pp. 150199–150212, 2020, doi:10.1109/ACCESS.2020.3015966.

## What is in here

| Path | Content |
|---|---|
| `data/ihsg_daily.csv` | IHSG daily OHLCV, 1995-01-02 to 2026-11-03 (7,745 rows) |
| `features.py` | CONT (rival, 10 indicators, n = 10), BIN (rival ±1 rules), SI (17 scale-invariant features), HYB (proposed = BIN + SI + consensus) |
| `run_experiments.py` | Next-day **direction** task: random 70/30 split (rival protocol) vs walk-forward 2010–2026, baselines, McNemar, same-day horizon diagnostic, SHAP |
| `run_downside.py` | **Downside** task (next-day return ≤ −1.5 %, plus −1.0 % / −2.0 %): walk-forward, baselines (climatology, vol-20 rule, LR, RF), block-bootstrap AUC differences, alerts, NSR, SHAP, latest signal |
| `make_figures.py` | Figures 1–3 of the paper |
| `results/` | `results.json`, `results_downside.json`, per-fold predictions, yearly AUC, SHAP importances, run logs |
| `figures/` | PNG figures (300 dpi) |
| `paper_build/` | script that fills the official IEEE conference template with the numbers in `results/` (`python paper_build/build_paper.py . out.docx paper_build/ieee_conference_template_transitional.docx`, run from `paper_build/`) |


## Key results (walk-forward 2010–2026, 4,079 test days)

**Next-day direction** (ROC-AUC, XGBoost, mean of 5 seeds)

| Input | Random split | Walk-forward |
|---|---|---|
| CONT (rival) | 0.544 | 0.514 |
| BIN (rival) | 0.506 | 0.504 |
| SI (ablation) | 0.526 | 0.509 |
| HYB (proposed) | 0.526 | 0.511 |

Same-day label diagnostic (BIN, random split): accuracy 0.912, F1 0.918.

**Downside day** (next-day return ≤ θ), walk-forward ROC-AUC

| Model / input | θ = −1.0% | θ = −1.5% | θ = −2.0% |
|---|---|---|---|
| CONT (rival) | 0.617 | 0.680 | 0.741 |
| BIN (rival) | 0.578 | 0.614 | 0.635 |
| SI (ablation) | 0.659 | 0.733 | 0.785 |
| HYB (proposed) | 0.652 | 0.730 | 0.774 |
| Vol-20 rule | 0.640 | 0.697 | 0.750 |
| LogReg (HYB) | 0.638 | 0.703 | 0.745 |
| RandomForest (HYB) | 0.655 | 0.734 | 0.772 |

HYB alerts at θ = −1.5%: precision 0.149, recall 0.497, lift 2.33, NSR 0.39.
Latest signal (2026-11-03): P(drop ≤ −1.5%) = 0.411, threshold 0.198 → WASPADA.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py data/ihsg_daily.csv results figures
python run_downside.py   data/ihsg_daily.csv results figures
python make_figures.py   results figures
```

Runtime on a 10-core laptop: about 6 min (direction) and 20–30 min (downside). XGBoost results are averaged over seeds 0–4.

## Protocol in one paragraph

Features on day *t* use prices up to *t* only. Walk-forward: for each test year Y = 2010…2026, train on all days before Y, use Y−2…Y−1 as the validation block (grid search over depth {2,3,4} × min_child_weight {1,10}, early stopping 50 rounds), refit on the full window, predict every day of Y. The alert threshold τ is the 80th percentile of validation-block scores, so it never sees the test year.

## Disclaimer

Research code for a course assignment. Outputs are model estimates, not investment advice.
