import json
import sys

from docbuild import Paper

EXP = sys.argv[1]          # experiment folder
OUT = sys.argv[2]          # output docx
R = json.load(open(f"{EXP}/results/results.json"))
D = json.load(open(f"{EXP}/results/results_downside.json"))
E = D["ev_15"]


def f3(x):
    return f"{x:.3f}".replace("-", "\u2212")


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def wf(name, k):
    return R["walk_forward"][name]["pooled_mean"][k]


def rs(name, k):
    return R["random_split"][name]["mean"][k]


def dm(name, k, ev=E):
    return ev[name]["mean"][k]


def ds(name, k):
    return E[name].get("std", {}).get(k)


import pandas as pd
YR = pd.read_csv(f"{EXP}/results/downside_yearly_auc.csv", index_col=0)
_nb = int((YR["HYB (proposed)"] > YR["BIN (rival)"]).sum())
_nc = int((YR["HYB (proposed)"] > YR["CONT (rival)"]).sum())
_nv = int((YR["HYB (proposed)"] > YR["Vol-20 rule"]).sum())
_lo, _hi = YR["HYB (proposed)"].idxmin(), YR["HYB (proposed)"].idxmax()
_ev = pd.read_csv(f"{EXP}/results/downside_pred_HYB.csv").groupby("year").y.sum()
YEARLY_SENTENCE = (
    f"Single-year scores are noisy because a test year holds a median of {int(_ev.median())} downside days "
    f"(from {int(_ev.min())} in {int(_ev.idxmin())} to {int(_ev.max())} in {int(_ev.idxmax())}). Within single "
    f"years HYB beats BIN in {_nb} of {len(YR)} cases, the volatility rule in {_nv} and CONT in {_nc} (Fig. 2), "
    f"with AUC from {YR['HYB (proposed)'].min():.3f} ({_lo}) to {YR['HYB (proposed)'].max():.3f} ({_hi}). Part of "
    f"the pooled advantage over CONT therefore comes from ranking calm and turbulent periods correctly against "
    f"each other, which is what an alert threshold fixed on validation data relies on.")
_sh = pd.read_csv(f"{EXP}/results/downside_shap_HYB.csv", index_col=0)["mean_abs_shap"]
SH_BIN = float(_sh[[i for i in _sh.index if i.startswith("b_") or i == "consensus"]].sum() / _sh.sum())
SH_VOLRET = float(_sh[[i for i in _sh.index if i.startswith(("vol_", "hl_range", "ret_"))]].sum() / _sh.sum())
BT = E["bootstrap_auc_diff"]
LAT = D["ev_15"]["latest"]
SD = R["sameday_diagnostic"]
n = R["data"]["n_rows"]
base15 = dm("HYB (proposed)", "base_rate")

P = Paper(sys.argv[3])
P.title("Hybrid Scale-Invariant Technical Indicators for Leakage-Free XGBoost Early Warning of IHSG Downside Days")
P.authors([
    ["1st Murti Wisnu Ragil Sastyawan", "Faculty of Intelligent Technology", "Universitas Dian Nuswantoro",
     "Semarang, Indonesia", "murti.sastyawan@unsoed.ac.id"],
    ["2nd Rindra Yusianto", "Faculty of Intelligent Technology", "Universitas Dian Nuswantoro",
     "Semarang, Indonesia", "rindra@dsn.dinus.ac.id"],
    ["3rd Guruh Fajar Shidik", "Faculty of Intelligent Technology", "Universitas Dian Nuswantoro",
     "Semarang, Indonesia", "guruh.fajar.shidik@dsn.dinus.ac.id"],
])

hyb_auc, bin_auc, cont_auc = dm("HYB (proposed)", "auc"), dm("BIN (rival)", "auc"), dm("CONT (rival)", "auc")
P.abstract(
    f"Nabipour et al. (IEEE Access, 2020) reported that converting ten technical indicators into binary trend "
    f"signals lifts the F1-score of stock-trend classifiers from about 68% to 85–90%. Their models were tested "
    f"on a random 70/30 split and the prediction horizon was not stated, so it is unclear how much of that gain "
    f"survives in a forecasting setting. We re-examine the continuous and binary representations on the Indonesia "
    f"Composite Index (IHSG) over {n:,} trading days (1996–2026) with an expanding-window walk-forward protocol of "
    f"17 annual folds. For next-day direction every representation stays near chance (ROC-AUC "
    f"{f3(wf('BIN (rival)', 'auc'))}–{f3(wf('CONT (rival)', 'auc'))}), while aligning the label with the "
    f"feature day pushes accuracy to {pc(SD['BIN (rival)']['random_sameday']['acc'])}, which shows how a horizon "
    f"mismatch can produce scores of the size reported. We then propose a hybrid representation that keeps the "
    f"binary signals, adds 17 scale-invariant magnitude and volatility features plus a bullish-consensus score, and "
    f"targets downside days (next-day return ≤ −1.5%). Under walk-forward testing the hybrid XGBoost model "
    f"reaches ROC-AUC {f3(hyb_auc)} against {f3(bin_auc)} for binary and {f3(cont_auc)} for continuous inputs; its "
    f"alerts flag {pc(dm('HYB (proposed)', 'recall'))} of downside days with a noise-to-signal ratio of "
    f"{dm('HYB (proposed)', 'nsr'):.2f}. An ablation shows that the scale-invariant features carry the gain, and the "
    f"margin over a plain 20-day volatility rule is small but significant (+{E['bootstrap_auc_diff']['Vol-20 rule']['diff']:.3f} AUC). "
    f"Code, data and predictions are released for replication.")
P.keywords("early warning system, IHSG, technical indicators, walk-forward validation, XGBoost, data leakage")

# ------------------------------------------------------------------ I. Introduction
P.h1("Introduction")
P.para(
    "Daily movements of the Indonesia Composite Index (IHSG) matter to retail investors, fund managers and "
    "regulators alike, and a steady stream of studies has tried to forecast them with machine learning. Reviews of "
    "this literature count well over a hundred journal papers and more than two thousand distinct input variables, "
    "with technical indicators as the most common family [1], [2]. Deep networks [3], [4] and gradient-boosted "
    "trees [5], [6] dominate recent work, and Indonesian studies have followed the same path [7], reporting "
    "LSTM and GRU forecasts of the IHSG itself [20], a Bi-LSTM variant conditioned on global indices [21], and "
    "a GARCH-MIDAS/XGBoost hybrid for its monthly volatility [22]. None of the three evaluates under "
    "leakage-free walk-forward conditions, a gap this paper returns to in Section V.")
P.para(
    "One widely cited result comes from Nabipour et al. [8], published in *IEEE Access*. Building on the "
    "trend-deterministic preparation of Patel et al. [9], they computed ten indicators with a 10-day window and "
    "compared two input forms: the continuous values, min-max scaled to (0, 1), and a binary version in which each "
    "indicator becomes +1 or −1 according to a fixed rule. On four Tehran Stock Exchange groups, nine machine "
    "learning models and two recurrent networks scored around 68% F1 with continuous inputs and 85–90% with "
    "binary inputs. The authors explain the jump by noting that the binary layer compares the value at day *t* "
    "with day *t*−1, so the input already carries a recognised trend [8].")
P.para(
    "That explanation is where our doubts begin. The test set in [8] is a random 30% of all days, which mixes "
    "future and past observations of a strongly autocorrelated series; time-series evaluation studies show that "
    "this kind of split tends to give optimistic estimates on non-stationary data [10]–[12]. The paper also "
    "does not say whether the label refers to the move from *t* to *t*+1 or from *t*−1 to *t*. If it is the "
    "latter, the binary rules partly restate the label. Finally, a ±1 encoding throws away magnitude: a 0.1% "
    "dip and a 4% fall look identical, yet magnitude and volatility are exactly what an early warning system "
    "needs.")
P.para("This paper makes four contributions:")
P.bullet("A replication of the continuous and binary representations of [8] on IHSG data from 1996 to 2026, scored "
         "under both the original random split and a leakage-free expanding-window walk-forward protocol.")
P.bullet("A horizon-alignment diagnostic showing that labels aligned with the feature day reproduce accuracy above "
         "90%, while true next-day labels stay near chance.")
P.bullet("A hybrid representation (HYB): the ten binary signals, 17 scale-invariant continuous features and a "
         "consensus score, which is a simple change to the input layer of [8] that keeps its trend signals.")
P.bullet("A downside-day early warning formulation with validation-calibrated alert thresholds, evaluated with "
         "precision, recall, lift and the noise-to-signal ratio, and explained with SHAP values.")

# ------------------------------------------------------------------ II. Related work
P.h1("Related Work")
P.h2("Technical-indicator representations")
P.para(
    "Patel et al. [9] introduced trend-deterministic data preparation, mapping indicators such as SMA, RSI and "
    "Williams %R into up/down opinions before classification, and reported large accuracy gains on Indian indices. "
    "Nabipour et al. [8] extended the comparison to eleven models and confirmed the gain on Tehran data. Basak et "
    "al. [6] used random forests and XGBoost on indicator features to predict direction several days ahead, and "
    "Ballings et al. [5] found ensembles such as random forests ahead of single classifiers for one-year direction. "
    "None of these studies isolates what a binary encoding loses when the goal is risk rather than direction.")
P.h2("Evaluating models on financial time series")
P.para(
    "Tashman [10] argued that forecasting accuracy should be measured out of sample with rolling origins. Bergmeir "
    "and Benítez [11] showed that standard cross-validation is only safe for stationary, purely autoregressive "
    "settings, and Cerqueira et al. [12] found that, for non-stationary real series, out-of-sample schemes that "
    "keep temporal order give the most accurate performance estimates. Fischer and Krauss [3] and Krauss et al. "
    "[13] trained on rolling study periods for S&P 500 constituents for the same reason. Sezer et al. [4] and "
    "Bustos and Pomares-Quimbaya [2] note that many published studies still rely on random splits.")
P.h2("Early warning and interpretability")
P.para(
    "Signal-based early warning systems trace back to the currency-crisis literature, where Kaminsky, Lizondo and "
    "Reinhart chose indicator thresholds that minimise the noise-to-signal ratio (NSR); Berg and Pattillo [14] "
    "tested that approach out of sample and discuss its strengths and limits. For market data, Tharavanij et al. "
    "[15] found that classic technical trading rules on Southeast Asian indices, including Jakarta, rarely beat "
    "buy-and-hold after costs, a reminder that direction is hard to time. On the modelling side, XGBoost [16] is a "
    "regularised gradient-boosting library that handles mixed feature types, and TreeSHAP [17] gives exact Shapley "
    "attributions for tree ensembles. Because class balance shifts across years, we also report the Matthews "
    "correlation coefficient (MCC), which Chicco and Jurman [18] recommend over accuracy and F1 for binary tasks.")

# ------------------------------------------------------------------ III. Method
P.h1("Materials and Methods")
P.figure(f"{EXP}/figures/fig1_framework.png", "Research framework: representations, walk-forward protocol, tasks and early-warning layer.")
P.h2("Data")
P.para(
    "We use daily open, high, low and close values of the IHSG from 2 January 1995 to 3 November 2026 (7,745 "
    "rows, no missing values). The 1995 records contain 103 days with identical high and low prices and 100 "
    "zero-volume days, so the sample starts in 1996; volume is excluded from all feature sets because 2026 again "
    f"shows 53 zero-volume days. After the 50-day warm-up of the longest window, {n:,} labelled days remain, from "
    f"18 March 1996 to 2 November 2026. Up days make up {pc(R['data']['up_share'])} of the sample, and "
    f"{pc(base15)} of next-day returns fall to −1.5% or lower.")
P.h2("Prediction tasks")
P.para("Let *C*~t~ be the close on day *t* and *r*~t+1~ = *C*~t+1~/*C*~t~ − 1. The direction task follows [8]:")
P.equation("*y*~t~^dir^ = 1 if *C*~t+1~ > *C*~t~, else 0.", 1)
P.para("The downside task, which an early warning system is meant to serve, marks large next-day losses:", style="Body Text")
P.equation("*y*~t~^dn^ = 1 if *r*~t+1~ ≤ θ, θ = −1.5%,", 2)
P.para("with θ = −1.0% and −2.0% as robustness checks. Every feature on day *t* uses prices up to and "
       "including *t* only.")
P.h2("Input representations")
P.para(
    "CONT reproduces the ten indicators of [8], [9] with *n* = 10: simple and weighted moving averages (SMA, WMA), "
    "momentum (MOM), stochastic %K and %D (STCK, STCD), RSI, MACD, Larry Williams %R (LWR), the A/D oscillator "
    "(ADO) and the commodity channel index (CCI). Values are min-max scaled using training data only. BIN applies "
    "the rules of [8], listed in Table I.")
P.table("Binary Trend Rules of the Rival Representation [8]",
        ["Indicator", "Rule for +1 (otherwise −1)"],
        [["SMA, WMA", "*C*~t~ ≥ moving average"],
         ["MOM", "MOM~t~ > 0"],
         ["STCK, STCD, LWR, MACD, ADO", "value~t~ > value~t−1~"],
         ["RSI", "RSI < 30, or 30–70 and rising; −1 if > 70"],
         ["CCI", "CCI < −200, or ±200 band and rising; −1 if > 200"]],
        [1.35, 1.95])
P.para(
    "The proposed HYB representation keeps the ten binary signals and adds two groups. The first is a set of 17 "
    "scale-invariant continuous features taken from the thesis feature plan: the last five daily returns, the "
    "5- and 20-day returns, the ratios of close to its 5-, 20- and 50-day moving averages, 5- and 20-day return "
    "volatility, the high-low range divided by the close, Wilder RSI(14), and MACD, its signal line and histogram "
    "divided by the close. None of these depends on the index level, which rose more than tenfold over the sample. "
    "The second is a consensus score,")
P.equation("*S*~t~ = Σ~k=1..10~ *b*~k,t~ ∈ {−10, −8, ..., 10},", 3)
P.para(
    "which counts how many binary signals agree. HYB therefore has 28 inputs. An ablation, SI, uses the 17 "
    "scale-invariant features alone.")
P.h2("Model and baselines")
P.para(
    "Each representation feeds the same XGBoost classifier [16] (learning rate 0.03, subsample 0.8, column sample "
    "0.8, L2 penalty 1). Tree depth {2, 3, 4} and minimum child weight {1, 10} are chosen by validation log-loss "
    "with early stopping after 50 rounds; the model is then refitted on the full training window with the "
    "selected number of trees. Baselines are the training-set majority class (climatology for the downside task), "
    "persistence (tomorrow repeats today), a rule that uses 20-day volatility as the risk score, logistic "
    "regression and a random forest (500 trees, minimum leaf 20), the last two on HYB inputs.")
P.h2("Evaluation protocols")
P.para(
    "*Random split.* As in [8], 30% of days are drawn at random for testing and 15% of the rest for early "
    "stopping, repeated over five seeds. *Walk-forward.* For each test year *Y* from 2010 to 2026, the model "
    "trains on all days before *Y*, uses years *Y*−2 and *Y*−1 as the validation block, and predicts every "
    "day of *Y* without refitting. The 17 folds yield 4,079 out-of-sample predictions, and XGBoost results are "
    "averaged over five seeds.")
P.para(
    "We report ROC-AUC and, for the imbalanced downside task, the area under the precision-recall curve (PR-AUC). "
    "An alert (level WASPADA) is raised when the predicted probability exceeds τ, the 80th percentile of "
    "validation-block scores for that fold, so the threshold never sees test data. For alerts we report precision, "
    "recall, lift (precision divided by the base rate) and")
P.equation("NSR = [*B*/(*B*+*D*)] / [*A*/(*A*+*C*)],", 4)
P.para(
    "where *A* counts correct alerts, *B* false alerts, *C* missed events and *D* quiet non-events; NSR below 1 "
    "means alerts are informative [14]. Direction classifiers are compared with McNemar's exact test [19], and "
    "AUC differences with a paired moving-block bootstrap (block length 20 days, 2,000 resamples).")

# ------------------------------------------------------------------ IV. Results
P.h1("Results")
P.h2("Next-day direction")
rows = []
for name in ["CONT (rival)", "BIN (rival)", "SI (ablation)", "HYB (proposed)"]:
    rows.append([name, f3(rs(name, "acc")), f3(rs(name, "f1")), f3(rs(name, "auc")),
                 f3(wf(name, "acc")), f3(wf(name, "mcc")), f3(wf(name, "auc"))])
for name, lab in [("Majority", "Majority"), ("Persistence", "Persistence"), ("LogReg (HYB)", "LogReg (HYB)"),
                  ("RandomForest (HYB)", "RF (HYB)")]:
    auc = 0.5 if name == "Majority" else wf(name, "auc")
    rows.append([lab, "–", "–", "–", f3(wf(name, "acc")), f3(wf(name, "mcc")), f3(auc)])
P.table("Next-Day Direction: Random Split vs Walk-Forward (XGBoost)",
        ["Input", "RS Acc", "RS F1", "RS AUC", "WF Acc", "WF MCC", "WF AUC"], rows,
        [1.0, 0.38, 0.38, 0.4, 0.4, 0.42, 0.4],
        note="RS: random 70/30 split, mean of five seeds. WF: walk-forward 2010–2026, 4,079 test days. "
             "Majority AUC is 0.5 by construction.")
mc = R["mcnemar"]
P.para(
    f"Table II shows that no representation predicts next-day direction in a useful way. Walk-forward AUC ranges "
    f"from {f3(wf('BIN (rival)', 'auc'))} (BIN) to {f3(wf('CONT (rival)', 'auc'))} (CONT), and the 95% block-bootstrap "
    f"interval of every XGBoost model contains 0.5. HYB is correct on {R['binom_vs_majority']['correct']:,} of "
    f"{R['binom_vs_majority']['n']:,} test days, which does not beat always predicting the majority class "
    f"(one-sided binomial *p* = {R['binom_vs_majority']['p_value']:.2f}). McNemar tests find no difference between "
    f"HYB and BIN (*p* = {mc['BIN (rival)']['p_value']:.2f}) or CONT (*p* = {mc['CONT (rival)']['p_value']:.2f}). "
    f"The random split is more generous to CONT (AUC {f3(rs('CONT (rival)', 'auc'))} against "
    f"{f3(wf('CONT (rival)', 'auc'))} walk-forward), consistent with level-type indicators such as SMA and WMA "
    f"letting the model match test days to their neighbours in time.")
P.para(
    f"The binary advantage reported in [8] does not appear in either protocol. To see where scores of that size "
    f"can come from, we repeated the experiment with the label aligned to the feature day, *y*~t~ = "
    f"1[*C*~t~ > *C*~t−1~]. BIN then reaches {pc(SD['BIN (rival)']['random_sameday']['acc'])} accuracy and "
    f"{pc(SD['BIN (rival)']['random_sameday']['f1'])} F1 on the random split and "
    f"{pc(SD['BIN (rival)']['walkfwd_sameday']['acc'])} under walk-forward, close to the 85–90% in [8]. We do "
    f"not claim that [8] used this alignment, since the paper does not specify it, but the diagnostic shows that "
    f"the horizon must be stated before binary-trend results can be read as forecasts.")

P.h2("Downside-day early warning")
rows = []
for name, lab in [("CONT (rival)", "CONT (rival)"), ("BIN (rival)", "BIN (rival)"), ("SI (ablation)", "SI (ablation)"),
                  ("HYB (proposed)", "**HYB (proposed)**"), ("Climatology", "Climatology"),
                  ("Vol-20 rule", "Vol-20 rule"), ("LogReg (HYB)", "LogReg (HYB)"), ("RandomForest (HYB)", "RF (HYB)")]:
    m = dict(E[name]["mean"])
    if name == "Climatology":
        m["auc"], m["pr_auc"] = 0.5, m["base_rate"]
    rows.append([lab, f3(m["auc"]), f3(m["pr_auc"]), pc(m["precision"]), pc(m["recall"]), f"{m['lift']:.2f}",
                 f"{m['nsr']:.2f}" if m["nsr"] == m["nsr"] else "–"])
P.table(f"Walk-Forward Downside Warning (θ = −1.5%, Base Rate {pc(base15)})",
        ["Model / input", "AUC", "PR-AUC", "Prec.", "Recall", "Lift", "NSR"], rows,
        [1.05, 0.38, 0.45, 0.42, 0.42, 0.33, 0.33], bold_rows=(3,),
        note="XGBoost rows: mean of five seeds. Alerts: score above the 80th percentile of validation scores. Climatology alerts on every day by construction.")
b_bin, b_vol, b_cont = BT["BIN (rival)"], BT["Vol-20 rule"], BT["CONT (rival)"]
P.para(
    f"Changing the target changes the picture (Table III). HYB reaches ROC-AUC {f3(hyb_auc)} and PR-AUC "
    f"{f3(dm('HYB (proposed)', 'pr_auc'))}, against {f3(bin_auc)} and {f3(dm('BIN (rival)', 'pr_auc'))} for BIN. The "
    f"paired bootstrap puts the HYB−BIN difference at {b_bin['diff']:+.3f} (95% CI "
    f"{b_bin['ci95'][0]:+.3f} to {b_bin['ci95'][1]:+.3f}) and the HYB−CONT difference at {b_cont['diff']:+.3f} "
    f"(95% CI {b_cont['ci95'][0]:+.3f} to {b_cont['ci95'][1]:+.3f}). HYB alerts cover "
    f"{pc(dm('HYB (proposed)', 'recall'))} of downside days while firing on about "
    f"{pc(dm('HYB (proposed)', 'alert_share'), 0)} of all days, a lift of {dm('HYB (proposed)', 'lift'):.2f} over "
    f"the base rate, and the NSR of {dm('HYB (proposed)', 'nsr'):.2f} compares with "
    f"{dm('BIN (rival)', 'nsr'):.2f} for BIN.")
P.para(
    f"The volatility rule is a strong baseline: a single feature, 20-day volatility, gives AUC "
    f"{f3(dm('Vol-20 rule', 'auc'))}. The HYB−Vol-20 difference is {b_vol['diff']:+.3f} (95% CI "
    f"{b_vol['ci95'][0]:+.3f} to {b_vol['ci95'][1]:+.3f}; one-sided *p* = {b_vol['p_one_sided']:.3f}). The SI "
    f"ablation scores {f3(dm('SI (ablation)', 'auc'))}, level with HYB, so the gain over the rival inputs comes "
    f"from the scale-invariant magnitude features and not from the binary signals; the binary block and consensus "
    f"score neither help nor hurt AUC, although they keep the alert readable in the trend vocabulary of [8]. The "
    f"learner matters less than the inputs: on HYB features a random forest reaches {f3(dm('RandomForest (HYB)', 'auc'))} "
    f"and logistic regression {f3(dm('LogReg (HYB)', 'auc'))}. {YEARLY_SENTENCE}")
P.figure(f"{EXP}/figures/fig2_yearly_auc.png", "Walk-forward ROC-AUC per test year for the downside task (seed 0).")
e10, e20 = D["ev_10"], D["ev_20"]
P.para(
    f"The result holds when the threshold moves. At θ = −1.0% (base rate "
    f"{pc(e10['HYB (proposed)']['mean']['base_rate'])}) HYB scores AUC {f3(e10['HYB (proposed)']['mean']['auc'])} "
    f"against {f3(e10['BIN (rival)']['mean']['auc'])} for BIN; at θ = −2.0% (base rate "
    f"{pc(e20['HYB (proposed)']['mean']['base_rate'])}) the figures are {f3(e20['HYB (proposed)']['mean']['auc'])} "
    f"and {f3(e20['BIN (rival)']['mean']['auc'])}. The random split again flatters level-based inputs: downside AUC "
    f"for CONT rises from {f3(cont_auc)} walk-forward to {f3(E['random_split_auc']['CONT (rival)']['auc_mean'])}, "
    f"while HYB moves only from {f3(hyb_auc)} to {f3(E['random_split_auc']['HYB (proposed)']['auc_mean'])}.")
P.h2("What drives the warnings")
sh = list(D["ev_15"]["shap_top10"].items())
P.figure(f"{EXP}/figures/fig3_shap.png", "Top ten HYB features by mean absolute SHAP value (downside task, all walk-forward test days). No binary or consensus input enters the top ten.")
P.para(
    f"TreeSHAP attributions over all walk-forward test days (Fig. 3) rank {sh[0][0].replace('_', chr(8209))}, "
    f"{sh[1][0].replace('_', chr(8209))} and {sh[2][0].replace('_', chr(8209))} highest. Volatility, range and "
    f"recent-return features take {pc(SH_VOLRET, 0)} of the total attribution, which matches the volatility-clustering "
    f"view of daily returns: large losses are far more likely after turbulent weeks, whatever their sign. The ten "
    f"binary signals and the consensus score together account for {pc(SH_BIN, 0)}, consistent with BIN on its own "
    f"failing to separate a quiet down-drift from the start of a sell-off.")
P.h2("Operational output")
P.para(
    f"Applied to the last available day ({LAT['date']}), with the model refitted on all labelled data, HYB "
    f"assigns a downside probability of {pc(LAT['p_event'])} against an alert threshold of {pc(LAT['tau'])}, so "
    f"the system issues {'a WASPADA alert' if LAT['alert'] else 'no alert'} for the next session. On that day the "
    f"index closed at 4,904.29 after a 20.8% fall over 20 sessions, RSI(14) stood at 25.8 and 20-day volatility "
    f"was 3.28%, more than twice its one-year median of 1.28%. This is a model output for illustration and not an "
    f"investment recommendation.")

# ------------------------------------------------------------------ V. Discussion
P.h1("Discussion")
P.para(
    "Two lessons follow. The evaluation protocol shapes the conclusion more than the model does: with "
    "the same features and learner, a random split and a stated horizon give very different stories, so studies "
    "that build on [8] should report the label definition and keep test data strictly after training data; the "
    "same caution applies to [20], whose held-out accuracy sits in the same 90%+ range as the horizon-misaligned "
    "diagnostic in Section IV-A, without the paper stating whether its split kept test days strictly after "
    "training days. Binary trend encoding is also a poor fit for risk tasks; it removes the magnitude information "
    "that separates ordinary noise from stress. The fix proposed here is cheap and keeps the interpretability of "
    "the binary signals, since the consensus score and the scale-invariant features are all readable by an "
    "analyst.")
P.para(
    "The study has limits. It covers one index and one market; the grid search is small; transaction costs and "
    "trading rules are not simulated; and the alert threshold is a fixed percentile rather than an NSR-optimised "
    "value. The thesis plan also includes an LSTM comparison, which we leave for the full study. The gain over "
    "the volatility rule is modest, so the main value of HYB lies in combining a volatility-aware risk score with "
    "trend signals in a single calibrated alert.")

# ------------------------------------------------------------------ VI. Conclusion
P.h1("Conclusion")
P.para(
    f"We replicated the continuous and binary indicator representations of Nabipour et al. [8] on 30 years of IHSG "
    f"data. Under leakage-free walk-forward testing neither predicts next-day direction better than chance, and the "
    f"high binary scores reappear only when the label is aligned with the feature day. Reframing the task as "
    f"downside warning and adding scale-invariant magnitude features to the binary signals raises walk-forward "
    f"ROC-AUC from {f3(bin_auc)} (binary) and {f3(cont_auc)} (continuous) to {f3(hyb_auc)}, with an NSR of "
    f"{dm('HYB (proposed)', 'nsr'):.2f}; the ablation shows that the scale-invariant features carry this gain. "
    f"Future work will "
    f"add cross-market tests, cost-aware evaluation and sequence models.")
P.para("Acknowledgment", style="Heading 5")
P.para("The first author thanks Universitas Dian Nuswantoro for academic support, and the second author for "
       "supervision of the thesis that this study extends. Code, data and walk-forward predictions are "
       "available at https://drive.google.com/drive/folders/1OqgT5gILTz5cM0UeoAguuHeZWRu-9pHc.")
P.para("References", style="Heading 5")
P.references([
    "M. M. Kumbure, C. Lohrmann, P. Luukka, and J. Porras, “Machine learning techniques and data for stock market forecasting: A literature review,” *Expert Syst. Appl.*, vol. 197, Art. no. 116659, 2022, doi: 10.1016/j.eswa.2022.116659.",
    "O. Bustos and A. Pomares-Quimbaya, “Stock market movement forecast: A systematic review,” *Expert Syst. Appl.*, vol. 156, Art. no. 113464, 2020, doi: 10.1016/j.eswa.2020.113464.",
    "T. Fischer and C. Krauss, “Deep learning with long short-term memory networks for financial market predictions,” *Eur. J. Oper. Res.*, vol. 270, no. 2, pp. 654–669, 2018, doi: 10.1016/j.ejor.2017.11.054.",
    "O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu, “Financial time series forecasting with deep learning: A systematic literature review: 2005–2019,” *Appl. Soft Comput.*, vol. 90, Art. no. 106181, 2020, doi: 10.1016/j.asoc.2020.106181.",
    "M. Ballings, D. Van den Poel, N. Hespeels, and R. Gryp, “Evaluating multiple classifiers for stock price direction prediction,” *Expert Syst. Appl.*, vol. 42, no. 20, pp. 7046–7056, 2015, doi: 10.1016/j.eswa.2015.05.013.",
    "S. Basak, S. Kar, S. Saha, L. Khaidem, and S. R. Dey, “Predicting the direction of stock market prices using tree-based classifiers,” *North Amer. J. Econ. Finance*, vol. 47, pp. 552–567, 2019, doi: 10.1016/j.najef.2018.06.013.",
    "W. Budiharto, “Data science approach to stock prices forecasting in Indonesia during Covid-19 using long short-term memory (LSTM),” *J. Big Data*, vol. 8, Art. no. 47, 2021, doi: 10.1186/s40537-021-00430-0.",
    "M. Nabipour, P. Nayyeri, H. Jabani, S. Shahab, and A. Mosavi, “Predicting stock market trends using machine learning and deep learning algorithms via continuous and binary data; a comparative analysis,” *IEEE Access*, vol. 8, pp. 150199–150212, 2020, doi: 10.1109/ACCESS.2020.3015966.",
    "J. Patel, S. Shah, P. Thakkar, and K. Kotecha, “Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques,” *Expert Syst. Appl.*, vol. 42, no. 1, pp. 259–268, 2015, doi: 10.1016/j.eswa.2014.07.040.",
    "L. J. Tashman, “Out-of-sample tests of forecasting accuracy: An analysis and review,” *Int. J. Forecast.*, vol. 16, no. 4, pp. 437–450, 2000, doi: 10.1016/S0169-2070(00)00065-0.",
    "C. Bergmeir and J. M. Benítez, “On the use of cross-validation for time series predictor evaluation,” *Inf. Sci.*, vol. 191, pp. 192–213, 2012, doi: 10.1016/j.ins.2011.12.028.",
    "V. Cerqueira, L. Torgo, and I. Mozetič, “Evaluating time series forecasting models: An empirical study on performance estimation methods,” *Mach. Learn.*, vol. 109, pp. 1997–2028, 2020, doi: 10.1007/s10994-020-05910-7.",
    "C. Krauss, X. A. Do, and N. Huck, “Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500,” *Eur. J. Oper. Res.*, vol. 259, no. 2, pp. 689–702, 2017, doi: 10.1016/j.ejor.2016.10.031.",
    "A. Berg and C. Pattillo, “Predicting currency crises: The indicators approach and an alternative,” *J. Int. Money Finance*, vol. 18, no. 4, pp. 561–586, 1999, doi: 10.1016/S0261-5606(99)00024-8.",
    "P. Tharavanij, V. Siraprapasiri, and K. Rajchamaha, “Performance of technical trading rules: Evidence from Southeast Asian stock markets,” *SpringerPlus*, vol. 4, Art. no. 552, 2015, doi: 10.1186/s40064-015-1334-7.",
    "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, San Francisco, CA, USA, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.",
    "S. M. Lundberg *et al.*, “From local explanations to global understanding with explainable AI for trees,” *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020, doi: 10.1038/s42256-019-0138-9.",
    "D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” *BMC Genomics*, vol. 21, Art. no. 6, 2020, doi: 10.1186/s12864-019-6413-7.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.",
    "A. F. Rasyid, D. Agushinta R., and D. T. Ediraras, “Deep learning methods in predicting Indonesia composite stock price index (IHSG),” *Int. J. Comput. Inf. Technol.*, vol. 10, no. 5, 2021, doi: 10.24203/ijcit.v10i5.153.",
    "M. A. Majid, P. D. Saputri, and Soehardjoepri, “Stock market index prediction using bi-directional long short-term memory,” *J. Appl. Inform. Comput.*, vol. 8, no. 1, pp. 55–61, 2024, doi: 10.30871/jaic.v8i1.7195.",
    "L. D. Siagian and W. Makaliwe, “Investor sentiment dynamics and market volatility in Indonesia: Hybrid approach using GARCH-MIDAS and machine learning,” *Eduvest J. Univers. Stud.*, vol. 5, no. 9, pp. 10849–10865, 2025, doi: 10.59188/eduvest.v5i9.51365.",
])
P.save(OUT)
print("saved", OUT)
