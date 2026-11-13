"""Figures for the paper (single-column IEEE width, 300 dpi).
Usage: python make_figures.py results figures
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

RES, FIG = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0})
W = 3.45


def fig_framework():
    fig, ax = plt.subplots(figsize=(W, 2.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    def box(x, y, w, h, t, fc="#ffffff", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=fc, ec="black", lw=0.6))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.0,
                fontweight="bold" if bold else "normal", wrap=True)
    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black"))
    box(0.1, 5.6, 2.3, 1.1, "IHSG daily OHLC\n1996-2026\n7,444 days")
    box(3.0, 5.6, 2.0, 1.1, "CONT (rival)\n10 indicators\nn = 10", fc="#eeeeee")
    box(5.3, 5.6, 2.0, 1.1, "BIN (rival)\n+1/-1 trend\nrules", fc="#eeeeee")
    box(7.6, 5.6, 2.3, 1.1, "HYB (proposed)\nBIN + 17 scale-\ninvariant + consensus", fc="#d9d9d9", bold=True)
    arr(2.4, 6.15, 3.0, 6.15)
    box(0.1, 3.3, 4.6, 1.5, "Walk-forward (expanding window)\ntrain 1996..Y-1, validate Y-2..Y-1,\ntest Y = 2010..2026 (17 folds)")
    box(5.3, 3.3, 4.6, 1.5, "XGBoost: grid on validation +\nearly stopping, refit on full train;\nbaselines: majority, persistence,\nvol-20 rule, LR, RF")
    for x in (4.0, 6.3, 8.75):
        arr(x, 5.6, x if x < 5 else x, 4.8)
    arr(4.7, 4.05, 5.3, 4.05)
    box(0.1, 0.6, 3.0, 1.9, "Tasks\n(a) next-day direction\n(b) next-day drop\n     <= -1.5 %")
    box(3.5, 0.6, 3.0, 1.9, "Evaluation\nAUC, PR-AUC, MCC,\nMcNemar, block\nbootstrap, SHAP")
    box(6.9, 0.6, 3.0, 1.9, "Early-warning signal\nWASPADA if P >= tau\n(tau = 80th pct of\nvalidation scores)", fc="#eeeeee")
    arr(7.6, 3.3, 8.4, 2.5); arr(5.0, 3.3, 5.0, 2.5); arr(2.0, 3.3, 1.6, 2.5)
    fig.tight_layout(pad=0.1)
    fig.savefig(f"{FIG}/fig1_framework.png", dpi=300)


def fig_yearly():
    d = pd.read_csv(f"{RES}/downside_yearly_auc.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(W, 1.9))
    style = {"CONT (rival)": ("s", "--", "#7f7f7f"), "BIN (rival)": ("^", ":", "#000000"),
             "Vol-20 rule": ("x", "-.", "#555555"), "HYB (proposed)": ("o", "-", "#000000")}
    for c in ["CONT (rival)", "BIN (rival)", "Vol-20 rule", "HYB (proposed)"]:
        m, ls, col = style[c]
        ax.plot(d.index, d[c], marker=m, ls=ls, color=col, ms=3, label=c)
    ax.axhline(0.5, color="#999999", lw=0.5)
    ax.set_ylabel("ROC-AUC"); ax.set_xlabel("Test year")
    ax.set_xticks(d.index[::2]); ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=4, fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=0.8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig2_yearly_auc.png", dpi=300)


def fig_shap():
    s = pd.read_csv(f"{RES}/downside_shap_HYB.csv", index_col=0)["mean_abs_shap"].head(10)[::-1]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    cols = ["#7f7f7f" if i.startswith("b_") or i == "consensus" else "#262626" for i in s.index]
    ax.barh(s.index, s.values, color=cols, height=0.6)
    ax.set_xlabel("Mean |SHAP| (log-odds), walk-forward test days")
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig3_shap.png", dpi=300)


if __name__ == "__main__":
    fig_framework()
    fig_yearly()
    fig_shap()
    print("figures ok")
