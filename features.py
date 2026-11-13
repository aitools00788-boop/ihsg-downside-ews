"""Feature construction for the IHSG next-day direction experiments.

Three input representations are built from daily OHLC data:
  * CONT  - the ten continuous technical indicators of Nabipour et al. (2020), n = 10
  * BIN   - the same indicators mapped to trend-deterministic +1/-1 signals (Nabipour et al., 2020)
  * SI    - scale-invariant continuous features (returns, price/MA ratios, volatility, RSI,
            MACD divided by the close)
  * HYB   - proposed hybrid: BIN + SI + a consensus score (sum of the ten binary signals)

Every feature at day t uses prices up to and including day t only. The label is
y_t = 1 if Close(t+1) > Close(t), else 0.
"""
import numpy as np
import pandas as pd

N = 10  # indicator window used by the rival paper

CONT_COLS = ["SMA", "WMA", "MOM", "STCK", "STCD", "RSI", "MACD", "LWR", "ADO", "CCI"]
BIN_COLS = [f"b_{c}" for c in CONT_COLS]
SI_COLS = ["ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_5d", "ret_20d",
           "ma_ratio_5", "ma_ratio_20", "ma_ratio_50", "vol_5", "vol_20", "hl_range",
           "rsi_14", "macd_n", "macd_signal_n", "macd_hist_n"]
HYB_COLS = BIN_COLS + SI_COLS + ["consensus"]


def rival_indicators(d, n=N):
    """Ten indicators following the definitions used by Nabipour et al. (2020) / Patel et al. (2015)."""
    C, H, L = d["Close"], d["High"], d["Low"]
    o = pd.DataFrame(index=d.index)
    o["SMA"] = C.rolling(n).mean()
    w = np.arange(1, n + 1, dtype=float)
    o["WMA"] = C.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    o["MOM"] = C - C.shift(n - 1)
    LL, HH = L.rolling(n).min(), H.rolling(n).max()
    rng = (HH - LL).replace(0, np.nan)
    o["STCK"] = (C - LL) / rng * 100
    o["STCD"] = o["STCK"].rolling(n).mean()
    diff = C.diff()
    up, dw = diff.clip(lower=0), -diff.clip(upper=0)
    o["RSI"] = 100 - 100 / (1 + up.rolling(n).mean() / dw.rolling(n).mean())
    ema12, ema26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean()
    o["MACD"] = (ema12 - ema26).ewm(span=n, adjust=False).mean()
    o["LWR"] = (HH - C) / rng * 100
    o["ADO"] = (H - C.shift(1)) / (H - L).replace(0, np.nan)
    M = (H + L + C) / 3
    SM = M.rolling(n).mean()
    D = M.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    o["CCI"] = (M - SM) / (0.015 * D)
    return o


def to_binary(o, close):
    """Trend-deterministic +1/-1 rules exactly as listed in Nabipour et al. (2020), Sec. II-B."""
    b = pd.DataFrame(index=o.index)
    b["b_SMA"] = np.where(close < o["SMA"], -1, 1)
    b["b_WMA"] = np.where(close < o["WMA"], -1, 1)
    b["b_MOM"] = np.where(o["MOM"] > 0, 1, -1)
    for c in ["STCK", "STCD", "LWR", "MACD", "ADO"]:
        b[f"b_{c}"] = np.where(o[c] > o[c].shift(1), 1, -1)
    r, rp = o["RSI"], o["RSI"].shift(1)
    b["b_RSI"] = np.where(r > 70, -1, np.where(r < 30, 1, np.where(r > rp, 1, -1)))
    c, cp = o["CCI"], o["CCI"].shift(1)
    b["b_CCI"] = np.where(c > 200, -1, np.where(c < -200, 1, np.where(c > cp, 1, -1)))
    return b[BIN_COLS]


def scale_invariant(d):
    """Scale-free continuous features (same set as Table 3.2 of the thesis proposal,
    with MACD terms divided by the close so that they do not depend on the index level)."""
    C = d["Close"]
    r = C.pct_change()
    o = pd.DataFrame(index=d.index)
    for k in range(1, 6):
        o[f"ret_{k}"] = r.shift(k - 1)
    o["ret_5d"] = C / C.shift(5) - 1
    o["ret_20d"] = C / C.shift(20) - 1
    for n in (5, 20, 50):
        o[f"ma_ratio_{n}"] = C / C.rolling(n).mean()
    o["vol_5"] = r.rolling(5).std()
    o["vol_20"] = r.rolling(20).std()
    o["hl_range"] = (d["High"] - d["Low"]) / C
    delta = C.diff()
    g, l = delta.clip(lower=0), -delta.clip(upper=0)
    ag = g.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = l.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    o["rsi_14"] = 100 - 100 / (1 + ag / al)
    e12, e26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    o["macd_n"] = macd / C
    o["macd_signal_n"] = sig / C
    o["macd_hist_n"] = (macd - sig) / C
    return o


def build(df, start="1996-01-01"):
    d = df.sort_values("Date").reset_index(drop=True)
    d = d[d["Date"] >= start].reset_index(drop=True)
    cont = rival_indicators(d)
    binr = to_binary(cont, d["Close"])
    si = scale_invariant(d)
    out = pd.concat([d[["Date", "Close"]], cont, binr, si], axis=1)
    out["consensus"] = binr.sum(axis=1)
    nxt = d["Close"].shift(-1)
    out["ret_next"] = nxt / d["Close"] - 1
    out["target"] = np.where(nxt.isna(), np.nan, (nxt > d["Close"]).astype(float))
    # same-day label, used ONLY for the horizon-alignment diagnostic
    out["target_sameday"] = (d["Close"] > d["Close"].shift(1)).astype(float)
    # drop warm-up rows (longest window = 50 days + 1 lag); keep the last unlabeled row separately
    feat_ok = out[CONT_COLS + SI_COLS].notna().all(axis=1)
    out = out[feat_ok].reset_index(drop=True)
    return out
