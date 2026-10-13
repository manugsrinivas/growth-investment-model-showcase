"""EX-US carve-out SIGNAL for the Growth slot (re-scoped 2026-07-05 to a GROWTH-ACCOUNT carve-out
-- NO separate account, NO new keys; it rides inside the existing Growth account via growth_slot).

This is the LIVE copy of the ex-US view signal. The research/validation study that chose these
thresholds is not published here; it read the same macro CSVs and used the same thresholds.

The view: a weak-dollar / valuation-mean-reversion regime favors ex-US growth over QQQ. It is a
BOUNDED FORWARD OPTION, explicitly NOT claimed as alpha (tactical ex-US rotation failed in earlier
testing). In that unpublished study it was bounded and diversifying: worst 12mo drag -0.7%,
+0.32pt/+0.05 Sharpe in the 2003-07 ex-US decade, corr 0.55 to QQQ when funded, near-costless
full-window. growth_slot carves k in [0, K_MAX=15%] of the
Growth budget into the ex-US basket when the view is on; (1-k) stays in the QQQ dial blend.

Gate = S3-mandatory: fund ONLY when S3 (momentum confirm) AND (S1 or S2) (a macro reason).
"""
import glob
import os

import numpy as np
import pandas as pd

MACRO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "macro")
STALE_MAX_DAYS = 10   # carve nothing if the macro cache is older than this (see current_allocation)
K_MAX = 0.15          # hard cap: ex-US <= 15% of the Growth budget (bounded-downside gate confirms)
MA_BAND = 0.03        # reuse the growth dial's hysteresis band
BASKET = {"EFA": 0.60, "EEM": 0.40}   # validated developed-tilt ex-US proxy (EFG/IEMG deferred)


def _px(s):
    return pd.read_csv(sorted(glob.glob(f"{MACRO}/{s}_daily_*.csv"))[-1],
                       index_col=0, parse_dates=True)["close"]


def _hyst(cond_below):
    """Hysteretic bool from a %-distance condition: risk-ON when < -band, OFF when > +band."""
    x = cond_below.to_numpy()
    out = np.zeros(len(x)); state = False
    for i in range(len(x)):
        if x[i] != x[i]:
            out[i] = 1.0 if state else 0.0; continue
        if state and x[i] > MA_BAND: state = False
        elif not state and x[i] < -MA_BAND: state = True
        out[i] = 1.0 if state else 0.0
    return pd.Series(out, index=cond_below.index)


def signals():
    """Monthly S1/S2/S3 flags + k (all leak-free). Returns (monthly_frame_with_k)."""
    daily = pd.DataFrame({s: _px(s) for s in ["UUP", "SPY", "QQQ", "EFA", "EEM"]}).dropna()
    M = daily.resample("ME").last()
    # S1 dollar downtrend: UUP below its 12mo MA (hysteretic).
    s1 = _hyst(M["UUP"] / M["UUP"].rolling(12).mean() - 1.0)
    # S2 valuation-gap proxy: EFA/SPY relative-price in the CHEAP tail of its expanding history.
    ratio = M["EFA"] / M["SPY"]
    pct = ratio.expanding(36).apply(lambda w: (w.iloc[-1] <= w).mean(), raw=False)
    s2 = (pct < 0.25).astype(float)
    # S3 relative-momentum confirm: 0.6EFA/0.4EEM beats QQQ 6mo AND is in its own uptrend.
    basket = 0.6 * M["EFA"] + 0.4 * M["EEM"]
    rel6 = basket.pct_change(6) - M["QQQ"].pct_change(6)
    up = _hyst(basket / basket.rolling(10).mean() - 1.0)
    s3 = ((rel6 > 0) & (up < 0.5)).astype(float)
    df = pd.DataFrame({"s1": s1, "s2": s2, "s3": s3}, index=M.index)
    # CANONICAL S3-mandatory gate: fund only when S3 AND (S1 or S2). Scaled 0->K_MAX by agreement.
    macro = ((df["s1"] + df["s2"]) > 0).astype(float)
    df["k"] = K_MAX * ((df["s1"] + df["s2"] + df["s3"]) / 3.0) * macro * df["s3"]
    return df


def current_allocation():
    """Live readout: today's ex-US weight k (of the Growth budget) and the three signal states.
    Fail-safe: any data problem -> k=0 (dormant), so a signal glitch never carves capital out."""
    try:
        df = signals()
        row = df.iloc[-1]
        k = float(row["k"])
        # STALENESS -> DORMANT (added 2026-08-27). The existing fail-safe only catches an
        # EXCEPTION; a stale-but-READABLE dated-filename cache returns a stale k that looks
        # perfectly valid. These sleeves read data/macro/*.csv via sorted(glob)[-1], which was
        # frozen 72 days back when this was found, and they can carve up to 15% of the Growth
        # budget on a signal computed from stale prices. Same failure class as any
        # dated-cache defect. Fails in the SAFE direction, matching this module's existing
        # philosophy: if we cannot trust the data, carve nothing.
        # Measure staleness on the DAILY source, never on df.index -- signals() resamples to
        # month-end, so its last label is a BIN (e.g. 2026-08-31) that stays current-looking even
        # when the underlying prices are weeks old. Checking the bin would catch a 72-day freeze
        # but silently miss a 3-week one.
        import datetime as _d
        _asof = _px("SPY").index[-1].date()
        _age = (_d.date.today() - _asof).days
        if _age > STALE_MAX_DAYS:
            return {"asof": str(_asof), "k_exus": 0.0,
                    "s1_dollar_downtrend": 0, "s2_valuation_gap": 0, "s3_momentum_confirm": 0,
                    "state": f"DORMANT (data STALE by {_age}d > {STALE_MAX_DAYS}d — refusing to "
                             f"carve on stale prices; k would have been {k:.4f})"}
        return {"asof": str(df.index[-1].date()), "k_exus": round(k, 4),
                "s1_dollar_downtrend": int(row["s1"]), "s2_valuation_gap": int(row["s2"]),
                "s3_momentum_confirm": int(row["s3"]),
                "state": "FUNDED" if k > 0 else "DORMANT (100% to QQQ dial)"}
    except Exception as e:
        return {"asof": "n/a", "k_exus": 0.0, "s1_dollar_downtrend": 0, "s2_valuation_gap": 0,
                "s3_momentum_confirm": 0, "state": f"DORMANT (signal unavailable: {str(e)[:60]})"}


if __name__ == "__main__":
    import json
    print(json.dumps(current_allocation(), indent=2))
