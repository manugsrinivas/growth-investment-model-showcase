"""XLE ENERGY-REGIME carve-out SIGNAL for the Growth slot (added 2026-07-05 by USER CONVICTION,
against neutral live evidence — see CAVEAT). Rides INSIDE the Growth account like the ex-US sleeve
(no separate account/keys); growth_slot carves k of the budget into XLE when the energy view is on.

The view: a commodity/inflation regime (weak-tech / energy super-cycle) favors energy over QQQ. XLE
is the best DIVERSIFIER found in the 2026-07-05 sector sweep (corr to QQQ just 0.32-0.42 when funded
vs ex-US 0.77), and in the 2001-08 energy super-cycle it added +1.02pt/yr (XLE +26.6%/yr vs a dead
post-dot-com QQQ +5.3%/yr).

  >>> CAVEAT (honest record): on the LIVE data path (USO-based oil signal, 2008-26 — crude futures
  >>> CL=F aren't Alpaca-tradeable) the sleeve is NEUTRAL (FULL 08-26 -0.03pt/+0.00 Sharpe), it
  >>> LAGGED QQQ in the one recent energy regime (2021-23: XLE +37.8% vs QQQ +45.3%), and it carries
  >>> a -2.8% worst-12mo tail at K_MAX=12% (FAILS the -1.5% bounded gate ex-US passed). The
  >>> compelling +1.02pt was the 2001-08 super-cycle, which REQUIRED QQQ to be dead — not a
  >>> repeatable condition. This is a deliberate forward VIEW bet on that regime returning, NOT an
  >>> evidence-supported add. It only pays if QQQ has a lost decade AND energy booms.

Gate = S3-mandatory: fund ONLY when E3 (XLE momentum confirm) AND (E1 or E2) (a macro reason).
E1 = USO (oil ETF) above its 12mo MA (live proxy; 80% month-match to CL=F crude).
E2 = XLE/SPY relative price in the cheap tail (energy underinvested).
E3 = XLE beats QQQ on 6mo AND XLE in its own uptrend (don't catch a falling knife).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

MACRO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "macro")
STALE_MAX_DAYS = 10   # carve nothing if the macro cache is older than this (see current_allocation)
K_MAX = 0.12          # user conviction cap (chose 12-15%); 12% keeps worst-12mo ~-3.4% vs -4.2% at
                      # 15%. To max energy capture, set to 0.15 (one line) -- accepts the -4.2% tail.
MA_BAND = 0.03
BASKET = {"XLE": 1.0}  # single-ETF sleeve (energy sector); no sub-basket


def _px(s):
    return pd.read_csv(sorted(glob.glob(f"{MACRO}/{s}_daily_*.csv"))[-1],
                       index_col=0, parse_dates=True)["close"]


def _hyst(cond_below):
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
    """Monthly E1/E2/E3 flags + k (leak-free). Reads live-available macro ETFs (USO, XLE, SPY, QQQ)."""
    daily = pd.DataFrame({s: _px(s) for s in ["USO", "XLE", "SPY", "QQQ"]}).dropna()
    M = daily.resample("ME").last()
    # E1 oil/inflation uptrend: USO above its 12mo MA (hysteretic). Live proxy for crude trend.
    e1 = _hyst(M["USO"] / M["USO"].rolling(12).mean() - 1.0)
    # E2 valuation: XLE/SPY relative price in the CHEAP tail of its expanding history.
    ratio = M["XLE"] / M["SPY"]
    pct = ratio.expanding(24).apply(lambda w: (w.iloc[-1] <= w).mean(), raw=False)
    e2 = (pct < 0.25).astype(float)
    # E3 momentum confirm: XLE beats QQQ 6mo AND XLE in its own uptrend.
    rel6 = M["XLE"].pct_change(6) - M["QQQ"].pct_change(6)
    up = _hyst(M["XLE"] / M["XLE"].rolling(10).mean() - 1.0)
    e3 = ((rel6 > 0) & (up < 0.5)).astype(float)
    df = pd.DataFrame({"e1": e1, "e2": e2, "e3": e3}, index=M.index)
    macro = ((df["e1"] + df["e2"]) > 0).astype(float)          # S3-mandatory: E3 AND (E1 or E2)
    df["k"] = K_MAX * ((df["e1"] + df["e2"] + df["e3"]) / 3.0) * macro * df["e3"]
    return df


def current_allocation():
    """Live readout: today's XLE weight k (of the Growth budget) + signal states. Fail-safe: any
    data problem -> k=0 (dormant), so a signal glitch never carves capital into XLE."""
    try:
        df = signals()
        row = df.iloc[-1]
        k = float(row["k"])
        # STALENESS -> DORMANT (added 2026-08-27; mirrors growth_exus). The existing fail-safe
        # only catches an EXCEPTION; a stale-but-READABLE dated-filename cache returns a stale k
        # that looks perfectly valid. This sleeve reads data/macro/*.csv via sorted(glob)[-1],
        # frozen 72 days back when this was found, and can carve up to 12% of the Growth budget
        # on a signal computed from stale prices. Measured on the DAILY source, never on
        # df.index -- signals() resamples to month-end, so its last label is a BIN that stays
        # current-looking while the underlying prices go weeks stale. Fails in the SAFE
        # direction, matching this module's existing philosophy: if we cannot trust the data,
        # carve nothing.
        import datetime as _d
        _asof = _px("XLE").index[-1].date()
        _age = (_d.date.today() - _asof).days
        if _age > STALE_MAX_DAYS:
            return {"asof": str(_asof), "k_xle": 0.0,
                    "e1_oil_uptrend": 0, "e2_valuation_gap": 0, "e3_momentum_confirm": 0,
                    "state": f"DORMANT (data STALE by {_age}d > {STALE_MAX_DAYS}d — refusing to "
                             f"carve on stale prices; k would have been {k:.4f})"}
        return {"asof": str(df.index[-1].date()), "k_xle": round(k, 4),
                "e1_oil_uptrend": int(row["e1"]), "e2_valuation_gap": int(row["e2"]),
                "e3_momentum_confirm": int(row["e3"]),
                "state": "FUNDED" if k > 0 else "DORMANT (100% to QQQ dial)"}
    except Exception as e:
        return {"asof": "n/a", "k_xle": 0.0, "e1_oil_uptrend": 0, "e2_valuation_gap": 0,
                "e3_momentum_confirm": 0, "state": f"DORMANT (signal unavailable: {str(e)[:60]})"}


def _gauntlet():
    """Reproducible LIVE-PATH gauntlet (USO-based E1, macro 2008-26) — the honest shipping validation.
    (The deeper 2001-26 CL=F validation used crude futures not in-repo; live path is USO.)"""
    df = signals(); M = pd.DataFrame({s: _px(s) for s in ["XLE", "QQQ"]}).dropna().resample("ME").last()
    R = M.pct_change()
    k = df["k"].shift(1).reindex(R.index).fillna(0.0)
    core, xle = R["QQQ"], R["XLE"]
    blend = (1 - k) * core + k * xle - (k - k.shift()).abs().fillna(0) * 0.0005

    def met(eq):
        r = eq.pct_change().dropna(); n = max(len(r), 1); tot = eq.iloc[-1] / eq.iloc[0] - 1
        return (1 + tot) ** (12 / n) - 1, (r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0)
    print(f"=== XLE energy sleeve — LIVE-PATH gauntlet (USO E1, K_MAX={K_MAX:.0%}) ===")
    print(f"  {'window':<16}{'coreQQQ':>14}{'blend':>14}{'delta':>15}{'fund%':>7}")
    for lbl, s, e in [("FULL 08-26", "2008-01-01", "2026-12-31"), ("tech 09-20", "2009-01-01", "2020-12-31"),
                      ("inflation 21-23", "2021-01-01", "2023-12-31"), ("recent 24-26", "2024-01-01", "2026-12-31")]:
        m = (R.index >= s) & (R.index <= e)
        ac, sc = met((1 + core[m].dropna()).cumprod()); ab, sb = met((1 + blend[m].dropna()).cumprod())
        print(f"  {lbl:<16}{ac*100:>7.1f}%/{sc:>4.2f}{ab*100:>7.1f}%/{sb:>4.2f}{(ab-ac)*100:>+8.2f}pt/{sb-sc:>+5.2f}{(k[m]>0).mean()*100:>6.0f}%")
    d12 = ((1 + blend).rolling(12).apply(np.prod) - (1 + core).rolling(12).apply(np.prod)).min()
    print(f"  GATE1 worst-12mo drag {d12*100:+.1f}% (bounded gate wants > -1.5% -> "
          f"{'PASS' if d12 > -0.015 else 'FAIL — accepted as a conviction bet'})")
    print(f"  CAVEAT: neutral on live data; a forward VIEW bet on a 2001-08-style regime returning.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauntlet", action="store_true", help="run the live-path validation")
    if ap.parse_args().gauntlet:
        _gauntlet()
    else:
        import json
        print(json.dumps(current_allocation(), indent=2))
