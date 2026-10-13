"""Reproduce every number in the README's results section.

    python results_table.py

Uses the same price data as dial_regression_test.py (rebuild it with
`python fetch_macro_bars_swing.py --etf-only`) and the same fixed window, 2008-01-01 to 2026-06-05.
Every row starts on the same day as the pinned test, so the rows are like-for-like.
"""
import numpy as np
import pandas as pd

import growth_dial as G

START, END = "2008-01-01", "2026-06-05"


def stats(eq):
    """CAGR, return/vol, Sharpe in excess of cash, max drawdown, 2022 return."""
    _, cagr, ret_vol, mdd = G._metrics(eq)
    r = eq.pct_change().dropna()
    cash, _ = G.rates_daily(r.index)
    ex = r - cash
    sharpe_ex = ex.mean() / ex.std() * np.sqrt(252)
    yearly = eq.resample("YE").last().pct_change()
    y22 = float(yearly[yearly.index.year == 2022].iloc[0])
    return cagr, ret_vol, sharpe_ex, mdd, y22


def row(name, eq, extra=""):
    cagr, rv, sx, mdd, y22 = stats(eq)
    print(f"  {name:<42} {cagr*100:6.1f}%  {rv:6.2f}  {sx:7.2f}  {mdd*100:7.1f}%  {y22*100:6.1f}%  {extra}")


def worst_after(eq, date):
    e = eq[eq.index >= date]
    dd = e / e.cummax() - 1
    return dd.min(), dd.idxmin().date()


qqq, lev, off = G.leverage_series()
rq = qqq.pct_change()
w = (rq.index >= START) & (rq.index <= END)
r, L = rq[w], lev[w]

sim, n_reb, _ = G._slot_sim(r, L, "gated")
simple, held, n_changes = G._bt_run(L, r)
bh, _, _ = G._bt_run(pd.Series(1.0, index=L.index), r)
static, n_static, _ = G._slot_sim(r, pd.Series(G.TARGET_LEV, index=L.index), "gated")

print(f"Window {sim.index[0].date()} .. {sim.index[-1].date()} ({len(sim) - 1} daily returns)\n")
print(f"  {'':<42} {'CAGR':>7}  {'Ret/vol':>6}  {'Sharpe*':>7}  {'MaxDD':>8}  {'2022':>7}")
row("Dial: QQQ/QLD/SGOV, simulated (_slot_sim)", sim, f"{n_reb} rebalances")
row("Dial: simple leverage model (_bt_run)", simple, f"avg exposure {held.mean():.2f}x, {n_changes} target changes")
row("QQQ, buy and hold", bh)
row(f"Constant {G.TARGET_LEV}x target, simulated, no dial", static, f"{n_static} rebalances")
print("  * Sharpe in excess of the 3-month T-bill. 'Ret/vol' = mean daily return / its std x sqrt(252).")

print("\nDrawdowns")
for name, eq in (("dial, simulated", sim), ("dial, simple model", simple), ("QQQ", bh)):
    dd = eq / eq.cummax() - 1
    low = dd.idxmin()
    peak = eq[:low].idxmax()
    w2, d2 = worst_after(eq, "2009-07-01")
    print(f"  {name:<20} worst {dd.min()*100:6.1f}% (peak {peak.date()}, low {low.date()}) | "
          f"worst from 2009-07-01: {w2*100:6.1f}% (low {d2})")
print(f"  Trend flags start on a risk-on seed for the first {G.MA - 1} sessions (to {L.index[G.MA - 2].date()}).")
print(f"  Dial held leverage: {held.loc[:'2008-10-15'].iloc[-1]:.2f}x on 2008-10-15, "
      f"{held.loc[:'2008-11-20'].iloc[-1]:.2f}x on the 2008-11-20 low.")
flags = off.loc["2008-10-15":"2008-11-20", ["BREADTH", "IEF", "HYG"]]   # the three flags the dial counts
print("  Risk-off share of days, 2008-10-15..2008-11-20: "
      + ", ".join(f"{c} {flags[c].mean()*100:.0f}%" for c in flags.columns))

print("\nStart-date sensitivity (simulated dial)")
r1 = rq.dropna()
r1 = r1[r1.index <= END]
sim1, _, _ = G._slot_sim(r1, lev.reindex(r1.index), "gated")
c1, rv1, _, _, _ = stats(sim1)
print(f"  starting one session later (skips {r.index[2].date()}, QQQ {r.iloc[2]*100:+.1f}% that day): "
      f"CAGR {c1*100:.1f}%, ret/vol {rv1:.2f}")

print("\nDeadband (simple model)")
saved = G.DEADBAND_LEV
for band in (0.0, saved, 0.5):
    G.DEADBAND_LEV = band
    try:
        _, lev_b, _ = G.leverage_series()
    finally:
        G.DEADBAND_LEV = saved
    eq_b, _, n_b = G._bt_run(lev_b[w], r)
    c, rv, _, mdd_b, y22_b = stats(eq_b)
    print(f"  deadband {band:.2f}x: {n_b:>5} target changes | CAGR {c*100:.2f}% | ret/vol {rv:.3f} | "
          f"MaxDD {mdd_b*100:.1f}% | 2022 {y22_b*100:.1f}%")
