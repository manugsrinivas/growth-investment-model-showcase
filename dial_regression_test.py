"""DIAL REGRESSION TEST. Run after ANY edit to
growth_dial.py / growth_slot.py -- guards the live money path against silent behavior changes.

Pins (1) the canonical backtest numbers on a FIXED window (immune to CSV data extensions),
(2) the structural config the A/B campaigns adopted, (3) the leverage->blend weight mapping,
(5) the LIVE dial's refusal to run on missing/insufficient input history, on real data, and
(6) the BACKTEST path's deliberately different handling of the same (abort on a missing CSV; seed
through the MA warm-up) -- so neither path can drift into the other's behaviour unnoticed.

  python dial_regression_test.py        # exit 0 = all green; nonzero = a pin broke
"""
import os
import sys

import numpy as np
import pandas as pd

# NO chdir (removed 2026-09-10): growth_dial and growth_slot anchor their own paths, and a
# suite that chdir's itself can never catch a working-directory regression. Run it from anywhere.
import growth_dial as G
import growth_slot as S

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'BROKEN'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# ---- 1. canonical backtest pins (FIXED window 2008-01-01..2026-06-05, band 3%) ----------------
qqq, lev, off = G.leverage_series()
rq = qqq.pct_change()
w = (rq.index >= "2008-01-01") & (rq.index <= "2026-06-05")
eq, lev1, flips = G._bt_run(lev[w], rq[w])
_, ann, sharpe, dd = G._metrics(eq)
yy = eq.resample("YE").last().pct_change()
y22 = float(yy[yy.index.year == 2022].iloc[0])

# RE-PINNED 2026-08-25 after CORE_FLOOR was REMOVED (0.25 -> 0.0; see growth_dial.py):
# 19.83% / 0.998 / -39.9% / 2022 -6% (retired flat-4% rate basis; superseded below). Removing the
# floor improved every measured axis; the deadband stays at 0.25x for its ~5x fewer leverage changes.
# RE-PINNED 2026-09-06 to the CORRECTED RATE BASIS (real daily 3-month bills + 50bp borrow,
# replacing the flat RF=0.04 that served as both borrow cost and cash yield). This is a CORRECTION,
# not an improvement: the headline barely moves (+0.13pp, because the book is a net borrower and the
# flat 4% overcharged borrow more than it overpaid cash), but 2022 moves materially from -6.1% to
# -7.6% because +1.46pp of the old number was cash interest the book never earned.
# The 2022 tolerance is TIGHTENED 0.03 -> 0.01: the old band was wide enough to pass on -8% while
# asserting "~-6%", which means it would NOT have caught a regression back to the wrong basis.
# Set growth_dial.RATES_REAL=False to reproduce the pre-2026-09-06 vintage (19.83/0.998/-39.9/-6.1).
check("AnnRet ~ 20.0%", abs(ann - 0.200) < 0.005, f"got {ann*100:.1f}%")
check("Sharpe ~ 1.00", abs(sharpe - 1.004) < 0.02, f"got {sharpe:.3f}")
check("MaxDD ~ -40%", abs(dd - (-0.40)) < 0.02, f"got {dd*100:.0f}%")
check("2022 ~ -7.6%", abs(y22 - (-0.076)) < 0.01, f"got {y22*100:.1f}%")

# ---- 2. structural config the A/B campaigns adopted --------------------------------------------
check("TARGET_LEV = 1.75", G.TARGET_LEV == 1.75)
check("VM semivol leg ON (tv .17)", G.VM_ON and abs(G.VM_TV - 0.17) < 1e-9)
check("discrete brake retired", G.VB_ENABLED is False)
check("breadth flag ON (thr .4, 9 ETFs)", G.BR_ON and G.BR_THR == 0.4 and len(G.BR_ETFS) == 9)
check("hysteresis band 3%", abs(G.BAND - 0.03) < 1e-9)
# knobs: CORE_FLOOR removed 2026-08-25; inelastic deadband retained
check("CORE_FLOOR REMOVED (0.0)", abs(G.CORE_FLOOR - 0.0) < 1e-9)
check("inelastic deadband 0.25x", abs(G.DEADBAND_LEV - 0.25) < 1e-9)
check("dial may de-risk to zero (SGOV is the sole brake)", G.CORE_FLOOR == 0.0)

# ---- 3. leverage -> QQQ/QLD/SGOV blend mapping --------------------------------------------------
# NB weights(0.0) -> {QQQ:0, SGOV:1}: with CORE_FLOOR removed the dial may go fully to cash.
for L, want in ((1.75, {"QQQ": 0.25, "QLD": 0.75}), (1.0, {"QQQ": 1.0, "QLD": 0.0}),
                (0.5, {"QQQ": 0.5, "SGOV": 0.5}), (0.0, {"QQQ": 0.0, "SGOV": 1.0})):
    got = S._weights_for_leverage(L)
    ok = all(abs(got.get(k, 0.0) - v) < 1e-9 for k, v in want.items())
    check(f"weights({L}x)", ok, f"got {got}")

# ---- 3b. semivol WARM-UP behaviour (pinned 2026-09-05) -------------------------------------------
# The canonical pins depend on pd.concat(...).max(axis=1) SKIPPING NaN: on days 5..20 the 21d
# window is undefined, so the leg falls back to sv5 and keeps a (shorter) spike guard. Under a
# literal max() the NaN would propagate and — because the downstream min(trend, semivol) also skips
# NaN — those days would run on the trend leg ALONE, unguarded, printing MaxDD -44.66% vs -39.93%.
# This pins the safer behaviour so a pandas default change cannot silently flip it.
import numpy as _np
import pandas as _pd
_ix = _pd.date_range("2020-01-01", periods=40, freq="B")
_px = _pd.Series(100.0 * (0.99 ** _np.arange(40)), index=_ix)     # steady decline -> nonzero semivol
_vm = G.vol_managed_series(_px)
check("semivol defined during warm-up (days 5..20)", bool(_vm.iloc[6:20].notna().all()),
      f"{int(_vm.iloc[6:20].isna().sum())} NaN in the warm-up span")
check("semivol warm-up falls back to sv5, not NaN", bool(_vm.iloc[6:20].notna().any()))
_neg = _px.pct_change().clip(upper=0.0)
_s5 = _np.sqrt((_neg ** 2).rolling(5).mean()) * _np.sqrt(252)
_s21 = _np.sqrt((_neg ** 2).rolling(21).mean()) * _np.sqrt(252)
check("warm-up uses sv5 while sv21 is undefined", bool(_s21.iloc[10] != _s21.iloc[10]
      and abs(float(_vm.iloc[10]) - float((G.VM_TV / max(_s5.iloc[10], G.VM_FLOOR)))) < 1e-9),
      "sv21 must be NaN at i=10 and the leg must equal the sv5-derived value")

# ---- 3c. _slot_sim SIGNAL LAG -- exactly ONE day, same as _bt_run (pinned 2026-09-06) ------------
# The cadence/instrument sim used to apply the day's return BEFORE rebalancing to Ld[i], so the
# exposure earning day i's return came from dial[i-2]: TWO days of lag where `Ld = dial.shift(1)`,
# the docstring and `_bt_run`'s `lev1 = levs.shift(1)` all claim ONE. It cost 1.34pp/yr of CAGR on
# the deployed 'gated' book and was 77% of the apparent "QLD is worse than clean leverage" penalty.
# TEETH: with RF=0 and QLD_FEE=0 the QQQ/QLD/cash blend returns exactly L*r, so equity_return/r
# reads the held leverage straight off the curve. Re-introduce the extra day and the first check
# fails while the second (an anti-pin for precisely that regression) starts passing.
_saved = (G.RATES_REAL, G.RF, G.QLD_FEE)
G.RATES_REAL, G.RF, G.QLD_FEE = False, 0.0, 0.0
try:
    _lix = _pd.bdate_range("2020-01-01", periods=9)
    _lr = _pd.Series([.01, .02, .03, .04, .05, .06, .07, .08, .09], index=_lix)
    _ld = _pd.Series([1.0, 1.0, 1.0, 1.75, 1.75, 1.75, 1.0, 1.0, 1.0], index=_lix)
    _leq, _, _ = G._slot_sim(_lr, _ld, "daily")
    _lret = _leq / _leq.shift(1) - 1.0
    _lret.iloc[0] = _leq.iloc[0] - 1.0                     # bar 1 carries a real return
    _held = _lret / _lr.reindex(_leq.index)
    _d1 = float((_held - _ld.shift(1).reindex(_leq.index)).abs().max())
    _d2 = float((_held - _ld.shift(2).reindex(_leq.index)).abs().max())
    check("_slot_sim lag == dial.shift(1) (ONE day, as documented)", _d1 < 1e-9,
          f"max|held - shift(1)| {_d1:.1e}")
    check("_slot_sim lag is NOT dial.shift(2) (anti-pin for the 2-day defect)", _d2 > 1e-3,
          f"max|held - shift(2)| {_d2:.3f}")
    # ...and where no QLD sleeve is held (L <= 1) the two engines must be the SAME model.
    _cq = qqq[w]
    _crq = _cq.pct_change()
    for _L in (0.0, 0.5, 1.0):
        _e1, _, _ = G._slot_sim(_crq.dropna(), _pd.Series(_L, index=_crq.dropna().index), "daily")
        _e2, _, _ = G._bt_run(_pd.Series(_L, index=_cq.index), _crq)
        _a = (_e1 / _e1.shift(1) - 1).dropna()
        _b = (_e2 / _e2.shift(1) - 1).dropna()
        _ii = _a.index.intersection(_b.index)
        _mx = float((_a.reindex(_ii) - _b.reindex(_ii)).abs().max())
        check(f"_slot_sim == _bt_run at constant {_L}x (no QLD sleeve, no ER)", _mx < 1e-12,
              f"max daily |diff| {_mx:.1e} over {len(_ii)} bars")
finally:
    G.RATES_REAL, G.RF, G.QLD_FEE = _saved

# ---- 3d. (removed in the public export: these pinned the after-tax engine's signal lag, and
#      that engine is not published because it is calibrated to a personal tax bracket) ----

# ---- 4. live-path plumbing ----------------------------------------------------------------------
check("dial_state_live exists", callable(getattr(G, "dial_state_live", None)))
check("breadth_series exists", callable(getattr(G, "breadth_series", None)))
b = G.breadth_series()
check("breadth in [0,1]", bool((b.dropna() >= 0).all() and (b.dropna() <= 1).all()),
      f"last {b.iloc[-1]:.2f}")

# ---- 5. LIVE DIAL: missing/insufficient input history => REFUSE (pinned 2026-09-13, real data) ------
# dial_state_live used to put a symbol with no CSV history on the ~420-calendar-day live stub and carry
# on with a printed warning. _hyst_off's seed is risk-ON and the stub has only ~89 bars with a valid
# 200d MA, so an established risk-OFF state the stub never re-crosses the band of reads as risk-ON.
# Scan 2010-01-04..2026-09-11 (4,198 sessions x 11 inputs): a missing IEF CSV would have RAISED the dial
# on 158 sessions, HYG on 70, max +0.583x, clearing the 0.25x deadband on 105 / 57 -- and lowered it on
# NONE. The two dates below sit mid-run in the longest such stretches (IEF 2021-10-19..11-24, weeks
# before the 2022 bear; HYG 2015-05-07..06-26). Runs on a TEMP COPY of the dial CSVs truncated at the
# as-of date, through a mock data client -- no network, never the live data dir.
# TEETH: the pre-fix code returns 1.75x on both "missing" cases instead of refusing.
import contextlib
import io
import shutil
import tempfile

_DSYMS = sorted(set(G.SIGNALS) | set(G.BR_ETFS))
_REAL = {s: G._load(s) for s in _DSYMS}


class _Bars:
    """Alpaca-shaped data client serving the real closes in (asof-420d, asof], stamped 04:00 UTC."""

    def __init__(self, asof):
        self.asof = pd.Timestamp(asof)

    def get_stock_bars(self, req):
        lo = self.asof - pd.Timedelta(days=420)
        rows = [(s, pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=4), float(v))
                for s, c in _REAL.items() for d, v in c[(c.index > lo) & (c.index <= self.asof)].items()]
        df = pd.DataFrame(rows, columns=["symbol", "timestamp", "close"])
        return type("R", (), {"df": df.set_index(["symbol", "timestamp"])})()


def _live_on_copy(asof, delete=()):
    """REAL dial_state_live on a temp copy of the dial CSVs truncated at `asof` -> (lev, flags) or the
    exception it raised."""
    root = tempfile.mkdtemp(prefix="dialreg_")
    os.makedirs(os.path.join(root, "data", "macro"))
    for s, c in _REAL.items():
        if s not in delete:
            c[c.index <= pd.Timestamp(asof)].to_frame("close").to_csv(
                os.path.join(root, "data", "macro", f"{s}_daily_2008-01-01_COPY.csv"))
    saved = G._HERE
    G._HERE = root
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return G.dial_state_live(_Bars(asof))
    except (Exception, SystemExit) as e:
        return e
    finally:
        G._HERE = saved
        shutil.rmtree(root, ignore_errors=True)


def _desc(r):
    return (f"raised {type(r).__name__}: {str(r)[:60]}" if isinstance(r, BaseException)
            else f"RETURNED {r[0]:.4f}x")


for _asof, _sym in (("2021-11-05", "IEF"), ("2015-06-03", "HYG")):
    _r = _live_on_copy(_asof)
    check(f"{_asof} full history: 1.1667x with {_sym} risk-OFF (the reference)",
          isinstance(_r, tuple) and abs(_r[0] - G.TARGET_LEV * 2 / 3) < 1e-9 and _r[1].get(_sym) is True,
          _desc(_r))
    _r = _live_on_copy(_asof, delete=(_sym,))
    check(f"{_asof} {_sym} CSV missing: REFUSES, naming {_sym} (pre-fix: the seed's 1.75x)",
          isinstance(_r, RuntimeError) and _sym in str(_r), _desc(_r))

_pinf = getattr(G, "_hyst_pinned", None)
_unpinned = [s for s in _DSYMS if _pinf is None or not _pinf(_REAL[s].dropna())]
check("deployed CSVs make every dial input data-determined (the guard cannot fire on them)",
      _pinf is not None and not _unpinned, f"unpinned: {_unpinned or 'none'}")
_latest = min(c.index[-1] for c in _REAL.values())
_r = _live_on_copy(_latest)
check(f"latest session {_latest.date()} on a copy: dial computes, no refusal", isinstance(_r, tuple),
      _desc(_r))

# ---- 6. BACKTEST path: deliberately NOT guarded, and pinned that way (2026-09-13) ------------------
# leverage_series()/breadth_series() have no live stub to degrade to, so a missing CSV ABORTS (_load's
# SystemExit) -- loud, never a degraded dial. They DO run each flag's risk-ON seed through the 200d-MA
# warm-up: that is the warm-up artifact the section-1 pins are built on (a separate warm-start test
# found it PESSIMISTIC: it deepens the 2008 drawdown). Applying the live guard here would drop 2008's
# warm-up and move every pin, so it is not applied -- change either only with a deliberate re-pin.
_empty = tempfile.mkdtemp(prefix="dialreg_empty_")
_saved_here = G._HERE
G._HERE = _empty
try:
    try:
        G.leverage_series()
        _bt = None
    except (Exception, SystemExit) as e:
        _bt = e
finally:
    G._HERE = _saved_here
    shutil.rmtree(_empty, ignore_errors=True)
check("backtest: a missing CSV ABORTS via SystemExit (no stub, no degraded dial)",
      isinstance(_bt, SystemExit), "returned a dial" if _bt is None else type(_bt).__name__)
_warm = off.iloc[:G.MA - 1]
check(f"backtest: first {G.MA - 1} bars run every trend flag on the risk-ON SEED (warm-up, pinned as-is)",
      bool((_warm[["QQQ", "IEF", "HYG", "BREADTH"]] == 0.0).all().all()),
      f"{_warm.index[0].date()}..{_warm.index[-1].date()}")
try:
    _dl = G.dial_leverage("2000-01-01")                  # an as-of before any data exists
except Exception as e:
    _dl = e
check("dial_leverage() with no history REFUSES (pre-fix: returned TARGET_LEV, the MAXIMUM)",
      isinstance(_dl, Exception), type(_dl).__name__ if isinstance(_dl, Exception) else f"returned {_dl!r}")

print(f"\n{'ALL GREEN' if not FAIL else 'REGRESSION BROKEN: ' + ', '.join(FAIL)}")
sys.exit(0 if not FAIL else 1)
