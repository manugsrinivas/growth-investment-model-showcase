"""Rule-based (NO ML) macro RISK DIAL for the growth slot -- autonomous of every other model.

Growth (long-duration tech) dies hard in rate-hike / risk-off regimes. This dial cuts the slot's
leverage in those regimes using ONLY price signals on liquid macro ETFs (fully self-contained).

  TREND LEG -- three risk-off flags, each a 200-day-MA trend with a 3% hysteresis band (flip
  risk-off below MA*0.97, back risk-on above MA*1.03, hold the state in between):
    BREADTH  -> fewer than 40% of 9 growth/intl ETFs (QQQ SPYG SMH IGV XLK XLY IWM EFA EEM) are in
                uptrend = broad equity breakdown. (QQQ's own flag is used only where breadth has
                no data in the backtest, and is shown for information live.)
    IEF      -> Treasuries falling = YIELDS RISING = rate-hike regime  (the growth-killer)
    HYG      -> high-yield weakening = credit stress
  n_off in 0..3  ->  trend leverage = TARGET_LEV * (1 - n_off/3), TARGET_LEV = 1.75:
    0 flags -> 1.75x   1 -> 1.17x   2 -> 0.58x   3 -> 0x (all SGOV, earns the T-bill rate)

  VOL LEG -- downside-semivol targeting: clip(0.17 / annualised downside semivol, 0.5, 2.0), where
  semivol is the larger of the 5-day and 21-day windows (a spike guard).

  leverage = min(trend leg, vol leg), clipped to [CORE_FLOOR = 0, TARGET_LEV = 1.75], then held by
  a 0.25x DEADBAND (the target only moves when the raw dial moves more than 0.25x).

These settings (band, breadth basket and threshold, semivol target, ceiling, deadband) were chosen
by in-sample A/B testing on 2008-2026 data; the 200-day MA itself is a standard choice. Figures in
the comments below are dated to the test that produced them; anything dated before 2026-09-06 is on
the retired flat-4% rate basis and is history, not a current pin (current pins:
dial_regression_test.py, section 1).

  python growth_dial.py --backtest      # hysteresis-band sweep vs static 1.75x and 1x QQQ
  python growth_dial.py --cadence       # rebalance-cadence sweep on the QQQ/QLD/SGOV slot simulation
  python growth_dial.py --today         # current dial leverage from the cached CSVs
  dial_leverage()                        # BACKTEST-path leverage as of a date (live = dial_state_live)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# --- PATH ANCHOR (2026-09-06; replaces an `os.chdir()` that ran AT IMPORT TIME) ---------------
# This module used to call `os.chdir(dirname(abspath(__file__)))` here, so merely IMPORTING the dial
# mutated the whole process's working directory as a side effect. That was not cosmetic: it was
# LOAD-BEARING for the live executor. `growth_slot.py` has no chdir of its own and resolved
# `logs/`, `state/growth_slot_lev.json` and `logs/PAUSE` -- the KILL SWITCH -- relatively, i.e. the
# live book's paths depended on an import side effect of a module that has no business owning them.
# Production only ever worked because every entry point is a .bat that `cd /d`s first.
# Every path in this file is now resolved from _HERE, and growth_slot does the same from its own
# _HERE, so both modules are cwd-independent and NOTHING moves the caller's cwd. Verified by running
# both suites AND the live entry points from an unrelated cwd -- the only way this defect shows up.
_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_LEV = 1.75    # risk-on ceiling (raised 1.5->1.75 in the 2026-07-05 semivol A/B; the VM
                     # leg still caps whenever downside vol is elevated, blend stays <=2x QLD-only)
# --- FINANCING AND CASH RATES (corrected 2026-09-06) ---
# WAS: a single flat `RF = 0.04` serving as BOTH the borrow cost on the levered portion AND the
# yield on cash when lev<1 -- two different rates collapsed into one constant, held flat across
# 2008-2026 while 3-month bills went ~5% -> 0.057% (2011-15 ZIRP mean) -> ~4.4%. Canonical-window
# mean is 1.41%, not 4.00%.
# The correction is SMALL on the headline and LARGE on one claim, which is why it was worth doing:
#   headline Ann +0.14pp at the 50bp default (the book is a NET BORROWER -- it borrows a mean
#   0.231 on 63% of days and lends 0.136 on 37% -- so a flat 4% charged phantom interest on ~1.7x
#   more notional than it credited; correcting it HELPS slightly, the opposite of the naive guess);
#   but 2022 goes -6.1% -> -7.6%, because +1.46pp of that celebrated number was cash interest the
#   book never earned. It held lev==0 on 223 of 251 days in 2022 while bills averaged 2.10%.
# The dial still beats QQQ in 2022 by +25.0pp (was +26.4pp) and the 2022 DRAWDOWN is unmoved.
RF = 0.04            # LEGACY flat rate. Retained ONLY to reproduce pre-2026-09-06 numbers via
                     # RATES_REAL=False, and as the fallback outside the DGS3MO series' coverage.
RATES_REAL = True    # use the real daily 3-month bill (data/rates/fred_DGS3MO.csv) for BOTH legs
BORROW_SPREAD_BP = 50.0
# 50bp over bills, chosen on ECONOMIC grounds and deliberately NOT tuned: the book's leverage lives
# inside QLD, financed at roughly institutional swap levels, so bills+~50bp is the right analogue
# (retail margin would be far higher; the 0.95% QLD expense ratio is charged separately in
# _slot_sim). NOTE the measured breakeven -- the spread at which this whole correction nets to zero
# on the headline -- is ~102bp. 50 was NOT picked to sit on either side of that. Sensitivity, so the
# reader can price it themselves: dAnn +0.27pp at 0bp, +0.20 at 25, +0.14 at 50, +0.00 at 100,
# -0.27 at 200. The 2022 restatement is -1.46pp at EVERY spread, because a book holding no leverage
# is untouched by the borrow rate.
QLD_FEE = 0.0095    # QLD expense ratio (approx) -- charged on the 2x sleeve in the cadence sim
MA = 200
SIGNALS = ("QQQ", "IEF", "HYG")
# --- BREADTH equity-trend flag (2026-07-05 A/B, adopted): the single-QQQ trend flag is replaced
# by growth-complex BREADTH — fraction of 9 growth/intl indexes in hysteretic uptrend; risk-off
# when < BR_THR (broad breakdown). Confirmatory voting: ignores single-index whipsaws, catches
# narrow-leadership tops earlier. A/B: 19.4%/0.98 -> 19.9%/1.00, wins 3/3 subperiods, robust
# across thr {0.4,0.5} x baskets {9-ETF, US-only}; stress windows/turnover identical.
# NOTE: breadth STANDALONE failed (chronic under-investment); it only works as this flag.
# LIVE PRICE BASIS — the alpaca Adjustment member name used by dial_state_live. Exposed at
# module level (not buried in the request) so a monitoring check (not part of this repo) can read
# the ACTUAL deployed basis instead of a hard-coded copy that could silently drift out of sync
# with this file. Must stay "ALL" to match the backtest CSVs' yfinance auto_adjust=True.
LIVE_ADJUSTMENT = "ALL"

# LIVE FEED — the alpaca DataFeed member name used by dial_state_live. Exposed here for the same
# reason as LIVE_ADJUSTMENT: a monitoring check can read it off this module, so it tracks the
# deployed feed instead of testing a hard-coded copy that can silently drift.
# SIP (2026-09-02, was IEX): IEX is a single venue carrying ~1.6% of consolidated volume (measured
# IEX/SIP volume ratio 0.016), while the backtest CSVs are yfinance CONSOLIDATED prices — so the
# deployed feed was a standing, if tiny, live-vs-backtest basis divergence of the same class as the
# 08-27 raw-close defect. The GROWTH key pair is already SIP-entitled; SIP is 15-minute DELAYED on
# this tier, which is irrelevant here because dial_state_live reads only COMPLETED daily sessions
# (see the `end` argument below). Verified before switching: same 289 bars, same last session, zero
# missing sessions, identical NaN counts, and an identical dial (trend 1.7500x, same flags) — close
# divergence is 0.019% mean / 0.038% max against a 3% hysteresis band.
# Honest detail: that band is NOT quite unflippable. Across 11 symbols x 289 sessions the two feeds
# disagreed on exactly ONE flag-day (IGV), where the price sat on the band edge — so "0.02% cannot
# cross 3%" is very nearly, but not strictly, true. No dial impact: IGV is a breadth constituent, and
# one ETF moves the breadth fraction by 1/9 (0.89 -> 0.78) against a 0.40 risk-off threshold.
# Confirmation the switch is the right direction: a parity check (not part of this repo) that measures
# the live feed against the yfinance CONSOLIDATED backtest basis improved ~10x — worst symbol went from
# IGV 0.038% on IEX to EEM 0.004% on SIP.
LIVE_FEED = "SIP"

BR_ON = True
BR_ETFS = ("QQQ", "SPYG", "SMH", "IGV", "XLK", "XLY", "IWM", "EFA", "EEM")
BR_THR = 0.4
BAND = 0.03          # hysteresis dead-band around the 200d MA (anti-whipsaw): flip risk-OFF below
                     # MA*(1-BAND), back risk-ON above MA*(1+BAND), HOLD state inside the band.
                     # Chosen 2026-06 from a {0,2,3,5%} sweep: 3% cuts whipsaw flips 86% (381->55)
                     # while keeping return/Sharpe and IMPROVING 2022 (-13%->-7%). 5% over-smooths.

# --- VOL EMERGENCY BRAKE (fast-crash blind-spot fix; conditional/extreme-only per FAJ 2020 /
# Harvey tail-thinning evidence; NOT continuous vol-targeting, which fails OOS per JFE 2020) ---
VB_ENABLED = False   # 2026-07-05: discrete brake RETIRED from the live dial — dominated in 8/8
                     # A/B cells by the continuous (semivol) VM leg, which cuts earlier & smoother.
                     # Code + --brake harness kept for reference.
VB_WIN = 20          # realized-vol lookback (days)
VB_ON = 0.90         # brake ENGAGES when 20d vol > expanding-window 90th percentile (extreme only)
VB_OFF = 0.80        # ...RELEASES when vol back below the 80th pct (hysteresis, anti-whipsaw)
VB_CAP = 0.5         # while engaged, leverage is CAPPED at this (brake only cuts, never adds)
VB_BURN = 504        # >=2yr burn-in before percentiles are trusted (brake disabled before that)

# --- CONTINUOUS VOL-MANAGED LEVERAGE (2nd, always-on brake; adopted 2026-07-04) ---
# Head-to-head 2008-26, identical costs: trend-dial Sharpe 0.81 (2022 -12%);
# vol-managed 0.84 (2022 -30%); min() HYBRID 0.89, maxDD -37%, 2022 -12%.
# Complementary failure modes: vol catches FAST crashes (2008/2020), trend flags
# catch SLOW grinds (2022) -> compose by taking the more defensive of the two.
# Validation: a 2008-26 head-to-head A/B (research script not published).
VM_ON = True         # compose lev = min(trend-dial, vol-managed) in live AND backtest
# 2026-07-05 A/B UPGRADE (vs total-vol 0.27): DOWNSIDE SEMIVOL targeting — upside vol no longer
# de-levers, so the leg re-enters V-recoveries (2009/2020) instead of missing them. Wide PASS
# plateau (ceil 1.6-1.75 x tv 0.17-0.19) on pooled + per-third subperiods + stress gates:
# AnnRet 18.2->19.4%, Sharpe 0.95->0.98, maxDD -38->-36%, COVID -23->-20%, 2022 -6/-7% intact.
VM_TV = 0.17         # DOWNSIDE-semivol target; L = clip(VM_TV/semivol, VM_LMIN, VM_LMAX).
#   PROVENANCE CORRECTED 2026-08-27 (comment only — the CONSTANT IS RIGHT, do not "fix" it).
#   The old comment read "~0.27/sqrt2 rescale", which does not produce this number:
#   0.27/sqrt(2) = 0.1909, i.e. 0.19. Under symmetry downside-semidev = total_vol/sqrt(2), so
#   0.17 corresponds to a 0.24 TOTAL-vol target (0.24/sqrt2 = 0.1697), not 0.27. The real
#   provenance is EMPIRICAL: 0.17 was selected from the A/B plateau above (tv 0.17-0.19 all
#   pass on pooled + per-third + stress gates), not derived from a total-vol target at all.
#   Worth knowing when re-tuning: 0.17 sits at the LOW EDGE of that validated plateau, not its
#   centre — 0.18 is the interior point if this is ever re-opened. The leg itself was
#   independently prosecuted 2026-08-27 and SURVIVED; only the derivation story was wrong.
VM_LMIN, VM_LMAX = 0.5, 2.0
VM_FLOOR = 0.056     # semivol floor so L can't explode on dead-calm data

# 2026-07-09 REDESIGN (research A/B, not published). Two knobs:
CORE_FLOOR = 0.0     # REMOVED 2026-08-25 (was 0.25). It existed ONLY to hold a "permanent QQQ core"
#   whose gains would defer to long-term rates. A lot-by-lot FIFO audit showed that deferral is
#   UNOBTAINABLE: Alpaca is FIFO-only and w_qqq = 2-lev falls as leverage rises, so FIFO consumes the
#   OLDEST lots first and just 0.68% of realized gains ever reached long-term. With the rationale
#   dead, the floor was measured as pure cost, and removing it improved every measured axis.
#   (The figures from that 2026-08-25 test are on the retired flat-4% rate basis and are not
#   quoted here; the current pins are in dial_regression_test.py.)
#   The floor bound on 89% of 2022 but 0% of 2008-09 and 0% of 2020 - i.e. its entire effect landed
#   in the one regime where it hurt. Set to a positive value ONLY with fresh evidence; the weight
#   mapping needs no floor (at lev=1.75 the QQQ leg is 2-1.75=0.25 by construction anyway).
DEADBAND_LEV = 0.25  # INELASTIC dial: hold leverage until the target moves > this, then snap.
#   Set 0.0 to restore the fully-elastic dial.
#
# ⚠ THE LATCH IS A TRADE, NOT A FREE WIN. Measured on the current code (pre-tax, _bt_run, fixed
# window 2008-01-01..2026-06-05), deadband OFF (0.00) vs the deployed 0.25:
#     AnnRet 19.97% -> 19.96%, Sharpe 1.004 -> 1.004          (return unchanged)
#     leverage changes 1,586 -> 298  (34.2% -> 6.4% of days, ~5x fewer)
#     worst 63-day return -31.0% -> -34.9%  (3.9pt WORSE)     MaxDD -38.1% -> -40.1%  (2.0pt WORSE)
# So the latch buys ~5x fewer rebalancing trades (each one realises short-term gains in a taxable
# account) and PAYS about 2-4pt of tail protection for it.
#
# ⚠ DO NOT "OPTIMISE" THIS PARAMETER ON THE SWEEP. A wider latch (0.50) scores better on every axis
# in that sweep -- on the same basis: AnnRet +0.26pp, Sharpe +0.039, worst 63-day +1.7pt, MaxDD
# +1.6pt, 194 fewer changes -- and it is NOT actionable. A 2026-09-05 study measured 38 purely
# SYMMETRIC latch values spanning 0.081 of Sharpe with sd 0.021, i.e. this parameter's own path
# noise is LARGER than the apparent gain. Two arms with the same rule snap on different days, their
# path noises decorrelate, and the difference looks like signal. Any change here needs its own
# pre-registered study with an exposure-matched control and a bootstrap against zero -- not a sweep.
# Recorded as a candidate, deliberately NOT adopted.


def _deadband(series, delta):
    """Inelasticity: hold the current level until the target deviates by > delta, then snap to it.
    Path-dependent (reconstructed from full history each call, like the hysteresis flags) so it needs
    no external state. NaNs pass through as the raw value (warm-up)."""
    a = series.to_numpy(); out = np.empty(len(a))
    cur = a[0]; out[0] = cur
    for i in range(1, len(a)):
        if a[i] == a[i] and (cur != cur or abs(a[i] - cur) > delta):
            cur = a[i]
        out[i] = cur
    return pd.Series(out, index=series.index)


def vol_managed_series(qqq_close):
    """Continuous vol-managed leverage on DOWNSIDE SEMIVOL. max(5d, 21d) is the SPIKE
    GUARD — never re-lever because vol is expected to mean-revert mid-crash; using only
    negative returns means big UP-days don't cut leverage. Leak-free: trailing windows."""
    r = qqq_close.pct_change()
    neg = r.clip(upper=0.0)
    # skipna=True is DELIBERATE and load-bearing, made explicit 2026-09-05 (it was pandas' silent
    # default, and the canonical pins depended on it without anyone having written that down).
    # On days 5..20 the 21d window is still NaN. Skipping it falls back to sv5, which keeps a
    # (shorter) spike guard. The literal max() the docstring implies would propagate NaN instead —
    # and because the downstream min(trend, semivol) ALSO skips NaN, that would drop the vol leg
    # entirely and run those days on the trend leg alone, i.e. UNGUARDED. Measured: the literal
    # version prints Sharpe 0.9663 and MaxDD -44.66% against the pinned 0.995 / -39.93%, a 4.7pt
    # worse tail — so skipna is not a quirk to be tidied away, it is the safer of the two and the
    # one the pins are built on. `dial_regression_test` now pins this behaviour so a pandas default
    # change cannot silently flip it. Live is unaffected either way: dial_state_live stitches full
    # history, so sv21 is always defined there — this only bites at a backtest window's first bars.
    sv = pd.concat([np.sqrt((neg ** 2).rolling(5).mean()),
                    np.sqrt((neg ** 2).rolling(21).mean())],
                   axis=1).max(axis=1, skipna=True) * np.sqrt(252)
    return (VM_TV / sv.clip(lower=VM_FLOOR)).clip(VM_LMIN, VM_LMAX)


def vol_brake_series(qqq_close):
    """Hysteretic extreme-vol flag (1.0 = brake ON). Leak-free: percentile thresholds use an
    EXPANDING window of history strictly up to each day."""
    rv = qqq_close.pct_change().rolling(VB_WIN).std() * np.sqrt(252)
    v = rv.to_numpy()
    on = rv.expanding(VB_BURN).quantile(VB_ON).to_numpy()
    off = rv.expanding(VB_BURN).quantile(VB_OFF).to_numpy()
    out = np.zeros(len(v))
    state = False
    for i in range(len(v)):
        if v[i] != v[i] or on[i] != on[i]:
            continue                       # burn-in / warm-up -> brake disabled
        if state:
            if v[i] < off[i]:
                state = False
        elif v[i] > on[i]:
            state = True
        out[i] = 1.0 if state else 0.0
    return pd.Series(out, index=qqq_close.index)


def _hyst_off(price, band=BAND):
    """Hysteretic risk-off flag for one signal (1.0 = risk-off). The dead-band kills the knife-edge
    whipsaws a raw close-vs-MA crossover produces. State is path-dependent (reconstructed from the
    full history each call), so it needs no external storage."""
    ma = price.rolling(MA).mean()
    lo = (ma * (1 - band)).to_numpy()
    hi = (ma * (1 + band)).to_numpy()
    p = price.to_numpy()
    out = np.empty(len(p), dtype=float)
    state = False
    for i in range(len(p)):
        if state:
            if p[i] == p[i] and hi[i] == hi[i] and p[i] > hi[i]:
                state = False
        elif p[i] == p[i] and lo[i] == lo[i] and p[i] < lo[i]:
            state = True
        out[i] = 1.0 if state else 0.0
    return pd.Series(out, index=price.index)


class DialInputError(RuntimeError):
    """A LIVE dial input is missing, or its history cannot determine its flag state.

    Raised only by dial_state_live. It is a RuntimeError on purpose: growth_slot._current_target
    catches `Exception` and turns it into HOLD-the-last-committed-target + _alert (or ABORT when
    nothing is persisted). _load's own SystemExit is NOT an Exception -- let that escape and the
    scheduled run dies with a traceback and no alert."""


def _hyst_pinned(price, band=BAND):
    """True when _hyst_off's CURRENT state is DATA-determined rather than SEED-determined.

    _hyst_off starts at the risk-ON seed and only leaves it on a close, with a valid MA, OUTSIDE the
    band: below MA*(1-band) forces risk-OFF and above MA*(1+band) forces risk-ON from EITHER prior
    state. Until one such close exists the flag is the seed, whatever the data would have implied;
    after it, the seed no longer matters. NaN compares False, so warm-up bars never pin. Pure, no I/O.
    Measured 2026-09-13 over 4,198 sessions x 11 dial inputs: every live-stub flag that disagreed with
    full history was unpinned by this test, and no pinned stub ever disagreed."""
    ma = price.rolling(MA).mean()
    return bool(((price < ma * (1 - band)) | (price > ma * (1 + band))).any())


def _load(sym):
    # _HERE-anchored (not cwd-relative): the dial must read the SAME vintage regardless of where
    # the caller was invoked from. sorted()[-1] still picks the newest dated vintage.
    pat = os.path.join(_HERE, "data", "macro", f"{sym}_daily_*.csv")
    fs = sorted(glob.glob(pat))
    if not fs:
        raise SystemExit(f"missing {pat}")
    return pd.read_csv(fs[-1], index_col=0, parse_dates=True)["close"]


# --- CSV<->LIVE STITCH ANCHOR (added 2026-09-02) -------------------------------------------------
# yfinance auto_adjust=True BACK-ADJUSTS the entire history every time it is pulled, so a CSV is
# anchored to its FETCH DATE, not to a fixed basis. The live Alpaca Adjustment.ALL bars are anchored
# to TODAY. Every distribution paid since the CSV was written is therefore a STEP at the seam where
# dial_state_live stitches the two together. Measured 2026-09-02 with CSVs of vintage 2026-06-17:
# HYG -1.483%, IEF -1.026%, QQQ -0.112% -- HYG's step is HALF the 3% hysteresis band, sitting inside
# the very series the 200d-MA flag is computed on. (Zero flag impact on that date, but that is luck,
# not design: a step of that size next to a band edge flips a trend flag.)
# The two segments are the SAME basis up to a CONSTANT factor, and 236 sessions of overlap are
# already fetched and thrown away -- so measure the factor there and put the segments on one anchor.
# WHY THIS CANNOT MASK THE 2026-08-27 RAW-BASIS DEFECT: an anchor lag is a CONSTANT ratio, whereas a
# raw-vs-adjusted basis error is a DRIFTING one (coupon drag accumulates). Measured on the same
# overlap: ratio dispersion under Adjustment.ALL is 0.0095-0.0335% for all 11 symbols, but under RAW
# it is 1.38% (HYG), 0.90% (IEF), 0.87% (EFA), 0.68% (EEM) and 44% (XLK/XLY split). So refusing to
# re-anchor whenever the ratio drifts leaves the defect fully exposed while fixing the benign case.
ANCHOR_MIN_OVERLAP = 60     # sessions of CSV/live overlap required before trusting a factor
ANCHOR_MAX_DISP = 0.005     # mean |ratio/median - 1| above this = a DRIFTING basis, not an anchor lag
ANCHOR_MAX_OFFSET = 0.25    # a factor this far from 1 is a split/scale mismatch, not a distribution


def _anchor_factor(hist, live):
    """Factor to put a CSV history segment on the live bars' adjustment anchor.

    -> (factor, note). factor is 1.0 (no change) whenever the offset cannot be trusted, and `note`
    is non-empty only when there is something an operator should see. Pure function: no I/O, so the
    test harness can drive it with synthetic constant-offset and drifting-offset series."""
    ov = hist.index.intersection(live.index)
    if len(ov) < ANCHOR_MIN_OVERLAP:
        return 1.0, (f"only {len(ov)} overlapping sessions (<{ANCHOR_MIN_OVERLAP}) — cannot measure "
                     f"the CSV/live anchor offset; stitching un-anchored")
    r = (live.reindex(ov) / hist.reindex(ov)).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if len(r) < ANCHOR_MIN_OVERLAP:
        return 1.0, f"only {len(r)} usable overlap ratios — stitching un-anchored"
    m = float(r.median())
    disp = float((r / m - 1.0).abs().mean())
    if disp > ANCHOR_MAX_DISP:
        return 1.0, (f"CSV/live ratio DRIFTS (dispersion {disp*100:.3f}% > {ANCHOR_MAX_DISP*100:.2f}%,"
                     f" median offset {(m-1)*100:+.3f}%) — that is a DIFFERENT PRICE BASIS, not an "
                     f"anchor lag; refusing to re-anchor so the guard can still see it")
    if abs(m - 1.0) > ANCHOR_MAX_OFFSET:
        return 1.0, (f"CSV/live offset {(m-1)*100:+.1f}% exceeds {ANCHOR_MAX_OFFSET*100:.0f}% — looks "
                     f"like a split/scale mismatch, not a distribution anchor; refusing to re-anchor")
    return m, ""


def breadth_series(band=BAND):
    """Fraction of BR_ETFS in hysteretic uptrend (each = NOT risk-off under the same band)."""
    bp = pd.DataFrame({s: _load(s) for s in BR_ETFS}).dropna()
    return pd.DataFrame({s: 1.0 - _hyst_off(bp[s], band) for s in BR_ETFS}).mean(axis=1)


def leverage_series(band=BAND):
    """Return (qqq_close, leverage_series, off_flags_df) over the full macro history.

    BACKTEST PATH -- deliberately NOT under dial_state_live's input-history guard (2026-09-13), and
    pinned that way in dial_regression_test section 6:
      * a missing CSV ABORTS the run (_load raises SystemExit). There is no live stub to degrade to,
        so the backtest can never silently compute a partial dial.
      * every flag runs on _hyst_off's risk-ON SEED until its first data-determined close: the 199-bar
        MA warm-up (2008-01-02..2008-10-14) for all inputs, and to 2008-11-17 for IEF (222 bars). That
        is the warm-up artifact the canonical pins are built on (a separate warm-start test found it
        pessimistic, i.e. it makes the 2008 drawdown deeper, not shallower). Applying the live guard
        here would drop that stretch and move every pin -- do it only with a deliberate re-pin."""
    px = pd.DataFrame({s: _load(s) for s in SIGNALS}).dropna()
    off = pd.DataFrame({s: _hyst_off(px[s], band) for s in SIGNALS}, index=px.index)
    eq_flag = off["QQQ"]
    if BR_ON:   # breadth flag replaces the single-QQQ trend flag (falls back to QQQ where no data)
        frac = breadth_series(band).reindex(off.index).ffill()
        off["BREADTH"] = (frac < BR_THR).astype(float).where(frac.notna(), off["QQQ"])
        eq_flag = off["BREADTH"]
    n_off = eq_flag + off["IEF"] + off["HYG"]
    lev = (TARGET_LEV * (1.0 - n_off / 3.0)).clip(lower=0.0)
    if VM_ON:   # hybrid: the more defensive of trend-dial and vol-managed (live==backtest)
        lev = pd.concat([lev, vol_managed_series(px["QQQ"])], axis=1).min(axis=1)
    # clip to [CORE_FLOOR (0), TARGET_LEV], then make it inelastic
    lev = _deadband(lev.clip(lower=CORE_FLOOR, upper=TARGET_LEV), DEADBAND_LEV)
    return px["QQQ"], lev, off


def _flags_and_lev(px, band=BAND):
    """px: DataFrame of recent daily closes (cols = SIGNALS) -> (leverage, {sym: is_risk_off}).
    Uses the hysteretic flag, so the current state is reconstructed from the fetched history."""
    flags = {}
    for s in SIGNALS:
        off = _hyst_off(px[s].dropna(), band)
        if not len(off):   # was `else False`: an EMPTY series read as risk-ON (2026-09-13)
            raise DialInputError(f"{s}: no prices at all -- there is no flag state to read")
        flags[s] = bool(off.iloc[-1])
    n_off = sum(flags.values())
    return max(0.0, TARGET_LEV * (1.0 - n_off / len(SIGNALS))), flags


def dial_state_live(data_client):
    """Current dial from LIVE Alpaca daily bars for QQQ/IEF/HYG and the breadth ETFs, composed with
    the downside-semivol leg. -> (target_leverage, flags dict; its 'VOL_BRAKE' entry is always False
    because the discrete brake is retired). Autonomous: depends only
    on these ETF prices, no other model. Raises DialInputError (a RuntimeError) when any input's
    history is missing or cannot determine its flag state -- it never returns a seed-determined dial."""
    import datetime as _dt
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    syms = sorted(set(SIGNALS) | (set(BR_ETFS) if BR_ON else set()))
    kw = dict(symbol_or_symbols=syms, timeframe=TimeFrame.Day,
              start=_dt.datetime.now() - _dt.timedelta(days=420),
              # EXCLUDE THE CURRENT, STILL-FORMING SESSION. With no `end`, Alpaca returns a daily
              # bar for TODAY that is only partially built -- the scheduled rebalance runs at 11:00 ET,
              # i.e. ~1.5 of the session's 6.5 hours. The backtest deploys shift(1) (yesterday's
              # completed close drives today's exposure), so reading a forming bar makes the live
              # dial a DIFFERENT rule, not a fresher one. Measured over 143 sessions: the raw dial
              # differed on 69 (48%), mean|d| 0.029x, max 0.622x; after the executor deadband the
              # HELD target differed on 88/143 (62%), max ~0.74x -- larger than the 0.583x raw-basis
              # defect. Bars are stamped 00:00 ET (04:00 UTC), so end=<today 00:00> keeps the last
              # COMPLETED session and drops the forming one. Also fixes the semivol leg, which was
              # computing its 5d window off a truncated newest day.
              end=_dt.datetime.combine(_dt.date.today(), _dt.time(0, 0)))
    # PRICE BASIS — MUST match the backtest, which builds its CSVs with yfinance
    # auto_adjust=True (fetch_macro_bars_swing.py, fetch_etf), i.e. split+dividend adjusted.
    # alpaca-py defaults `adjustment` to None => the server returns RAW closes. A 200d-MA
    # trend flag on a distribution-paying bond ETF is a DIFFERENT SIGNAL on the two bases:
    # raw HYG/IEF are dragged down by every coupon, so they cross their own MA at times the
    # adjusted series never does. Measured 2026-08-27 over the live 420d window:
    #   raw vs yfinance-adjusted   mean |err| HYG 3.24% · IEF 2.13% · QQQ 0.24% · XLK 37.46%
    #   Adjustment.ALL vs same     mean |err| HYG 0.014% · IEF 0.009% · QQQ 0.012% · XLK 0.018%
    # so ALL reproduces the backtest basis to ~1-2bp, while raw does not. XLK's 37% is the
    # unadjusted 2025-12-05 2:1 split, which SPLIT alone would fix but is only one of the
    # two failure modes -- the coupon drag needs the dividend leg too. Hence ALL, not SPLIT.
    # Deployed-vs-validated impact when this was found: raw dial 1.1667x vs validated 1.7500x.
    # NO SILENT FALLBACK: falling back to raw is precisely the defect, so an import failure
    # here must be loud rather than quietly re-deploying an unvalidated strategy.
    from alpaca.data.enums import Adjustment
    kw["adjustment"] = getattr(Adjustment, LIVE_ADJUSTMENT)
    # Feed fails OPEN, deliberately, unlike `adjustment` above. The asymmetry is proportionate:
    # a wrong ADJUSTMENT changes the signal (raw vs adjusted flips 200d-MA flags — that was the
    # 08-27 defect), whereas a wrong FEED moves closes by ~0.02%, which cannot cross a 3% hysteresis
    # band. Trading an availability risk for a cosmetic one would be the wrong bargain here.
    try:
        from alpaca.data.enums import DataFeed
        kw["feed"] = getattr(DataFeed, LIVE_FEED)
    except Exception:
        pass
    df = data_client.get_stock_bars(StockBarsRequest(**kw)).df
    px = df.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    # HYSTERESIS STATE SEEDING — stitch CSV history UNDER the live bars before computing flags.
    # _hyst_off is PATH-DEPENDENT (its docstring assumes "reconstructed from the full history
    # each call") but the live path was handing it only 420 calendar days => ~89 days with a
    # valid 200d MA. If the price never crosses a band inside that stub, the flag is still the
    # SEED value (state=False, i.e. risk-ON) rather than anything the data implied, and live can
    # silently disagree with the backtest's state machine. Measured 2026-08-27: QQQ and all 9
    # breadth ETFs pin inside the window (2026-04-21/05-27), but **HYG and IEF NEVER pin** --
    # 2 of the 3 trend signals were running on the seed. They happened to agree with full
    # history that day, which is luck, not correctness: the bias is one-directional (the seed is
    # risk-ON), so the failure mode is silently carrying MORE risk than the validated strategy,
    # exactly when an established risk-off state should have persisted.
    # FAIL-SAFE, NOT FAIL-OPEN (2026-09-13). This block used to degrade a symbol with no CSV history
    # to that 420d stub, print a warning, and carry on -- i.e. it re-created the very seed defect
    # described above whenever a CSV went missing, with only stdout to say so. Measured on real data
    # 2010-01-04..2026-09-11 (repro on a temp copy through a mock data client): a missing IEF CSV would
    # have RAISED the dial on 158 sessions and a missing HYG CSV on 70 -- max +0.583x, clearing the
    # 0.25x deadband on 105 / 57 -- and lowered it on NONE (e.g. IEF 2021-10-19..11-24, weeks before the
    # 2022 bear: 1.17x -> 1.75x; 2018-12-24: cash -> 0.50x). On 2026-09-11 it prints the right number
    # by luck. Every dial input must now have (1) live bars, (2) CSV history stitched under them, and
    # (3) a DATA-determined flag state (_hyst_pinned) -- otherwise DialInputError, which
    # growth_slot._current_target turns into HOLD-the-committed-target + _alert (ABORT if nothing is
    # persisted): "the book keeps yesterday's target" (drift toward it is still corrected), never
    # "the book reads its seed". A missing CSV is
    # refused even when the stub happens to pin: without the CSV the anchor check cannot run, and a
    # pinned stub matches full history only if the live bars are gap-free -- agreement would be luck.
    # The CSV segment and the live segment are the same basis on DIFFERENT ANCHORS (see
    # _anchor_factor): re-anchor the CSV segment onto the live bars using the overlap, so the
    # 200d MA is not computed across a distribution-sized step. Self-nullifying — the factor goes
    # to 1.0 the moment the CSVs are re-fetched.
    _stitched, _bad, _anchored, _anchor_notes = {}, [], [], []
    for _s in syms:                       # EVERY required input -- not just what the API returned
        _lv = px[_s].dropna() if _s in px.columns else pd.Series(dtype=float)
        if _lv.empty:
            _bad.append(f"{_s}: no live bars returned")
            continue
        try:
            _h = _load(_s)
            _h.index = pd.to_datetime(_h.index).tz_localize(None).normalize()
            _fac, _note = _anchor_factor(_h, _lv)
            if _note:
                _anchor_notes.append(f"{_s}: {_note}")
            if abs(_fac - 1.0) > 1e-4:
                _anchored.append(f"{_s} {(_fac-1)*100:+.3f}%")
            _stitched[_s] = pd.concat([_h[~_h.index.isin(_lv.index)] * _fac, _lv]).sort_index()
        except (Exception, SystemExit) as _e:   # _load raises SystemExit, which is NOT an Exception;
            #                                     it must become a DialInputError below, not escape.
            _bad.append(f"{_s}: no CSV history to stitch ({_e})")
    if _anchored:
        print(f"  [dial] CSV history re-anchored onto the live basis (stale-CSV distribution drift): "
              f"{', '.join(_anchored)}")
    for _n in _anchor_notes:
        print(f"  [dial] WARNING: stitch anchor — {_n}")
    px = pd.DataFrame(_stitched)
    for _s in _stitched:
        _ser = px[_s].dropna()
        if not _hyst_pinned(_ser):
            _bad.append(f"{_s}: {len(_ser)} bars never closed outside the {BAND:.0%} band of a valid "
                        f"{MA}d MA -- its flag would be the risk-ON seed, not data")
    if _bad:
        raise DialInputError("dial input history missing or insufficient -- refusing to compute "
                             "flags on the risk-ON seed: " + "; ".join(_bad))
    lev, flags = _flags_and_lev(px)
    # BREADTH flag replaces the single-QQQ trend flag (same stitched-history hysteresis treatment as
    # the other live flags).
    # NO FALLBACK TO THE QQQ FLAG (2026-09-13). This used to be "any problem -> keep the QQQ-flag
    # leverage", which runs a DIFFERENT rule (breadth replaced QQQ in the 2026-07-05 A/B) and is not
    # risk-neutral: over the backtest's 4,481 data-determined sessions the QQQ-flag dial would have
    # been HIGHER on 11 (max +0.366x) and lower on 61. The input guard above already refuses the
    # missing-data cases that used to land here, so anything that still raises is a real fault.
    if BR_ON:
        try:
            ups = [1.0 - _hyst_off(px[s].dropna()).iloc[-1] for s in BR_ETFS]
        except Exception as e:
            raise DialInputError(f"breadth flag could not be computed ({e}) -- refusing the QQQ-flag "
                                 f"fallback, which is a different, unvalidated rule") from e
        frac = float(np.mean(ups))
        b_off = bool(frac < BR_THR)
        flags["BREADTH"] = b_off               # the BINDING equity-trend flag (QQQ flag shown for info)
        print(f"  [dial] breadth {frac:.2f} of {len(BR_ETFS)} growth indexes in uptrend "
              f"(risk-off < {BR_THR})")
        n_off = float(b_off) + float(flags["IEF"]) + float(flags["HYG"])
        lev = max(0.0, TARGET_LEV * (1.0 - n_off / 3.0))
    # Stitch historical CSV (2008->) with live bars ONCE — the vol brake's expanding
    # percentiles and the vol-managed leg both need full history so live thresholds
    # match the backtested ones exactly. (The brake is retired; the semivol leg raises, never fails open.)
    # NB (2026-09-02): since the per-symbol stitch above was added, px["QQQ"] is ALREADY
    # the full stitched+anchored series, so `hist[~hist.index.isin(live_q.index)]` is empty and this
    # block is a pass-through of that same series. Kept as-is (it is the fail-open path if the
    # per-symbol stitch degraded to the live stub) — but do NOT read it as a second, different basis.
    full = None
    try:
        hist = _load("QQQ")
        live_q = px["QQQ"].dropna()
        live_q.index = pd.to_datetime(live_q.index).tz_localize(None).normalize()
        hist.index = pd.to_datetime(hist.index).normalize()
        full = pd.concat([hist[~hist.index.isin(live_q.index)], live_q]).sort_index()
    except Exception as e:
        print(f"  [dial] history stitch unavailable ({e}); trend dial alone")
    # VOL BRAKE (extreme-only, hysteretic) — RETIRED (VB_ENABLED=False): superseded by the
    # semivol VM leg (dominated in 8/8 A/B cells). Guard kept so it can be re-armed if needed.
    flags["VOL_BRAKE"] = False
    if VB_ENABLED and full is not None:
        try:
            if bool(vol_brake_series(full).iloc[-1]):
                flags["VOL_BRAKE"] = True
                lev = min(lev, VB_CAP)     # brake only CUTS leverage, never adds
        except Exception as e:
            print(f"  [dial] vol-brake unavailable ({e})")
    # VOL-MANAGED leg (continuous; hybrid = more defensive of the two dials)
    # NO SILENT FAIL-OPEN. This is the ONLY always-on crash brake left (VB_ENABLED=False retired
    # the discrete one), so dropping it does not "fall back to the trend dial" -- it DELETES the
    # model's fast-crash defence and raises leverage. Measured on a readable-but-corrupt QQQ CSV:
    # the book goes from a mean 1.32x to a flat 1.75x, +0.43x average (+0.22x over full history),
    # +1.25x worst day, with no alert; the semivol leg was the binding (more defensive) leg on
    # 201 of 289 recent sessions, so this is the common case, not an edge case.
    # The stitch exists for the vol BRAKE's expanding percentiles; the semivol leg needs only the
    # 5d/21d windows, so the ~289-bar live series is always sufficient for it. Fall back to that
    # rather than dropping the leg, and only if BOTH sources fail do we refuse to raise leverage.
    if VM_ON:
        _src = full if full is not None else px["QQQ"].dropna()
        try:
            vm = float(vol_managed_series(_src).iloc[-1])
            if not np.isfinite(vm):
                # Too little QQQ history for even the 5d window. `min(lev, nan)` returns `lev` in
                # Python, so this used to DROP the crash brake silently -- the silent fail-open
                # defect above again, through a NaN instead of an exception (2026-09-13).
                raise ValueError(f"semivol is {vm} (insufficient QQQ history)")
            if full is None:
                print("  [dial] WARNING: semivol leg computed on the LIVE window (history stitch "
                      "unavailable) — value is sound, but the vol brake would not be")
            flags["VOL_MGD_LEV"] = round(vm, 2)
            lev = min(lev, vm)
        except Exception as e:
            # RAISE, do not continue. Continuing here would return a trend-only leverage that
            # LOOKS like a valid dial while the crash brake is silently gone. Raising composes
            # correctly with growth_slot._current_target, which now HOLDS the last committed
            # target and alerts instead of setting a new one -- so the failure mode is "the book
            # keeps yesterday's target", not "the book quietly levers up".
            raise RuntimeError(
                f"semivol leg could not be computed ({e}) -- refusing to return a trend-only "
                f"dial, because that silently removes the only always-on crash brake") from e
    lev = min(max(lev, CORE_FLOOR), TARGET_LEV)   # clip to [CORE_FLOOR (0), TARGET_LEV];
    #   the INELASTIC deadband is applied by the executor against its persisted held-target, so the
    #   raw (floored) target is returned here.
    return lev, flags


def _metrics(eq):
    r = eq.pct_change().dropna()
    n = max(len(r), 1)
    tot = eq.iloc[-1] / eq.iloc[0] - 1.0
    ann = (1.0 + tot) ** (252.0 / n) - 1.0
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    # DRAWDOWN measured on the EQUITY CURVE ITSELF, not on a curve rebuilt from returns.
    # BUG FIXED 2026-09-02: this was `((1+r).cumprod()/cummax - 1).min()` with r = pct_change().
    # `(1+r).cumprod()` begins at bar 2, so eq.iloc[0] could never BE the high-water mark — any
    # curve whose peak sits on the FIRST bar had its drawdown measured from a lower reference and
    # silently under-reported. Synthetic proof: 1.00 -> 0.90 -> 0.95 -> 0.92 reported -3.16% where
    # the truth is -10.00%, understated by 6.8pt.
    # It bit for real: the OHLC range-estimator arm's headline "-36.8% vs -39.9%, 3.1pt better
    # drawdown" was this artifact (that curve peaks on bar 1); on the standard convention it is
    # -39.84% vs -39.93% = +0.09pt, and at the GFC trough that arm's investor was 0.37% POORER.
    # The bias is ASYMMETRIC and flatters whichever arm happens to peak first, which is exactly
    # what you do not want in an A/B harness.
    # The canonical pin is UNAFFECTED: the deployed baseline's deepest drawdown runs from an
    # interior peak, so it reads -39.93% on both conventions and the pinned -40% stands.
    dd = (eq / eq.cummax() - 1.0).min()
    return tot, ann, sh, float(dd)


_RATE_CACHE = {}


def rates_daily(index):
    """(cash_daily, borrow_daily) as DAILY-decimal series aligned to `index`.

    Cash = the real 3-month bill (FRED DGS3MO, cached at data/rates/fred_DGS3MO.csv).
    Borrow = the same bill + BORROW_SPREAD_BP. Splitting them is the whole point: a single
    constant cannot be both what you PAY to borrow and what you EARN on cash.
    Forward-fill only (bills are not published on holidays); dates outside the series' coverage
    fall back to the legacy flat RF rather than silently becoming zero -- a zero rate would be a
    free-money bug, so the fallback is deliberately the conservative direction.
    RATES_REAL=False restores the exact pre-2026-09-06 behaviour for reproducing old numbers."""
    if not RATES_REAL:
        f = pd.Series(RF / 252.0, index=index)
        return f, f
    key = (id(index), len(index), index[0], index[-1])
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    # _HERE is now defined unconditionally at module level (2026-09-06), so the old
    # `if "_HERE" in globals() else <cwd-relative>` fallback is dead and has been dropped --
    # a cwd-relative fallback here would silently read a DIFFERENT rate file (or none) and is
    # exactly the class of defect this change removes.
    p = os.path.join(_HERE, "data", "rates", "fred_DGS3MO.csv")
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0] / 100.0
    s = s.reindex(index.union(s.index)).ffill().reindex(index).fillna(RF)
    cash = s / 252.0
    borrow = (s + BORROW_SPREAD_BP / 1e4) / 252.0
    _RATE_CACHE[key] = (cash, borrow)
    return cash, borrow


def _pnl(lev, r, cash, borrow):
    """Levered P&L with the two rate legs SEPARATED: you pay `borrow` on the borrowed fraction and
    earn `cash` on the un-invested fraction. The old single term -(lev-1)*RF conflated them."""
    return lev * r - (lev - 1.0).clip(lower=0.0) * borrow + (1.0 - lev).clip(lower=0.0) * cash


def _curve(lev, rq):
    cash, borrow = rates_daily(rq.index)
    return (1 + _pnl(lev, rq, cash, borrow)).cumprod()


def _bt_run(levs, rq):
    lev1 = levs.shift(1)                           # LEAK-FREE: yesterday's signal -> today's exposure
    idx = rq.index[lev1.notna() & rq.notna()]
    lev1, r = lev1.reindex(idx), rq.reindex(idx)
    cash, borrow = rates_daily(idx)
    eq = (1 + _pnl(lev1, r, cash, borrow)).cumprod()
    flips = int((lev1.diff().abs() > 1e-9).sum())  # # of leverage changes = whipsaw/round-trip proxy
    return eq, lev1, flips


def _stress_dd(eq, s, e):
    w = eq[(eq.index >= s) & (eq.index <= e)]
    return ((w / w.cummax()) - 1).min() * 100 if len(w) else float("nan")


def backtest():
    qqq, _, _ = leverage_series(0.0)
    rq = qqq.pct_change()

    def y2022(eq):
        y = eq.resample("YE").last().pct_change()
        v = y[y.index.year == 2022]
        return float(v.iloc[0]) * 100 if len(v) else float("nan")

    print("\n=== GROWTH RISK-DIAL: HYSTERESIS-BAND SWEEP 2008-2026 (rule-based, no ML) ===")
    print(f"  {'strategy':<14} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>7} {'2022yr':>7} {'meanLev':>8} {'flips':>6}")
    for name, levs in [("STATIC 1.75x", pd.Series(TARGET_LEV, index=qqq.index)),
                       ("QQQ 1.0x", pd.Series(1.0, index=qqq.index))]:
        eq, lev1, flips = _bt_run(levs, rq)
        _, a, s, d = _metrics(eq)
        print(f"  {name:<14} {a*100:>6.1f}% {s:>7.2f} {d*100:>6.0f}% {y2022(eq):>6.0f}% {lev1.mean():>7.2f}x {flips:>6}")
    print("  " + "-" * 64)
    for b in (0.00, 0.02, 0.03, 0.05):
        _, levs, _ = leverage_series(b)
        eq, lev1, flips = _bt_run(levs, rq)
        _, a, s, d = _metrics(eq)
        star = "  <- default" if abs(b - BAND) < 1e-9 else ""
        print(f"  DIAL band={b*100:>2.0f}%   {a*100:>6.1f}% {s:>7.2f} {d*100:>6.0f}% {y2022(eq):>6.0f}% "
              f"{lev1.mean():>7.2f}x {flips:>6}{star}")
    # 2022 drawdown-window protection at band 0 vs default
    _, l0, _ = leverage_series(0.0); e0, _, _ = _bt_run(l0, rq)
    _, lb, _ = leverage_series(BAND); eb, _, _ = _bt_run(lb, rq)
    print(f"\n  2022 rate-hike drawdown window: band 0% {_stress_dd(e0,'2022-01-01','2022-12-31'):.0f}% "
          f"-> band {BAND*100:.0f}% {_stress_dd(eb,'2022-01-01','2022-12-31'):.0f}%")
    print("  Want: keep the 2022 protection while CUTTING flips vs band=0% (fewer whipsaws).\n")


def _slot_sim(r, dial, trigger, drift=0.03):
    """Realistic QQQ/QLD/cash slot: QLD = 2x QQQ daily reset (with expense+financing), cash earns
    the bill rate. trigger: daily|weekly|monthly|quarterly = calendar rebalance; 'gated' = the
    DEPLOYED logic (rebalance only on a dial change OR a leg drifting > `drift` of equity). Returns
    (equity, n_rebalances, total_turnover_fraction).

    SIGNAL LAG IS EXACTLY ONE DAY, matching `_bt_run`'s `lev1 = levs.shift(1)` and the live path
    (`dial_state_live` excludes the forming session, so yesterday's completed close drives today's
    exposure). See the ordering note in the loop -- this was TWO days until 2026-09-06."""
    Ld = dial.shift(1).to_numpy()                       # leak-free: yesterday's dial -> today's target
    rr = r.to_numpy()
    # Rate legs separated here too (2026-09-06): QLD's embedded financing is the BORROW rate, and
    # the idle sleeve earns the CASH rate. QLD_FEE (the 0.95% expense ratio) is charged on top of
    # financing and is NOT part of either rate.
    _c, _b = rates_daily(r.index)
    qld_r = 2.0 * rr - (_b.to_numpy() + QLD_FEE / 252.0)
    cash_dr = _c.to_numpy()
    per = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}.get(trigger)

    def wts(L):
        L = max(0.0, min(2.0, L))
        return (2 - L, L - 1, 0.0) if L >= 1 else (L, 0.0, 1 - L)   # (qqq, qld, cash)

    start = 0
    while start < len(Ld) and Ld[start] != Ld[start]:
        start += 1
    if start >= len(Ld):
        return pd.Series(dtype=float), 0, 0.0
    vq, vl, vc = wts(Ld[start])                          # book opens at equity 1.0, weights Ld[start]
    eq, n_reb, turn, last_L, since = [], 0, 0.0, Ld[start], 0
    for i in range(start, len(rr)):
        # ORDER MATTERS, AND IT WAS WRONG UNTIL 2026-09-06. This loop used to grow the positions by
        # r[i] FIRST and rebalance to Ld[i] afterwards, so day i's return was earned on the weights
        # set from Ld[i-1] = dial[i-2] -- TWO days of lag, where the `shift(1)` above and
        # `_bt_run`'s `lev1 = levs.shift(1)` both claim ONE. Proof (with RF=0 and QLD_FEE=0 the
        # sleeve blend returns exactly L*r, so eq_ret/r reads the held leverage straight off the
        # curve): on a dial stepping 1.00 -> 1.75, the old loop's implied held leverage equalled
        # dial.shift(2) on every single day and dial.shift(1) on none. Cost of the extra day over
        # 2008-2026 on the deployed 'gated' book: 1.34pp/yr of CAGR at the current config
        # (1.16pp on the earlier CORE_FLOOR=0.25 + flat-RF configuration).
        # That was 77% of the apparent "QLD is worse than clean leverage" penalty -- a measurement
        # artifact, not an instrument effect. Constant-dial books are lag-invariant, so every
        # constant-weight CONTROL is unmoved to 0.0000pp and only the dial rows restate.
        # Rebalancing first also puts the DRIFT test on the book as it stood at the previous close,
        # which is what the 11:00 ET executor actually sees when it decides.
        E = vq + vl + vc
        since += 1
        L = Ld[i]
        if per is not None:
            do = since >= per
        else:
            tq, tl, _ = wts(L)
            do = (abs(L - last_L) > 1e-9) or (max(abs(vq / E - tq), abs(vl / E - tl)) > drift)
        if do and E > 0:
            tq, tl, tc = wts(L)
            nvq, nvl, nvc = tq * E, tl * E, tc * E
            turn += (abs(nvq - vq) + abs(nvl - vl)) / E
            vq, vl, vc = nvq, nvl, nvc
            n_reb += 1; since = 0; last_L = L
        vq *= (1 + rr[i]); vl *= (1 + qld_r[i]); vc *= (1 + cash_dr[i])
        eq.append(vq + vl + vc)
    return pd.Series(eq, index=r.index[start:]), n_reb, turn


def cadence_backtest():
    qqq, dial, _ = leverage_series()
    r = qqq.pct_change().dropna()
    dial = dial.reindex(r.index).ffill()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    print("\n=== REBALANCE-CADENCE SWEEP 2008-2026 (dial-driven QQQ/QLD/cash slot, 3% band) ===")
    print(f"  {'cadence':<12} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>7} {'2022':>7} {'#rebal':>7} {'turnover/yr':>12}")
    for trig in ("daily", "weekly", "monthly", "quarterly", "gated"):
        eq, nreb, turn = _slot_sim(r, dial, trig)
        _, a, s, d = _metrics(eq)
        y = eq.resample("YE").last().pct_change()
        v = y[y.index.year == 2022]
        y22 = float(v.iloc[0]) * 100 if len(v) else float("nan")
        label = "gated(LIVE)" if trig == "gated" else trig
        print(f"  {label:<12} {a*100:>6.1f}% {s:>7.2f} {d*100:>6.0f}% {y22:>6.0f}% {nreb:>7} {turn/yrs*100:>11.0f}%")
    print("  'gated(LIVE)' = the DEPLOYED logic (rebalance only on a dial change or >3% leg drift).\n")


def brake_backtest():
    """Full-window test of the conditional vol EMERGENCY BRAKE layered on the trend dial."""
    qqq, lev, _ = leverage_series()
    rq = qqq.pct_change()
    brake = vol_brake_series(qqq)
    lev_b = lev.where(brake < 0.5, lev.clip(upper=VB_CAP))     # brake ON -> cap leverage at VB_CAP
    print(f"\n=== VOL EMERGENCY BRAKE BACKTEST 2008-2026 (20d vol > exp {VB_ON:.0%} pct -> cap {VB_CAP}x, "
          f"release < {VB_OFF:.0%} pct) ===")
    print(f"  brake engaged {brake.mean()*100:.1f}% of days | episodes: "
          f"{int(((brake.diff() > 0).sum()))}")
    print(f"  {'strategy':<16} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>7} {'COVID20':>8} {'2022':>7} {'flips':>6}")
    rows = [("STATIC 1.75x", pd.Series(TARGET_LEV, index=qqq.index)),
            ("DIAL only", lev), ("DIAL + BRAKE", lev_b)]
    for name, ls in rows:
        eq, lev1, flips = _bt_run(ls, rq)
        _, a, s, d = _metrics(eq)
        cov = _stress_dd(eq, "2020-02-01", "2020-04-30")
        y = eq.resample("YE").last().pct_change()
        v = y[y.index.year == 2022]
        y22 = float(v.iloc[0]) * 100 if len(v) else float("nan")
        print(f"  {name:<16} {a*100:>6.1f}% {s:>7.2f} {d*100:>6.0f}% {cov:>7.0f}% {y22:>6.0f}% {flips:>6}")
    # brake-episode dates (sanity: should be 2008/2011/2015/2018/2020/2022-style spikes, not calm years)
    eps = brake[brake.diff() > 0].index
    print("  brake engagements:", ", ".join(d.strftime("%Y-%m") for d in eps))


def dial_leverage(as_of=None):
    """Last (or as-of) leverage on the BACKTEST path, from the macro CSVs -- `--today` and research.
    NOT the live path: growth_slot uses dial_state_live (live bars, forming session excluded)."""
    _, lev, _ = leverage_series()
    if as_of is not None:
        lev = lev[lev.index <= pd.Timestamp(as_of)]
    if not len(lev):
        # was `else TARGET_LEV`: no history resolved to the MAXIMUM leverage (fixed 2026-09-13)
        raise ValueError(f"no dial history on or before {as_of} -- refusing to report a leverage")
    return float(lev.iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--cadence", action="store_true", help="rebalance-cadence sweep (daily/weekly/monthly/...)")
    ap.add_argument("--brake", action="store_true", help="vol emergency-brake backtest (fast-crash fix)")
    ap.add_argument("--today", action="store_true", help="print the current dial leverage")
    a = ap.parse_args()
    if a.today:
        print(f"current dial leverage: {dial_leverage():.2f}x")
    elif a.cadence:
        cadence_backtest()
    elif a.brake:
        brake_backtest()
    else:
        backtest()


if __name__ == "__main__":
    main()
