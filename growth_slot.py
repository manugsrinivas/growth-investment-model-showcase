"""GROWTH SLOT executor -- LEVERED PASSIVE QQQ, with leverage set by a rule-based risk dial.

Rationale: in earlier testing (not published here), active growth stock-picking variants did not
beat passive QQQ, so this slot holds QQQ itself, levered to the dial's target. With a CASH account
(no margin), leverage L between 1x and 2x is built from QQQ and QLD (QLD = 2x QQQ daily):
weights QQQ = 2 - L, QLD = L - 1 (at the 1.75x ceiling: 25% QQQ / 75% QLD). Below 1x the
remainder sits in SGOV (T-bills). Periodic rebalancing controls the QLD daily-reset drift.

  python growth_slot.py --status                 # no orders: equity, positions, weights, leverage
                                                 # (read-only at the broker; appends a local log row)
  python growth_slot.py --rebalance              # DRY-RUN: show the orders it would place
  python growth_slot.py --rebalance --live       # actually submit the rebalance orders

Target leverage is set by the rule-based RISK DIAL (growth_dial.py): up to 1.75x when growth
breadth, IEF and HYG are all in uptrends, stepping down (1.17x / 0.58x / 0x) as those trends break,
and capped further by a downside-volatility leg. At 0x the slot is entirely in SGOV.

Keys: ALPACA_API_KEY_GROWTH / ALPACA_SECRET_KEY_GROWTH. Trades the Alpaca PAPER endpoint unless
ALPACA_BASE_URL_GROWTH is set to a URL that does not contain "paper".
Run --rebalance during market hours (notional/fractional orders need an open market). Runs DAILY now
(so the dial can act within a day of a signal flip); the drift guard makes calm-day runs no-ops.
"""
import argparse
import datetime as _dt
import os
import sys
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockHistoricalDataClient

import growth_dial   # rule-based, autonomous macro RISK DIAL -> sets the slot's target leverage
import growth_exus   # rule-based ex-US VIEW signal -> carves k of the budget into EFA/EEM when on
import growth_xle    # rule-based ENERGY VIEW signal -> carves k of the budget into XLE when on

# ---- slot configuration -------------------------------------------------------
# Target leverage comes from the rule-based risk dial (growth_dial: breadth/IEF/HYG 200d-trend
# flags plus a downside-volatility leg), so the slot self-de-levers in rate-hike / risk-off
# regimes. The QQQ/QLD/SGOV blend is derived to hit the dial's leverage each run.
# REGIME CARVE-OUTS (2026-07-05): two bounded VIEW sleeves ride INSIDE this same Growth account (no
# separate accounts/keys). growth_exus sets k_ex in [0,15%] (EFA/EEM, weak-dollar regime); growth_xle
# sets k_xl in [0,12%] (XLE, energy/inflation regime). When a view is on we reserve its k of the
# budget for its basket and scale the QQQ dial blend to (1 - total_k). When BOTH DORMANT (default)
# behavior is IDENTICAL to before. EFA/EEM/XLE are unlevered (1x); SGOV ~0 beta. NOTE (honest): the
# energy carve is a USER CONVICTION bet -- neutral on live data, -2.8% tail at 12% (see growth_xle).
LEVERAGE = {"QQQ": 1.0, "QLD": 2.0, "SGOV": 0.0, "EFA": 1.0, "EEM": 1.0, "XLE": 1.0}
SLEEVES = ("QQQ", "QLD", "SGOV", "EFA", "EEM", "XLE")   # +EFA/EEM ex-US, +XLE energy carve-outs
MAX_LEV = growth_dial.TARGET_LEV       # 1.75x when the dial is fully risk-on
DEPLOY = 0.99                          # deploy 99% of equity; tiny buffer for fractional rounding
                                       # A fixed dollar deploy cap was REMOVED 2026-09-21: it was
                                       # live-only (the backtest always sized on equity*DEPLOY), so it
                                       # froze invested capital while equity compounded.
DRIFT_PCT = 0.03                       # only trade a leg if it has drifted > 3% of equity from target
MIN_ORDER_USD = 25.0                   # don't bother with sub-$25 adjustments

# --- PATH ANCHOR (2026-09-06) -----------------------------------------------------------------
# Until today this module had NO chdir of its own and resolved logs/, state/ and logs/PAUSE
# RELATIVE TO THE CALLER'S CWD. It happened to work only because `import growth_dial`
# ran `os.chdir(dirname(__file__))` as an import side effect. That made the LIVE BOOK'S KILL SWITCH
# depend on the import order of an unrelated module: anything that resolved a path before the dial
# was imported, or any future entry point that did not `cd` first, would have looked for
# `logs/PAUSE` in the wrong directory -- NOT FINDING THE PAUSE FILE MEANS TRADING WHEN A HUMAN HAS
# SAID STOP. The dial's chdir is gone; every path below is now anchored to THIS file's directory,
# so the executor is correct no matter where it is invoked from and no matter what it imports.
# Do not reintroduce a bare relative path here. (Verified by running from an unrelated cwd.)
_HERE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(_HERE, "logs")
STATE_DIR = os.path.join(_HERE, "state")
LOG = os.path.join(LOGS_DIR, "growth_slot.csv")

KEY = os.environ.get("ALPACA_API_KEY_GROWTH")
SECRET = os.environ.get("ALPACA_SECRET_KEY_GROWTH")
PAPER = "paper" in (os.environ.get("ALPACA_BASE_URL_GROWTH", "paper") or "paper").lower()


def _alert(msg):
    """Append a line to today's alert file, logs/ALERTS/alert_YYYYMMDD.txt, for an operator to
    read. Best-effort: alerting must never be able to take
    down the executor, so every failure here is swallowed. Used for conditions that are NOT normal
    operation and must not scroll past in a scheduled run's stdout."""
    try:
        import datetime as _d
        _adir = os.path.join(LOGS_DIR, "ALERTS")
        os.makedirs(_adir, exist_ok=True)
        p = os.path.join(_adir, f"alert_{_d.date.today():%Y%m%d}.txt")
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"[{_d.datetime.now():%Y-%m-%d %H:%M}] growth_slot: {msg}\n")
    except Exception:
        pass


def _client():
    if not (KEY and SECRET):
        sys.exit("ALPACA_API_KEY_GROWTH / ALPACA_SECRET_KEY_GROWTH not set in the environment.")
    return TradingClient(KEY, SECRET, paper=PAPER)


def _data_client():
    return StockHistoricalDataClient(KEY, SECRET)


def _weights_for_leverage(lev):
    """Map target leverage to QQQ(1x)/QLD(2x)/SGOV weights (cash acct, no margin). Leverage is
    clipped to [CORE_FLOOR, 2 - CORE_FLOOR]; with growth_dial.CORE_FLOOR = 0 that is [0, 2], so the
    dial can de-risk fully to SGOV (lev 0 -> 100% SGOV). De-levered capital parks in SGOV
    (T-bill yield)."""
    c = growth_dial.CORE_FLOOR
    lev = max(c, min(2.0 - c, lev))                      # [core floor, 2-core ceiling]
    if lev >= 1.0:
        return {"QQQ": 2.0 - lev, "QLD": lev - 1.0}       # QQQ + QLD overlay
    return {"QQQ": lev, "SGOV": 1.0 - lev}                # QQQ, rest in SGOV


HELD_STATE = os.path.join(STATE_DIR, "growth_slot_lev.json")


def _sticky_lev(raw, update):
    """INELASTIC dial: hold the last-committed target leverage until the raw dial target moves more
    than growth_dial.DEADBAND_LEV away, then snap to it. Persisted (survives restarts). The executor
    still rebalances the blend to this sticky target every run, so the drift band keeps correcting
    QLD daily-reset drift WITHIN the deadband -- only the TARGET is sticky, not the maintenance."""
    import json
    db = growth_dial.DEADBAND_LEV
    held = None
    if os.path.exists(HELD_STATE):
        try:
            held = float(json.load(open(HELD_STATE)).get("held_lev"))
        except Exception:
            held = None
    if held is None or abs(raw - held) > db:
        held = raw
        if update:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(HELD_STATE, "w") as f:
                json.dump({"held_lev": round(held, 4), "raw": round(raw, 4),
                           "ts": _dt.datetime.now().isoformat(timespec="seconds")}, f)
    return held


def _current_target(dc, update_held=False):
    """Risk dial -> QQQ/QLD/SGOV blend, then carve the TWO regime satellites (growth_exus, growth_xle)
    out of the SAME Growth budget: k_ex to EFA/EEM, k_xl to XLE, (1 - total_k) to the QQQ blend.
    Returns (target weights, HELD leverage, flags, carves). On a dial hiccup -> HOLD the last
    committed target (never a fresh guess; the caller still corrects drift toward it), or ABORT if
    no target was ever persisted. On a carve-signal hiccup -> that k=0 (no carve), which means a
    currently funded EFA/EEM/XLE carve is sold back into the QQQ blend on that run.
    The dial's raw (floored) target is passed through the INELASTIC deadband -> a sticky HELD target."""
    try:
        lev, flags = growth_dial.dial_state_live(dc)
    except Exception as e:
        # FAIL-OPEN REMOVED 2026-08-27. This used to resolve a dial exception to MAX_LEV (1.75x)
        # and call it "fails safe" -- for a book whose mandate is drawdown protection, defaulting
        # to MAXIMUM leverage is the least safe thing available. Measured: the validated target
        # sits below 1.50x on 3532/4643 sessions (76.1%), mean jump if this fired +0.841x, max
        # +1.750x, and the jump clears DEADBAND_LEV on those same 76.1% of sessions so it reaches
        # the book rather than being absorbed. Cost of it firing through a stress window, dial vs
        # stuck-at-1.75x (2026-08-27, retired flat-4% rate basis; indicative): 2022 -7.1% -> -55.3%, 2008 Lehman -22.3% -> -56.1%, COVID -8.6% -> -22.6%.
        # It also silently DEFEATED growth_dial's own "NO SILENT FALLBACK" price-basis guard: the
        # dial raises loudly on a bad basis, and this swallowed it and deployed full risk-on.
        # A dial hiccup is not information, so it must not move the TARGET in either direction:
        # hold the last committed target. With no persisted state there is nothing honest to hold,
        # so ABORT rather than invent a number -- an aborted run leaves the book exactly as it is.
        import json as _json
        _held = None
        if os.path.exists(HELD_STATE):
            try:
                _held = float(_json.load(open(HELD_STATE)).get("held_lev"))
            except Exception:
                _held = None
        if _held is None:
            raise RuntimeError(
                f"dial failed ({e}) and there is no persisted target to hold -- ABORTING rather "
                f"than guessing a leverage. Fix the dial, then re-run.") from e
        print(f"  [dial] live signal fetch FAILED ({e}); HOLDING last committed target {_held:g}x "
              f"(no new target on a dial hiccup). THIS IS AN ALERT, not a normal path.")
        _alert(f"dial fetch failed: {e} -- held {_held:g}x, no fresh signal")
        lev, flags = _held, {s: False for s in growth_dial.SIGNALS}
        flags["DIAL_FAILED"] = True
    raw_lev = lev
    lev = _sticky_lev(raw_lev, update_held)                   # inelastic: hold until target moves >band
    flags["RAW_LEV"] = round(raw_lev, 2); flags["HELD_LEV"] = round(lev, 2)
    qqq_w = _weights_for_leverage(lev)                        # QQQ/QLD/SGOV, sums to 1
    ex = growth_exus.current_allocation(); k_ex = ex["k_exus"]   # 0..0.15 (weak-dollar regime)
    xl = growth_xle.current_allocation();  k_xl = xl["k_xle"]    # 0..0.12 (energy regime)
    total_k = min(k_ex + k_xl, 0.90)                          # caps sum to <=27%; clamp is insurance
    w = {s: (1.0 - total_k) * wt for s, wt in qqq_w.items()}  # scale the QQQ blend down by total_k...
    for sym, frac in growth_exus.BASKET.items():
        w[sym] = w.get(sym, 0.0) + k_ex * frac               # ...reserve k_ex for the ex-US basket...
    for sym, frac in growth_xle.BASKET.items():
        w[sym] = w.get(sym, 0.0) + k_xl * frac               # ...and k_xl for the energy sleeve
    flags["EXUS_K"] = k_ex; flags["XLE_K"] = k_xl
    return w, lev, flags, {"exus": ex, "xle": xl, "total_k": total_k}


def _fetch_snapshot(tc):
    acct = tc.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    mv = {p.symbol: float(p.market_value) for p in tc.get_all_positions()}
    return acct, equity, cash, mv


def _snapshot(tc, retries=4):
    """Robust snapshot. Alpaca intermittently returns an EMPTY positions list with equity collapsed
    to just cash -- a transient read glitch, not a wiped account. That glitch once tripped the
    auto-pause and stranded funds, so we now RE-FETCH before trusting a book that appears to have
    vanished vs yesterday's equity."""
    acct, equity, cash, mv = _fetch_snapshot(tc)
    for attempt in range(retries - 1):
        le = float(acct.last_equity or 0)
        glitch = (not mv) and le > 0 and equity / le < 0.75    # positions gone vs yesterday's book
        if not glitch:
            break
        print(f"  [snapshot] positions empty & equity ${equity:,.0f} vs last ${le:,.0f} -- "
              f"likely API glitch, re-fetching ({attempt + 1}/{retries - 1})")
        time.sleep(2)
        acct, equity, cash, mv = _fetch_snapshot(tc)
    return acct, equity, cash, mv


def _log(row):
    os.makedirs(LOGS_DIR, exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if new:
            f.write("ts,action,symbol,side,notional,equity,note\n")
        f.write(",".join(str(x) for x in row) + "\n")


def status(tc, dc):
    acct, equity, cash, mv = _snapshot(tc)
    target, lev, flags, carves = _current_target(dc)
    ex, xl, k = carves["exus"], carves["xle"], carves["total_k"]
    invested = sum(mv.values())
    eff_lev = sum(mv.get(s, 0.0) * LEVERAGE.get(s, 1.0) for s in set(list(mv) + list(SLEEVES))) / equity if equity else 0
    riskoff = [s for s, off in flags.items() if off is True]   # bool flags only (skip *_K/VOL_MGD_LEV floats)
    _mult = float(getattr(acct, "multiplier", 1) or 1)
    _atype = "MARGIN (same-day rotations)" if _mult > 1 else "CASH (T+1 settlement -> rotations lag a day)"
    print(f"\n  GROWTH SLOT  account {acct.account_number}  ({'PAPER' if PAPER else 'LIVE'})  [{_atype}]")
    _base = equity * DEPLOY
    print(f"  equity ${equity:,.0f} | cash ${cash:,.0f} | invested ${invested:,.0f} | "
          f"buying-power ${float(acct.buying_power):,.0f} (mult {_mult:g})")
    print(f"  deploy base ${_base:,.0f}  (99% of equity, no cap)")
    print(f"  RISK DIAL -> HELD leverage {lev:.2f}x (raw {flags.get('RAW_LEV', lev):.2f}x, "
          f"inelastic band {growth_dial.DEADBAND_LEV:g}x, floor {growth_dial.CORE_FLOOR:g}x core, "
          f"max {MAX_LEV:g}x) | risk-off: {', '.join(riskoff) if riskoff else 'none (full risk-on)'}")
    print(f"  EX-US VIEW  -> {ex['state']}  k={ex['k_exus']*100:.1f}% "
          f"(S1={ex['s1_dollar_downtrend']} S2={ex['s2_valuation_gap']} S3={ex['s3_momentum_confirm']})")
    print(f"  ENERGY VIEW -> {xl['state']}  k={xl['k_xle']*100:.1f}% "
          f"(E1={xl['e1_oil_uptrend']} E2={xl['e2_valuation_gap']} E3={xl['e3_momentum_confirm']})")
    print(f"  {'symbol':>6} {'mkt_val':>12} {'weight':>8} {'target':>8} {'drift':>8}")
    for s in SLEEVES:
        tw = target.get(s, 0.0)
        if mv.get(s, 0.0) <= 0 and tw <= 0:
            continue                                     # hide sleeves with no position and no target
        w = mv.get(s, 0.0) / equity if equity else 0
        print(f"  {s:>6} {mv.get(s,0.0):>12,.0f} {w*100:>7.1f}% {tw*100:>7.1f}% {(w-tw)*100:>+7.1f}%")
    other = {s: v for s, v in mv.items() if s not in SLEEVES}
    if other:
        print(f"  OFF-TARGET holdings: {other}")
    print(f"  >>> effective QQQ leverage now: {eff_lev:.2f}x  (dial target {lev:.2f}x)\n")
    _daily_log(equity, cash, eff_lev, lev, riskoff, mv, carves)
    return equity, mv


DAILY_HDR = ("date,equity,cash,eff_lev,dial_lev,riskoff,qqq_mv,qld_mv,sgov_mv,"
             "efa_mv,eem_mv,exus_k,xle_mv,xle_k,post_lev")
#  eff_lev  = leverage BOOKED at the 11:00 read, i.e. BEFORE this run's orders.
#  post_lev = leverage this run's SUBMITTED orders steer the book to. Blank on days nothing was
#             submitted (no drift, dry-run, market closed, paused) -- blank therefore means
#             "eff_lev IS the post-trade state", which is what a downstream consumer needs to know.


def _daily_file():
    """Path to the daily state log, with the header migrated in place if a column was added since
    the file was created. `_daily_log` only writes a header when the file is NEW, so without this a
    new column would never appear in the live file and every name-keyed reader (e.g. a
    csv.DictReader downstream) would silently ignore it. Old rows
    are padded with empty trailing fields -- honest: that value was never recorded for those days."""
    fn = os.path.join(LOGS_DIR, "growth_slot_daily.csv")
    lines = open(fn).read().splitlines() if os.path.exists(fn) else []
    if lines and lines[0] != DAILY_HDR and lines[0].startswith("date,"):
        ncol = DAILY_HDR.count(",") + 1
        tmp = fn + ".tmp"                                      # atomic: never leave a half-written
        with open(tmp, "w") as f:                              # history behind on a crash
            f.write(DAILY_HDR + "\n")
            for ln in lines[1:]:
                f.write(ln + "," * max(0, ncol - 1 - ln.count(",")) + "\n")
        os.replace(tmp, fn)
    return fn


def _daily_log(equity, cash, eff_lev, dial_lev, riskoff, mv, carves=None):
    """One state row per day -> logs/growth_slot_daily.csv (daily state history).
    Written from `status()`, i.e. BEFORE any orders -- deliberately, because it must be
    deterministic. `_log_post_lev` fills in the post-order column after the order loop."""
    try:
        k_ex = (carves or {}).get("exus", {}).get("k_exus", 0.0)
        k_xl = (carves or {}).get("xle", {}).get("k_xle", 0.0)
        os.makedirs(LOGS_DIR, exist_ok=True)
        fn = _daily_file()
        today = _dt.date.today().isoformat()
        if os.path.exists(fn) and any(ln.startswith(today) for ln in open(fn).read().splitlines()[-5:]):
            return                                             # already logged today
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            if new:
                f.write(DAILY_HDR + "\n")
            f.write(f"{today},{equity:.2f},{cash:.2f},{eff_lev:.3f},{dial_lev:.3f},"
                    f"{'|'.join(riskoff) or 'none'},{mv.get('QQQ',0):.0f},{mv.get('QLD',0):.0f},"
                    f"{mv.get('SGOV',0):.0f},{mv.get('EFA',0):.0f},{mv.get('EEM',0):.0f},{k_ex:.4f},"
                    f"{mv.get('XLE',0):.0f},{k_xl:.4f},\n")
    except Exception as e:
        print(f"  [daily-log] skipped ({e})")


def _log_post_lev(post_lev):
    """Fill TODAY's `post_lev` -- the exposure implied by the orders this run actually SUBMITTED
    (pre-trade market values + submitted notionals). Deliberately NOT a re-snapshot of the broker:
    market orders fill asynchronously, so a post-trade read can catch a half-filled book and would
    be nondeterministic where the pre-trade row is at least reproducible. So this is
    SUBMITTED, not FILLED -- the next run's `eff_lev` is the booked truth, and the pair brackets it.
    Rewrites one field of the last row only; no-op if today's row is missing."""
    try:
        fn = _daily_file()
        lines = open(fn).read().splitlines() if os.path.exists(fn) else []
        today = _dt.date.today().isoformat()
        if not lines or not lines[-1].startswith(today):
            return
        cells = lines[-1].split(",")
        cells[-1] = f"{post_lev:.3f}"                          # post_lev is the last column
        lines[-1] = ",".join(cells)
        tmp = fn + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, fn)
    except Exception as e:
        print(f"  [daily-log] post_lev not recorded ({e})")


PAUSE_FILE = os.path.join(LOGS_DIR, "PAUSE")   # the KILL SWITCH -- _HERE-anchored, see the note at
                                               # LOGS_DIR. This must never be cwd-relative again.


def _sanity_check(acct, mv, eff_lev):
    """Two guard classes (both refuse to trade, but they differ in persistence):
      LATCHING -> write logs/PAUSE, block until a human clears it. Genuinely broken states that
        need review: impossible leverage, alien holdings, runaway order count.
      TRANSIENT -> skip THIS run only, retry next run, NO pause latched. A suspected data glitch
        (equity cratered vs yesterday's close). Latching on a read hiccup once stranded funds --
        glitches must self-heal on the next clean snapshot, not block for days.
    Returns (latching_reason | None, transient_reason | None). Does NOT fire on real market crashes:
    a true crash persists, so next day's last_equity resets and the dial de-risks normally."""
    reasons = []
    if eff_lev > 2.2:
        reasons.append(f"impossible leverage {eff_lev:.2f}x (blend caps at 2.0)")
    alien = {s: v for s, v in mv.items() if s not in SLEEVES and abs(v) > 100}
    if alien:
        reasons.append(f"unexpected holdings {list(alien)}")
    try:
        today = _dt.date.today().isoformat()
        n_today = sum(1 for ln in open(LOG).read().splitlines() if ln.startswith(today))
        if n_today > 8:
            reasons.append(f"runaway orders ({n_today} today)")
    except FileNotFoundError:
        pass
    transient = None
    le = float(acct.last_equity or 0)
    if le > 0 and abs(float(acct.equity) / le - 1.0) > 0.25:
        transient = (f"equity {(float(acct.equity)/le-1)*100:+.0f}% vs last close"
                     + (" (positions missing -> read glitch)" if not mv else " (suspected data glitch)"))
    if reasons:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(PAUSE_FILE, "w") as f:
            f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} AUTO-PAUSE: "
                    + "; ".join(reasons) + "\n")
        return "; ".join(reasons), transient
    return None, transient


def rebalance(tc, dc, live):
    if live:
        if os.path.exists(PAUSE_FILE):
            print(f"  !! PAUSED -- {open(PAUSE_FILE).read().strip()}")
            print("  (delete logs/PAUSE to resume live trading)")
            return
        clk = tc.get_clock()
        if not clk.is_open:
            print(f"  market CLOSED (next open {clk.next_open}) -- skipping live rebalance.")
            return
    acct, equity, cash, mv = _snapshot(tc)
    # raw Position objects — needed for .qty on full-liquidation sells (FULL-LIQUIDATION FIX below).
    # Fails soft: an empty dict just means every order falls back to the notional path.
    try:
        _pos = {p.symbol: p for p in tc.get_all_positions()}
    except Exception as e:
        print(f"  [positions] qty fetch failed ({e}); notional-only this run")
        _pos = {}
    if live:
        eff = sum(mv.get(s, 0.0) * LEVERAGE.get(s, 1.0) for s in mv) / equity if equity else 0.0
        latch, transient = _sanity_check(acct, mv, eff)
        if latch:
            print(f"  !! AUTO-PAUSE tripped: {latch} -- no orders submitted (delete logs/PAUSE to resume)")
            return
        if transient:
            print(f"  !! transient anomaly: {transient} -- skipping THIS run, NO pause latched; "
                  f"the next scheduled run retries on a fresh snapshot.")
            return
    # update_held=True: THIS is the acting path, so commit the inelastic held-target (status is read-only)
    target, lev, flags, carves = _current_target(dc, update_held=(dc is not None and live))
    base = equity * DEPLOY   # no cap: the book compounds with equity (cap removed 2026-09-21)
    orders = []
    for s in SLEEVES:
        target_mv = target.get(s, 0.0) * base
        delta = target_mv - mv.get(s, 0.0)
        if abs(delta) < max(MIN_ORDER_USD, DRIFT_PCT * equity):
            continue
        orders.append((s, OrderSide.BUY if delta > 0 else OrderSide.SELL, round(abs(delta), 2)))
    # sells first (free cash), then buys
    orders.sort(key=lambda o: 0 if o[1] == OrderSide.SELL else 1)
    if not orders:
        print(f"  in balance at dial target {lev:.2f}x -- no rebalance needed (within drift band).")
        return
    # Buying power available for BUYS this run:
    #  - MARGIN account (multiplier>1): buying power is instant, no T+1 -> use full buying_power so
    #    a rotation (sell one sleeve, buy another) completes SAME DAY.
    #  - CASH account: sale proceeds settle T+1, so use only SETTLED funds (non_marginable_buying_power)
    #    and let the next daily run finish the buy once they settle. (No over-leverage either way:
    #    the blend deploys <=99% of equity, so margin borrowing is never actually used.)
    _mult = float(getattr(acct, "multiplier", 1) or 1)
    if _mult > 1:
        settled = float(acct.buying_power or 0.0)
    else:
        settled = float(getattr(acct, "non_marginable_buying_power", None) or acct.buying_power or 0.0)
    _kt = carves["total_k"]
    _cv = (f", carve {_kt*100:.0f}% (ex-US {carves['exus']['k_exus']*100:.0f}%/energy "
           f"{carves['xle']['k_xle']*100:.0f}%)") if _kt > 0 else ""
    print(f"  {'DRY-RUN' if not live else 'LIVE'} rebalance to dial target {lev:.2f}x{_cv} "
          f"(equity ${equity:,.0f}, settled-cash ${settled:,.0f}, weights "
          f"{ {s: round(target[s],3) for s in SLEEVES if target.get(s,0)>0} }):")
    remaining_buy = settled
    placed = {}      # symbol -> signed $ SUBMITTED this run; feeds post_lev (see _log_post_lev)
    for sym, side, notional in orders:
        tag = "BUY " if side == OrderSide.BUY else "SELL"
        amt = notional
        if side == OrderSide.BUY:
            amt = min(notional, remaining_buy)
            if amt < MIN_ORDER_USD:
                print(f"    BUY  {sym}: ${notional:,.0f} DEFERRED -- only ${remaining_buy:,.0f} settled "
                      f"(completes on the next settled day)")
                continue
        deferred = f"  (of ${notional:,.0f}; ${notional-amt:,.0f} deferred)" if amt < notional - 0.5 else ""
        print(f"    {tag} ${amt:,.2f} {sym}{deferred}")
        if live:
            try:
                # FULL-LIQUIDATION FIX. A notional SELL that consumes ~the whole position can be
                # rejected by Alpaca (code 40310000 "insufficient qty available"): notional ->
                # shares can round a hair ABOVE the shares that exist, so the sell fails and a
                # rotation (e.g. SGOV -> QLD) never completes, leaving the book under its target.
                # Sell by SHARE QUANTITY whenever the order would clear >=98% of the position;
                # notional is fine for partial trims.
                p = _pos.get(sym)
                p_mv = abs(float(p.market_value)) if p is not None else 0.0
                if side == OrderSide.SELL and p is not None and p_mv > 0 and amt >= 0.98 * p_mv:
                    q = str(getattr(p, "qty_available", None) or p.qty)
                    req = MarketOrderRequest(symbol=sym, qty=q, side=side,
                                             time_in_force=TimeInForce.DAY)
                    print(f"      (full-position sell -> submitting qty={q} instead of notional)")
                else:
                    req = MarketOrderRequest(symbol=sym, notional=round(amt, 2), side=side,
                                             time_in_force=TimeInForce.DAY)
                o = tc.submit_order(req)
                placed[sym] = placed.get(sym, 0.0) + (amt if side == OrderSide.BUY else -amt)
                _log([_dt.datetime.now().isoformat(timespec="seconds"), "rebalance", sym,
                      tag.strip(), round(amt, 2), round(equity, 2), f"lev={lev:.2f};order_id={o.id}"])
            except Exception as e:
                print(f"      !! order failed (won't crash the run): {e}")
                _log([_dt.datetime.now().isoformat(timespec="seconds"), "rebalance", sym,
                      tag.strip(), round(amt, 2), round(equity, 2), f"lev={lev:.2f};FAILED:{str(e)[:80]}"])
        if not live:                                   # dry-run: no order, but show the same math
            placed[sym] = placed.get(sym, 0.0) + (amt if side == OrderSide.BUY else -amt)
        if side == OrderSide.BUY:
            remaining_buy -= amt
    # POST-TRADE EXPOSURE (2026-09-21). The daily row is written pre-trade by design, so on
    # every day the book trades it understated the change. Report the exposure the submitted orders
    # steer to, from pre-trade market values + submitted notionals -- deterministic, and it captures
    # what a downstream consumer cannot otherwise see: legs left alone inside the 3% drift band and
    # buys truncated by T+1 settlement are both already in this number.
    _post = {s: mv.get(s, 0.0) + placed.get(s, 0.0) for s in set(list(mv) + list(placed))}
    post_lev = sum(_post[s] * LEVERAGE.get(s, 1.0) for s in _post) / equity if equity else 0.0
    print(f"  >>> submitted orders steer the book to {post_lev:.2f}x (dial target {lev:.2f}x)")
    if live:
        _log_post_lev(post_lev)
    else:
        print("    (dry-run -- re-run with --live to submit these paper orders during market hours)")


def main():
    ap = argparse.ArgumentParser(description="Levered-QQQ (QQQ/QLD/SGOV, dial-set leverage up to 1.75x) "
                                             "growth-slot executor.")
    ap.add_argument("--status", action="store_true",
                    help="positions/leverage report (no orders; appends a local log row)")
    ap.add_argument("--rebalance", action="store_true", help="rebalance the blend to target (dry-run unless --live)")
    ap.add_argument("--live", action="store_true",
                    help="actually submit orders (paper unless ALPACA_BASE_URL_GROWTH says otherwise)")
    a = ap.parse_args()
    tc = _client()
    dc = _data_client()
    if a.status or not (a.status or a.rebalance):
        status(tc, dc)
    if a.rebalance:
        status(tc, dc)
        rebalance(tc, dc, a.live)


if __name__ == "__main__":
    main()
