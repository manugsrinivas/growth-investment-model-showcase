"""
fetch_macro_bars_swing.py
=========================
Macro data fetcher for the growth dial and its carve-out sleeves.

Fetches up to two categories of data and merges them into a single wide CSV:

  1. ETF OHLCV  — 48 ETFs via yfinance daily, 2008-01-01 → today
     Output columns: {ticker}_open, _high, _low, _close, _volume
     Source: yfinance (split/dividend-adjusted, 2008+). This is the ONLY block the dial,
     growth_exus and growth_xle read (the per-ETF cache files below).

  2. OPTIONAL FRED macro series — 47 economic series via the FRED API, forward-filled
     to match the daily ETF calendar. Fetched only when FRED_API_KEY is set AND --etf-only
     is NOT passed; nothing in the dial uses them.
     Output columns: fred_{series_id} (lowercase)
     Source: fredapi. Some of these series are third-party copyrighted (e.g. Moody's DAAA and
     BAA10Y, S&P CSUSHPINSA, University of Michigan UMCSENT) and carry their owners' terms.
     This product uses the FRED® API but is not endorsed or certified by the Federal Reserve
     Bank of St. Louis.

All output is written under data/macro/, which is gitignored: none of this data is redistributed.

OUTPUT
------
  data/macro/macro_daily_bars.csv
    Wide CSV, one row per trading day, DatetimeIndex (tz-naive date strings)
    ~240 ETF OHLCV columns, plus the FRED series and derived composites when FRED is fetched

  data/macro/{TICKER}_daily_{start}_{end}.csv
    Per-ETF cache files (skip re-fetch if fresh within cache_grace_days)

USAGE
-----
  python fetch_macro_bars_swing.py --etf-only  # ETF block only (everything the dial needs)
  python fetch_macro_bars_swing.py             # ETF block, plus FRED if FRED_API_KEY is set
  python fetch_macro_bars_swing.py --force     # force re-fetch even if fresh

FRED API KEY
------------
  Set FRED_API_KEY in the environment to fetch the optional FRED block.
  If not set, the FRED section is skipped and macro_daily_bars.csv contains ETF data only.
"""

import os
import sys

# Force UTF-8 output so box-drawing chars in comments don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import argparse
import datetime as dt
from pathlib import Path

import numpy  as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install -r requirements.txt"); sys.exit(1)

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR    = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "trader_config_swing.json"
MACRO_DIR   = BASE_DIR / "data" / "macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV  = MACRO_DIR / "macro_daily_bars.csv"

HISTORY_START = "2008-01-01"
YF_RATE_LIMIT = 1.0   # seconds between yfinance calls

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

CACHE_GRACE_DAYS = CFG["data_pipeline"].get("cache_grace_days", 3)

# ── 48-ETF macro universe (from trader_config_swing.json) ───────────────────
_u = CFG["macro_etf_universe"]
ETF_UNIVERSE = (
    _u["broad_indices"]
    + _u["style"]
    + _u["treasuries"]
    + _u["credit_bonds"]
    + _u["commodities"]
    + _u["intl_fx"]
    + _u["volatility"]
    + _u["sectors"]
)
ETF_UNIVERSE = sorted(set(ETF_UNIVERSE))   # deduplicate, stable sort

# ── FRED series (from trader_config_swing.json) ─────────────────────────────
_f = CFG["fred_series"]
FRED_SERIES = (
    _f["rates"]
    + _f["financial_conditions"]
    + _f["employment"]
    + _f["inflation"]
    + _f["money"]
    + _f["activity"]
    + _f["consumer"]
    + _f["housing"]
    + _f["fiscal"]
    + _f["fx"]
)
FRED_SERIES = list(dict.fromkeys(FRED_SERIES))   # deduplicate, preserve order


# ============================================================================
# HELPERS
# ============================================================================

def _today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def _cache_path(ticker: str) -> Path:
    return MACRO_DIR / f"{ticker}_daily_{HISTORY_START}_{_today_str()}.csv"


def _cache_is_fresh(ticker: str) -> bool:
    p = _cache_path(ticker)
    if not p.exists():
        return False
    age_days = (dt.date.today() - dt.date.fromtimestamp(p.stat().st_mtime)).days
    return age_days <= CACHE_GRACE_DAYS


def _load_cache(ticker: str) -> pd.DataFrame:
    p = _cache_path(ticker)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    p = _cache_path(ticker)
    df.to_csv(p)


# ============================================================================
# ETF FETCH
# ============================================================================

def fetch_etf(ticker: str, force: bool = False) -> pd.DataFrame | None:
    """Fetch daily OHLCV for one ETF via yfinance. Returns DataFrame or None."""
    if not force and _cache_is_fresh(ticker):
        print(f"  {ticker:<8} [cache]")
        return _load_cache(ticker)

    try:
        raw = yf.download(
            ticker,
            start=HISTORY_START,
            end=_today_str(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"  {ticker:<8} ERROR: {e}")
        return None

    if raw is None or len(raw) == 0:
        print(f"  {ticker:<8} NO DATA")
        return None

    # Flatten MultiIndex if present (yfinance sometimes wraps columns)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index).normalize()

    # Keep only OHLCV
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    df   = raw[cols].copy()
    df.dropna(subset=["close"], inplace=True)

    earliest = df.index.min().date()
    print(f"  {ticker:<8} {str(earliest):>12}  {len(df):>7,} bars")

    _save_cache(ticker, df)
    time.sleep(YF_RATE_LIMIT)
    return df


def build_etf_block(force: bool = False) -> pd.DataFrame:
    """Fetch every ETF in the universe (48), prefix columns, outer-join on date index."""
    print()
    print("=" * 70)
    print(f"FETCHING {len(ETF_UNIVERSE)} ETFs via yfinance daily  ({HISTORY_START} -> today)")
    print("=" * 70)

    frames = {}
    for ticker in ETF_UNIVERSE:
        df = fetch_etf(ticker, force=force)
        if df is not None and len(df) > 0:
            df.columns = [f"{ticker.lower()}_{c}" for c in df.columns]
            frames[ticker] = df

    if not frames:
        print("ERROR: No ETF data fetched.")
        return pd.DataFrame()

    combined = pd.concat(frames.values(), axis=1, join="outer")
    combined.sort_index(inplace=True)
    print(f"\n  ETF block: {len(combined)} rows × {len(combined.columns)} columns")
    return combined


# ============================================================================
# FRED FETCH
# ============================================================================

def _build_fred_client():
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        return None
    if not FRED_AVAILABLE:
        print("  WARNING: fredapi not installed. Run: pip install -r requirements.txt")
        return None
    return fredapi.Fred(api_key=key)


def fetch_fred_series(fred_client, series_id: str, date_index: pd.DatetimeIndex) -> pd.Series | None:
    """Fetch one FRED series, resample to daily, forward-fill to match date_index."""
    try:
        raw = fred_client.get_series(series_id, observation_start=HISTORY_START)
        if raw is None or len(raw) == 0:
            return None
        raw.index = pd.to_datetime(raw.index)
        # Expand to full calendar (including weekends) so weekly/monthly releases
        # on non-trading days propagate forward into trading days correctly.
        full_cal = pd.date_range(raw.index.min(), date_index.max(), freq="D")
        s = raw.reindex(full_cal).ffill().reindex(date_index)
        s.name = f"fred_{series_id.lower()}"
        return s
    except Exception as e:
        print(f"  {series_id:<30} ERROR: {str(e)[:60]}")
        return None


def build_fred_block(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fetch all FRED series and return DataFrame aligned to date_index."""
    print()
    print("=" * 70)
    print(f"FETCHING {len(FRED_SERIES)} FRED series  ({HISTORY_START} -> today)")
    print("=" * 70)

    fred = _build_fred_client()
    if fred is None:
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            print("  SKIPPED — FRED_API_KEY not set (optional; the dial uses none of these series)")
        return pd.DataFrame(index=date_index)

    series_frames = {}
    ok = skip = 0
    for sid in FRED_SERIES:
        s = fetch_fred_series(fred, sid, date_index)
        if s is not None:
            n_valid = int(s.notna().sum())
            if n_valid == 0:
                print(f"  {sid:<30} SKIPPED (0 non-null after reindex)")
                skip += 1
            else:
                series_frames[s.name] = s   # key = "fred_xxx" prefix applied in fetch_fred_series
                print(f"  {sid:<30} {n_valid:>6,} non-null values")
                ok += 1
        else:
            print(f"  {sid:<30} SKIPPED")
            skip += 1
        time.sleep(0.1)   # FRED is lenient but be polite

    print(f"\n  FRED block: {ok} OK, {skip} skipped")
    if not series_frames:
        return pd.DataFrame(index=date_index)

    return pd.DataFrame(series_frames, index=date_index)


# ============================================================================
# ENGINEERED COMPOSITES
# ============================================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of derived macro features computed from the fetched data.
    None of them is read by the growth dial or its sleeves.
    """
    df = df.copy()   # defragment to suppress PerformanceWarning
    cols = df.columns.tolist()

    # ── Yield curve slopes ────────────────────────────────────────────────────
    if "fred_dgs10" in cols and "fred_dgs2" in cols:
        df["fred_yield_10y2y"] = df["fred_dgs10"] - df["fred_dgs2"]
    if "fred_dgs10" in cols and "fred_fedfunds" in cols:
        df["fred_yield_10yff"] = df["fred_dgs10"] - df["fred_fedfunds"]
    if "fred_dgs30" in cols and "fred_dgs10" in cols:
        df["fred_yield_30y10y"] = df["fred_dgs30"] - df["fred_dgs10"]

    # ── Inflation breakevens ───────────────────────────────────────────────────
    if "fred_t10yie" in cols and "fred_t5yie" in cols:
        df["fred_breakeven_slope"] = df["fred_t10yie"] - df["fred_t5yie"]

    # ── Credit quality spread (BAA − AAA = IG quality ladder) ────────────────
    # BAA10Y  = Moody's BAA corporate yield - 10yr treasury (full history on FRED; © Moody's)
    # DAAA    = Moody's AAA corporate bond yield (full history on FRED; © Moody's)
    #   Both are fetched only with FRED_API_KEY set and without --etf-only, and are never
    #   redistributed (the output CSV is gitignored).
    # DGS10   = 10yr treasury yield
    # fred_credit_quality_spread = BAA-AAA spread: widens during credit stress
    #   even within investment grade — cleaner than the restricted BAML/ICE series
    if "fred_daaa" in cols and "fred_dgs10" in cols:
        # AAA spread over 10yr treasury
        df["fred_aaa_spread"] = df["fred_daaa"] - df["fred_dgs10"]
    if "fred_baa10y" in cols and "fred_aaa_spread" in df.columns:
        # BAA-AAA quality differential: pure credit quality signal (no rate noise)
        df["fred_credit_quality_spread"] = df["fred_baa10y"] - df["fred_aaa_spread"]

    # ── Macro stress composite (z-score sum of stress indicators) ─────────────
    # Uses only series with confirmed full FRED coverage.
    # Removed BAMLH0A0HYM2 / BAMLC0A0CM — ICE licensing restricts to last ~3yr.
    stress_cols = [c for c in ["fred_nfci", "fred_stlfsi4",
                               "fred_baa10y", "fred_credit_quality_spread"] if c in df.columns]
    if len(stress_cols) >= 2:
        z_scores = []
        for c in stress_cols:
            s  = df[c].dropna()
            mu = s.mean(); sd = s.std()
            if sd > 0:
                z_scores.append((df[c] - mu) / sd)
        if z_scores:
            df["fred_macro_stress_score"] = pd.concat(z_scores, axis=1).mean(axis=1)

    # ── Growth vs value rotation proxy ────────────────────────────────────────
    if "spyg_close" in cols and "spyv_close" in cols:
        df["macro_growth_value_ratio"] = df["spyg_close"] / df["spyv_close"]

    # ── Equity risk premium proxy (earnings yield - 10yr rate) ────────────────
    if "spy_close" in cols and "fred_dgs10" in cols:
        # P/E for SPY is not directly available without options data;
        # use inverse of YoY return as crude proxy for now
        spy_ret = df["spy_close"].pct_change(252)
        df["macro_erp_proxy"] = spy_ret - df["fred_dgs10"] / 100.0

    # ── VXX momentum (5-day) ──────────────────────────────────────────────────
    if "vxx_close" in cols:
        df["macro_vxx_5d_chg"] = df["vxx_close"].pct_change(5)

    # ── HYG/TLT ratio (risk-on/off signal) ────────────────────────────────────
    if "hyg_close" in cols and "tlt_close" in cols:
        df["macro_hyg_tlt_ratio"] = df["hyg_close"] / df["tlt_close"]

    return df


# ============================================================================
# COVERAGE AUDIT
# ============================================================================

def audit_coverage(df: pd.DataFrame) -> None:
    """Print coverage statistics — flag ETFs with <80% fill after 2014."""
    print()
    print("=" * 70)
    print("COVERAGE AUDIT (post-2014 — IPO-limited tickers expected before)")
    print("=" * 70)
    ref_start = pd.Timestamp("2014-01-01")
    sub = df[df.index >= ref_start]
    n_rows = len(sub)
    if n_rows == 0:
        print("  No data after 2014."); return

    etf_close_cols = [c for c in df.columns if c.endswith("_close") and not c.startswith("fred_")]
    print(f"\n  {'Ticker':<8} {'First date':>12}  {'Post-2014 fill':>14}  Status")
    print(f"  {'------':<8} {'----------':>12}  {'----------':>14}  ------")
    for col in sorted(etf_close_cols):
        ticker = col.replace("_close", "").upper()
        full_series = df[col].dropna()
        if len(full_series) == 0:
            print(f"  {ticker:<8} {'NO DATA':>12}")
            continue
        first_date = full_series.index.min().date()
        post2014   = sub[col].notna().sum()
        pct        = post2014 / n_rows * 100
        flag       = "" if pct >= 80 else " [LOW]"
        print(f"  {ticker:<8} {str(first_date):>12}  {post2014:>7,}/{n_rows:<6,} ({pct:>5.1f}%){flag}")

    fred_cols = [c for c in df.columns if c.startswith("fred_")]
    print(f"\n  FRED series: {len(fred_cols)} columns")
    if fred_cols:
        fills = [(c, sub[c].notna().sum() / n_rows * 100) for c in fred_cols]
        low   = [(c, p) for c, p in fills if p < 50]
        if low:
            print(f"  Low-fill FRED series (post-2014 <50%):")
            for c, p in low:
                print(f"    {c:<40} {p:.1f}%")
        else:
            print(f"  All FRED series ≥50% fill post-2014  (monthly/weekly series expected lower)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Macro data fetcher: ETF bars for the growth dial "
                                                 "(plus optional FRED series if FRED_API_KEY is set)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-fetch even if cache is fresh")
    parser.add_argument("--etf-only", action="store_true",
                        help="Skip FRED fetch (ETF block only; everything the dial needs)")
    parser.add_argument("--fred-only", action="store_true",
                        help="Skip ETF fetch, reload from cache + add FRED")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("MACRO DATA FETCHER")
    print(f"  Start: {HISTORY_START}  ->  End: {_today_str()}")
    print(f"  ETFs:  {len(ETF_UNIVERSE)} symbols")
    print(f"  FRED:  {len(FRED_SERIES)} series")
    print(f"  Out:   {OUTPUT_CSV}")
    print("=" * 70)

    # ── Step 1: ETF block ────────────────────────────────────────────────────
    if args.fred_only and OUTPUT_CSV.exists():
        print("\nLoading ETF block from existing output CSV...")
        etf_df = pd.read_csv(OUTPUT_CSV, index_col=0, parse_dates=True)
        etf_df.index = pd.to_datetime(etf_df.index).normalize()
        etf_cols = [c for c in etf_df.columns if not c.startswith("fred_")]
        etf_df = etf_df[etf_cols]
    else:
        etf_df = build_etf_block(force=args.force)

    if etf_df.empty:
        print("ERROR: ETF block is empty — cannot proceed."); sys.exit(1)

    date_index = etf_df.index   # use ETF trading calendar as master

    # ── Step 2: FRED block ────────────────────────────────────────────────────
    if args.etf_only:
        fred_df = pd.DataFrame(index=date_index)
        print("\n  FRED skipped (--etf-only)")
    else:
        fred_df = build_fred_block(date_index)

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("MERGING ETF + FRED + ENGINEERED COMPOSITES")
    print("=" * 70)

    if not fred_df.empty:
        combined = etf_df.join(fred_df, how="left")
    else:
        combined = etf_df.copy()

    combined = add_engineered_features(combined)
    combined.sort_index(inplace=True)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    combined.to_csv(OUTPUT_CSV)
    print(f"\n  Saved: {OUTPUT_CSV}")
    print(f"  Shape: {combined.shape[0]:,} rows × {combined.shape[1]} columns")
    print(f"  Date range: {combined.index.min().date()} -> {combined.index.max().date()}")

    # ── Step 5: Coverage audit ────────────────────────────────────────────────
    audit_coverage(combined)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
