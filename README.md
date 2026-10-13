# Growth dial — a rule-based leverage dial on QQQ

This is the research and engineering record of a growth strategy I built and run on a paper trading
account. Every day after the close, a rule-based dial decides how much Nasdaq-100 exposure to hold —
anywhere from all cash to 1.75× — and an executor holds it through three plain ETFs in a cash account:
**QQQ**, **QLD** (a 2× daily-reset QQQ fund) and **SGOV** (short T-bills). It never uses margin; the
leverage comes from holding QLD. Every input is a public end-of-day price, and no machine learning makes
any decision.

What's published here is the method, the backtest that can be reproduced from this repository, and the
engineering. The live account's balances and returns stay private. It is research code, not investment
advice.

## Why I built it

Levered index exposure is the obvious way to beat an index over a long horizon, and the obvious way to
get wiped out: a constant 1.75× QQQ book compounds faster than QQQ until a 2008 or a 2022, then gives back
years. I wanted to know whether a small set of transparent rules could keep most of the upside while
cutting the deep drawdowns — and whether the answer would survive treating my own backtest as
adversarially as I'd treat anyone else's. That meant pinning the results in a regression test, writing
down the numbers that changed when I found bugs, and refusing to tune anything the noise could explain.

## What I'd highlight

1. **Two signals with opposite blind spots, combined by taking the more defensive one.** A volatility
   leg reacts within days to fast crashes (2008, 2020) but re-levers into slow grinds; a trend leg (200-day
   averages on market breadth, Treasuries and high-yield credit) catches slow bears (2022) but lags fast
   ones. The dial holds `min(trend, volatility)`, so each covers the other's failure mode.
2. **I didn't adopt the setting the sweep liked best.** Widening the deadband from 0.25× to 0.50× improves
   return, return/vol and worst drawdown on the published code (return/vol +0.04; 2022 gets about a point
   worse). But in an unpublished study (2008–2026, on the earlier flat-rate basis), 38 equivalent settings
   of the same parameter spanned 0.081 of return/vol — the parameter's own path noise is wider than the
   gain. It stays at 0.25×, and the reasoning is written next to the constant in `growth_dial.py` so
   nobody "optimises" it later.
3. **Design rationales get re-tested, and some die.** A minimum-exposure floor existed to defer taxes by
   holding a permanent core. An unpublished tax-lot audit (2008–2026) showed the broker's oldest-first lot
   matching meant only 0.68% of realised gains ever reached long-term rates — the rationale was false, so the floor was pure
   cost and was removed. The volatility target's documented derivation turned out to be wrong too; the
   comment now records its real, empirical provenance.
4. **Live is not the backtest, so the executor is engineered for the gaps.** It trades a leg only when it
   drifts more than 3% of equity from target, sizes buys from settled cash in a cash account, sells by
   share quantity when clearing a whole position (selling by dollar amount can round past the shares held
   and get rejected), and will not invent leverage when its inputs are missing.

## How the dial works

```
L = min(trend leg, volatility leg), clipped to [0, 1.75], then held unless it moves by more than 0.25×
```

- **Trend leg.** Three on/off risk flags, each "price vs its 200-day average" with a 3% hysteresis band:
  - **Breadth** — the share of 9 growth/international ETFs in an uptrend; below 40% the flag is off.
  - **IEF** (7–10y Treasuries) and **HYG** (high-yield credit) — stress in rates or credit.

  Each flag that is off removes a third of the 1.75× ceiling: 1.75× → 1.17× → 0.58× → 0×.
- **Volatility leg.** `0.17 / max(downside volatility over 5 days, over 21 days)`, clipped to 0.5–2.0.
  Only downside moves count, and taking the larger of a fast and a slow window lets leverage come back
  when a selloff genuinely calms, but not on a two-day bounce.
- **Deadband.** The held target changes only when the computed one moves by more than 0.25×. That cut
  target changes about five-fold in the backtest (1,586 → 298), and it isn't free: the worst drawdown is
  about 2 points deeper (−38.1% → −40.1%).

The target maps to the three ETFs:

| Target `L` | QQQ | QLD | SGOV |
|---|---|---|---|
| 1.75× | 25% | 75% | — |
| 1.0× | 100% | — | — |
| 0.5× | 50% | — | 50% |
| 0× | — | — | 100% |

Above 1×: QQQ = 2 − L, QLD = L − 1. Below 1×: QQQ = L, SGOV = 1 − L.

## Backtest results

Window **2008-01-03 to 2026-06-05** (4,634 daily returns), daily closes adjusted for splits and
dividends. The signal is lagged one day: yesterday's close sets today's exposure. Every number here is
printed by `python results_table.py`; the second row's CAGR, return/vol, drawdown and 2022 are also
pinned by `dial_regression_test.py`.

| | CAGR | Return/vol | Sharpe vs cash | Worst drawdown | 2022 |
|---|---|---|---|---|---|
| **Dial — QQQ/QLD/SGOV, simulated** | **19.7%** | **0.99** | **0.92** | −40.3% | −7.6% |
| Dial — simple leverage model | 20.0% | 1.00 | 0.93 | −40.1% | −7.6% |
| QQQ, buy and hold | 16.3% | 0.79 | 0.73 | −49.4% | −32.6% |
| Constant 1.75× target, simulated, no dial | 23.5% | 0.74 | 0.70 | −73.4% | −54.5% |

How each row is computed:

- **Dial, simulated** (`growth_dial._slot_sim`). QLD is modelled as 2× QQQ's daily return less its
  financing (3-month T-bill + 0.50%) and its 0.95% expense ratio, compounding day by day, so its
  daily-reset drag is in. SGOV is modelled as the 3-month T-bill rate over the whole window (the fund
  itself launched in 2020). The book rebalances whenever the dial changes or a leg drifts more than 3% —
  314 times in the window.
- **Dial, simple leverage model** (`growth_dial._bt_run`). QQQ's return × `L`, paying T-bill + 0.50% on
  exposure above 1× and earning the T-bill below 1×; it re-levers daily and charges no expense ratio.
  Average exposure was 1.10×.
- **QQQ, buy and hold.** Total return, dividends reinvested, same window.
- **Constant 1.75×.** The same simulation as the first row with a fixed 1.75× target, rebalanced when a leg
  drifts 3% (37 times). 1.75× is the target, not a measured average.
- **Return/vol** is mean daily return ÷ its standard deviation × √252. **Sharpe vs cash** uses returns in
  excess of the 3-month T-bill.
- **Not included anywhere:** trading costs, bid/ask spread, taxes.

## What the numbers do and don't say

- **The dial is risk control, not extra return.** Holding 1.75× all the time earned more (23.5% a year)
  but fell 73% at its worst and 54.5% in 2022. The dial gave up some of that return to lose 7.6% in 2022
  instead of QQQ's 32.6%, and to keep its worst drawdown near 40%.
- **Leverage through QLD isn't free.** There's no margin interest, but QLD pays to borrow inside the
  fund, charges about 0.95% a year, and loses value in choppy markets because it resets to 2× daily. The
  simulated row charges all of that; the simple model charges the financing but not the fee.
- **The worst drawdown is the 2008 crash** (peak 2008-05-16, low 2008-11-20). For the first 199 sessions,
  to 2008-10-14, the 200-day averages are still filling in and every trend flag starts risk-on, so only
  the volatility leg could cut leverage. From mid-October through the November low, breadth and credit
  were risk-off but the Treasury flag never turned — Treasuries rallied — so the trend leg bottomed at
  0.58× instead of zero. With the volatility leg at its 0.5× floor the two legs computed 0.50×, and the
  0.25× deadband kept the held target at 0.63× — it couldn't step down to cash. **From July 2009 on, the dial's worst drawdown was −22% (low 2025-04-08), against −35% for QQQ
  (low 2022-11-03).**
- **The results are in-sample.** The 1.75× ceiling, the bands, the volatility target and the deadband were
  chosen by testing on this same period. Read the table as a description of the rule's behaviour, not a
  forecast.
- **Point estimates are fragile.** Starting the simulation one session later — skipping 2008-01-04, when
  QQQ fell 4.4% while the dial sat at 1.75× — lifts the first row from 19.7% to 20.2% a year. Differences
  of a few tenths between rows mean little.
- **It trades often, so taxes matter.** In an unpublished after-tax test at one tax bracket, the dial gave
  back all of its edge over QQQ in a taxable account over 2008–2026 — its weakest after-tax period — so it
  belongs in a tax-advantaged account. The after-tax engine isn't published because it is calibrated to
  one person's bracket.
- **The live executor differs from the simulation.** It trades only the legs whose own gap exceeds 3% of
  equity (and $25), invests 99% of equity, waits for T+1 settlement before buying with sale proceeds, trades
  around 11:00 ET rather than at the close, and has two optional sleeves (international and energy, up to
  27% combined) that the simulation leaves out. None of that is modelled here.
- **No live record is published.** The paper account's balances and returns are deliberately kept out.

## Bugs I caught in my own pipeline

Each is documented where it was fixed, with the number it moved:

- **A two-day signal lag.** The QQQ/QLD simulation grew positions by the day's return *before*
  rebalancing, so each day earned the weights set two days earlier while the docs said one. Over
  2008–2026 it cost 1.34 points a year (`growth_dial._slot_sim`).
- **A drawdown measure that under-reported.** Rebuilding the equity curve from returns dropped the first
  bar, so a curve that peaked on day one had its drawdown measured from a lower point. It made one
  rejected variant look about 3 points better than it was, on the earlier rate basis
  (`growth_dial._metrics`).
- **One rate doing two jobs.** A flat 4% stood in for both the borrowing cost and the cash yield. Replacing
  it with real T-bill rates barely moved the headline, but 2022 went from −6.1% to −7.6%: 1.46 points of the
  old figure was cash interest the book never earned (`dial_regression_test.py`).
- **A live path that failed open.** The executor used to answer any dial error with the maximum, 1.75×;
  separately, a missing price history started its trend flag risk-on. Both now refuse instead, and the
  regression test checks the dial's refusal on real data.

## What I tested and didn't adopt

The growth slot started as an **ML stock-picker**. It — and later a cross-sectional ML ranker — couldn't
beat simply holding the index in honest tests, so the slot became this rule-based dial, and machine
learning plays no part in any decision. Every idea below was a transparent rule, tested A/B against the
dial as deployed. Their study scripts aren't in this repo, so they're described in words, not figures.

- **Credit spreads as a timing signal** — rejected across two rounds and nineteen variants. Credit stress
  moved with volatility rather than ahead of it, so it added nothing the volatility leg didn't already see.
- **Valuation (CAPE) timing** — rejected, and the whole valuation family closed after a test back to 1871.
- **Re-levering faster once a volatility spike has peaked** — rejected.
- **Volatility estimated from intraday high/low ranges** instead of closes — rejected.
- **A discrete volatility "brake"** — retired; the continuous volatility leg beat it in all eight tests.
- **Swapping QQQ for a broader growth fund** (VUG, VOOG, SPYG) — rejected.
- **A wider deadband** — not adopted: its gain sat inside the parameter's own noise (see above).

Kept after the same kind of test: the breadth flag in place of a single QQQ trend flag, downside-only
volatility in place of total volatility, rebalancing on a dial change or 3% drift rather than on a
calendar, and all three trend flags (each was tested for whether it earns its place).

## Reproduce the results

Tested on Python 3.14 with the pinned versions in `requirements.txt` (Windows paths shown).

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python fetch_macro_bars_swing.py --etf-only
venv\Scripts\python dial_regression_test.py
venv\Scripts\python results_table.py
```

The third step downloads the ETF price histories into `data/macro/` (a few minutes, no API key). The price
data isn't committed — its source's terms don't allow redistributing it — so you rebuild it yourself. The
regression test should end with `ALL GREEN`: it pins the simple-model row (all but its Sharpe vs cash) and the rule's structure (the
settings, the leverage-to-ETF mapping, the one-day signal lag, and the live dial's refusal to run on
missing inputs). `results_table.py` prints every number in the results section.

Because the prices are re-downloaded, dividend adjustments can shift the last decimal over time; the
test's tolerances (±0.5pp return, ±0.02 return/vol) allow for that.

## Running it live (paper account)

`growth_slot.py` drives an [Alpaca](https://alpaca.markets) account. Set `ALPACA_API_KEY_GROWTH` and
`ALPACA_SECRET_KEY_GROWTH` as environment variables — never in a file. It uses the paper endpoint unless
`ALPACA_BASE_URL_GROWTH` points elsewhere.

```
venv\Scripts\python growth_slot.py --status               # no orders; writes a row to a local log
venv\Scripts\python growth_slot.py --rebalance            # dry run: prints the orders it would place
venv\Scripts\python growth_slot.py --rebalance --live     # places them
```

It places no orders when the market is closed or when a `logs/PAUSE` file exists. If a dial input is
missing, it doesn't compute a new target: it keeps the last saved one (still correcting drift toward it)
and raises an alert, and it stops only if no target has ever been saved.

## Files

| File | What it is |
|---|---|
| `growth_dial.py` | The dial: signals, leverage rule, and both backtest models |
| `growth_slot.py` | Executor for an Alpaca account: target weights, drift band, settlement-aware buys |
| `growth_exus.py`, `growth_xle.py` | Two optional sleeves (international, energy); stay at zero unless their own conditions fire |
| `dial_regression_test.py` | Pins the simple-model row's figures and the rule's structure; run after any change |
| `results_table.py` | Prints every number in the results section |
| `fetch_macro_bars_swing.py`, `trader_config_swing.json` | Downloads the price data (and, with a key, optional FRED series) |
| `wait_for_network.py`, `keep_awake.py` | Scheduling helpers: wait for the network after a wake; keep the PC awake during a run |
| `data/rates/fred_DGS3MO.csv` | Daily 3-month Treasury rate — cash yield and financing cost |

## What's not in this repo, and why

This is a fresh-history export of a private working repo. The method and code are public; operational
data is not:

- No credentials or API keys, ever.
- No broker account IDs, no live account equity or P&L, and no personal tax details.
- No raw `data/`, `logs/` or `state/` — the one exception is `data/rates/fred_DGS3MO.csv`, a public-domain
  T-bill series the dial reads — and no third-party data whose licence bars redistribution.
- Every performance number carries its window and basis. Figures later found to be wrong are corrected or
  labelled as superseded, never quietly kept.

## Data sources

- ETF prices: Yahoo Finance, via [yfinance](https://github.com/ranaroussi/yfinance), downloaded locally —
  not included.
- 3-month Treasury rate: Board of Governors of the Federal Reserve System (US), *Market Yield on U.S.
  Treasury Securities at 3-Month Constant Maturity, Quoted on an Investment Basis* [DGS3MO], retrieved from
  FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DGS3MO (observations through
  2026-09-03). Public domain; refresh it from that page.
- This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
  The optional FRED series the fetcher can download are not used by the dial and are never committed.
