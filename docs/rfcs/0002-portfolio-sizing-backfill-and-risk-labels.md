# RFC 0002: Portfolio Sizing Backfill and Risk Label Clarity

- **Status**: Draft
- **Author**: Hartej Grewal
- **Date**: 2026-04-15
- **Related code**:
  - [`indicators/risk.py`](../../src/clenow_momentum/indicators/risk.py)
  - [`scripts/run_analysis.py`](../../scripts/run_analysis.py)
  - [`utils/config.py`](../../src/clenow_momentum/utils/config.py)

## Summary
[summary]: #summary

Make portfolio construction fill up to `MAX_POSITIONS` from the ranked eligible universe after sizing, not before sizing. When a candidate is dropped, preserve and display the concrete reason. Separate the strategy's one-ATR sizing target from the wider stop-loss exposure shown in portfolio output.

## Motivation
[motivation]: #motivation

The Russell 1000 run exposed two related issues in portfolio construction and reporting.

First, the strategy found hundreds of eligible names but produced fewer than `MAX_POSITIONS` holdings. The root cause was ordering: `run_analysis.py` sliced the filtered momentum list to the top 20 before sizing. Then the risk layer dropped several of those names because they sized to zero shares or lacked usable ATR. The pipeline did not backfill from ranks 21 onward, so the portfolio ended with fewer than 20 positions even though many valid candidates remained.

In the observed cached Russell 1000 run, the pre-sizing top 20 were:

```text
SNDK, LITE, CIEN, LYB, DAR, DOW, IRDM, GLW, WLK, WDC,
CF, ACHC, MRNA, VRT, TER, FTI, SOLS, TPL, TIGO, MTZ
```

`SNDK` and `LITE` sized to zero shares because the configured target ATR impact was `$50`, while their ATR values were approximately `$59.44` and `$68.80`. `SOLS` had enough valid recent OHLC rows, but leading missing rows caused the ATR calculation to return `NaN`, so it disappeared during the ATR merge. Without backfill and clear drop reasons, the portfolio looked arbitrarily incomplete.

Second, the output used the generic label `Risk $` for stop-loss exposure while also printing `Risk per Trade: 0.100% ($50)`. These are not the same measure:

- `0.100% ($50)` is the target one-ATR portfolio impact used for sizing.
- `Risk $` was actually `shares * ATR * STOP_LOSS_MULTIPLIER`, which is stop exposure at the configured ATR stop distance.

For `LYB`:

```text
ATR impact = 13 shares * $3.77 ATR = about $49
3x ATR stop exposure = 13 shares * $3.77 ATR * 3 = about $147
```

Both numbers are useful, but presenting them under one "risk" label makes the output look internally inconsistent.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Portfolio construction should be read as a ranked-candidate selection process:

1. Calculate momentum for the full universe.
2. Apply market and stock filters.
3. Walk the filtered candidates in momentum rank order.
4. Size each candidate using ATR risk parity.
5. Drop candidates that cannot become valid positions, recording the reason.
6. Continue walking the ranked list until `MAX_POSITIONS` valid holdings are selected or candidates are exhausted.

The selected holdings are therefore "the first N valid sized candidates", not "the first N filtered candidates after later invalid names are removed."

### Drop reasons

Dropped candidates should be reported with concrete reasons such as:

```text
SNDK - sized to 0 shares because target ATR impact $50 is below ATR $59.44
LITE - sized to 0 shares because target ATR impact $50 is below ATR $68.80
FIX  - sized to 0 shares because target ATR impact $50 is below ATR $73.35
```

This makes it clear whether the issue is volatility, price/whole-share rounding, missing data, minimum position size, or cash constraints.

### Risk labels

The portfolio table should distinguish:

| Concept | Formula | Meaning |
|---|---|---|
| Target ATR impact | `strategy_allocation * risk_per_trade` | Desired one-ATR portfolio impact per position |
| Actual ATR impact | `shares * ATR` | Realized one-ATR impact after whole-share rounding |
| Stop risk | `shares * ATR * stop_loss_multiplier` | Exposure at the configured ATR stop distance |

The table should avoid the generic column name `Risk $`. Prefer:

```text
ATR Impact $
Stop Risk $
```

The summary should say:

```text
Target ATR Impact per Position: 0.100% ($50)
Total 3x ATR Stop Exposure: $...
Weighting Method: Risk Parity (Equal ATR Impact)
```

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Candidate selection

`check_and_filter_stocks()` should pass all filtered candidates to portfolio construction:

```python
stocks_for_portfolio = filtered_stocks
```

`build_portfolio()` owns final candidate selection because it has the data needed to know whether a candidate is valid after ATR, share rounding, minimum position size, and cash constraints.

The portfolio builder accepts:

```python
max_positions: int | None
min_position_value: float | None
```

It iterates over sized candidates in momentum rank order, selecting valid rows until `max_positions` is reached. Invalid candidates are skipped and appended to `portfolio_df.attrs["drop_reasons"]`.

### Drop reason schema

Drop reasons are dictionaries stored on the returned DataFrame:

```python
{
    "ticker": "SNDK",
    "stage": "position_size",
    "reason": "sized to 0 shares because target ATR impact $50 is below ATR $59.44",
}
```

Initial stages:

| Stage | Meaning |
|---|---|
| `ATR` | Candidate could not be merged with a valid ATR value |
| `position_size` | Candidate sized below the minimum position requirement or to zero shares |
| `cash` | Candidate could not fit within remaining account allocation |
| `max_positions` | Candidate was valid but below the final cap after filtering |

`run_analysis.py` displays the drop reasons after the portfolio summary.

### ATR calculation with leading missing rows

Some tickers have valid recent OHLC rows but leading empty rows in the one-year download frame. ATR should be calculated over valid OHLC rows:

```python
ticker_data = ticker_data[["High", "Low", "Close"]].dropna()
```

If the valid row count is still below `atr_period`, the ticker is dropped with an ATR reason. Otherwise the current ATR should be calculated from the cleaned series.

### Risk fields

`calculate_position_size()` returns explicit fields:

```python
{
    "target_atr_impact": account_value * risk_per_trade,
    "actual_atr_impact": shares * atr,
    "stop_loss_risk": shares * atr * stop_loss_multiplier,
}
```

Backward-compatible aliases remain:

```python
"target_risk" == "target_atr_impact"
"actual_risk" == "stop_loss_risk"
```

The aliases preserve existing trading/rebalancing behavior while the display layer migrates to clearer names.

### Configuration naming

The config loader should tolerate both:

```text
RISK_PER_TRADE=0.1
RISK_PER_TRADE_PCT=0.1
```

`RISK_PER_TRADE` should be the preferred documented name, while `RISK_PER_TRADE_PCT` remains an accepted alias for existing local configs.

## Drawbacks
[drawbacks]: #drawbacks

- Building the portfolio from all filtered candidates can do more sizing work than slicing to `MAX_POSITIONS` first. For a Russell 1000 universe this is still small compared with data download and indicator calculation.
- Showing dropped candidates adds more output. This is useful in interactive runs, but the table could become noisy if many candidates fail.
- DataFrame `.attrs` is convenient for attaching diagnostics, but it is easy to lose if downstream code copies or serializes the DataFrame without preserving metadata.
- Keeping `actual_risk` as a backward-compatible alias for stop risk can still confuse readers of the code until all call sites migrate to explicit names.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

### Why backfill inside `build_portfolio()`?

The risk layer is the first place where all constraints are known: ATR availability, whole-share rounding, minimum position value, maximum position percent, and cash usage. Backfilling in `run_analysis.py` would require duplicating sizing logic or making repeated calls with arbitrary larger slices.

### Why not just increase the pre-sizing slice?

Using `filtered_stocks.head(MAX_POSITIONS * 2)` would reduce the failure rate but not solve the invariant. A concentrated run could still lose more candidates than expected. Walking all filtered candidates until the final valid count is reached is deterministic and easier to reason about.

### Why keep unused cash?

This RFC does not change the risk model. Low capital utilization can be an expected result of conservative ATR sizing. The fix is to ensure the strategy fills the requested number of valid positions, not to force 100% capital deployment.

### Why keep `actual_risk`?

Trading and rebalancing code already reads `actual_risk`. Renaming it everywhere should be a separate cleanup once the behavior is stable. This RFC adds explicit fields while preserving compatibility.

## Prior art
[prior-art]: #prior-art

ATR-based risk parity commonly sizes positions by volatility rather than by equal capital allocation. In whole-share portfolios, high-priced or highly volatile instruments can legitimately size to zero shares when the configured risk budget is small. Portfolio constructors usually handle this by skipping untradeable candidates and continuing down the ranked list.

The distinction between one-unit volatility impact and stop-loss exposure is also common in discretionary and systematic trading systems. The former is a sizing primitive; the latter is an adverse-move scenario.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- Should fractional shares be supported for brokers/accounts that allow them? That would reduce zero-share drops for high-priced/high-ATR names.
- Should `drop_reasons` be returned as a separate structured object instead of `DataFrame.attrs`?
- Should the CLI display all dropped candidates or only the candidates skipped before the final portfolio filled?
- Should cash remaining above a threshold produce an informational note explaining that conservative ATR sizing can leave capital idle?

## Future possibilities
[future-possibilities]: #future-possibilities

- Add a portfolio construction report object with selected positions, dropped candidates, and summary diagnostics.
- Add optional fractional-share sizing.
- Add configurable cash deployment modes, for example "strict ATR risk parity" vs "use residual cash subject to max position caps."
- Add a warning when many candidates size to zero under the current risk setting, with suggested configuration levers.
- Fully migrate trading/rebalancing code from `actual_risk` to `stop_loss_risk`.

## Acceptance criteria
[acceptance-criteria]: #acceptance-criteria

The implementation should satisfy all of:

1. **Backfill behavior.** If a top-ranked candidate is invalid after sizing, the next-ranked filtered candidate is considered until `MAX_POSITIONS` valid positions are selected or no candidates remain.
2. **Drop reasons.** Candidates dropped for missing ATR, zero shares, minimum position value, or cash constraints are recorded with ticker, stage, and human-readable reason.
3. **Risk labels.** CLI output distinguishes `ATR Impact $` from `Stop Risk $`; the portfolio summary uses `Target ATR Impact per Position` and `Total Nx ATR Stop Exposure`.
4. **ATR robustness.** Leading missing OHLC rows do not cause a ticker with enough valid recent rows to produce `NaN` ATR.
5. **Config compatibility.** Both `RISK_PER_TRADE` and `RISK_PER_TRADE_PCT` are accepted, with `RISK_PER_TRADE` preferred when both are present.
6. **Regression tests.** Unit tests cover leading-missing-row ATR handling and backfill/drop-reason behavior.
7. **Russell 1000 smoke test.** A cached Russell 1000 run that previously returned 17 positions now fills to 20 valid positions when enough candidates exist, and reports skipped zero-share candidates.
