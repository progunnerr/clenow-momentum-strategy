# RFC 0003: Universe Constituent Metadata

## Status

Accepted

## Context

The strategy already fetches constituent tables from Wikipedia for supported
universes, but the runtime only keeps ticker symbols. That is enough for price
downloads, momentum, filtering, and position sizing, but it leaves the final
portfolio output without useful context such as company name and sector.

Russell 1000 and S&P 500 Wikipedia constituent tables include metadata columns
such as company/security name and GICS sector. Since this metadata belongs to
the universe definition rather than to price data, it should be parsed and
cached as part of the universe constituent pipeline.

This RFC is intentionally separate from RFC 0002. RFC 0002 covers portfolio
sizing, risk labels, and backfilling. This RFC covers constituent metadata and
diagnostic columns shown in the portfolio table.

## Decision

Add a universe metadata path that returns normalized constituent records with:

- `ticker`: yfinance-normalized ticker used by analysis and joins.
- `source_symbol`: raw symbol as listed by the source table.
- `company_name`: source company/security name when available.
- `sector`: source sector when available.

The existing ticker-only API remains compatible. `get_universe_tickers()` will
continue returning a list of normalized tickers, while a new
`get_universe_constituents()` API returns the richer metadata DataFrame.

## Universe Schema

`UniverseSpec` will define optional metadata column candidates:

- `company_column_candidates`: `("Security", "Company", "Company Name", "Name")`
- `sector_column_candidates`: `("GICS Sector", "Sector")`

The parser will use `symbol_column_candidates` for the required source symbol
and the metadata candidates for optional company and sector values.

Missing company or sector columns are non-fatal. The parser should log a
warning and return missing values for the unavailable metadata field.

## Cache Behavior

Full constituent metadata is cached in:

`data/cache/{SYMBOL}_constituents.pkl`

The cache payload stores:

- `constituents`: DataFrame with `ticker`, `source_symbol`, `company_name`,
  and `sector`.
- `timestamp`: cache creation time.
- `count`: number of constituent records.

The legacy ticker cache remains compatible:

`data/cache/{SYMBOL}_tickers.pkl`

When fresh constituent metadata is fetched, the provider may update both the
metadata cache and the legacy ticker cache. Existing callers that only need
tickers should not need code changes.

## Portfolio Diagnostics

The main portfolio table will include these additional columns:

- `Company`
- `Sector`
- `R²`
- `90d Return %`
- `Price vs MA %`

The diagnostics come from existing analysis outputs where available:

- `r_squared`
- `period_return_pct`
- `price_vs_ma`

The metadata comes from the universe constituent cache:

- `company_name`
- `sector`

Missing values should display as `N/A` and should not block analysis or
portfolio construction.

Long company and sector names should be shortened for table readability without
changing the underlying stored values.

## Acceptance Criteria

- S&P 500 and Russell 1000 Wikipedia parser tests cover company and GICS sector
  extraction.
- `get_universe_constituents()` returns normalized `ticker`, preserves
  `source_symbol`, and includes `company_name` and `sector`.
- Dot-style source tickers such as `BRK.B` join through normalized yfinance
  tickers such as `BRK-B`.
- `get_universe_tickers()` continues returning a list of normalized tickers.
- Portfolio construction preserves optional metadata and diagnostics through
  sizing.
- The main portfolio table shows `Company`, `Sector`, `R²`, `90d Return %`,
  and `Price vs MA %`.
- Missing metadata displays as `N/A` and does not fail the run.
