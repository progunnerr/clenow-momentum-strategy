# RFC 0001: Switchable Market Universe

- **Status**: Draft — amended after first review (2026-04-14)
- **Author**: Hartej Grewal
- **Date**: 2026-04-14
- **Related ADR**: [0001-universe-registry-dispatch](../adr/0001-universe-registry-dispatch.md)

## Changelog

- **2026-04-14 v2** (post-review). Four material changes:
  1. **Unified vocabulary**: reuse the existing `IndexSymbol` Literal (`SP500`, `NASDAQ100`, `DOW30`, `RUSSELL2000`) as the canonical universe key. Extend it with `RUSSELL1000`. Drop the proposed lowercase `"sp500"` / `"russell1000"` strings — a single vocabulary, not two.
  2. **Full benchmark plumbing**: the active universe now controls *both* constituents and the benchmark used for market-regime detection. Hard-coded `"SPY"` in `MarketDataSource.get_market_data`, `YFinanceMarketDataAdapter.get_market_data`, and `MarketRegimeDetector._get_spy_data` must be replaced by a benchmark resolved from the active `UniverseSpec`.
  3. **Deterministic cache hash**: replace Python's built-in `hash(tuple(...))` (non-stable across interpreter runs) with a SHA-256 hex digest over the sorted ticker list.
  4. **Stricter Wikipedia fallback**: "use the first table" is a silent correctness bet. Replace with: pick the first table whose columns contain one of `symbol_column_candidates` *and* whose row count falls inside `expected_row_range`. Fail loudly with a clear error if zero or more than one table matches.

## Summary
[summary]: #summary

Introduce a first-class notion of a *market universe* in the data layer. Add a `UniverseSpec` registry keyed by the existing `IndexSymbol` values, expose generic `get_universe_tickers(symbol)` / `get_benchmark_data(symbol)` entry points, and drive selection from a single `MARKET_UNIVERSE` environment variable. The active universe controls *both* the tradable constituent list *and* the benchmark series used by market-regime detection. Ship **Russell 1000** as the second supported universe alongside the existing S&P 500.

## Motivation
[motivation]: #motivation

The project today is hard-coded to the S&P 500 in several places:

1. The constituents fetcher (`data/sources/sp500_wikipedia.py`) knows one Wikipedia URL and one HTML table id.
2. The benchmark series for the strategy provider is hard-coded to `^GSPC` in `provider.get_sp500_index_data()`.
3. The benchmark series for *market-regime detection* is hard-coded to `SPY` in:
   - `MarketDataSource.get_market_data` ([interfaces.py:114](../../src/clenow_momentum/data/interfaces.py))
   - `YFinanceMarketDataAdapter.get_market_data` ([yfinance_adapter.py:94](../../src/clenow_momentum/data/sources/yfinance_adapter.py))
   - `MarketRegimeDetector._get_spy_data` ([regime_detector.py:96](../../src/clenow_momentum/market_analysis/regime_detector.py))
4. The disk cache (`data/cache.py`) uses a magic ticker-count range (`450 < len < 550`) as an implicit "this is the S&P 500" fingerprint to stabilize the cache key across minor constituent churn. Anything outside that range — Russell 1000 sits near 1,000 tickers — falls into a hash-based `custom_*` key that defeats the stabilization and collides with any ad-hoc ticker list of similar size. The hash itself is built with `hash(tuple(...))`, which is *not* stable across interpreter runs, silently breaking cache reuse across sessions when `PYTHONHASHSEED` is unset.

Clenow's *Stocks on the Move* methodology is written against a broader US universe than the S&P 500 — typically the Russell 1000 or the S&P 1500 composite. Restricting ourselves to the S&P 500 departs from the reference strategy and prevents running comparable backtests against each other. If we only half-switch the universe (tickers change but regime detection stays on SPY), we ship a subtly incorrect feature.

### Use cases

- **Faithful Clenow replication**: run the strategy against the Russell 1000, as in the book.
- **Backtest breadth comparison**: run the same strategy over S&P 500 vs Russell 1000 and compare risk/return with regime detection matched to each.
- **Faster iteration**: during development, switch to S&P 500 for faster runs; promote to Russell 1000 for validation.
- **Future extension**: the registry makes it a one-entry change to add S&P 400, S&P 600, Nasdaq-100, or an international index.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

A **universe** is the set of tickers the strategy considers, together with the benchmark series used for market-regime filtering. The project ships with two universes:

| IndexSymbol | Display | Regime benchmark (ETF) | Index quote | Approx. size |
|---|---|---|---|---|
| `SP500` | S&P 500 | `SPY` | `^GSPC` | ~503 |
| `RUSSELL1000` | Russell 1000 | `IWB` | `^RUI` | ~1,000 |

### Selecting a universe

Set `MARKET_UNIVERSE` in your `.env`:

```
MARKET_UNIVERSE=RUSSELL1000
```

Values are normalized to upper case and validated against the `UNIVERSES` registry via `get_universe_spec()`. If unset, the universe defaults to `SP500` (existing behavior — nothing changes for existing users). A value can exist in `IndexSymbol` but still be unavailable at runtime until it has a registered `UniverseSpec` (for example, deferred `RUSSELL2000` support).

Running `uv run python scripts/run_analysis.py` now reads the configured universe and uses it everywhere: constituents list, market-regime ETF (`SPY`/`IWB`), and benchmark index quote.

### Programmatic access

```python
from clenow_momentum.data import get_universe_tickers, get_benchmark_data

tickers = get_universe_tickers("RUSSELL1000")            # ~1000 US mid/large caps
benchmark = get_benchmark_data("RUSSELL1000", period="2y")  # IWB OHLCV (regime-detection ETF)
```

`get_sp500_tickers()` and `get_sp500_index_data()` remain as back-compat wrappers that delegate with `"SP500"`.

### Errors

Passing an unknown or unregistered universe raises a `ValueError` that lists registered names, so typos fail fast:

```
ValueError: unknown universe 'SPX'. Registered: ['SP500', 'RUSSELL1000']
```

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Vocabulary

There is exactly **one** universe naming system: the existing `IndexSymbol` Literal in `data/interfaces.py`. We extend it:

```python
# data/interfaces.py
IndexSymbol = Literal["SP500", "NASDAQ100", "DOW30", "RUSSELL1000", "RUSSELL2000"]
```

All public APIs (`get_universe_tickers`, `get_benchmark_data`, `TickerSource.get_tickers_for_index`, `UNIVERSES` registry keys, `MARKET_UNIVERSE` env var) use these uppercase symbols. No lowercase `"sp500"` strings anywhere.

### Registry

A new module `src/clenow_momentum/data/universes.py`:

```python
from dataclasses import dataclass
from .interfaces import IndexSymbol

@dataclass(frozen=True)
class UniverseSpec:
    symbol: IndexSymbol                         # "SP500"
    display_name: str                           # "S&P 500"
    wiki_url: str                               # full Wikipedia URL
    wiki_table_id: str | None                   # HTML table id, or None
    symbol_column_candidates: tuple[str, ...]   # ("Symbol", "Ticker")
    expected_row_range: tuple[int, int]         # (450, 560) for SP500
    benchmark_etf: str                          # "SPY"   (used by regime detection)
    benchmark_index: str                        # "^GSPC" (used by index data fetch)

UNIVERSES: dict[IndexSymbol, UniverseSpec] = {
    "SP500": UniverseSpec(
        symbol="SP500",
        display_name="S&P 500",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        wiki_table_id="constituents",
        symbol_column_candidates=("Symbol",),
        expected_row_range=(450, 560),
        benchmark_etf="SPY",
        benchmark_index="^GSPC",
    ),
    "RUSSELL1000": UniverseSpec(
        symbol="RUSSELL1000",
        display_name="Russell 1000",
        wiki_url="https://en.wikipedia.org/wiki/List_of_companies_in_the_Russell_1000_Index",
        wiki_table_id=None,
        symbol_column_candidates=("Symbol", "Ticker"),
        expected_row_range=(900, 1100),
        benchmark_etf="IWB",
        benchmark_index="^RUI",
    ),
}

def get_universe_spec(symbol: str) -> UniverseSpec:
    key = symbol.upper()
    if key not in UNIVERSES:
        raise ValueError(
            f"unknown universe {symbol!r}. Registered: {list(UNIVERSES)}"
        )
    return UNIVERSES[key]  # type: ignore[index]
```

### Generalized Wikipedia fetcher (strict)

`data/sources/sp500_wikipedia.py` becomes a generic `wikipedia_index.py` (with the old module kept as a back-compat shim). The fetcher selects the right table by filter, not by "first":

```python
def fetch_index_tickers_from_wikipedia(spec: UniverseSpec, *, timeout=10, headers=None) -> list[str]:
    # 1. If spec.wiki_table_id is set, find <table id=...>. Require match.
    # 2. Otherwise: parse all tables via pd.read_html. Select tables where:
    #       - at least one of spec.symbol_column_candidates is in df.columns, AND
    #       - spec.expected_row_range[0] <= len(df) <= spec.expected_row_range[1]
    #    If zero match: raise ValueError listing table shapes/columns seen.
    #    If multiple match: raise ValueError listing the ambiguous candidates.
    # 3. Pick the first symbol_column_candidate present in the chosen table.
    # 4. Return the ticker list.
```

This removes the silent "first table" correctness bet. Russell 1000's Wikipedia layout can shift without silently corrupting downstream data — the parser either finds one unambiguous table or fails with actionable error text.

### Provider

`data/provider.py` gains:

```python
def get_universe_tickers(symbol: IndexSymbol, use_cache=True, max_age_hours=24) -> list[str]: ...
def get_benchmark_data(symbol: IndexSymbol, period="1y", use_cache=True) -> pd.DataFrame | None: ...
def get_index_data(symbol: IndexSymbol, period="1y", use_cache=True) -> pd.DataFrame | None: ...
```

Two distinct entry points for benchmarks:
- `get_benchmark_data(symbol)` returns the **ETF** (`SPY` for SP500, `IWB` for RUSSELL1000). This is what regime detection and market diagnostics consume.
- `get_index_data(symbol)` is the provider-level registered-universe helper that returns the **index quote** (`^GSPC`, `^RUI`). This is what the S&P 500 index legacy path uses.

Market breadth does not consume the benchmark ETF. It consumes the active constituent list plus stock OHLCV data, so its switch point is `ticker_source.get_tickers_for_index(active_symbol)`.

Per-universe disk cache files: `data/cache/{SYMBOL}_tickers.pkl`. `get_sp500_tickers()` / `get_sp500_index_data()` become one-liners delegating to `"SP500"`.

### Cache keying (deterministic)

`data/cache.py` changes:

- **Drop** the `450 < len < 550` magic range.
- **Add** optional `universe: IndexSymbol | None` and `data_kind: str | None` kwargs to `DataCache.get()`, `DataCache.save()`, and `_get_cache_key()`.
- **Named-universe path**: key is `{SYMBOL}_{data_kind}_{period}` (stable across minor constituent churn and separated by data purpose).
  - Universe OHLCV: `{SYMBOL}_universe_{period}`
  - Regime ETF benchmark OHLCV: `{SYMBOL}_benchmark_etf_{period}`
  - Index quote OHLCV: `{SYMBOL}_benchmark_index_{period}`
- **Custom path** (no universe): replace `hash(tuple(sorted(tickers)))` with `hashlib.sha256("\n".join(sorted(tickers)).encode()).hexdigest()[:16]`. Deterministic across processes, `PYTHONHASHSEED`-independent, no more invisible cache misses between runs.

Migration: existing cache files under old keys are ignored (treated as cache miss and re-fetched). No in-place rewrite; acceptable because the cache is a rebuildable optimization, not data of record.

### Config

`utils/config.load_config()` reads `MARKET_UNIVERSE`, uppercases it, and stores it as `config["universe"]` (type `IndexSymbol`). Default `"SP500"`. `validate_config()` asserts the value is in `UNIVERSES` via a lazy import. This is runtime availability validation, not just type-vocabulary validation; deferred symbols like `RUSSELL2000` remain invalid until registered.

### Call sites — full plumbing

| File | Current | After |
|---|---|---|
| `scripts/run_analysis.py:330` | `get_sp500_tickers()` | `get_universe_tickers(config["universe"])` |
| `scripts/warm_cache.py:27` | `get_sp500_tickers()` | `get_universe_tickers(config["universe"])` |
| `data/interfaces.py:114` `MarketDataSource.get_market_data` | hard-coded `"SPY"` | takes `symbol` / reads active universe spec; resolves `spec.benchmark_etf` and calls the raw-symbol data-source method |
| `data/sources/yfinance_adapter.py:94` `YFinanceMarketDataAdapter.get_market_data` | hard-coded `"SPY"` | same — resolves ETF from active `UniverseSpec` |
| `data/sources/yfinance_adapter.py:120` (ticker dispatch) | `get_sp500_tickers_from_wikipedia()` | dispatches through `fetch_index_tickers_from_wikipedia(get_universe_spec(symbol))` |
| `market_analysis/regime_detector.py:96` `_get_spy_data` | hard-coded `"SPY"` | renamed to `_get_regime_benchmark_data`; takes universe or pulls from injected `UniverseSpec` |
| `market_analysis/market_breadth.py:148` | `ticker_source.get_sp500_tickers()` | `ticker_source.get_tickers_for_index(active_symbol)` |
| `data/cache.py:55-73, 107-140` | magic-range + unstable hash | universe-aware key + SHA-256 hash |

Back-compat surface kept: `get_sp500_tickers()`, `get_sp500_index_data()`, `TickerSource.get_sp500_tickers()` all remain as one-line wrappers. The internal `_get_spy_data` is an underscore method so the rename is safe.

### Corner cases

- **Column name drift**: `symbol_column_candidates` tries each candidate in order; first match wins. Russell 1000 has used both `Symbol` and `Ticker`.
- **Dot tickers**: `convert_ticker_for_yfinance()` already maps `BRK.B → BRK-B` universe-agnostically. No change.
- **Benchmark availability**: `^RUI` and `IWB` both exist on yfinance; verified at smoke-test time (see Verification).
- **Row-range too tight**: `expected_row_range` has ±10% margin around the canonical count. If a membership rebalance pushes past the margin, the parser fails loudly with a clear message — better than silently fetching an adjacent calendar-year table.

## Drawbacks
[drawbacks]: #drawbacks

- **Wikipedia scraping remains fragile per-universe.** Each new universe inherits the same HTML-parsing risk. Mitigated by the strict table-selection filter, which fails loudly rather than silently corrupting.
- **Survivorship bias.** We still only see current constituents. Out of scope — separate RFC.
- **No point-in-time membership.** Same caveat as today.
- **Interface-name debt is still only partially addressed.** `TickerSource.get_sp500_tickers()` stays as a back-compat method on the interface. The interface's *primary* surface (`get_tickers_for_index`) is already correct; the convenience method is debt we live with to avoid churning tests and docs.
- **Cache reset on upgrade.** Because we're changing the key format, first run on existing installs re-fetches. Acceptable — it's tickers + prices, both cheap to redownload.
- **Env-var-only selection** is still a library-level default; individual callers can pass any registered symbol to `get_universe_tickers()`. Intentional, but operators must understand the env var is not a hard global.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

### Why reuse `IndexSymbol` instead of introducing lowercase names?
The codebase already speaks in `SP500`/`NASDAQ100`/`DOW30`/`RUSSELL2000`. Adding a second (lowercase) vocabulary would force every boundary crossing to translate between them — the exact kind of duplication this RFC set out to remove.

### Why two benchmark fields (`benchmark_etf` vs `benchmark_index`)?
Market-regime detection uses a **tradable ETF** (SPY, IWB) because it's what the code treats as OHLCV for signal generation. The legacy S&P 500 provider entry point exposes the **index quote** (`^GSPC`, `^RUI`) for display and analytics. Conflating them breaks either semantics. Keeping both fields explicit is cheap.

### Why a registry rather than per-universe modules?
A registry centralizes the catalog, parametrizes tests trivially, and keeps the fetcher single-responsibility. Per-universe modules would duplicate 90% of the scraping logic.

### Why env var rather than CLI flag?
Matches the existing `utils/config.load_config()` pattern. A CLI flag is additive later.

### Why keep back-compat wrappers?
The rename is multi-file churn for no runtime gain. Deferred.

### Impact of not doing this
We can't run Clenow's methodology as written. We keep a silent cross-session cache-miss bug. And any "switch to Russell 1000" attempt would ship a half-switched feature where regime detection still runs on SPY.

## Prior art
[prior-art]: #prior-art

- **Zipline** data bundles — registry pattern, keyed by name.
- **Backtrader** stores + feeds — more general, heavier.
- **QuantConnect / Lean** universes as runtime-selectable objects.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- Should `market_analysis/market_breadth.py` be universe-aware, or is "whatever `TickerSource` was injected" the right semantic? Today it's implicit — we'll thread the active symbol through explicitly in this RFC.
- Do we ship S&P 400 + S&P 600 + Nasdaq-100 in the same PR to prove the registry is genuinely general?
- Should the S&P 500 regime ETF be per-user configurable (`SPY` vs `VOO` vs `IVV`)? Not blocking.
- Do we back-fill `RUSSELL2000` (already in `IndexSymbol`) as part of this RFC, or defer? Russell 2000's Wikipedia page is unreliable; probably needs an ETF-holdings (IWM CSV) source. Defer.

## Future possibilities
[future-possibilities]: #future-possibilities

- **International indexes**: FTSE 100, DAX, CAC 40, Nikkei 225, Hang Seng, ASX 200, TSX 60. Need suffix-aware tickers (`.L`, `.DE`, `.PA`, `.T`, `.HK`, `.AX`, `.TO`), plus session/currency awareness in the engine. Registry is ready; the engine isn't.
- **More US indexes**: S&P 400, S&P 600, Nasdaq-100, Dow 30 — all near-free once this lands.
- **Non-Wikipedia sources**: ETF holdings CSV (iShares IWM for Russell 2000, IWB for Russell 1000 as a more stable alternative to Wikipedia). Registry would grow a `source_type` discriminator.
- **Point-in-time constituents**: separate RFC.
- **Custom user universes**: register an arbitrary ticker list under a name.

## Acceptance criteria
[acceptance-criteria]: #acceptance-criteria

The implementation PR must satisfy all of:

1. **Whole-stack switching.** Setting `MARKET_UNIVERSE=RUSSELL1000` switches constituents, regime-detection benchmark (SPY→IWB), and index-data benchmark (`^GSPC`→`^RUI`). No code path still references `"SPY"` or `"^GSPC"` as a hard-coded string outside the `UNIVERSES` registry.
2. **Single vocabulary.** All universe keys across `IndexSymbol`, `UNIVERSES`, `MARKET_UNIVERSE`, public provider functions, and tests use the uppercase `IndexSymbol` values. No lowercase `"sp500"` strings introduced.
3. **Back-compat preserved.** `get_sp500_tickers()`, `get_sp500_index_data()`, and `TickerSource.get_sp500_tickers()` return identical results to pre-change, verified by the existing test suite passing unchanged.
4. **Deterministic cache.** `DataCache`'s custom-path key is SHA-256-based and stable across Python processes (test: two subprocesses produce identical keys for the same sorted ticker list).
5. **Universe-aware cache keys.** No magic ticker-count range in `_get_cache_key`. Named path includes the data kind: `{SYMBOL}_{data_kind}_{period}` (for example, `SP500_universe_1y`, `SP500_benchmark_etf_1y`, `SP500_benchmark_index_1y`). Custom path: `custom_{n}tickers_{period}_{sha256[:16]}`.
6. **Strict Wikipedia parser.** Fetcher raises a clear `ValueError` when zero or multiple tables match the spec's filter; never silently picks the first.
7. **Test coverage**:
   - `test_universes.py`: registry lookup, unknown-universe error, spec shape.
   - `test_wikipedia_parser.py`: fixture-based tests for SP500 and Russell 1000 HTML (including an ambiguous-multi-table fixture that must fail).
   - `test_cache.py`: universe-keyed path, custom SHA-256 path, cross-process determinism.
   - `test_provider.py`: parametrized `get_universe_tickers` / `get_benchmark_data` over both universes.
   - Regime detector test that asserts the benchmark ETF is resolved from the active spec, not `"SPY"`.
8. **Smoke test** (manual, documented in PR description):
   - `MARKET_UNIVERSE=RUSSELL1000 uv run python scripts/run_analysis.py` completes, logs show `Russell 1000`, ~1000 tickers, and `IWB`/`^RUI` as benchmarks.
   - `data/cache/RUSSELL1000_tickers.pkl` and `data/cache/SP500_tickers.pkl` coexist without collision.
