# ADR 0001: Use a registry with dispatch for market universes

- **Issue**: The project hard-codes the S&P 500 in the constituent fetcher, the *regime-detection* benchmark (`SPY` in three places: `MarketDataSource.get_market_data`, `YFinanceMarketDataAdapter.get_market_data`, `MarketRegimeDetector._get_spy_data`), the index-data benchmark (`^GSPC`), and the disk cache key (magic `450 < len < 550` range + non-deterministic `hash(tuple(...))`). Adding a second universe (Russell 1000) without an abstraction would duplicate each of these. The codebase already has a partial abstraction — `IndexSymbol = Literal["SP500", "NASDAQ100", "DOW30", "RUSSELL2000"]` and `TickerSource.get_tickers_for_index(index)` — that we should extend rather than compete with.

- **Decision**: Introduce a `UniverseSpec` frozen dataclass and a `UNIVERSES` dict registry keyed by the existing uppercase `IndexSymbol` values (extended with `RUSSELL1000`). Each spec carries both a **regime-detection ETF** (`SPY`/`IWB`) and an **index quote symbol** (`^GSPC`/`^RUI`) — these are semantically distinct and previously conflated. Generic provider functions (`get_universe_tickers`, `get_benchmark_data`, `get_index_data`) and a strict Wikipedia fetcher dispatch on the spec. Selection is driven by `MARKET_UNIVERSE` env var (uppercased, validated against `UNIVERSES` via `get_universe_spec()`, not merely against the `IndexSymbol` vocabulary). Cache keys become universe- and data-kind-aware (`{SYMBOL}_{data_kind}_{period}`); the custom path switches from `hash()` to SHA-256 for cross-process determinism.

- **Status**: Amended post-review, approved (pending RFC 0001 v2 merge).

- **Group**: Data.

- **Assumptions**:
  - Wikipedia remains the primary constituent source for S&P 500 and Russell 1000.
  - yfinance supports `^RUI` (Russell 1000 index) and `IWB` (Russell 1000 ETF) alongside `^GSPC` / `SPY`.
  - The existing `convert_ticker_for_yfinance()` handles all punctuation quirks (`.` → `-`) universe-agnostically.
  - Users prefer env-var config over CLI flags, consistent with the rest of `utils/config.py`.
  - The existing `IndexSymbol` Literal and `get_tickers_for_index` interface are the canonical universe vocabulary — no parallel naming system is introduced.

- **Constraints**:
  - Must be additive on the public surface: `get_sp500_tickers()`, `get_sp500_index_data()`, and `TickerSource.get_sp500_tickers()` stay callable with unchanged semantics.
  - One-time cache invalidation on upgrade is acceptable (cache is a rebuildable optimization). Old key format is ignored and re-fetched; no corrupt reads.
  - Must be testable without live HTTP (HTML fixtures for the Wikipedia parser, mocked `requests`/`yfinance` elsewhere).
  - The active universe must control *every* benchmark consumer (regime detection, breadth, index-data provider) — no half-switched state.

- **Positions considered**:

  1. **Registry + dispatch (chosen).** One module (`data/universes.py`) holds every spec; one generalized fetcher reads from the spec; one generic provider function dispatches by name.

  2. **Per-universe modules** (`sp500_wikipedia.py`, `russell1000_wikipedia.py`, ...). Rejected: 90% duplicate scraping code per module, no central catalog, every new universe requires editing the provider too.

  3. **Subclass hierarchy** (`BaseUniverse` with `SP500Universe(BaseUniverse)`, etc.). Rejected: inheritance is overkill for what is essentially configuration data. A frozen dataclass is clearer and diff-friendly.

  4. **External config file (YAML/TOML)**. Rejected for now: adds a parsing layer, a schema, and a second place to look for universe definitions. Python dataclasses give us type checking and IDE completion for free. Revisit if users start adding universes faster than we review PRs.

- **Argument**: The registry pattern is proven in the Python ecosystem (zipline bundles, pytest markers, django apps). Implementation cost is low — one registry module, a generalized fetcher, and generic provider functions. The central abstraction lives in `data/`, while the integration path necessarily touches runtime orchestration and market analysis so the active universe controls every benchmark consumer. Back-compat is preserved because the existing S&P 500 entry points become one-line wrappers.

- **Implications**:
  - `DataCache._get_cache_key` gains optional `universe: IndexSymbol | None` and `data_kind: str | None` kwargs; the magic `450 < len < 550` range check goes away, named keys use `{SYMBOL}_{data_kind}_{period}`, and the custom path switches to SHA-256 for cross-process determinism.
  - Three currently-hard-coded `"SPY"` call sites (`MarketDataSource.get_market_data`, `YFinanceMarketDataAdapter.get_market_data`, `MarketRegimeDetector._get_spy_data`) become benchmark-ETF-from-spec. `_get_spy_data` is renamed internally to `_get_regime_benchmark_data`.
  - `TickerSource.get_sp500_tickers()` on the interface remains as a back-compat convenience; the primary surface (`get_tickers_for_index`) is already correct.
  - `MarketBreadthAnalyzer._get_tickers` switches from `ticker_source.get_sp500_tickers()` to `ticker_source.get_tickers_for_index(active_symbol)` — surfaces an implicit coupling we're making explicit.
  - Adding S&P 400, S&P 600, Nasdaq-100 is a one-entry change to the registry. `RUSSELL2000` remains registered in `IndexSymbol` but is not wired up in this ADR (its Wikipedia page is unreliable).
  - One-time cache wipe on upgrade (acceptable — rebuildable).

- **Related decisions**: none yet; this is ADR 0001.

- **Related requirements**: faithful Clenow methodology replication requires a broader-than-S&P-500 US universe (Russell 1000 is the canonical choice in the book). Regime detection must track the traded universe, not an unrelated benchmark.

- **Related artifacts**:
  - [RFC 0001: Switchable Market Universe](../rfcs/0001-switchable-market-universe.md)
  - `src/clenow_momentum/data/interfaces.py` (extend `IndexSymbol`)
  - `src/clenow_momentum/data/provider.py`
  - `src/clenow_momentum/data/sources/sp500_wikipedia.py` (→ generic `wikipedia_index.py`)
  - `src/clenow_momentum/data/sources/yfinance_adapter.py` (three SPY / sp500-dispatch sites)
  - `src/clenow_momentum/data/cache.py`
  - `src/clenow_momentum/market_analysis/regime_detector.py` (`_get_spy_data` → `_get_regime_benchmark_data`)
  - `src/clenow_momentum/market_analysis/market_breadth.py`
  - `src/clenow_momentum/utils/config.py`

- **Related principles**: prefer data-driven configuration over code branches; prefer additive changes over renames when the cost/benefit is unclear.

- **Notes**: The env var is only a *default*. Library callers can pass any registered name to `get_universe_tickers()` directly. This is intentional — the library function should not reach into global config.
