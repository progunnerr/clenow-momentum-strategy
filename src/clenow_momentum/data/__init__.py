"""Data module for the Clenow momentum strategy.

Provides a clean API for fetching market data from various sources.
Supports multiple market universes via the universe registry.

Generic entry points (universe-aware):
    get_universe_tickers(symbol)  — constituents for any registered universe
    get_universe_constituents(symbol) — constituents with metadata
    get_benchmark_data(symbol)    — regime-detection ETF OHLCV
    get_index_data(symbol)        — index-quote OHLCV
    get_stock_data(tickers)       — OHLCV for an arbitrary ticker list

Back-compat wrappers (S&P 500, unchanged semantics):
    get_sp500_tickers()           — delegates to get_universe_tickers("SP500")
    get_sp500_index_data()        — delegates to get_index_data("SP500")

Universe registry:
    UNIVERSES                     — dict of registered UniverseSpec objects
    get_universe_spec(symbol)     — look up a spec by IndexSymbol
"""

from .cache import DataCache
from .caching import CachedDataSource
from .interfaces import (
    DataSourceError,
    IndexSymbol,
    MarketDataSource,
    TickerSource,
)
from .provider import (
    get_benchmark_data,
    get_index_data,
    get_sp500_index_data,
    get_sp500_tickers,
    get_stock_data,
    get_universe_constituents,
    get_universe_tickers,
)
from .universes import UNIVERSES, UniverseSpec, get_universe_spec

__all__ = [
    # Generic universe-aware entry points
    "get_universe_tickers",
    "get_universe_constituents",
    "get_benchmark_data",
    "get_index_data",
    "get_stock_data",
    # Back-compat wrappers
    "get_sp500_tickers",
    "get_sp500_index_data",
    # Universe registry
    "UNIVERSES",
    "UniverseSpec",
    "get_universe_spec",
    # Caching
    "DataCache",
    "CachedDataSource",
    # Interfaces
    "MarketDataSource",
    "TickerSource",
    "IndexSymbol",
    "DataSourceError",
]
