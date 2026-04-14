"""
Data module for the Clenow momentum strategy.

This module provides a clean API for fetching market data from various sources
for the S&P 500 market universe. Uses Wikipedia (for market universe constituents)
and yfinance (for OHLCV data).

Main functions:
    get_sp500_tickers(): Get current S&P 500 market universe constituents
    get_stock_data(): Fetch OHLCV data for market universe securities
    get_sp500_index_data(): Fetch S&P 500 index data (market benchmark)
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
    get_sp500_index_data,
    get_sp500_tickers,
    get_stock_data,
)

__all__ = [
    # Main functions
    "get_sp500_tickers",
    "get_stock_data",
    "get_sp500_index_data",
    # Caching
    "DataCache",
    "CachedDataSource",
    # Interfaces
    "MarketDataSource",
    "TickerSource",
    "IndexSymbol",
    "DataSourceError",
]
