"""
Data module for the Clenow momentum strategy.

This module provides a clean API for fetching market data from various sources
including Wikipedia (for S&P 500 tickers) and yfinance (for OHLCV data).

Main functions:
    get_sp500_tickers(): Get current S&P 500 ticker list
    get_stock_data(): Fetch OHLCV data for multiple stocks
    get_sp500_index_data(): Fetch S&P 500 index data
"""

from .cache import DataCache
from .provider import (
    get_sp500_index_data,
    get_sp500_tickers,
    get_stock_data,
)

__all__ = [
    "get_sp500_tickers",
    "get_stock_data",
    "get_sp500_index_data",
    "DataCache",
]
