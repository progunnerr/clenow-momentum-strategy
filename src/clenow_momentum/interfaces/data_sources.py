"""Data source interfaces for market data access.

These interfaces define contracts for accessing market data from various sources
(yfinance, Bloomberg, Alpha Vantage, etc.) without coupling business logic to
specific implementations.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    pass


class MarketDataSource(ABC):
    """Abstract interface for market data sources.

    Provides standardized access to stock and index data regardless of
    the underlying data provider (yfinance, Bloomberg, etc.).
    """

    @abstractmethod
    def get_stock_data(
        self,
        tickers: list[str],
        period: str = "1y",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for multiple stocks.

        Args:
            tickers: List of stock symbols
            period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start_date: Start date (alternative to period)
            end_date: End date (alternative to period)

        Returns:
            MultiIndex DataFrame with (ticker, price_type) columns

        Raises:
            DataSourceError: If data cannot be retrieved
        """
        pass

    @abstractmethod
    def get_index_data(
        self,
        symbol: str,
        period: str = "1y",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for a market index.

        Args:
            symbol: Index symbol (e.g., 'SPY', '^GSPC')
            period: Period string
            start_date: Start date (alternative to period)
            end_date: End date (alternative to period)

        Returns:
            DataFrame with OHLCV columns

        Raises:
            DataSourceError: If data cannot be retrieved
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available.

        Returns:
            True if data source can be accessed
        """
        pass

    def get_current_price(self, ticker: str) -> float | None:
        """Get current/latest price for a ticker.

        Default implementation fetches recent data and takes last close price.
        Override for real-time data sources.

        Args:
            ticker: Stock symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            data = self.get_stock_data([ticker], period="1d")
            if data.empty:
                return None

            if isinstance(data.columns, pd.MultiIndex):
                close_col = (ticker, "Close")
                if close_col in data.columns:
                    return float(data[close_col].iloc[-1])
            else:
                if "Close" in data.columns:
                    return float(data["Close"].iloc[-1])

            return None
        except Exception:
            return None


class TickerSource(ABC):
    """Abstract interface for ticker list sources.

    Provides access to predefined lists of stocks (S&P 500, NASDAQ 100, etc.)
    from various sources without coupling to specific implementations.
    """

    @abstractmethod
    def get_sp500_tickers(self) -> list[str]:
        """Get current S&P 500 ticker symbols.

        Returns:
            List of S&P 500 ticker symbols

        Raises:
            DataSourceError: If ticker list cannot be retrieved
        """
        pass

    @abstractmethod
    def get_nasdaq100_tickers(self) -> list[str]:
        """Get current NASDAQ 100 ticker symbols.

        Returns:
            List of NASDAQ 100 ticker symbols

        Raises:
            DataSourceError: If ticker list cannot be retrieved
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the ticker source is available.

        Returns:
            True if ticker source can be accessed
        """
        pass


class DataSourceError(Exception):
    """Exception raised when data cannot be retrieved from a source."""

    def __init__(self, message: str, source: str = "", original_error: Exception = None):
        self.message = message
        self.source = source
        self.original_error = original_error
        super().__init__(f"{source}: {message}" if source else message)


class CachedDataSource(MarketDataSource):
    """Base class for data sources with caching capabilities.

    Provides common caching functionality that can be mixed with
    any concrete data source implementation.
    """

    def __init__(self, cache_ttl_hours: int = 24):
        self.cache_ttl_hours = cache_ttl_hours
        self._cache: dict[str, dict[str, Any]] = {}

    def _get_cache_key(self, tickers: list[str], period: str) -> str:
        """Generate cache key for request parameters."""
        tickers_str = ",".join(sorted(tickers))
        return f"{tickers_str}_{period}"

    def _is_cache_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        if "timestamp" not in cache_entry:
            return False

        cache_age_hours = (datetime.now(UTC) - cache_entry["timestamp"]).total_seconds() / 3600
        return cache_age_hours < self.cache_ttl_hours

    def _get_from_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Get data from cache if available and valid."""
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry["data"]
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache with timestamp."""
        self._cache[cache_key] = {"data": data.copy(), "timestamp": datetime.now(UTC)}
