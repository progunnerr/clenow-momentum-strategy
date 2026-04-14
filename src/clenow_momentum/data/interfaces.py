"""Data source interfaces for market data access.

These interfaces define contracts for accessing market data from various sources
(yfinance, Bloomberg, Alpha Vantage, etc.) without coupling business logic to
specific implementations.

Co-located with data module for easier navigation and maintenance.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

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
        """Get OHLCV data for market universe securities.

        Args:
            tickers: List of ticker symbols from the market universe
            period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start_date: Start date (alternative to period)
            end_date: End date (alternative to period)

        Returns:
            MultiIndex DataFrame with (ticker, price_type) columns

        Raises:
            DataSourceError: If market universe data cannot be retrieved
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

    def get_market_data(self, period: str = "1y") -> pd.DataFrame:
        """Get market benchmark data (SPY for S&P 500).
        
        Convenience method that wraps get_index_data for SPY.
        
        Args:
            period: Period string
            
        Returns:
            DataFrame with SPY OHLCV data
        """
        return self.get_index_data("SPY", period=period)


# Supported market universe indices
IndexSymbol = Literal["SP500", "NASDAQ100", "DOW30", "RUSSELL2000"]


class TickerSource(ABC):
    """Abstract interface for market universe ticker sources.

    Provides access to market universe constituents for various indices
    without coupling to specific implementations. This generic approach
    allows adding new indices without breaking existing implementations.
    """

    @abstractmethod
    def get_tickers_for_index(self, index: IndexSymbol) -> list[str]:
        """Get constituents for a given market index.

        Args:
            index: The symbol of the index to retrieve (e.g., 'SP500', 'NASDAQ100')

        Returns:
            List of ticker symbols in the specified market universe

        Raises:
            DataSourceError: If market universe constituents cannot be retrieved
            NotImplementedError: If the index is not supported by this source
        """
        pass

    @abstractmethod
    def get_supported_indices(self) -> list[IndexSymbol]:
        """Get list of indices supported by this ticker source.

        Returns:
            List of index symbols that this source can provide
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the ticker source is available.

        Returns:
            True if ticker source can be accessed
        """
        pass

    def is_index_supported(self, index: IndexSymbol) -> bool:
        """Check if a specific index is supported by this source.

        Args:
            index: Index symbol to check

        Returns:
            True if the index is supported
        """
        return index in self.get_supported_indices()

    # Convenience methods for backward compatibility
    def get_sp500_tickers(self) -> list[str]:
        """Get S&P 500 market universe constituents.
        
        Convenience method that wraps get_tickers_for_index('SP500').
        
        Returns:
            List of ticker symbols in the S&P 500 market universe
            
        Raises:
            DataSourceError: If market universe constituents cannot be retrieved
            NotImplementedError: If SP500 is not supported by this source
        """
        return self.get_tickers_for_index("SP500")
        
    def get_nasdaq100_tickers(self) -> list[str]:
        """Get NASDAQ 100 ticker symbols.
        
        Convenience method that wraps get_tickers_for_index('NASDAQ100').
        
        Returns:
            List of ticker symbols in the NASDAQ 100
            
        Raises:
            DataSourceError: If ticker list cannot be retrieved
            NotImplementedError: If NASDAQ100 is not supported by this source
        """
        return self.get_tickers_for_index("NASDAQ100")


class DataSourceError(Exception):
    """Exception raised when data cannot be retrieved from a source."""

    def __init__(self, message: str, source: str = "", original_error: Exception = None):
        self.message = message
        self.source = source
        self.original_error = original_error
        super().__init__(f"{source}: {message}" if source else message)


