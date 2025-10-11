"""YFinance adapter implementing MarketDataSource interface.

This adapter wraps yfinance to implement the MarketDataSource interface,
allowing market analysis components to work with yfinance through the
abstract interface.
"""

from datetime import datetime

import pandas as pd
import yfinance as yf
from loguru import logger

from ...interfaces.data_sources import DataSourceError, MarketDataSource, TickerSource


class YFinanceMarketDataAdapter(MarketDataSource):
    """YFinance implementation of MarketDataSource interface."""

    def get_stock_data(
        self,
        tickers: list[str],
        period: str = "1y",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for multiple stocks via yfinance."""
        try:
            logger.debug(f"Fetching stock data for {len(tickers)} tickers...")
            data = yf.download(
                tickers,
                period=period,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if data is None or data.empty:
                raise DataSourceError("No data returned for tickers", source="yfinance")

            return data

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise DataSourceError(
                f"Failed to fetch stock data: {str(e)}",
                source="yfinance",
                original_error=e,
            )

    def get_index_data(
        self,
        symbol: str,
        period: str = "1y",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for a market index."""
        try:
            logger.debug(f"Fetching index data for {symbol}...")
            data = yf.download(
                symbol,
                period=period,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )

            if data is None or data.empty:
                raise DataSourceError(
                    f"No data returned for {symbol}", source="yfinance"
                )

            # Handle potential MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten to single level
                data.columns = data.columns.get_level_values(0)

            return data

        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            raise DataSourceError(
                f"Failed to fetch index data: {str(e)}",
                source="yfinance",
                original_error=e,
            )

    def get_market_data(self, period: str = "1y") -> pd.DataFrame:
        """Get S&P 500 (SPY) market data.

        This is a convenience method specifically for market regime analysis.
        """
        return self.get_index_data("SPY", period=period)

    def is_available(self) -> bool:
        """Check if yfinance is available."""
        try:
            # Try to fetch a small amount of data to check availability
            test_data = yf.download("SPY", period="1d", progress=False)
            return test_data is not None and not test_data.empty
        except Exception:
            return False


class WikipediaTickerAdapter(TickerSource):
    """Wikipedia S&P 500 ticker source implementation."""

    def get_sp500_tickers(self) -> list[str]:
        """Get S&P 500 tickers from Wikipedia."""
        try:
            from .sp500_wikipedia import get_sp500_tickers_wikipedia

            tickers = get_sp500_tickers_wikipedia()
            if not tickers:
                raise DataSourceError(
                    "No tickers returned from Wikipedia", source="wikipedia"
                )

            return tickers

        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            raise DataSourceError(
                f"Failed to fetch tickers: {str(e)}",
                source="wikipedia",
                original_error=e,
            )

    def get_nasdaq100_tickers(self) -> list[str]:
        """Get NASDAQ 100 tickers (not implemented)."""
        raise NotImplementedError("NASDAQ 100 ticker fetching not implemented")

    def is_available(self) -> bool:
        """Check if Wikipedia is available."""
        try:
            tickers = self.get_sp500_tickers()
            return len(tickers) > 0
        except Exception:
            return False
