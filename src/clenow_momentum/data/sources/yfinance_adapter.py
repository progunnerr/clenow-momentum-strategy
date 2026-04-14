"""YFinance adapter implementing MarketDataSource and TickerSource interfaces.

This adapter wraps yfinance to implement the MarketDataSource interface,
allowing market analysis components to work with yfinance through the
abstract interface.
"""

from datetime import datetime

import pandas as pd
import yfinance as yf
from loguru import logger

from ..interfaces import DataSourceError, IndexSymbol, MarketDataSource, TickerSource


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
            ) from e

    def get_index_data(
        self,
        symbol: str,
        period: str = "1y",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data for any market index or ETF symbol."""
        try:
            logger.debug(f"Fetching index data for {symbol}...")
            data = yf.download(
                symbol,
                period=period,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )

            if data is None or data.empty:
                raise DataSourceError(
                    f"No data returned for {symbol}", source="yfinance"
                )

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data

        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            raise DataSourceError(
                f"Failed to fetch index data: {str(e)}",
                source="yfinance",
                original_error=e,
            ) from e

    def get_market_data(self, period: str = "1y", benchmark_ticker: str = "SPY") -> pd.DataFrame:
        """Get market benchmark data for regime detection.

        Benchmark technicals use unadjusted Close values so moving averages
        match standard broker/chart displays. Universe stock data remains
        adjusted through get_stock_data() for momentum calculations.

        Args:
            period:           yfinance period string.
            benchmark_ticker: ETF or index symbol (default "SPY").
                              Pass the active universe's benchmark_etf here
                              (e.g. "IWB" for Russell 1000).
        """
        return self.get_index_data(benchmark_ticker, period=period)

    def is_available(self) -> bool:
        """Check if yfinance is importable and minimally functional."""
        try:
            return hasattr(yf, "download")
        except Exception:
            return False


class WikipediaTickerAdapter(TickerSource):
    """Wikipedia ticker source implementation.

    Supports all universes registered in data/universes.py.
    """

    def get_tickers_for_index(self, index: IndexSymbol) -> list[str]:
        """Get constituents for a given market index.

        Dispatches through the universe registry so any registered universe
        (SP500, RUSSELL1000, …) is supported without code changes here.

        Args:
            index: IndexSymbol value (e.g. "SP500", "RUSSELL1000").

        Returns:
            List of yfinance-normalised ticker symbols.

        Raises:
            DataSourceError: If fetch fails or index is not registered.
            NotImplementedError: If the index is not in the universe registry.
        """
        try:
            from ..universes import get_universe_spec
            from .sp500_wikipedia import fetch_index_tickers_from_wikipedia
            from .yfinance_source import convert_ticker_for_yfinance
        except ImportError as e:
            raise DataSourceError(
                f"Import error in WikipediaTickerAdapter: {e}", source="wikipedia"
            ) from e

        try:
            spec = get_universe_spec(index)
        except ValueError:
            from ..universes import UNIVERSES
            raise NotImplementedError(
                f"Index '{index}' is not registered in the universe registry. "
                f"Registered: {list(UNIVERSES)}"
            ) from None

        try:
            raw_tickers = fetch_index_tickers_from_wikipedia(spec)
            if not raw_tickers:
                raise DataSourceError(
                    f"No tickers returned from Wikipedia for {spec.display_name}",
                    source="wikipedia",
                )
            return [convert_ticker_for_yfinance(t) for t in raw_tickers]
        except Exception as e:
            logger.error(f"Error fetching {spec.display_name} tickers: {e}")
            raise DataSourceError(
                f"Failed to fetch {index} tickers: {str(e)}",
                source="wikipedia",
                original_error=e,
            ) from e

    def get_supported_indices(self) -> list[IndexSymbol]:
        """Get list of indices supported by this ticker source (all registered universes)."""
        try:
            from ..universes import UNIVERSES
            return list(UNIVERSES.keys())  # type: ignore[return-value]
        except ImportError:
            return ["SP500"]

    def is_available(self) -> bool:
        """Check if Wikipedia ticker fetching dependencies are importable."""
        try:
            import requests
            resp = requests.head("https://en.wikipedia.org", timeout=5)
            return resp.status_code < 500
        except Exception:
            return False
