"""Market breadth analysis for Clenow momentum strategy.

This module focuses solely on calculating market breadth metrics (percentage of
stocks above moving averages), following the Single Responsibility Principle.
"""

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from ..data.interfaces import DataSourceError, MarketDataSource, TickerSource
from .debug_utils import DebugDataManager


@dataclass(frozen=True)
class BreadthMetrics:
    """Market breadth analysis results.

    Immutable value object containing all breadth calculation results.
    """

    breadth_pct: float
    above_ma: int
    below_ma: int
    total_checked: int
    no_data: int
    sample_size: int
    breadth_strength: str
    ma_period: int
    error: str | None = None


class MarketBreadthAnalyzer:
    """Analyzes market breadth by calculating percentage of stocks above moving averages.

    Market breadth is a key indicator of market health - high breadth indicates
    a healthy bull market, while low breadth suggests weakness.
    """

    def __init__(
        self,
        market_data_source: MarketDataSource,
        ticker_source: TickerSource,
        debug_manager: DebugDataManager | None = None,
    ):
        """Initialize market breadth analyzer.

        Args:
            market_data_source: Data source for stock data
            ticker_source: Source for ticker lists
            debug_manager: Optional debug data manager
        """
        self.market_data_source = market_data_source
        self.ticker_source = ticker_source
        self.debug_manager = debug_manager or DebugDataManager()

    def calculate_breadth(
        self,
        ma_period: int = 200,
        tickers: list[str] | None = None,
        stock_data: pd.DataFrame | None = None,
    ) -> BreadthMetrics:
        """Calculate market breadth - percentage of stocks above their moving average.

        Args:
            ma_period: Moving average period (default 200)
            tickers: List of tickers (if None, will fetch S&P 500)
            stock_data: Pre-fetched stock data (optional, for efficiency)

        Returns:
            BreadthMetrics with breadth statistics
        """
        logger.info(f"Calculating market breadth (% of stocks above {ma_period}-day MA)...")

        try:
            # Get tickers if not provided
            if tickers is None:
                tickers = self._get_tickers()
                if not tickers:
                    return BreadthMetrics(
                        breadth_pct=0.0,
                        above_ma=0,
                        below_ma=0,
                        total_checked=0,
                        no_data=0,
                        sample_size=0,
                        breadth_strength="Unknown",
                        ma_period=ma_period,
                        error="Could not fetch tickers",
                    )

            # Get stock data if not provided
            if stock_data is None:
                stock_data = self._get_stock_data(tickers)
                if stock_data is None or stock_data.empty:
                    return BreadthMetrics(
                        breadth_pct=0.0,
                        above_ma=0,
                        below_ma=0,
                        total_checked=0,
                        no_data=len(tickers),
                        sample_size=len(tickers),
                        breadth_strength="Unknown",
                        ma_period=ma_period,
                        error="Could not fetch stock data",
                    )

            # Validate data sufficiency
            if self._insufficient_data_for_ma(stock_data, ma_period):
                available_days = len(stock_data.index) if hasattr(stock_data, "index") else 0
                return BreadthMetrics(
                    breadth_pct=0.0,
                    above_ma=0,
                    below_ma=0,
                    total_checked=0,
                    no_data=0,
                    sample_size=len(tickers),
                    breadth_strength="Unknown",
                    ma_period=ma_period,
                    error=f"Insufficient data: {available_days} days available, {ma_period} needed",
                )

            # Calculate breadth metrics
            return self._calculate_breadth_metrics(stock_data, tickers, ma_period)

        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return BreadthMetrics(
                breadth_pct=0.0,
                above_ma=0,
                below_ma=0,
                total_checked=0,
                no_data=0,
                sample_size=len(tickers) if tickers else 0,
                breadth_strength="Unknown",
                ma_period=ma_period,
                error=str(e),
            )

    def _get_tickers(self) -> list[str]:
        """Get S&P 500 tickers for breadth analysis.

        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            tickers = self.ticker_source.get_sp500_tickers()
            logger.debug(f"Retrieved {len(tickers)} tickers for breadth analysis")
            return tickers
        except DataSourceError as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    def _get_stock_data(self, tickers: list[str]) -> pd.DataFrame | None:
        """Get stock data for breadth analysis.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            # Get 1 year of data to ensure we have enough for 200-day MA
            logger.debug(f"Fetching stock data for {len(tickers)} tickers...")
            data = self.market_data_source.get_stock_data(tickers, period="1y")

            if data is not None and not data.empty:
                logger.debug(f"Successfully fetched stock data: {data.shape}")

                # Save debug data if enabled
                self.debug_manager.save_debug_data(data, "breadth_stock_data")

                return data

            logger.warning("Failed to fetch stock data for breadth analysis")
            return None

        except DataSourceError as e:
            logger.error(f"Data source error fetching stock data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching stock data: {e}")
            return None

    def _insufficient_data_for_ma(self, stock_data: pd.DataFrame, ma_period: int) -> bool:
        """Check if there's insufficient data for MA calculation.

        Args:
            stock_data: Stock data DataFrame
            ma_period: Moving average period

        Returns:
            True if insufficient data
        """
        if not hasattr(stock_data, "index"):
            return True

        available_days = len(stock_data.index)
        return available_days < ma_period

    def _calculate_breadth_metrics(
        self, stock_data: pd.DataFrame, tickers: list[str], ma_period: int
    ) -> BreadthMetrics:
        """Calculate breadth metrics from stock data.

        Args:
            stock_data: Stock data DataFrame
            tickers: List of ticker symbols
            ma_period: Moving average period

        Returns:
            BreadthMetrics with calculated values
        """
        above_ma_count = 0
        below_ma_count = 0
        no_data_count = 0

        # Log data structure for debugging
        self._log_data_structure(stock_data)

        # Process each ticker
        for ticker in tickers:
            try:
                close_prices = self._extract_close_prices(stock_data, ticker)

                if close_prices is None or len(close_prices) < ma_period:
                    no_data_count += 1
                    continue

                # Calculate moving average
                ma = close_prices.rolling(window=ma_period, min_periods=ma_period).mean()

                # Check if current price is above MA
                current_price = close_prices.iloc[-1]
                current_ma = ma.iloc[-1]

                if pd.notna(current_ma) and pd.notna(current_price):
                    if current_price > current_ma:
                        above_ma_count += 1
                    else:
                        below_ma_count += 1
                else:
                    no_data_count += 1

            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                no_data_count += 1

        # Log processing results
        logger.info(
            f"Breadth calculation complete: {above_ma_count} above MA, "
            f"{below_ma_count} below MA, {no_data_count} no data"
        )

        # Calculate breadth percentage and strength
        valid_stocks = above_ma_count + below_ma_count
        if valid_stocks == 0:
            return BreadthMetrics(
                breadth_pct=0.0,
                above_ma=above_ma_count,
                below_ma=below_ma_count,
                total_checked=valid_stocks,
                no_data=no_data_count,
                sample_size=len(tickers),
                breadth_strength="Unknown",
                ma_period=ma_period,
                error="No valid data for breadth calculation",
            )

        breadth_pct = (above_ma_count / valid_stocks) * 100
        breadth_strength = self._determine_breadth_strength(breadth_pct)

        return BreadthMetrics(
            breadth_pct=round(breadth_pct, 1),
            above_ma=above_ma_count,
            below_ma=below_ma_count,
            total_checked=valid_stocks,
            no_data=no_data_count,
            sample_size=len(tickers),
            breadth_strength=breadth_strength,
            ma_period=ma_period,
        )

    def _log_data_structure(self, stock_data: pd.DataFrame) -> None:
        """Log data structure for debugging.

        Args:
            stock_data: Stock data DataFrame to analyze
        """
        if not hasattr(stock_data, "columns"):
            logger.debug("Data has no columns attribute")
            return

        if isinstance(stock_data.columns, pd.MultiIndex):
            logger.debug(f"MultiIndex levels[0] sample: {list(stock_data.columns.levels[0])[:5]}")
            if len(stock_data.columns.levels) > 1:
                logger.debug(
                    f"MultiIndex levels[1] sample: {list(stock_data.columns.levels[1])[:5]}"
                )
        else:
            logger.debug(f"Regular columns sample: {list(stock_data.columns)[:5]}")

    def _extract_close_prices(self, stock_data: pd.DataFrame, ticker: str) -> pd.Series | None:
        """Extract close prices for a specific ticker from stock data.

        Args:
            stock_data: Stock data DataFrame
            ticker: Ticker symbol to extract

        Returns:
            Series of close prices or None if not found
        """
        try:
            # Handle MultiIndex columns (group_by='ticker' from yfinance)
            if hasattr(stock_data, "columns") and isinstance(stock_data.columns, pd.MultiIndex):
                close_col = (ticker, "Close")
                if close_col in stock_data.columns:
                    return stock_data[close_col].dropna()

            # Handle simple column structure
            elif "Close" in stock_data.columns:
                return stock_data["Close"].dropna()

            return None

        except Exception as e:
            logger.debug(f"Error extracting close prices for {ticker}: {e}")
            return None

    def _determine_breadth_strength(self, breadth_pct: float) -> str:
        """Determine breadth strength based on percentage.

        Args:
            breadth_pct: Breadth percentage

        Returns:
            Breadth strength description
        """
        if breadth_pct >= 70:
            return "Strong Bullish"
        if breadth_pct >= 60:
            return "Bullish"
        if breadth_pct >= 50:
            return "Neutral-Bullish"
        if breadth_pct >= 40:
            return "Neutral-Bearish"
        if breadth_pct >= 30:
            return "Bearish"
        return "Strong Bearish"
