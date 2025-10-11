"""Market regime detection for Clenow momentum strategy.

This module focuses solely on detecting market regimes (bull/bear) based on
S&P 500 vs moving average comparison, following the Single Responsibility Principle.
"""

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from ..interfaces.data_sources import DataSourceError, MarketDataSource
from .debug_utils import DebugDataManager


@dataclass(frozen=True)
class RegimeStatus:
    """Market regime status information.

    Immutable value object containing all regime detection results.
    """

    regime: str  # "bullish", "bearish", or "unknown"
    current_price: float | None
    ma_value: float | None
    price_vs_ma_pct: float | None
    trading_allowed: bool
    latest_date: str | None
    ma_period: int
    error: str | None = None


class MarketRegimeDetector:
    """Detects market regime based on S&P 500 vs moving average.

    According to Clenow's strategy, momentum trading should only be active
    when the S&P 500 is above its moving average (bullish regime).
    """

    def __init__(
        self, market_data_source: MarketDataSource, debug_manager: DebugDataManager | None = None
    ):
        """Initialize regime detector.

        Args:
            market_data_source: Data source for S&P 500 data
            debug_manager: Optional debug data manager
        """
        self.market_data_source = market_data_source
        self.debug_manager = debug_manager or DebugDataManager()

    def check_regime(self, ma_period: int = 200) -> RegimeStatus:
        """Check current market regime based on S&P 500 vs moving average.

        Args:
            ma_period: Moving average period for regime check (default 200)

        Returns:
            RegimeStatus with all regime information
        """
        logger.info(f"Checking market regime (SPY vs {ma_period}-day MA)...")

        try:
            # Get S&P 500 data (using SPY ETF as proxy)
            spy_data = self._get_spy_data()

            if spy_data is None or spy_data.empty:
                logger.error("Could not fetch S&P 500 data for market regime check")
                return RegimeStatus(
                    regime="unknown",
                    current_price=None,
                    ma_value=None,
                    price_vs_ma_pct=None,
                    trading_allowed=False,
                    latest_date=None,
                    ma_period=ma_period,
                    error="Could not fetch market data",
                )

            # Calculate regime status
            return self._calculate_regime_status(spy_data, ma_period)

        except Exception as e:
            logger.error(f"Error checking market regime: {e}")
            return RegimeStatus(
                regime="unknown",
                current_price=None,
                ma_value=None,
                price_vs_ma_pct=None,
                trading_allowed=False,
                latest_date=None,
                ma_period=ma_period,
                error=str(e),
            )

    def _get_spy_data(self) -> pd.DataFrame | None:
        """Get S&P 500 data using SPY ETF as proxy.

        Returns:
            DataFrame with SPY OHLCV data or None if failed
        """
        try:
            # Get 1 year of data to ensure we have enough for 200-day MA
            spy_data = self.market_data_source.get_index_data("SPY", period="1y")

            if spy_data is not None and not spy_data.empty:
                logger.debug(f"Successfully fetched SPY data: {spy_data.shape}")

                # Save debug data if enabled
                self.debug_manager.save_debug_data(spy_data, "spy_regime_data")

                return spy_data

            logger.warning("Failed to fetch SPY data - empty response")
            return None

        except DataSourceError as e:
            logger.error(f"Data source error fetching SPY data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching SPY data: {e}")
            return None

    def _calculate_regime_status(self, spy_data: pd.DataFrame, ma_period: int) -> RegimeStatus:
        """Calculate regime status from SPY data.

        Args:
            spy_data: SPY OHLCV data
            ma_period: Moving average period

        Returns:
            RegimeStatus with calculated values
        """
        if "Close" not in spy_data.columns:
            raise ValueError("SPY data must contain 'Close' column")

        # Calculate moving average
        close_prices = spy_data["Close"].dropna()
        ma = close_prices.rolling(window=ma_period, min_periods=ma_period).mean()

        # Get latest values - ensure scalar values
        current_price = float(close_prices.iloc[-1])
        ma_value = float(ma.iloc[-1])

        if pd.isna(ma_value):
            logger.warning(f"Insufficient data for {ma_period}-day MA calculation")
            return RegimeStatus(
                regime="unknown",
                current_price=current_price,
                ma_value=None,
                price_vs_ma_pct=None,
                trading_allowed=False,
                latest_date=spy_data.index[-1].strftime("%Y-%m-%d"),
                ma_period=ma_period,
                error=f"Insufficient data for {ma_period}-day MA",
            )

        # Determine regime
        price_vs_ma_pct = (current_price / ma_value - 1.0) * 100
        is_bullish = current_price > ma_value
        regime = "bullish" if is_bullish else "bearish"

        # Get date of latest data
        latest_date = spy_data.index[-1].strftime("%Y-%m-%d")

        result = RegimeStatus(
            regime=regime,
            current_price=round(current_price, 2),
            ma_value=round(ma_value, 2),
            price_vs_ma_pct=round(price_vs_ma_pct, 2),
            trading_allowed=is_bullish,
            latest_date=latest_date,
            ma_period=ma_period,
        )

        logger.info(
            f"Market Regime: {regime.upper()} "
            f"(SPX: ${current_price:.2f} vs {ma_period}MA: ${ma_value:.2f}, "
            f"{price_vs_ma_pct:+.2f}%)"
        )

        return result

    def should_trade_momentum(
        self, regime_status: RegimeStatus | None = None, ma_period: int = 200
    ) -> tuple[bool, str]:
        """Determine if momentum trading should be active based on market regime.

        Args:
            regime_status: Pre-calculated regime status (will check if None)
            ma_period: Moving average period if checking is needed

        Returns:
            Tuple of (should_trade, reason)
        """
        if regime_status is None:
            regime_status = self.check_regime(ma_period)

        if regime_status.error:
            return False, f"Market data error: {regime_status.error}"

        if regime_status.trading_allowed:
            reason = (
                f"Market regime is {regime_status.regime} (SPX above {regime_status.ma_period}MA)"
            )
        else:
            reason = (
                f"Market regime is {regime_status.regime} "
                f"(SPX below {regime_status.ma_period}MA) - momentum trading suspended"
            )

        return regime_status.trading_allowed, reason
