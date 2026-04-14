"""Absolute momentum analysis for S&P 500.

This module focuses solely on calculating absolute momentum metrics for the S&P 500,
following the Single Responsibility Principle.
"""

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from ..data.interfaces import MarketDataSource
from .debug_utils import DebugDataManager


@dataclass(frozen=True)
class MomentumMetrics:
    """Absolute momentum analysis results."""

    period_return: float
    period_months: int
    current_price: float
    past_price: float
    momentum_strength: str
    all_returns: dict[str, float]
    bullish: bool
    error: str | None = None


@dataclass(frozen=True)
class DailyPerformance:
    """Daily performance metrics."""

    current_price: float
    previous_close: float
    daily_change: float
    daily_change_pct: float
    daily_trend: str
    five_day_return: float | None = None
    latest_date: str | None = None
    error: str | None = None


class AbsoluteMomentumAnalyzer:
    """Analyzes absolute momentum of S&P 500 (N-month returns)."""

    def __init__(self, debug_manager: DebugDataManager | None = None):
        self.debug_manager = debug_manager or DebugDataManager()

    def calculate_momentum(self, spy_data: pd.DataFrame, period_months: int = 12) -> MomentumMetrics:
        """Calculate absolute momentum of S&P 500.

        Args:
            spy_data: S&P 500 market data (should have at least period_months of history)
            period_months: Number of months for return calculation (default 12)

        Returns:
            MomentumMetrics with calculated momentum values
        """
        logger.info(f"Calculating S&P 500 absolute momentum ({period_months}-month return)...")

        try:
            if spy_data is None or spy_data.empty:
                logger.error("Could not fetch S&P 500 data for momentum calculation")
                return MomentumMetrics(
                    period_return=0.0,
                    period_months=period_months,
                    current_price=0.0,
                    past_price=0.0,
                    momentum_strength="Unknown",
                    all_returns={},
                    bullish=False,
                    error="Could not fetch market data",
                )

            close_prices = spy_data["Close"]
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.squeeze()

            # Calculate returns for different periods
            current_price = float(close_prices.iloc[-1])

            # Calculate N-month return
            days_in_month = 21  # Trading days
            lookback_days = period_months * days_in_month

            if len(close_prices) > lookback_days:
                past_price = float(close_prices.iloc[-lookback_days])
                period_return = ((current_price / past_price) - 1) * 100
            else:
                # Fallback to earliest available
                past_price = float(close_prices.iloc[0])
                actual_days = len(close_prices)
                period_return = ((current_price / past_price) - 1) * 100
                logger.warning(
                    f"Only {actual_days} days available for {period_months}-month return calculation"
                )

            # Calculate other useful momentum periods
            returns = {}

            # 1-month return
            if len(close_prices) > 21:
                returns["1m"] = ((current_price / float(close_prices.iloc[-21])) - 1) * 100

            # 3-month return
            if len(close_prices) > 63:
                returns["3m"] = ((current_price / float(close_prices.iloc[-63])) - 1) * 100

            # 6-month return
            if len(close_prices) > 126:
                returns["6m"] = ((current_price / float(close_prices.iloc[-126])) - 1) * 100

            # 12-month return
            if len(close_prices) > 252:
                returns["12m"] = ((current_price / float(close_prices.iloc[-252])) - 1) * 100

            # Determine momentum strength
            if period_return > 20:
                momentum_strength = "Strong Positive"
            elif period_return > 10:
                momentum_strength = "Positive"
            elif period_return > 0:
                momentum_strength = "Weak Positive"
            elif period_return > -10:
                momentum_strength = "Weak Negative"
            elif period_return > -20:
                momentum_strength = "Negative"
            else:
                momentum_strength = "Strong Negative"

            return MomentumMetrics(
                period_return=round(period_return, 2),
                period_months=period_months,
                current_price=round(current_price, 2),
                past_price=round(past_price, 2),
                momentum_strength=momentum_strength,
                all_returns={k: round(v, 2) for k, v in returns.items()},
                bullish=period_return > 0,
            )

        except Exception as e:
            logger.error(f"Error calculating absolute momentum: {e}")
            return MomentumMetrics(
                period_return=0.0,
                period_months=period_months,
                current_price=0.0,
                past_price=0.0,
                momentum_strength="Unknown",
                all_returns={},
                bullish=False,
                error=str(e),
            )

    def calculate_daily_performance(self, spy_data: pd.DataFrame) -> DailyPerformance:
        """Calculate S&P 500 daily performance metrics.

        Args:
            spy_data: Recent S&P 500 market data (at least 6 days for 5-day return)

        Returns:
            DailyPerformance with calculated daily metrics
        """
        logger.info("Calculating S&P 500 daily performance...")

        try:
            if spy_data is None or spy_data.empty:
                logger.error("Could not fetch S&P 500 data for daily performance")
                return DailyPerformance(
                    current_price=0.0,
                    previous_close=0.0,
                    daily_change=0.0,
                    daily_change_pct=0.0,
                    daily_trend="Unknown",
                    error="Could not fetch market data",
                )

            close_prices = spy_data["Close"]
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.squeeze()

            # Ensure we have enough data
            if len(close_prices) < 2:
                return DailyPerformance(
                    current_price=0.0,
                    previous_close=0.0,
                    daily_change=0.0,
                    daily_change_pct=0.0,
                    daily_trend="Unknown",
                    error="Insufficient data for daily calculation",
                )

            # Calculate daily change
            current_price = float(close_prices.iloc[-1])
            previous_close = float(close_prices.iloc[-2])
            daily_change = current_price - previous_close
            daily_change_pct = ((current_price / previous_close) - 1) * 100

            # Calculate 5-day performance
            five_day_return = None
            if len(close_prices) >= 6:
                five_days_ago = float(close_prices.iloc[-6])
                five_day_return = ((current_price / five_days_ago) - 1) * 100

            # Determine trend
            if daily_change_pct > 0.5:
                daily_trend = "Strong Up"
            elif daily_change_pct > 0:
                daily_trend = "Up"
            elif daily_change_pct > -0.5:
                daily_trend = "Down"
            else:
                daily_trend = "Strong Down"

            latest_date = spy_data.index[-1].strftime("%Y-%m-%d")

            return DailyPerformance(
                current_price=round(current_price, 2),
                previous_close=round(previous_close, 2),
                daily_change=round(daily_change, 2),
                daily_change_pct=round(daily_change_pct, 2),
                daily_trend=daily_trend,
                five_day_return=round(five_day_return, 2) if five_day_return else None,
                latest_date=latest_date,
            )

        except Exception as e:
            logger.error(f"Error calculating daily performance: {e}")
            return DailyPerformance(
                current_price=0.0,
                previous_close=0.0,
                daily_change=0.0,
                daily_change_pct=0.0,
                daily_trend="Unknown",
                error=str(e),
            )
