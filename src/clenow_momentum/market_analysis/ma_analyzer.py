"""Moving average analysis for S&P 500.

This module focuses solely on analyzing S&P 500 position relative to multiple
moving averages, following the Single Responsibility Principle.
"""

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from ..data.interfaces import MarketDataSource
from .debug_utils import DebugDataManager


@dataclass(frozen=True)
class ShortTermMAs:
    """Short-term moving average values."""

    current_price: float
    ema_10: float | None = None
    ema_10_distance: float | None = None
    sma_20: float | None = None
    sma_20_distance: float | None = None
    sma_50: float | None = None
    sma_50_distance: float | None = None
    latest_date: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MAPosition:
    """Moving average position analysis results."""

    current_price: float
    mas_above: int
    mas_below: int
    total_mas: int
    market_structure: str
    ma_positions: list[tuple[str, str, float]]
    mas_aligned: bool
    short_term_mas: ShortTermMAs | None = None
    ma_200: float | None = None
    error: str | None = None


class MovingAverageAnalyzer:
    """Analyzes S&P 500 position relative to multiple moving averages."""

    def __init__(self, debug_manager: DebugDataManager | None = None):
        self.debug_manager = debug_manager or DebugDataManager()

    def calculate_short_term_mas(self, spy_data: pd.DataFrame) -> ShortTermMAs:
        """Calculate short-term moving averages (10 EMA, 20 SMA, 50 SMA).

        Args:
            spy_data: S&P 500 market data (must have at least 50 days for 50-day MA)

        Returns:
            ShortTermMAs with calculated values or error
        """
        logger.info("Calculating short-term moving averages...")

        try:
            if spy_data is None or spy_data.empty:
                logger.error("Could not fetch S&P 500 data for MA calculation")
                return ShortTermMAs(
                    current_price=0.0, error="Could not fetch market data"
                )

            close_prices = spy_data["Close"]
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.squeeze()

            current_price = float(close_prices.iloc[-1])
            latest_date = spy_data.index[-1].strftime("%Y-%m-%d")

            # Calculate various MAs
            ema_10 = None
            ema_10_distance = None
            if len(close_prices) >= 10:
                ema_10_series = close_prices.ewm(span=10, adjust=False).mean()
                ema_10 = float(ema_10_series.iloc[-1])
                ema_10_distance = ((current_price / ema_10) - 1) * 100

            sma_20 = None
            sma_20_distance = None
            if len(close_prices) >= 20:
                sma_20_series = close_prices.rolling(window=20).mean()
                sma_20 = float(sma_20_series.iloc[-1])
                sma_20_distance = ((current_price / sma_20) - 1) * 100

            sma_50 = None
            sma_50_distance = None
            if len(close_prices) >= 50:
                sma_50_series = close_prices.rolling(window=50).mean()
                sma_50 = float(sma_50_series.iloc[-1])
                sma_50_distance = ((current_price / sma_50) - 1) * 100

            return ShortTermMAs(
                current_price=current_price,
                ema_10=ema_10,
                ema_10_distance=ema_10_distance,
                sma_20=sma_20,
                sma_20_distance=sma_20_distance,
                sma_50=sma_50,
                sma_50_distance=sma_50_distance,
                latest_date=latest_date,
            )

        except Exception as e:
            logger.error(f"Error calculating short-term MAs: {e}")
            return ShortTermMAs(current_price=0.0, error=str(e))

    def analyze_position(
        self,
        long_term_ma: float | None = None,
        short_term_mas: ShortTermMAs | None = None,
        spy_data: pd.DataFrame | None = None,
    ) -> MAPosition:
        """Analyze S&P 500 position relative to all moving averages.

        Args:
            long_term_ma: Optional 200-day MA value
            short_term_mas: Optional pre-calculated short-term MAs (if None, spy_data required)
            spy_data: Optional S&P 500 data for calculating short-term MAs

        Returns:
            MAPosition with analysis results
        """
        try:
            # Calculate short-term MAs if not provided
            if short_term_mas is None:
                if spy_data is None:
                    return MAPosition(
                        current_price=0.0,
                        mas_above=0,
                        mas_below=0,
                        total_mas=0,
                        market_structure="Unknown",
                        ma_positions=[],
                        mas_aligned=False,
                        error="Either short_term_mas or spy_data must be provided",
                    )
                short_term_mas = self.calculate_short_term_mas(spy_data)

            if short_term_mas.error:
                return MAPosition(
                    current_price=0.0,
                    mas_above=0,
                    mas_below=0,
                    total_mas=0,
                    market_structure="Unknown",
                    ma_positions=[],
                    mas_aligned=False,
                    error=short_term_mas.error,
                )

            current_price = short_term_mas.current_price

            # Count how many MAs we're above
            mas_above = 0
            mas_below = 0
            ma_positions = []

            # Check each MA
            if short_term_mas.ema_10 is not None:
                if current_price > short_term_mas.ema_10:
                    mas_above += 1
                    ma_positions.append(
                        ("10-day EMA", "above", short_term_mas.ema_10_distance or 0.0)
                    )
                else:
                    mas_below += 1
                    ma_positions.append(
                        ("10-day EMA", "below", short_term_mas.ema_10_distance or 0.0)
                    )

            if short_term_mas.sma_20 is not None:
                if current_price > short_term_mas.sma_20:
                    mas_above += 1
                    ma_positions.append(
                        ("20-day SMA", "above", short_term_mas.sma_20_distance or 0.0)
                    )
                else:
                    mas_below += 1
                    ma_positions.append(
                        ("20-day SMA", "below", short_term_mas.sma_20_distance or 0.0)
                    )

            if short_term_mas.sma_50 is not None:
                if current_price > short_term_mas.sma_50:
                    mas_above += 1
                    ma_positions.append(
                        ("50-day SMA", "above", short_term_mas.sma_50_distance or 0.0)
                    )
                else:
                    mas_below += 1
                    ma_positions.append(
                        ("50-day SMA", "below", short_term_mas.sma_50_distance or 0.0)
                    )

            if long_term_ma and long_term_ma > 0:
                distance_200 = ((current_price / long_term_ma) - 1) * 100
                if current_price > long_term_ma:
                    mas_above += 1
                    ma_positions.append(("200-day SMA", "above", distance_200))
                else:
                    mas_below += 1
                    ma_positions.append(("200-day SMA", "below", distance_200))

            # Determine market structure
            total_mas = mas_above + mas_below
            if total_mas == 0:
                market_structure = "Unknown"
            elif mas_above == total_mas:
                market_structure = "Strong Bullish (Above all MAs)"
            elif mas_above >= 3:
                market_structure = "Bullish"
            elif mas_above >= 2:
                market_structure = "Mixed"
            elif mas_above >= 1:
                market_structure = "Weak"
            else:
                market_structure = "Bearish (Below all MAs)"

            # Check MA alignment (bullish when 10 > 20 > 50 > 200)
            aligned = False
            if (
                short_term_mas.ema_10 is not None
                and short_term_mas.sma_20 is not None
                and short_term_mas.sma_50 is not None
            ):
                if (
                    short_term_mas.ema_10 > short_term_mas.sma_20
                    and short_term_mas.sma_20 > short_term_mas.sma_50
                ):
                    if long_term_ma:
                        aligned = short_term_mas.sma_50 > long_term_ma
                    else:
                        aligned = True

            return MAPosition(
                current_price=current_price,
                mas_above=mas_above,
                mas_below=mas_below,
                total_mas=total_mas,
                market_structure=market_structure,
                ma_positions=ma_positions,
                mas_aligned=aligned,
                short_term_mas=short_term_mas,
                ma_200=long_term_ma,
            )

        except Exception as e:
            logger.error(f"Error analyzing MA position: {e}")
            return MAPosition(
                current_price=0.0,
                mas_above=0,
                mas_below=0,
                total_mas=0,
                market_structure="Unknown",
                ma_positions=[],
                mas_aligned=False,
                error=str(e),
            )
