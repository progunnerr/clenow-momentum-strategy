"""
Market regime detection for Clenow momentum strategy.

This module implements market regime filtering based on S&P 500 moving average.
According to Clenow's strategy, momentum trading should only be active when
the overall market (SPX) is above its 200-day moving average.
"""


import pandas as pd
import yfinance as yf
from loguru import logger


def get_sp500_data(period: str = "1y") -> pd.DataFrame | None:
    """
    Get S&P 500 index data using SPY ETF as proxy.

    Args:
        period: Data period (1y, 2y, etc.)

    Returns:
        DataFrame with S&P 500 OHLCV data or None if failed
    """
    try:
        logger.debug("Fetching S&P 500 data via SPY ETF...")
        spy_data = yf.download("SPY", period=period, auto_adjust=True)

        if spy_data is not None and not spy_data.empty:
            logger.debug(f"Successfully fetched SPY data: {spy_data.shape}")
            return spy_data
        logger.warning("Failed to fetch SPY data - empty response")
        return None

    except Exception as e:
        logger.error(f"Error fetching S&P 500 data: {e}")
        return None


def calculate_market_ma(spy_data: pd.DataFrame, period: int = 200) -> pd.Series:
    """
    Calculate moving average for S&P 500 index.

    Args:
        spy_data: S&P 500 OHLCV data
        period: Moving average period (default 200 days)

    Returns:
        Series with moving average values
    """
    if 'Close' not in spy_data.columns:
        raise ValueError("SPY data must contain 'Close' column")

    close_prices = spy_data['Close'].dropna()
    return close_prices.rolling(window=period, min_periods=period).mean()



def check_market_regime(period: int = 200) -> dict:
    """
    Check current market regime based on S&P 500 vs its moving average.

    According to Clenow's strategy, momentum trading should only be active
    when SPX > 200-day MA (bullish regime).

    Args:
        period: Moving average period for regime check (default 200)

    Returns:
        Dictionary with market regime information
    """
    logger.info(f"Checking market regime (SPX vs {period}-day MA)...")

    # Get S&P 500 data
    spy_data = get_sp500_data("1y")  # Get 1 year to ensure we have enough data for 200-day MA

    if spy_data is None or spy_data.empty:
        logger.error("Could not fetch S&P 500 data for market regime check")
        return {
            'regime': 'unknown',
            'current_price': None,
            'ma_value': None,
            'price_vs_ma': None,
            'trading_allowed': False,
            'error': 'Could not fetch market data'
        }

    try:
        # Calculate moving average
        ma = calculate_market_ma(spy_data, period)

        # Get latest values - ensure we get scalar values, not Series
        current_price = float(spy_data['Close'].iloc[-1])
        ma_value = float(ma.iloc[-1])

        if pd.isna(ma_value):
            logger.warning(f"Insufficient data for {period}-day MA calculation")
            return {
                'regime': 'unknown',
                'current_price': current_price,
                'ma_value': None,
                'price_vs_ma': None,
                'trading_allowed': False,
                'error': f'Insufficient data for {period}-day MA'
            }

        # Determine regime
        price_vs_ma = (current_price / ma_value) - 1.0
        is_bullish = current_price > ma_value
        regime = 'bullish' if is_bullish else 'bearish'

        # Get date of latest data
        latest_date = spy_data.index[-1].strftime('%Y-%m-%d')

        result = {
            'regime': regime,
            'current_price': round(current_price, 2),
            'ma_value': round(ma_value, 2),
            'price_vs_ma': round(price_vs_ma, 4),
            'trading_allowed': is_bullish,
            'latest_date': latest_date,
            'ma_period': period
        }

        logger.info(f"Market Regime: {regime.upper()} (SPX: ${current_price:.2f} vs {period}MA: ${ma_value:.2f}, {price_vs_ma:+.2%})")

        return result

    except Exception as e:
        logger.error(f"Error calculating market regime: {e}")
        return {
            'regime': 'unknown',
            'current_price': None,
            'ma_value': None,
            'price_vs_ma': None,
            'trading_allowed': False,
            'error': str(e)
        }


def get_sp500_ma_status(period: int = 200) -> dict:
    """
    Get detailed S&P 500 moving average status for analysis.

    This provides more detailed information than check_market_regime,
    including historical context.

    Args:
        period: Moving average period (default 200)

    Returns:
        Dictionary with detailed MA status
    """
    logger.info(f"Getting detailed S&P 500 {period}-day MA status...")

    # Get more data for historical context
    spy_data = get_sp500_data("2y")

    if spy_data is None or spy_data.empty:
        logger.error("Could not fetch S&P 500 data")
        return {'error': 'Could not fetch market data'}

    try:
        # Calculate moving average
        ma = calculate_market_ma(spy_data, period)
        close_prices = spy_data['Close']

        # Get recent data (last 30 days)
        recent_data = spy_data.tail(30)
        recent_ma = ma.tail(30)

        # Calculate statistics - ensure scalar values
        current_price = float(close_prices.iloc[-1])
        current_ma = float(ma.iloc[-1])

        if pd.isna(current_ma):
            return {'error': f'Insufficient data for {period}-day MA calculation'}

        # Days above/below MA in recent period
        recent_above_ma = (recent_data['Close'] > recent_ma).sum()
        recent_below_ma = len(recent_data) - recent_above_ma

        # Trend analysis
        ma_slope = (recent_ma.iloc[-1] - recent_ma.iloc[-10]) / 10 if len(recent_ma) >= 10 else 0
        ma_trend = 'rising' if ma_slope > 0 else 'falling' if ma_slope < 0 else 'flat'

        # Price momentum vs MA
        price_vs_ma_ratio = current_price / current_ma

        # Historical context - how long above/below
        above_ma_series = (close_prices > ma).iloc[-60:]  # Last 60 days

        # Count consecutive days above/below
        consecutive_days = 0
        current_state = bool(above_ma_series.iloc[-1])  # Ensure boolean value

        for i in range(len(above_ma_series) - 1, -1, -1):
            if above_ma_series.iloc[i] == current_state:
                consecutive_days += 1
            else:
                break

        return {
            'current_price': round(current_price, 2),
            'ma_value': round(current_ma, 2),
            'price_vs_ma_pct': round((price_vs_ma_ratio - 1) * 100, 2),
            'above_ma': current_price > current_ma,
            'ma_trend': ma_trend,
            'ma_slope_10d': round(ma_slope, 2),
            'consecutive_days_current_regime': consecutive_days,
            'recent_30d_above_ma': recent_above_ma,
            'recent_30d_below_ma': recent_below_ma,
            'latest_date': spy_data.index[-1].strftime('%Y-%m-%d'),
            'ma_period': period
        }


    except Exception as e:
        logger.error(f"Error getting S&P 500 MA status: {e}")
        return {'error': str(e)}


def should_trade_momentum(market_regime: dict = None) -> tuple[bool, str]:
    """
    Determine if momentum trading should be active based on market regime.

    Args:
        market_regime: Market regime dict from check_market_regime() (optional)

    Returns:
        Tuple of (should_trade, reason)
    """
    if market_regime is None:
        market_regime = check_market_regime()

    if 'error' in market_regime:
        return False, f"Market data error: {market_regime['error']}"

    trading_allowed = market_regime.get('trading_allowed', False)
    regime = market_regime.get('regime', 'unknown')

    if trading_allowed:
        reason = f"Market regime is {regime} (SPX above {market_regime.get('ma_period', 200)}MA)"
    else:
        reason = f"Market regime is {regime} (SPX below {market_regime.get('ma_period', 200)}MA) - momentum trading suspended"

    return trading_allowed, reason
