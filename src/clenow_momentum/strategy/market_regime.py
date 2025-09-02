"""
Market regime detection for Clenow momentum strategy.

This module implements market regime filtering based on S&P 500 moving average.
According to Clenow's strategy, momentum trading should only be active when
the overall market (SPX) is above its 200-day moving average.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger


def is_debug_mode() -> bool:
    """
    Check if debug mode is enabled via environment variable.
    
    Returns:
        True if DEBUG_MARKET_DATA environment variable is set to 'true' or '1'
    """
    debug_env = os.getenv('DEBUG_MARKET_DATA', '').lower()
    return debug_env in ('true', '1', 'yes')


def save_debug_data(data: pd.DataFrame, filename: str, data_dir: Path = None) -> Path | None:
    """
    Save data for debugging purposes if debug mode is enabled.
    
    Args:
        data: DataFrame to save
        filename: Name of the file (without extension)
        data_dir: Directory to save data (default: data/debug)
    
    Returns:
        Path to saved file or None if debug mode is disabled
    """
    # Only save if debug mode is enabled
    if not is_debug_mode():
        return None
        
    if data_dir is None:
        data_dir = Path("data/debug")
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Use reverse timestamp format for better sorting (most recent first)
    # Format: YYYYMMDD_HHMMSS becomes a sortable string where recent dates are "larger"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # For reverse sorting, we can use a complementary timestamp
    # Or simpler: just rely on file system sorting which will sort YYYYMMDD properly
    
    # Save as CSV for easy inspection
    csv_path = data_dir / f"{timestamp}_{filename}.csv"
    data.to_csv(csv_path)
    logger.debug(f"Saved debug data to {csv_path}")
    
    # Also save metadata as JSON
    metadata = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "shape": list(data.shape),
        "columns": list(data.columns) if not isinstance(data.columns, pd.MultiIndex) else str(data.columns),
        "index_type": str(type(data.index)),
        "dtypes": {str(col): str(dtype) for col, dtype in data.dtypes.items()},
        "has_multiindex_columns": isinstance(data.columns, pd.MultiIndex)
    }
    
    json_path = data_dir / f"{timestamp}_{filename}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Saved metadata to {json_path}")
    
    # Clean up old debug files (keep only last 10 sets)
    cleanup_old_debug_files(data_dir, keep_last=10)
    
    return csv_path


def cleanup_old_debug_files(data_dir: Path, keep_last: int = 10):
    """
    Clean up old debug files, keeping only the most recent ones.
    
    Args:
        data_dir: Directory containing debug files
        keep_last: Number of file sets to keep (each set = csv + json)
    """
    try:
        # Get all CSV files, sorted by name (which includes timestamp)
        csv_files = sorted(data_dir.glob("*.csv"), reverse=True)
        
        # Keep only the most recent files
        if len(csv_files) > keep_last:
            for old_file in csv_files[keep_last:]:
                # Remove CSV and corresponding metadata JSON
                old_file.unlink()
                json_file = old_file.with_suffix('.json')
                if json_file.exists():
                    json_file.unlink()
                # Also remove metadata files
                metadata_file = data_dir / f"{old_file.stem}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                logger.debug(f"Removed old debug file: {old_file.name}")
    except Exception as e:
        logger.warning(f"Error cleaning up old debug files: {e}")


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
        spy_data = yf.download("SPY", period=period, auto_adjust=True, progress=False)

        if spy_data is not None and not spy_data.empty:
            logger.debug(f"Successfully fetched SPY data: {spy_data.shape}")
            
            # Save debug data if debug mode is enabled
            save_debug_data(spy_data, "spy_raw_data")
            
            # Handle potential MultiIndex columns from yfinance
            if isinstance(spy_data.columns, pd.MultiIndex):
                # If MultiIndex, get the first level (the actual column names)
                spy_data.columns = spy_data.columns.get_level_values(0)
                logger.debug("Flattened MultiIndex columns to single level")
            
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
    including historical context and crossover dates.

    Args:
        period: Moving average period (default 200)

    Returns:
        Dictionary with detailed MA status including crossover dates
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
        
        # Ensure we're working with Series, not DataFrame
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()
        if isinstance(ma, pd.DataFrame):
            ma = ma.squeeze()

        # Get recent data (last 30 days)
        recent_data = spy_data.tail(30)
        recent_ma = ma.tail(30)

        # Calculate statistics - ensure scalar values using iloc[0] if needed
        if isinstance(close_prices.iloc[-1], pd.Series):
            current_price = float(close_prices.iloc[-1].iloc[0])
        else:
            current_price = float(close_prices.iloc[-1])
            
        if isinstance(ma.iloc[-1], pd.Series):
            current_ma = float(ma.iloc[-1].iloc[0])
        else:
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

        # Historical context - how long above/below (KEEPING ORIGINAL LOGIC)
        above_ma_series = (close_prices > ma).iloc[-60:]  # Last 60 days

        # Count consecutive days above/below (ORIGINAL CALCULATION)
        consecutive_days = 0
        current_state = bool(above_ma_series.iloc[-1])  # Ensure boolean value

        for i in range(len(above_ma_series) - 1, -1, -1):
            if above_ma_series.iloc[i] == current_state:
                consecutive_days += 1
            else:
                break

        # NEW: Find the last crossover date (looking further back)
        above_ma_full = (close_prices > ma).dropna()
        crossover_date = None
        days_since_crossover = 0
        
        # Look for the last crossover in all available data
        for i in range(len(above_ma_full) - 1, 0, -1):
            # Compare boolean values directly
            if above_ma_full.iloc[i] != above_ma_full.iloc[i-1]:
                # Found a crossover
                crossover_date = above_ma_full.index[i]
                days_since_crossover = (above_ma_full.index[-1] - crossover_date).days
                break
        
        # If no crossover found in the data, we've been in this regime the whole time
        if crossover_date is None:
            crossover_date = above_ma_full.index[0]  # Use earliest date as reference
            days_since_crossover = (above_ma_full.index[-1] - crossover_date).days
        
        # Determine regime type and crossover type
        if current_state:  # Currently above MA (bull market)
            regime_type = "Bull Market"
            crossover_type = "crossed above"
        else:  # Currently below MA (bear market)
            regime_type = "Bear Market" 
            crossover_type = "crossed below"

        # Calculate additional metrics - distance from MA over time
        # Get data since crossover for min/max calculations
        if crossover_date and crossover_date in close_prices.index:
            regime_prices = close_prices.loc[crossover_date:]
            regime_ma = ma.loc[crossover_date:]
            price_vs_ma_series = (regime_prices / regime_ma - 1) * 100
            
            if len(price_vs_ma_series) > 0:
                max_distance = float(price_vs_ma_series.max())
                min_distance = float(price_vs_ma_series.min())
                avg_distance = float(price_vs_ma_series.mean())
            else:
                max_distance = min_distance = avg_distance = 0
        else:
            max_distance = min_distance = avg_distance = 0

        return {
            'current_price': round(current_price, 2),
            'ma_value': round(current_ma, 2),
            'price_vs_ma_pct': round((price_vs_ma_ratio - 1) * 100, 2),
            'above_ma': current_price > current_ma,
            'ma_trend': ma_trend,
            'ma_slope_10d': round(ma_slope, 2),
            'consecutive_days_current_regime': consecutive_days,  # KEPT ORIGINAL
            'regime_type': regime_type,  # NEW
            'crossover_date': crossover_date.strftime('%Y-%m-%d'),  # NEW
            'crossover_type': crossover_type,  # NEW
            'days_since_crossover': days_since_crossover,  # NEW
            'max_distance_from_ma_pct': round(max_distance, 2),  # NEW
            'min_distance_from_ma_pct': round(min_distance, 2),  # NEW
            'avg_distance_from_ma_pct': round(avg_distance, 2),  # NEW
            'recent_30d_above_ma': recent_above_ma,
            'recent_30d_below_ma': recent_below_ma,
            'latest_date': spy_data.index[-1].strftime('%Y-%m-%d'),
            'ma_period': period
        }


    except Exception as e:
        logger.error(f"Error getting S&P 500 MA status: {e}")
        return {'error': str(e)}


def calculate_market_breadth(tickers: list = None, period: int = 200, stock_data: pd.DataFrame = None) -> dict:
    """
    Calculate market breadth - percentage of S&P 500 stocks above their moving average.
    
    Args:
        tickers: List of S&P 500 tickers (if None, will fetch them)
        period: Moving average period (default 200)
        stock_data: Pre-fetched stock data (optional, for efficiency)
    
    Returns:
        Dictionary with breadth statistics
    """
    logger.info(f"Calculating market breadth (% of stocks above {period}-day MA)...")
    
    # Check if we have enough data
    if stock_data is not None and hasattr(stock_data, 'index'):
        available_days = len(stock_data.index)
        if available_days < period:
            logger.error(f"Insufficient data: Only {available_days} days available but {period} days needed for MA calculation")
            return {'error': f'Insufficient data: {available_days} days available, {period} needed'}
    
    try:
        # Get tickers if not provided
        if tickers is None:
            from clenow_momentum.data.fetcher import get_sp500_tickers
            tickers = get_sp500_tickers()
            if not tickers:
                logger.error("Could not fetch S&P 500 tickers")
                return {'error': 'Could not fetch tickers'}
        
        # Use ALL tickers for accurate breadth
        sample_tickers = tickers  # Analyze all stocks, not just a sample
        sample_size = len(sample_tickers)
        
        # Use pre-fetched data if available, otherwise fetch
        if stock_data is not None:
            logger.debug(f"Using pre-fetched data for {sample_size} stocks")
            logger.debug(f"Stock data shape: {stock_data.shape if hasattr(stock_data, 'shape') else 'unknown'}")
            logger.debug(f"Stock data type: {type(stock_data)}")
            
            # Save debug data to understand structure
            if is_debug_mode():
                save_debug_data(stock_data, "breadth_stock_data")
            
            data = stock_data
        else:
            # Fetch data for all tickers
            logger.debug(f"Fetching data for {sample_size} stocks to calculate breadth...")
            import yfinance as yf
            
            # Download data for multiple tickers - this will take a moment
            # Using threads=True for parallel downloads
            data = yf.download(
                sample_tickers,
                period="1y",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True
            )
            
            if data is None or data.empty:
                logger.error("Could not fetch stock data for breadth calculation")
                return {'error': 'Could not fetch stock data'}
        
        # Calculate how many stocks are above their MA
        above_ma_count = 0
        below_ma_count = 0
        no_data_count = 0
        
        # Log data structure for debugging
        logger.debug(f"Data columns type: {type(data.columns) if hasattr(data, 'columns') else 'no columns'}")
        if hasattr(data, 'columns'):
            if isinstance(data.columns, pd.MultiIndex):
                logger.debug(f"MultiIndex levels[0] sample: {list(data.columns.levels[0])[:5]}")
                logger.debug(f"MultiIndex levels[1] sample: {list(data.columns.levels[1])[:5] if len(data.columns.levels) > 1 else 'N/A'}")
            else:
                logger.debug(f"Regular columns sample: {list(data.columns)[:5]}")
        
        for ticker in sample_tickers:
            try:
                close_prices = None
                
                # Extract close prices for this ticker
                # Handle both single ticker and multi-ticker data structures
                if hasattr(data, 'columns') and isinstance(data.columns, pd.MultiIndex):
                    # Multi-ticker with MultiIndex columns (group_by='ticker')
                    # The structure from yfinance is: MultiIndex[('TICKER', 'Price')]
                    # where Price can be 'Open', 'High', 'Low', 'Close', 'Volume'
                    
                    # Check if ticker exists in the data
                    if (ticker, 'Close') in data.columns:
                        try:
                            close_prices = data[(ticker, 'Close')]
                        except Exception as e:
                            logger.debug(f"Error extracting {ticker}: {e}")
                            no_data_count += 1
                            continue
                    else:
                        # Ticker not in data
                        no_data_count += 1
                        continue
                elif 'Close' in data.columns:
                    # Simple structure with 'Close' column
                    close_prices = data['Close']
                else:
                    # Unknown structure
                    logger.debug(f"Unknown data structure for {ticker}")
                    no_data_count += 1
                    continue
                
                if close_prices is None or (hasattr(close_prices, 'empty') and close_prices.empty):
                    no_data_count += 1
                    continue
                
                # Drop NaN values from close prices
                close_prices = close_prices.dropna()
                
                if len(close_prices) < period:
                    # Not enough data for MA calculation
                    no_data_count += 1
                    continue
                
                # Calculate MA
                ma = close_prices.rolling(window=period, min_periods=period).mean()
                
                # Check if current price is above MA
                if len(close_prices) > 0 and len(ma) > 0:
                    current_price = close_prices.iloc[-1]
                    current_ma = ma.iloc[-1]
                else:
                    no_data_count += 1
                    continue
                
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
        
        # Log results
        logger.info(f"Breadth calculation complete: {above_ma_count} above MA, {below_ma_count} below MA, {no_data_count} no data")
        
        # Calculate percentages
        valid_stocks = above_ma_count + below_ma_count
        if valid_stocks > 0:
            breadth_pct = (above_ma_count / valid_stocks) * 100
            
            # Determine breadth strength
            if breadth_pct >= 70:
                breadth_strength = "Strong Bullish"
            elif breadth_pct >= 60:
                breadth_strength = "Bullish"
            elif breadth_pct >= 50:
                breadth_strength = "Neutral-Bullish"
            elif breadth_pct >= 40:
                breadth_strength = "Neutral-Bearish"
            elif breadth_pct >= 30:
                breadth_strength = "Bearish"
            else:
                breadth_strength = "Strong Bearish"
            
            return {
                'breadth_pct': round(breadth_pct, 1),
                'above_ma': above_ma_count,
                'below_ma': below_ma_count,
                'total_checked': valid_stocks,
                'no_data': no_data_count,
                'sample_size': sample_size,
                'breadth_strength': breadth_strength,
                'ma_period': period
            }
        else:
            return {'error': 'Could not calculate breadth - no valid data'}
            
    except Exception as e:
        logger.error(f"Error calculating market breadth: {e}")
        return {'error': str(e)}


def calculate_absolute_momentum(period_months: int = 12) -> dict:
    """
    Calculate absolute momentum of S&P 500 (12-month return).
    
    Args:
        period_months: Number of months for return calculation (default 12)
    
    Returns:
        Dictionary with absolute momentum metrics
    """
    logger.info(f"Calculating S&P 500 absolute momentum ({period_months}-month return)...")
    
    try:
        # Get S&P 500 data for the period
        period_str = f"{period_months + 1}mo"  # Extra month to ensure we have enough data
        spy_data = get_sp500_data(period_str)
        
        if spy_data is None or spy_data.empty:
            logger.error("Could not fetch S&P 500 data for momentum calculation")
            return {'error': 'Could not fetch market data'}
        
        close_prices = spy_data['Close']
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
            logger.warning(f"Only {actual_days} days available for {period_months}-month return calculation")
        
        # Also calculate other useful momentum periods
        returns = {}
        
        # 1-month return
        if len(close_prices) > 21:
            returns['1m'] = ((current_price / float(close_prices.iloc[-21])) - 1) * 100
        
        # 3-month return
        if len(close_prices) > 63:
            returns['3m'] = ((current_price / float(close_prices.iloc[-63])) - 1) * 100
        
        # 6-month return
        if len(close_prices) > 126:
            returns['6m'] = ((current_price / float(close_prices.iloc[-126])) - 1) * 100
        
        # 12-month return
        if len(close_prices) > 252:
            returns['12m'] = ((current_price / float(close_prices.iloc[-252])) - 1) * 100
        
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
        
        return {
            'period_return': round(period_return, 2),
            'period_months': period_months,
            'current_price': round(current_price, 2),
            'past_price': round(past_price, 2),
            'momentum_strength': momentum_strength,
            'all_returns': {k: round(v, 2) for k, v in returns.items()},
            'bullish': period_return > 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating absolute momentum: {e}")
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
