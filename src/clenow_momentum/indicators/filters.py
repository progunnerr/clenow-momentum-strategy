"""
Trading filters for Clenow momentum strategy.

This module implements the filtering criteria used in Andreas Clenow's momentum strategy:
- 100-day moving average filter (price > MA)
- Gap detection filter (exclude stocks with significant gaps)
"""

import pandas as pd
from loguru import logger


def calculate_moving_average(prices: pd.Series, period: int = 100) -> pd.Series:
    """
    Calculate simple moving average for a price series.

    Args:
        prices: Series of stock prices
        period: Number of days for moving average (default 100)

    Returns:
        Series of moving average values
    """
    return prices.rolling(window=period, min_periods=period).mean()


def process_ticker_ma(ticker, ticker_data, ma_period: int) -> dict | None:
    """
    Process a single ticker for moving average filtering.

    Args:
        ticker: Stock ticker symbol
        ticker_data: Price data for the ticker
        ma_period: Moving average period

    Returns:
        Dictionary with MA results or None if processing failed
    """
    try:
        # Extract close prices based on data structure
        if isinstance(ticker_data, pd.DataFrame) and 'Close' in ticker_data.columns:
            prices = ticker_data['Close'].dropna()
        elif isinstance(ticker_data, pd.Series):
            prices = ticker_data.dropna()
        else:
            logger.debug(f"No Close price data for {ticker}")
            return None

        if len(prices) < ma_period:
            logger.debug(f"Insufficient data for {ticker}: {len(prices)} < {ma_period}")
            return None

        # Calculate moving average
        ma = calculate_moving_average(prices, ma_period)

        # Get the latest price and MA value
        latest_price = prices.iloc[-1]
        latest_ma = ma.iloc[-1]

        if pd.isna(latest_ma):
            logger.debug(f"No MA value for {ticker}")
            return None

        # Check if price is above MA
        above_ma = latest_price > latest_ma

        return {
            'ticker': ticker,
            'latest_price': latest_price,
            f'ma_{ma_period}': latest_ma,
            'price_vs_ma': latest_price / latest_ma - 1.0,  # Percentage above/below MA
            'above_ma': above_ma
        }
    except (KeyError, AttributeError) as e:
        logger.debug(f"Error processing {ticker} for MA filter: {e}")
        return None


def filter_above_ma(data: pd.DataFrame, ma_period: int = 100) -> pd.DataFrame:
    """
    Filter stocks that are trading above their moving average.

    According to Clenow's strategy, only stocks trading above their 100-day MA
    should be considered for momentum trading.

    Args:
        data: Stock data with MultiIndex columns (ticker, OHLCV)
        ma_period: Moving average period (default 100 days)

    Returns:
        DataFrame containing tickers that pass the MA filter with their latest prices and MA values
    """
    # Validate input
    if data is None or data.empty:
        logger.warning("Input data is None or empty for MA filter")
        return pd.DataFrame()

    results = []
    logger.info(f"Applying {ma_period}-day moving average filter...")

    # Extract tickers based on column structure
    if isinstance(data.columns, pd.MultiIndex):
        tickers = data.columns.get_level_values(0).unique()
        def ticker_data_func(t):
            return data[t]
    else:
        logger.warning("Simple column structure detected - assuming columns are price data")
        tickers = data.columns
        def ticker_data_func(t):
            return data[t]

    # Process each ticker
    for ticker in tickers:
        ticker_data = ticker_data_func(ticker)
        result = process_ticker_ma(ticker, ticker_data, ma_period)
        if result:
            results.append(result)

    df = pd.DataFrame(results)

    # Count results
    passed_count = len(df[df['above_ma'] == True]) if not df.empty else 0
    failed_count = len(df[df['above_ma'] == False]) if not df.empty else 0
    # Add skipped tickers (those that had no results at all)
    skipped_count = len(tickers) - len(results)
    failed_count += skipped_count

    logger.info(f"MA Filter Results: {passed_count} passed, {failed_count} failed")

    return df


def detect_gaps(data: pd.DataFrame, gap_threshold: float = 0.15) -> pd.DataFrame:
    """
    Detect stocks with significant price gaps.

    According to Clenow's strategy, stocks with gaps > 15% should be excluded
    as they may indicate news events or other anomalies that affect momentum.

    Args:
        data: Stock data with MultiIndex columns (ticker, OHLCV)
        gap_threshold: Maximum allowed gap as percentage (default 0.15 = 15%)

    Returns:
        DataFrame with gap analysis for each ticker
    """
    results = []
    logger.info(f"Detecting price gaps > {gap_threshold:.1%}...")

    if isinstance(data.columns, pd.MultiIndex):
        tickers = data.columns.get_level_values(0).unique()

        for ticker in tickers:
            try:
                ticker_data = data[ticker]
                if 'Close' not in ticker_data.columns or 'Open' not in ticker_data.columns:
                    logger.debug(f"Missing OHLC data for {ticker}")
                    continue

                close_prices = ticker_data['Close'].dropna()
                open_prices = ticker_data['Open'].dropna()

                if len(close_prices) < 2 or len(open_prices) < 2:
                    continue

                # Calculate overnight gaps: (Open[t] - Close[t-1]) / Close[t-1]
                prev_close = close_prices.shift(1)
                gaps = (open_prices - prev_close) / prev_close
                gaps = gaps.dropna()

                if len(gaps) == 0:
                    continue

                # Find the maximum absolute gap in recent period (last 30 days)
                recent_gaps = gaps.tail(30)
                max_gap = recent_gaps.abs().max()
                max_gap_date = recent_gaps.abs().idxmax() if not pd.isna(max_gap) else None

                # Check if any gap exceeds threshold
                has_large_gap = max_gap > gap_threshold

                # Determine gap direction safely to avoid KeyError
                if max_gap_date is not None:
                    direction = 'up' if recent_gaps.loc[max_gap_date] > 0 else 'down'
                else:
                    direction = 'none'

                results.append({
                    'ticker': ticker,
                    'max_gap': max_gap,
                    'max_gap_date': max_gap_date,
                    'has_large_gap': has_large_gap,
                    'gap_direction': direction
                })

            except Exception as e:
                logger.debug(f"Error detecting gaps for {ticker}: {e}")
                continue

    df = pd.DataFrame(results)

    if not df.empty:
        # Sort by gap size (largest gaps first)
        df = df.sort_values('max_gap', ascending=False)

        large_gaps = df[df['has_large_gap']].shape[0]
        clean_stocks = df[~df['has_large_gap']].shape[0]
        logger.info(f"Gap Analysis: {large_gaps} stocks with gaps > {gap_threshold:.1%}, {clean_stocks} clean stocks")

    return df


def filter_exclude_gaps(momentum_df: pd.DataFrame, stock_data: pd.DataFrame, gap_threshold: float = 0.15) -> pd.DataFrame:
    """
    Apply gap filter to momentum-ranked stocks.

    Args:
        momentum_df: DataFrame with momentum rankings
        stock_data: Raw stock data for gap analysis
        gap_threshold: Gap threshold for exclusion

    Returns:
        Filtered momentum DataFrame excluding stocks with large gaps
    """
    logger.info(f"Applying gap exclusion filter (threshold: {gap_threshold:.1%})...")

    # Get gap analysis
    gap_df = detect_gaps(stock_data, gap_threshold)

    if gap_df.empty:
        logger.warning("No gap data available - returning original momentum rankings")
        return momentum_df

    # Create a set of tickers with large gaps for quick lookup
    gapped_tickers = set(gap_df[gap_df['has_large_gap']]['ticker'].tolist())

    # Filter out gapped stocks
    original_count = len(momentum_df)
    filtered_df = momentum_df[~momentum_df['ticker'].isin(gapped_tickers)].copy()
    excluded_count = original_count - len(filtered_df)

    logger.info(f"Gap Filter: Excluded {excluded_count} stocks with large gaps, {len(filtered_df)} remaining")

    return filtered_df


def apply_all_filters(momentum_df: pd.DataFrame, stock_data: pd.DataFrame,
                     ma_period: int = 100, gap_threshold: float = 0.15) -> pd.DataFrame:
    """
    Apply all trading filters to momentum-ranked stocks.

    This applies the complete Clenow filtering process:
    1. Moving average filter (price > MA)
    2. Gap exclusion filter

    Args:
        momentum_df: DataFrame with momentum rankings
        stock_data: Raw stock data for filter analysis
        ma_period: Moving average period (default 100)
        gap_threshold: Gap threshold for exclusion (default 0.15)

    Returns:
        Filtered momentum DataFrame with additional filter columns
    """
    logger.info("Applying all trading filters...")

    # Start with momentum rankings
    filtered_df = momentum_df.copy()
    original_count = len(filtered_df)

    # 1. Apply moving average filter
    ma_filter_df = filter_above_ma(stock_data, ma_period)
    if not ma_filter_df.empty:
        # Keep only stocks above MA
        above_ma_tickers = set(ma_filter_df[ma_filter_df['above_ma']]['ticker'].tolist())
        filtered_df = filtered_df[filtered_df['ticker'].isin(above_ma_tickers)].copy()

        # Add MA data to results
        ma_data = ma_filter_df.set_index('ticker')[['latest_price', f'ma_{ma_period}', 'price_vs_ma']]
        filtered_df = filtered_df.set_index('ticker').join(ma_data, how='left').reset_index()

        ma_excluded = original_count - len(filtered_df)
        logger.info(f"MA Filter: Excluded {ma_excluded} stocks, {len(filtered_df)} remaining")

    # 2. Apply gap exclusion filter
    if not filtered_df.empty:
        filtered_df = filter_exclude_gaps(filtered_df, stock_data, gap_threshold)

    final_count = len(filtered_df)
    total_excluded = original_count - final_count

    logger.info(f"All Filters Applied: {total_excluded} stocks excluded, {final_count} stocks passed all filters")

    return filtered_df
