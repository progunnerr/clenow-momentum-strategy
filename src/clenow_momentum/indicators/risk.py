"""
Risk management and position sizing for Clenow momentum strategy.

This module implements the risk management components of Andreas Clenow's momentum strategy:
- Average True Range (ATR) calculation for volatility measurement
- Position sizing based on 0.1% account risk per trade
- Portfolio construction with equal risk weighting
"""

import pandas as pd
from loguru import logger


def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate True Range for each period.

    True Range is the maximum of:
    1. High - Low
    2. High - Previous Close
    3. Previous Close - Low

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Series of True Range values
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility measurement.

    ATR is a technical indicator that measures volatility by calculating
    the average of true ranges over a specified period.

    Args:
        data: OHLC data with columns ['Open', 'High', 'Low', 'Close']
        period: Number of periods for ATR calculation (default 14)

    Returns:
        Series of ATR values
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Data must contain '{col}' column")

    # Calculate True Range
    true_range = calculate_true_range(data['High'], data['Low'], data['Close'])

    # Calculate ATR using Wilder's smoothing
    # Wilder's smoothing is an EMA with alpha=1/period
    # In pandas, this is achieved with alpha=1/period or span=2*period-1
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()

    return atr


def calculate_atr_for_universe(stock_data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ATR for all stocks in the universe.

    Args:
        stock_data: MultiIndex DataFrame with stock data (yfinance format)
        period: ATR calculation period (default 14)

    Returns:
        DataFrame with ticker and ATR values
    """
    results = []
    processed_count = 0
    failed_count = 0

    logger.info(f"Calculating {period}-day ATR for stock universe...")

    if isinstance(stock_data.columns, pd.MultiIndex):
        # Handle yfinance group_by="ticker" structure
        tickers = stock_data.columns.get_level_values(0).unique()

        for ticker in tickers:
            try:
                ticker_data = stock_data[ticker]

                # Check for required columns
                required_cols = ['High', 'Low', 'Close']
                if not all(col in ticker_data.columns for col in required_cols):
                    failed_count += 1
                    continue

                # Calculate ATR
                atr_series = calculate_atr(ticker_data, period)
                current_atr = atr_series.iloc[-1]

                if pd.isna(current_atr):
                    failed_count += 1
                    continue

                results.append({
                    'ticker': ticker,
                    'atr': float(current_atr),
                    'atr_period': period
                })
                processed_count += 1

            except Exception as e:
                logger.debug(f"Error calculating ATR for {ticker}: {e}")
                failed_count += 1
                continue

    else:
        # Simple column structure - assume columns are tickers with OHLC data
        logger.warning("Simple column structure detected - ATR calculation may not work properly")

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values('atr', ascending=False)
        logger.info(f"ATR calculation complete: {processed_count} successful, {failed_count} failed")

        if processed_count > 0:
            avg_atr = df['atr'].mean()
            logger.info(f"Average ATR across all stocks: ${avg_atr:.2f}")
    else:
        logger.warning("No ATR values calculated")

    return df


def calculate_position_size(account_value: float, risk_per_trade: float,
                          stock_price: float, atr: float,
                          max_position_pct: float = 0.05,
                          stop_loss_multiplier: float = 3.0) -> dict:
    """
    Calculate position size based on Clenow's risk management rules.

    The position size is determined by:
    1. Risk per trade (default 0.1% of account)
    2. Stock's ATR (volatility) with stop loss multiplier
    3. Maximum position size limit (default 5% of account)

    Args:
        account_value: Total account value
        risk_per_trade: Risk per trade as percentage (e.g., 0.001 for 0.1%)
        stock_price: Current stock price
        atr: Stock's Average True Range
        max_position_pct: Maximum position as percentage of account (default 0.05 = 5%)
        stop_loss_multiplier: ATR multiplier for stop loss (default 3.0 = 3x ATR per Clenow)

    Returns:
        Dictionary with position sizing details
    """
    if atr <= 0:
        raise ValueError("ATR must be positive")

    if stock_price <= 0:
        raise ValueError("Stock price must be positive")

    # Calculate risk amount in dollars
    risk_amount = account_value * risk_per_trade

    # Calculate stop loss distance (typically 3x ATR per Clenow)
    stop_loss_distance = atr * stop_loss_multiplier

    # Calculate position size based on ATR and stop loss
    # Position size = Risk Amount / Stop Loss Distance
    shares_based_on_risk = risk_amount / stop_loss_distance

    # Calculate maximum shares based on position limit
    max_position_value = account_value * max_position_pct
    max_shares = max_position_value / stock_price

    # Use the smaller of the two (risk-based or position limit)
    shares = min(shares_based_on_risk, max_shares)
    shares = max(0, int(shares))  # Ensure positive integer

    # Calculate actual investment amount
    investment_amount = shares * stock_price

    # Calculate actual risk (based on stop loss distance)
    actual_risk = shares * stop_loss_distance

    # Calculate position as percentage of account
    position_pct = investment_amount / account_value

    # Calculate stop loss price
    stop_loss_price = stock_price - stop_loss_distance

    return {
        'shares': shares,
        'investment_amount': investment_amount,
        'position_pct': position_pct,
        'target_risk': risk_amount,
        'actual_risk': actual_risk,
        'risk_utilization': actual_risk / risk_amount if risk_amount > 0 else 0,
        'limited_by': 'position_limit' if shares == max_shares else 'risk_limit',
        'stop_loss_price': stop_loss_price,
        'stop_loss_distance': stop_loss_distance,
        'stop_loss_multiplier': stop_loss_multiplier
    }


def build_portfolio(filtered_stocks: pd.DataFrame, stock_data: pd.DataFrame,
                   account_value: float = 1000000, risk_per_trade: float = 0.001,
                   atr_period: int = 14, allocation_method: str = "equal_risk",
                   stop_loss_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Build complete portfolio with position sizing for all filtered stocks.

    Args:
        filtered_stocks: DataFrame with filtered momentum stocks
        stock_data: Raw stock data for ATR calculation
        account_value: Total account value (default $1M)
        risk_per_trade: Risk per trade as percentage (default 0.1%)
        atr_period: ATR calculation period (default 14)
        allocation_method: "equal_risk" or "equal_dollar"
        stop_loss_multiplier: ATR multiplier for stop loss (default 3.0 = 3x ATR per Clenow)

    Returns:
        DataFrame with complete portfolio including position sizes
    """
    logger.info(f"Building portfolio for {len(filtered_stocks)} stocks...")
    logger.info(f"Account Value: ${account_value:,.0f}")
    logger.info(f"Allocation Method: {allocation_method}")
    
    # Handle empty filtered stocks
    if filtered_stocks.empty:
        logger.warning("No stocks to build portfolio with")
        return pd.DataFrame()
    
    if allocation_method == "equal_risk":
        logger.info(f"Risk per trade: {risk_per_trade:.3%}")
    else:
        logger.info(f"Equal dollar allocation: ${account_value / len(filtered_stocks):,.0f} per position")

    # Calculate ATR for all stocks
    atr_df = calculate_atr_for_universe(stock_data, atr_period)

    if atr_df.empty:
        logger.error("No ATR data available for portfolio construction")
        return pd.DataFrame()

    # Merge filtered stocks with ATR data
    portfolio_df = filtered_stocks.merge(
        atr_df[['ticker', 'atr']],
        on='ticker',
        how='inner'
    ).copy()

    if portfolio_df.empty:
        logger.warning("No stocks remained after ATR merge")
        return pd.DataFrame()

    # Calculate position sizes based on allocation method
    portfolio_results = []
    total_investment = 0
    total_risk = 0

    if allocation_method == "equal_dollar":
        # Equal dollar allocation
        position_value = account_value / len(portfolio_df)
        logger.info(f"Target position size: ${position_value:,.0f}")

        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']

            # Get current price (prefer latest_price from filters, fall back to current_price)
            current_price = row.get('latest_price', row.get('current_price', 0))

            if pd.isna(current_price) or current_price <= 0:
                logger.warning(f"No valid price for {ticker}, skipping")
                continue

            # Calculate shares for equal dollar amount
            shares = int(position_value / current_price)
            if shares == 0:
                logger.warning(f"Position too small for {ticker} at ${current_price:.2f}, skipping")
                continue

            investment = shares * current_price
            position_pct = investment / account_value

            # Calculate stop loss and risk
            stop_loss_distance = row['atr'] * stop_loss_multiplier
            stop_loss_price = current_price - stop_loss_distance
            actual_risk = shares * stop_loss_distance

            result = {
                'ticker': ticker,
                'momentum_score': row.get('momentum_score', 0),
                'current_price': current_price,
                'atr': row['atr'],
                'shares': shares,
                'investment': investment,
                'position_pct': position_pct,
                'target_risk': position_value * risk_per_trade,  # What risk would be
                'actual_risk': actual_risk,  # Actual stop-loss based risk
                'risk_utilization': 1.0,  # Always 100% for equal dollar
                'limited_by': 'equal_dollar',
                'stop_loss_price': stop_loss_price,
                'stop_loss_distance': stop_loss_distance,
                'stop_loss_multiplier': stop_loss_multiplier
            }

            # Add filter-specific data if available
            if 'price_vs_ma' in row:
                result['price_vs_ma'] = row['price_vs_ma']

            portfolio_results.append(result)
            total_investment += investment
            total_risk += actual_risk

    else:
        # Risk-based allocation (original method)
        for _, row in portfolio_df.iterrows():
            ticker = row['ticker']

            # Get current price (prefer latest_price from filters, fall back to current_price)
            current_price = row.get('latest_price', row.get('current_price', 0))

            if pd.isna(current_price) or current_price <= 0:
                logger.warning(f"No valid price for {ticker}, skipping")
                continue

            try:
                # Calculate position size
                position = calculate_position_size(
                    account_value=account_value,
                    risk_per_trade=risk_per_trade,
                    stock_price=current_price,
                    atr=row['atr'],
                    stop_loss_multiplier=stop_loss_multiplier
                )

                # Add to results
                result = {
                    'ticker': ticker,
                    'momentum_score': row.get('momentum_score', 0),
                    'current_price': current_price,
                    'atr': row['atr'],
                    'shares': position['shares'],
                    'investment': position['investment_amount'],
                    'position_pct': position['position_pct'],
                    'target_risk': position['target_risk'],
                    'actual_risk': position['actual_risk'],
                    'risk_utilization': position['risk_utilization'],
                    'limited_by': position['limited_by'],
                    'stop_loss_price': position['stop_loss_price'],
                    'stop_loss_distance': position['stop_loss_distance'],
                    'stop_loss_multiplier': position['stop_loss_multiplier']
                }

                # Add filter-specific data if available
                if 'price_vs_ma' in row:
                    result['price_vs_ma'] = row['price_vs_ma']

                portfolio_results.append(result)
                total_investment += position['investment_amount']
                total_risk += position['actual_risk']

            except Exception as e:
                logger.warning(f"Error calculating position for {ticker}: {e}")
                continue

    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame(portfolio_results)

    if portfolio_df.empty:
        logger.warning("No valid positions created")
        return pd.DataFrame()

    # Sort by investment amount (largest positions first)
    portfolio_df = portfolio_df.sort_values('investment', ascending=False).reset_index(drop=True)

    # Add portfolio statistics
    portfolio_df['portfolio_rank'] = range(1, len(portfolio_df) + 1)

    # Calculate portfolio statistics
    total_positions = len(portfolio_df)
    cash_remaining = account_value - total_investment
    capital_utilization = total_investment / account_value
    avg_position_size = total_investment / total_positions if total_positions > 0 else 0

    logger.info("Portfolio Construction Complete:")
    logger.info(f"  Total Positions: {total_positions}")
    logger.info(f"  Total Investment: ${total_investment:,.0f}")
    logger.info(f"  Cash Remaining: ${cash_remaining:,.0f}")
    logger.info(f"  Capital Utilization: {capital_utilization:.1%}")
    logger.info(f"  Average Position Size: ${avg_position_size:,.0f}")
    logger.info(f"  Total Portfolio Risk: ${total_risk:,.0f}")

    return portfolio_df


def apply_risk_limits(portfolio_df: pd.DataFrame, max_positions: int = 20,
                     min_position_value: float = 10000,
                     skip_minimum_filter: bool = False) -> pd.DataFrame:
    """
    Apply risk limits to the portfolio.

    Args:
        portfolio_df: Portfolio DataFrame
        max_positions: Maximum number of positions (default 20)
        min_position_value: Minimum position value (default $10,000)

    Returns:
        Filtered portfolio DataFrame
    """
    if portfolio_df.empty:
        return portfolio_df

    original_count = len(portfolio_df)

    # Filter by minimum position value (skip for equal dollar allocation)
    if not skip_minimum_filter:
        portfolio_df = portfolio_df[portfolio_df['investment'] >= min_position_value].copy()
        logger.info(f"Minimum position filter applied: ${min_position_value:,.0f}")
    else:
        logger.info("Minimum position filter skipped for equal dollar allocation")

    # Limit to maximum number of positions (keep largest positions)
    portfolio_df = portfolio_df.head(max_positions).copy()

    # Re-rank after filtering
    portfolio_df['portfolio_rank'] = range(1, len(portfolio_df) + 1)

    final_count = len(portfolio_df)
    excluded_count = original_count - final_count

    logger.info(f"Risk limits applied: {excluded_count} positions excluded, {final_count} positions remain")

    return portfolio_df
