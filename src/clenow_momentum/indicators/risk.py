"""
Risk management and position sizing for Clenow momentum strategy.

This module implements the risk management components of Andreas Clenow's momentum strategy:
- Average True Range (ATR) calculation for volatility measurement
- Position sizing based on 0.1% account risk per trade
- Portfolio construction with equal risk weighting
"""

import numpy as np
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

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


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
    required_columns = ["High", "Low", "Close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Data must contain '{col}' column")

    # Calculate True Range
    tr = calculate_true_range(data["High"], data["Low"], data["Close"])

    # Initialize ATR using Wilder's method
    # Start with simple average of first period TR values, then use smoothed formula
    atr = tr.copy()

    # Seed with simple average of first period True Range values
    first_atr = tr.iloc[:period].mean()
    atr.iloc[:period] = np.nan
    atr.iloc[period - 1] = first_atr

    # Apply Wilder's smoothing formula for subsequent values
    # ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period
    for i in range(period, len(atr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

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
                required_cols = ["High", "Low", "Close"]
                if not all(col in ticker_data.columns for col in required_cols):
                    failed_count += 1
                    continue

                # Some constituents have leading empty rows in a one-year
                # download window. Calculate ATR on valid OHLC rows only.
                ticker_data = ticker_data[required_cols].dropna()
                if len(ticker_data) < period:
                    logger.debug(
                        f"Insufficient valid OHLC data for {ticker}: {len(ticker_data)} < {period}"
                    )
                    failed_count += 1
                    continue

                # Calculate ATR
                atr_series = calculate_atr(ticker_data, period)
                current_atr = atr_series.iloc[-1]

                if pd.isna(current_atr):
                    failed_count += 1
                    continue

                results.append({"ticker": ticker, "atr": float(current_atr), "atr_period": period})
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
        df = df.sort_values("atr", ascending=False)
        logger.info(
            f"ATR calculation complete: {processed_count} successful, {failed_count} failed"
        )

        if processed_count > 0:
            avg_atr = df["atr"].mean()
            logger.info(f"Average ATR across all stocks: ${avg_atr:.2f}")
    else:
        logger.warning("No ATR values calculated")

    return df


def calculate_position_size(
    account_value: float,
    risk_per_trade: float,
    stock_price: float,
    atr: float,
    max_position_pct: float = 0.05,
    stop_loss_multiplier: float = 3.0,
) -> dict:
    """
    Calculate position size based on Clenow's risk management rules.

    Using Clenow's formula: Shares = (Account Value × Risk Factor) / ATR

    This targets a specific daily portfolio impact per position based on the stock's
    average daily volatility (ATR). The risk_per_trade parameter sets the target
    daily impact as a percentage of account value.

    The position size is determined by:
    1. Target daily impact (risk_per_trade × account_value)
    2. Stock's average daily volatility (ATR)
    3. Maximum position size limit constraint

    Args:
        account_value: Total account value
        risk_per_trade: Target daily impact as decimal (e.g., 0.001 for 0.1%)
        stock_price: Current stock price
        atr: Stock's Average True Range (daily volatility)
        max_position_pct: Maximum position as percentage of account (default 0.05 = 5%)
        stop_loss_multiplier: ATR multiplier for stop loss calculation (default 3.0)

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

    # Calculate position size using Clenow's correct formula:
    # Shares = (Account Value × Risk Factor) / ATR
    # This targets a specific daily impact per position, NOT stop loss risk
    shares_based_on_risk = risk_amount / atr

    # Calculate maximum shares based on position limit
    max_position_value = account_value * max_position_pct
    max_shares = max_position_value / stock_price

    # Determine limiting factor before rounding to avoid misidentification
    limited_by_position = shares_based_on_risk > max_shares

    # Use the smaller of the two (risk-based or position limit)
    shares_float = min(shares_based_on_risk, max_shares)
    shares = max(0, int(shares_float))  # Ensure positive integer

    # Calculate actual investment amount
    investment_amount = shares * stock_price

    # Calculate realized ATR impact after whole-share rounding, and the
    # wider stop exposure implied by the configured ATR stop multiplier.
    actual_atr_impact = shares * atr
    stop_loss_risk = shares * stop_loss_distance

    # Calculate position as percentage of account
    position_pct = investment_amount / account_value

    # Calculate stop loss price
    stop_loss_price = stock_price - stop_loss_distance

    return {
        "shares": shares,
        "investment_amount": investment_amount,
        "position_pct": position_pct,
        "target_atr_impact": risk_amount,
        "actual_atr_impact": actual_atr_impact,
        "stop_loss_risk": stop_loss_risk,
        # Backward-compatible aliases used by trading/rebalancing code.
        "target_risk": risk_amount,
        "actual_risk": stop_loss_risk,
        "risk_utilization": actual_atr_impact / risk_amount if risk_amount > 0 else 0,
        "stop_loss_risk_utilization": stop_loss_risk / risk_amount if risk_amount > 0 else 0,
        "limited_by": "position_limit" if limited_by_position else "risk_limit",
        "stop_loss_price": stop_loss_price,
        "stop_loss_distance": stop_loss_distance,
        "stop_loss_multiplier": stop_loss_multiplier,
    }


def calculate_equal_dollar_position(
    row, account_value, num_stocks, risk_per_trade, stop_loss_multiplier
):
    """
    Calculate position size for equal dollar allocation.

    Args:
        row: Stock data row
        account_value: Total account value
        num_stocks: Number of stocks in portfolio
        risk_per_trade: Risk per trade as percentage
        stop_loss_multiplier: ATR multiplier for stop loss

    Returns:
        Position dictionary or None if invalid
    """
    ticker = row["ticker"]
    position_value = account_value / num_stocks

    # Get current price
    current_price = row.get("latest_price", row.get("current_price", 0))

    if pd.isna(current_price) or current_price <= 0:
        logger.warning(f"No valid price for {ticker}, skipping")
        return None

    # Calculate shares for equal dollar amount
    shares = int(position_value / current_price)
    if shares == 0:
        logger.warning(f"Position too small for {ticker} at ${current_price:.2f}, skipping")
        return None

    investment = shares * current_price
    position_pct = investment / account_value

    # Calculate stop loss and risk
    stop_loss_distance = row["atr"] * stop_loss_multiplier
    stop_loss_price = current_price - stop_loss_distance
    actual_atr_impact = shares * row["atr"]
    stop_loss_risk = shares * stop_loss_distance
    target_atr_impact = position_value * risk_per_trade

    result = {
        "ticker": ticker,
        "momentum_score": row.get("momentum_score", 0),
        "current_price": current_price,
        "atr": row["atr"],
        "shares": shares,
        "investment": investment,
        "position_pct": position_pct,
        "target_atr_impact": target_atr_impact,
        "actual_atr_impact": actual_atr_impact,
        "stop_loss_risk": stop_loss_risk,
        "target_risk": target_atr_impact,
        "actual_risk": stop_loss_risk,
        "risk_utilization": 1.0,
        "limited_by": "equal_dollar",
        "stop_loss_price": stop_loss_price,
        "stop_loss_distance": stop_loss_distance,
        "stop_loss_multiplier": stop_loss_multiplier,
    }

    # Add filter-specific data if available
    if "price_vs_ma" in row:
        result["price_vs_ma"] = row["price_vs_ma"]

    return result


def calculate_risk_based_position(
    row, account_value, risk_per_trade, max_position_pct, stop_loss_multiplier
):
    """
    Calculate position size for risk-based allocation.

    Args:
        row: Stock data row
        account_value: Total account value
        risk_per_trade: Risk per trade as percentage
        max_position_pct: Maximum position size as percentage
        stop_loss_multiplier: ATR multiplier for stop loss

    Returns:
        Position dictionary or None if invalid
    """
    ticker = row["ticker"]
    current_price = row.get("latest_price", row.get("current_price", 0))

    if pd.isna(current_price) or current_price <= 0:
        logger.warning(f"No valid price for {ticker}, skipping")
        return None

    try:
        # Calculate position size
        position = calculate_position_size(
            account_value=account_value,
            risk_per_trade=risk_per_trade,
            stock_price=current_price,
            atr=row["atr"],
            max_position_pct=max_position_pct,
            stop_loss_multiplier=stop_loss_multiplier,
        )

        # Add to results
        result = {
            "ticker": ticker,
            "momentum_score": row.get("momentum_score", 0),
            "current_price": current_price,
            "atr": row["atr"],
            "shares": position["shares"],
            "investment": position["investment_amount"],
            "position_pct": position["position_pct"],
            "target_atr_impact": position["target_atr_impact"],
            "actual_atr_impact": position["actual_atr_impact"],
            "stop_loss_risk": position["stop_loss_risk"],
            "target_risk": position["target_risk"],
            "actual_risk": position["actual_risk"],
            "risk_utilization": position["risk_utilization"],
            "stop_loss_risk_utilization": position["stop_loss_risk_utilization"],
            "limited_by": position["limited_by"],
            "stop_loss_price": position["stop_loss_price"],
            "stop_loss_distance": position["stop_loss_distance"],
            "stop_loss_multiplier": position["stop_loss_multiplier"],
        }

        # Add filter-specific data if available
        if "price_vs_ma" in row:
            result["price_vs_ma"] = row["price_vs_ma"]

        return result

    except Exception as e:
        logger.warning(f"Error calculating position for {ticker}: {e}")
        return None


def build_portfolio(
    filtered_stocks: pd.DataFrame,
    stock_data: pd.DataFrame,
    account_value: float = 1000000,
    risk_per_trade: float = 0.001,
    atr_period: int = 14,
    allocation_method: str = "equal_risk",
    stop_loss_multiplier: float = 3.0,
    max_position_pct: float = 0.05,
    max_positions: int | None = None,
    min_position_value: float | None = None,
) -> pd.DataFrame:
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
        max_position_pct: Maximum position size as percentage of account (default 0.05 = 5%)
        max_positions: Optional maximum number of valid positions to keep after sizing
        min_position_value: Optional minimum investment required after sizing

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

    drop_reasons = []

    if allocation_method == "equal_risk":
        logger.info(f"Risk per trade: {risk_per_trade:.3%}")
    else:
        logger.info(
            f"Equal dollar allocation: ${account_value / len(filtered_stocks):,.0f} per position"
        )

    # Calculate ATR for all stocks
    atr_df = calculate_atr_for_universe(stock_data, atr_period)

    if atr_df.empty:
        logger.error("No ATR data available for portfolio construction")
        return pd.DataFrame()

    # Merge filtered stocks with ATR data
    portfolio_df = filtered_stocks.merge(atr_df[["ticker", "atr"]], on="ticker", how="inner").copy()

    portfolio_tickers = set(portfolio_df["ticker"])
    missing_atr_tickers = [
        ticker for ticker in filtered_stocks["ticker"].tolist() if ticker not in portfolio_tickers
    ]
    for ticker in missing_atr_tickers:
        reason = f"ATR unavailable after {atr_period}-day calculation"
        drop_reasons.append({"ticker": ticker, "stage": "ATR", "reason": reason})
        logger.info(f"Dropping {ticker}: {reason}")

    if portfolio_df.empty:
        logger.warning("No stocks remained after ATR merge")
        empty = pd.DataFrame()
        empty.attrs["drop_reasons"] = drop_reasons
        return empty

    # Calculate position sizes based on allocation method
    portfolio_results = []

    if allocation_method == "equal_dollar":
        # Equal dollar allocation
        position_value = account_value / len(portfolio_df)
        logger.info(f"Target position size: ${position_value:,.0f}")

        for _, row in portfolio_df.iterrows():
            result = calculate_equal_dollar_position(
                row, account_value, len(portfolio_df), risk_per_trade, stop_loss_multiplier
            )
            if result:
                portfolio_results.append(result)
    else:
        # Risk-based allocation (original method)
        for _, row in portfolio_df.iterrows():
            result = calculate_risk_based_position(
                row, account_value, risk_per_trade, max_position_pct, stop_loss_multiplier
            )
            if result:
                portfolio_results.append(result)

    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame(portfolio_results)

    if portfolio_df.empty:
        logger.warning("No valid positions created")
        empty = pd.DataFrame()
        empty.attrs["drop_reasons"] = drop_reasons
        return empty

    # The incoming data is already sorted by momentum score. Preserve this ranking.
    portfolio_df = portfolio_df.reset_index(drop=True)

    # Apply validity and cash constraints in momentum rank order. If a higher
    # ranked candidate is dropped, lower ranked candidates can backfill until
    # max_positions valid holdings have been selected.
    selected_positions = []
    cumulative_investment = 0.0

    for _, row in portfolio_df.iterrows():
        ticker = row["ticker"]
        position_cost = row["investment"]

        if min_position_value is not None and position_cost < min_position_value:
            if row["shares"] <= 0:
                reason = (
                    f"sized to 0 shares because target ATR impact "
                    f"${row['target_atr_impact']:,.0f} is below ATR ${row['atr']:,.2f}"
                )
            else:
                reason = (
                    f"investment ${position_cost:,.0f} is below minimum position "
                    f"value ${min_position_value:,.0f}"
                )
            drop_reasons.append({"ticker": ticker, "stage": "position_size", "reason": reason})
            logger.info(f"Dropping {ticker}: {reason}")
            continue

        # Check if we can afford this position.
        if cumulative_investment + position_cost > account_value:
            reason = (
                f"insufficient cash for ${position_cost:,.0f} position; "
                f"${account_value - cumulative_investment:,.0f} remaining"
            )
            drop_reasons.append({"ticker": ticker, "stage": "cash", "reason": reason})
            logger.info(
                f"Cash constraint: Excluding {ticker} "
                f"(would need ${position_cost:,.0f}, "
                f"only ${account_value - cumulative_investment:,.0f} remaining)"
            )
            continue

        selected_positions.append(row)
        cumulative_investment += position_cost

        if max_positions is not None and len(selected_positions) >= max_positions:
            break

    # Replace portfolio with constrained version.
    if selected_positions:
        portfolio_df = pd.DataFrame(selected_positions)
        portfolio_df = portfolio_df.reset_index(drop=True)
    else:
        logger.warning("No positions fit within position and cash constraints")
        empty = pd.DataFrame()
        empty.attrs["drop_reasons"] = drop_reasons
        return empty

    # Add portfolio statistics
    portfolio_df["portfolio_rank"] = range(1, len(portfolio_df) + 1)
    portfolio_df.attrs["drop_reasons"] = drop_reasons

    # Calculate portfolio statistics
    total_positions = len(portfolio_df)
    total_investment = portfolio_df["investment"].sum()
    total_atr_impact = portfolio_df["actual_atr_impact"].sum()
    total_stop_loss_risk = portfolio_df["stop_loss_risk"].sum()
    cash_remaining = account_value - total_investment
    capital_utilization = total_investment / account_value
    avg_position_size = total_investment / total_positions if total_positions > 0 else 0

    logger.info("Portfolio Construction Complete:")
    logger.info(f"  Total Positions: {total_positions}")
    logger.info(f"  Total Investment: ${total_investment:,.0f}")
    logger.info(f"  Cash Remaining: ${cash_remaining:,.0f}")
    logger.info(f"  Capital Utilization: {capital_utilization:.1%}")
    logger.info(f"  Average Position Size: ${avg_position_size:,.0f}")
    logger.info(f"  Total ATR Impact: ${total_atr_impact:,.0f}")
    logger.info(
        f"  Total {stop_loss_multiplier:g}x ATR Stop Exposure: ${total_stop_loss_risk:,.0f}"
    )

    return portfolio_df


def apply_risk_limits(
    portfolio_df: pd.DataFrame,
    max_positions: int = 20,
    min_position_value: float = 10000,
    skip_minimum_filter: bool = False,
) -> pd.DataFrame:
    """
    Apply risk limits to the portfolio.

    Args:
        portfolio_df: Portfolio DataFrame
        max_positions: Maximum number of positions. If None, this filter is skipped. (default 20)
        min_position_value: Minimum position value (default $10,000)

    Returns:
        Filtered portfolio DataFrame
    """
    if portfolio_df.empty:
        return portfolio_df

    drop_reasons = list(portfolio_df.attrs.get("drop_reasons", []))
    original_count = len(portfolio_df)

    # Filter by minimum position value (skip for equal dollar allocation)
    if not skip_minimum_filter:
        below_minimum = portfolio_df[portfolio_df["investment"] < min_position_value]
        for _, row in below_minimum.iterrows():
            if row.get("shares", 0) <= 0 and "target_atr_impact" in row:
                reason = (
                    f"sized to 0 shares because target ATR impact "
                    f"${row['target_atr_impact']:,.0f} is below ATR ${row['atr']:,.2f}"
                )
            else:
                reason = (
                    f"investment ${row['investment']:,.0f} is below minimum "
                    f"position value ${min_position_value:,.0f}"
                )
            drop_reasons.append(
                {"ticker": row["ticker"], "stage": "position_size", "reason": reason}
            )

        portfolio_df = portfolio_df[portfolio_df["investment"] >= min_position_value].copy()
        logger.info(f"Minimum position filter applied: ${min_position_value:,.0f}")
    else:
        logger.info("Minimum position filter skipped for equal dollar allocation")

    # Limit to maximum number of positions (keep highest momentum stocks)
    if max_positions is not None:
        excluded_by_rank = portfolio_df.iloc[max_positions:]
        for _, row in excluded_by_rank.iterrows():
            drop_reasons.append(
                {
                    "ticker": row["ticker"],
                    "stage": "max_positions",
                    "reason": f"below top {max_positions} after risk filters",
                }
            )
        portfolio_df = portfolio_df.head(max_positions).copy()

    # Re-rank after filtering
    portfolio_df["portfolio_rank"] = range(1, len(portfolio_df) + 1)
    portfolio_df.attrs["drop_reasons"] = drop_reasons

    final_count = len(portfolio_df)
    excluded_count = original_count - final_count

    logger.info(
        f"Risk limits applied: {excluded_count} positions excluded, {final_count} positions remain"
    )

    return portfolio_df
