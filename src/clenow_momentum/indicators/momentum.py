
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def calculate_exponential_regression_slope(
    prices: pd.Series, period: int = 90
) -> tuple[float, float]:
    """
    Calculate the annualized exponential regression slope and R-squared.

    This is the core momentum calculation from Clenow's strategy.

    Args:
        prices: Series of stock prices
        period: Number of days to look back for regression (default 90)

    Returns:
        Tuple of (annualized_slope, r_squared)
    """
    if len(prices) < period:
        return np.nan, np.nan

    # Get the last 'period' prices
    recent_prices = prices.iloc[-period:].copy()

    # Create log prices for exponential regression
    log_prices = np.log(recent_prices)

    # Create x values (days)
    x = np.arange(len(log_prices))

    # Remove any NaN values
    mask = ~np.isnan(log_prices)
    if mask.sum() < 10:  # Need at least 10 data points
        return np.nan, np.nan

    x_clean = x[mask]
    y_clean = log_prices[mask]

    # Perform linear regression on log prices
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    # Annualize the slope (multiply by 252 trading days)
    annualized_slope = slope * 252

    # R-squared
    r_squared = r_value**2

    return annualized_slope, r_squared


def calculate_momentum_score(prices: pd.Series, period: int = 90) -> float:
    """
    Calculate the momentum score (slope weighted by R-squared).

    Args:
        prices: Series of stock prices
        period: Number of days to look back

    Returns:
        Momentum score (annualized slope * R-squared)
    """
    slope, r_squared = calculate_exponential_regression_slope(prices, period)

    if np.isnan(slope) or np.isnan(r_squared):
        return np.nan

    # Weight the momentum by R-squared (measure of trend strength)
    return slope * r_squared



def calculate_momentum_for_universe(data: pd.DataFrame, period: int = 90) -> pd.DataFrame:
    """
    Calculate momentum scores for all stocks in the universe.

    Args:
        data: DataFrame with stock price data from yfinance with group_by="ticker"
        period: Number of days to look back

    Returns:
        DataFrame with momentum metrics for each stock
    """
    results = []
    processed_count = 0
    skipped_count = 0

    # Handle yfinance group_by="ticker" structure
    # From your screenshot: data structure is like data['AAPL'] -> DataFrame with ['Open', 'High', 'Low', 'Close', 'Volume']
    if isinstance(data.columns, pd.MultiIndex):
        # Get unique ticker symbols from the first level
        tickers = data.columns.get_level_values(0).unique()
        logger.info(f"Calculating momentum scores for {len(tickers)} stocks with group_by ticker structure (period: {period} days)")

        # Log basic data structure info
        logger.debug(f"Processing {len(tickers)} tickers from MultiIndex DataFrame (shape: {data.shape})")
        for ticker in tickers:
            try:
                # Access the ticker's data - this should be a DataFrame with OHLCV columns
                ticker_data = data[ticker]

                # Get the Close prices
                if 'Close' in ticker_data.columns:
                    prices = ticker_data['Close']
                else:
                    logger.debug(f"No Close column found for {ticker}, columns: {ticker_data.columns.tolist()}")
                    skipped_count += 1
                    continue
            except (KeyError, AttributeError) as e:
                logger.debug(f"Error accessing {ticker}: {e}")
                skipped_count += 1
                continue
            # Skip if not enough data
            if len(prices.dropna()) < period:
                logger.debug(f"Skipping {ticker}: insufficient data ({len(prices.dropna())} < {period})")
                skipped_count += 1
                continue

            slope, r_squared = calculate_exponential_regression_slope(prices, period)
            momentum_score = (
                slope * r_squared if not (np.isnan(slope) or np.isnan(r_squared)) else np.nan
            )

            # Calculate additional metrics
            current_price = prices.iloc[-1] if not pd.isna(prices.iloc[-1]) else np.nan
            period_return = (
                ((current_price / prices.iloc[-period]) - 1) * 100
                if len(prices) >= period and not pd.isna(prices.iloc[-period])
                else np.nan
            )

            results.append(
                {
                    "ticker": ticker,
                    "momentum_score": momentum_score,
                    "annualized_slope": slope,
                    "r_squared": r_squared,
                    "current_price": current_price,
                    "period_return_pct": period_return,
                }
            )
            processed_count += 1
    else:
        # Fallback: Simple column structure (single level)
        # This handles cases where data might be pre-processed or single ticker
        tickers = data.columns
        logger.info(f"Calculating momentum scores for {len(tickers)} stocks with simple columns (period: {period} days)")
        for ticker in tickers:
            # Simple column structure - assume it's already price data
            prices = data[ticker]
            # Skip if not enough data
            if len(prices.dropna()) < period:
                logger.debug(f"Skipping {ticker}: insufficient data ({len(prices.dropna())} < {period})")
                skipped_count += 1
                continue

            slope, r_squared = calculate_exponential_regression_slope(prices, period)
            momentum_score = (
                slope * r_squared if not (np.isnan(slope) or np.isnan(r_squared)) else np.nan
            )

            # Calculate additional metrics
            current_price = prices.iloc[-1] if not pd.isna(prices.iloc[-1]) else np.nan
            period_return = (
                ((current_price / prices.iloc[-period]) - 1) * 100
                if len(prices) >= period and not pd.isna(prices.iloc[-period])
                else np.nan
            )

            results.append(
                {
                    "ticker": ticker,
                    "momentum_score": momentum_score,
                    "annualized_slope": slope,
                    "r_squared": r_squared,
                    "current_price": current_price,
                    "period_return_pct": period_return,
                }
            )
            processed_count += 1

    df = pd.DataFrame(results)

    # Sort by momentum score (descending)
    df = df.sort_values("momentum_score", ascending=False, na_position="last")
    df_final = df.reset_index(drop=True)
    # Log summary
    valid_scores = df_final.dropna(subset=['momentum_score'])
    logger.success(f"Momentum calculation complete: {processed_count} processed, {skipped_count} skipped, {len(valid_scores)} valid scores")
    if len(valid_scores) > 0:
        top_stock = valid_scores.iloc[0]
        logger.info(f"Top momentum stock: {top_stock['ticker']} (score: {top_stock['momentum_score']:.3f})")
    return df_final



def get_top_momentum_stocks(momentum_df: pd.DataFrame, top_pct: float = 0.20) -> pd.DataFrame:
    """
    Get the top momentum stocks by percentile.

    Args:
        momentum_df: DataFrame from calculate_momentum_for_universe
        top_pct: Top percentage to select (0.20 = top 20%)

    Returns:
        DataFrame with top momentum stocks
    """
    # Remove NaN momentum scores
    valid_scores = momentum_df.dropna(subset=["momentum_score"])
    logger.info(f"Selecting top {top_pct:.0%} from {len(valid_scores)} stocks with valid momentum scores")

    # Calculate number of stocks to select
    n_stocks = int(len(valid_scores) * top_pct)
    n_stocks = max(1, n_stocks)  # At least 1 stock

    # Select top stocks
    top_stocks = valid_scores.head(n_stocks).copy()
    top_stocks["rank"] = range(1, len(top_stocks) + 1)

    logger.success(f"Selected {len(top_stocks)} top momentum stocks")
    if len(top_stocks) > 0:
        avg_score = top_stocks['momentum_score'].mean()
        logger.info(f"Average momentum score of selected stocks: {avg_score:.3f}")

    return top_stocks
