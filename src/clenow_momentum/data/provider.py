import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# Specialized data-source helpers
from .sources.sp500_wikipedia import fetch_sp500_tickers_from_wikipedia
from .sources.yfinance_source import (
    convert_ticker_for_yfinance,
    download_index_data,
    download_stock_data,
)


def get_sp500_tickers(use_cache: bool = True, max_age_hours: int = 24) -> list[str]:
    """
    Get S&P 500 market universe constituents from Wikipedia with optional caching.

    Args:
        use_cache: Whether to use cached market universe data (default True)
        max_age_hours: Maximum cache age in hours (default 24)

    Returns:
        List of ticker symbols in the S&P 500 market universe (e.g., ['AAPL', 'MSFT', ...])

    Raises:
        RuntimeError: If unable to fetch market universe constituents from Wikipedia
    """
    cache_dir = Path("data/cache")
    cache_file = cache_dir / "sp500_tickers.pkl"

    # Try to load from cache if enabled
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            cache_age = datetime.now() - cache_data["timestamp"]
            if cache_age < timedelta(hours=max_age_hours):
                tickers = cache_data["tickers"]
                logger.info(
                    f"Loaded {len(tickers)} tickers from cache "
                    f"(age: {cache_age.total_seconds()/3600:.1f}h)"
                )
                return tickers
            logger.debug(f"Cache expired (age: {cache_age}), fetching fresh data")
        except Exception as e:
            logger.warning(f"Failed to load ticker cache: {e}")

    # Fetch from Wikipedia
    try:
        raw_tickers = fetch_sp500_tickers_from_wikipedia(timeout=10)
        if not raw_tickers:
            raise RuntimeError("Wikipedia returned no tickers")

        tickers = [convert_ticker_for_yfinance(ticker) for ticker in raw_tickers]
        logger.success(
            f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia"
        )

        # Save to cache
        if use_cache:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    "tickers": tickers,
                    "timestamp": datetime.now(),
                    "count": len(tickers),
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.debug(f"Saved {len(tickers)} tickers to cache")
            except Exception as e:
                logger.warning(f"Failed to save ticker cache: {e}")

        return tickers

    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 tickers from Wikipedia: {e}")
        raise RuntimeError(f"Unable to fetch S&P 500 tickers: {e}") from e


def get_stock_data(
    tickers: list[str], period: str = "1y", use_cache: bool = True
) -> pd.DataFrame | None:
    """
    Fetch historical market data for market universe securities using yfinance.

    Uses the DataCache system for efficient caching to data/cache folder.

    Args:
        tickers: List of ticker symbols from the market universe
        period: Period for data (1y, 2y, 5y, etc.)
        use_cache: Whether to use cached data (default True)

    Returns:
        DataFrame with OHLCV data for all market universe securities, or None if fetch fails
    """
    # Check cache first if enabled
    if use_cache:
        from .cache import DataCache

        cache = DataCache(cache_dir="data/cache")
        cached_data = cache.get(tickers, period, max_age_hours=24)
        if cached_data is not None:
            return cached_data

    try:
        import yfinance as yf

        logger.info(
            f"Fetching stock data for {len(tickers)} tickers (period: {period})"
        )
        data = download_stock_data(tickers, period=period, yf=yf)

        if data is not None and not data.empty:
            logger.success(f"Successfully fetched stock data: {data.shape}")
            # Save to cache for next time
            if use_cache:
                cache.save(data, tickers, period)
            return data
        logger.warning("Received empty stock data from yfinance")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch stock data: {e}")
        return None


def get_sp500_index_data(
    period: str = "1y", use_cache: bool = True
) -> pd.DataFrame | None:
    """
    Fetch S&P 500 index data (market universe benchmark) for market regime analysis.

    Args:
        period: Period for data (1y, 2y, 5y, etc.)
        use_cache: Whether to use cached data (default True)

    Returns:
        DataFrame with S&P 500 market benchmark OHLCV data, or None if fetch fails
    """
    # Check cache first if enabled
    if use_cache:
        from .cache import DataCache

        cache = DataCache(cache_dir="data/cache")
        cached_data = cache.get(["^GSPC"], period, max_age_hours=24)
        if cached_data is not None:
            # Extract single ticker data from MultiIndex format
            if isinstance(cached_data.columns, pd.MultiIndex):
                return cached_data["^GSPC"]
            return cached_data

    try:
        import yfinance as yf

        logger.info(f"Fetching S&P 500 index data (period: {period})")
        data = download_index_data("^GSPC", period=period, yf=yf)

        if data is not None and not data.empty:
            logger.success(f"Successfully fetched S&P 500 index data: {data.shape}")
            # Save to cache for next time
            if use_cache:
                cache.save(data, ["^GSPC"], period)
            return data
        logger.warning("Received empty S&P 500 index data from yfinance")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 index data: {e}")
        return None
