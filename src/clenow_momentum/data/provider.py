"""Data provider: generic universe-aware entry points + back-compat wrappers.

Public API
----------
get_universe_tickers(symbol)   — constituents for any registered universe
get_benchmark_data(symbol)     — regime-detection ETF OHLCV (SPY, IWB, …)
get_index_data(symbol)         — index-quote OHLCV (^GSPC, ^RUI, …)
get_stock_data(tickers)        — OHLCV for an arbitrary ticker list

Back-compat wrappers (unchanged semantics)
-----------------------------------------
get_sp500_tickers()            → get_universe_tickers("SP500")
get_sp500_index_data()         → get_index_data("SP500")
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from .sources.sp500_wikipedia import fetch_index_constituents_from_wikipedia
from .sources.yfinance_source import (
    convert_ticker_for_yfinance,
    download_index_data,
    download_stock_data,
)
from .universes import get_universe_spec

# ---------------------------------------------------------------------------
# Generic universe entry points
# ---------------------------------------------------------------------------

def get_universe_tickers(
    symbol: str,
    use_cache: bool = True,
    max_age_hours: int = 24,
) -> list[str]:
    """Get market universe constituents for any registered universe.

    Args:
        symbol:        IndexSymbol string ("SP500", "RUSSELL1000", …).
        use_cache:     Whether to use cached constituent list (default True).
        max_age_hours: Maximum cache age in hours (default 24).

    Returns:
        List of yfinance-normalised ticker symbols.

    Raises:
        ValueError:   If symbol is not registered in UNIVERSES.
        RuntimeError: If unable to fetch constituents from Wikipedia.
    """
    spec = get_universe_spec(symbol)
    cache_dir = Path("data/cache")
    cache_file = cache_dir / f"{spec.symbol}_tickers.pkl"

    if use_cache:
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            cache_age = datetime.now() - cache_data["timestamp"]  # noqa: DTZ005
            if cache_age < timedelta(hours=max_age_hours):
                tickers = cache_data["tickers"]
                logger.info(
                    f"Loaded {len(tickers)} {spec.display_name} tickers from cache "
                    f"(age: {cache_age.total_seconds() / 3600:.1f}h)"
                )
                return tickers
            logger.debug(f"Ticker cache expired for {spec.symbol} (age: {cache_age}), fetching fresh")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to load ticker cache for {spec.symbol}: {e}")

    try:
        constituents = get_universe_constituents(
            spec.symbol, use_cache=use_cache, max_age_hours=max_age_hours
        )
        tickers = constituents["ticker"].dropna().astype(str).tolist()
        if not tickers:
            raise RuntimeError(f"Wikipedia returned no tickers for {spec.display_name}")

        return tickers

    except Exception as e:
        logger.error(f"Failed to fetch {spec.display_name} tickers: {e}")
        raise RuntimeError(f"Unable to fetch {spec.display_name} tickers: {e}") from e


def get_universe_constituents(
    symbol: str,
    use_cache: bool = True,
    max_age_hours: int = 24,
) -> pd.DataFrame:
    """Get normalized universe constituents with optional metadata.

    Args:
        symbol:        IndexSymbol string ("SP500", "RUSSELL1000", ...).
        use_cache:     Whether to use cached constituent metadata.
        max_age_hours: Maximum cache age in hours.

    Returns:
        DataFrame with ticker, source_symbol, company_name, and sector.

    Raises:
        ValueError:   If symbol is not registered in UNIVERSES.
        RuntimeError: If unable to fetch constituents from Wikipedia.
    """
    spec = get_universe_spec(symbol)
    cache_dir = Path("data/cache")
    constituents_cache_file = cache_dir / f"{spec.symbol}_constituents.pkl"
    ticker_cache_file = cache_dir / f"{spec.symbol}_tickers.pkl"

    if use_cache:
        try:
            with open(constituents_cache_file, "rb") as f:
                cache_data = pickle.load(f)
            cache_age = datetime.now() - cache_data["timestamp"]  # noqa: DTZ005
            if cache_age < timedelta(hours=max_age_hours):
                constituents = cache_data["constituents"]
                logger.info(
                    f"Loaded {len(constituents)} {spec.display_name} constituents from cache "
                    f"(age: {cache_age.total_seconds() / 3600:.1f}h)"
                )
                return constituents
            logger.debug(
                f"Constituent cache expired for {spec.symbol} (age: {cache_age}), fetching fresh"
            )
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to load constituent cache for {spec.symbol}: {e}")

    try:
        raw_constituents = fetch_index_constituents_from_wikipedia(spec, timeout=10)
        if raw_constituents.empty:
            raise RuntimeError(f"Wikipedia returned no constituents for {spec.display_name}")

        constituents = raw_constituents.copy()
        constituents["source_symbol"] = constituents["source_symbol"].astype(str)
        constituents["ticker"] = constituents["source_symbol"].map(convert_ticker_for_yfinance)

        for col in ("company_name", "sector"):
            if col not in constituents.columns:
                constituents[col] = pd.NA

        constituents = constituents[
            ["ticker", "source_symbol", "company_name", "sector"]
        ].reset_index(drop=True)
        tickers = constituents["ticker"].dropna().astype(str).tolist()

        logger.success(
            f"Fetched {len(constituents)} {spec.display_name} constituents from Wikipedia"
        )

        if use_cache:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now()  # noqa: DTZ005
                with open(constituents_cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "constituents": constituents,
                            "timestamp": timestamp,
                            "count": len(constituents),
                        },
                        f,
                    )
                with open(ticker_cache_file, "wb") as f:
                    pickle.dump(
                        {"tickers": tickers, "timestamp": timestamp, "count": len(tickers)},
                        f,
                    )
                logger.debug(
                    f"Saved {spec.symbol} constituent cache ({len(constituents)} rows)"
                )
            except Exception as e:
                logger.warning(f"Failed to save constituent cache for {spec.symbol}: {e}")

        return constituents

    except Exception as e:
        logger.error(f"Failed to fetch {spec.display_name} constituents: {e}")
        raise RuntimeError(
            f"Unable to fetch {spec.display_name} constituents: {e}"
        ) from e


def _fetch_single_ticker_data(
    ticker: str,
    period: str,
    universe: str,
    display_name: str,
    data_kind: str,
    use_cache: bool,
) -> pd.DataFrame | None:
    """Fetch OHLCV data for a single ticker with caching.

    Shared implementation for get_benchmark_data and get_index_data.
    """
    cache = None
    if use_cache:
        from .cache import DataCache
        cache = DataCache(cache_dir="data/cache")
        cached = cache.get([ticker], period, max_age_hours=24, universe=universe, data_kind=data_kind)
        if cached is not None:
            if isinstance(cached.columns, pd.MultiIndex):
                return cached[ticker]
            return cached

    try:
        import yfinance as yf
        logger.info(f"Fetching {display_name} {data_kind} ({ticker}) data (period: {period})")
        data = download_index_data(ticker, period=period, yf=yf)
        if data is not None and not data.empty:
            logger.success(f"Fetched {display_name} {data_kind} data: {data.shape}")
            if cache is not None:
                cache.save(data, [ticker], period, universe=universe, data_kind=data_kind)
            return data
        logger.warning(f"Empty {data_kind} data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch {data_kind} data for {display_name}: {e}")
        return None


def get_benchmark_data(
    symbol: str,
    period: str = "1y",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Fetch regime-detection ETF data for a registered universe.

    Returns unadjusted OHLCV data for the universe's benchmark_etf (SPY for
    SP500, IWB for RUSSELL1000, etc.). This is the broker-compatible price
    series consumed by MarketRegimeDetector and market breadth analysis.

    Args:
        symbol:    IndexSymbol string ("SP500", "RUSSELL1000", …).
        period:    yfinance period string (default "1y").
        use_cache: Whether to use cached data (default True).

    Returns:
        DataFrame with benchmark ETF OHLCV, or None if fetch fails.
    """
    spec = get_universe_spec(symbol)
    return _fetch_single_ticker_data(
        ticker=spec.benchmark_etf,
        period=period,
        universe=spec.symbol,
        display_name=spec.display_name,
        data_kind="benchmark_etf_raw",
        use_cache=use_cache,
    )


def get_index_data(
    symbol: str,
    period: str = "1y",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Fetch index-quote data for a registered universe.

    Returns unadjusted OHLCV data for the universe's benchmark_index (^GSPC
    for SP500, ^RUI for RUSSELL1000, etc.). Used for analytics and display.

    Args:
        symbol:    IndexSymbol string ("SP500", "RUSSELL1000", …).
        period:    yfinance period string (default "1y").
        use_cache: Whether to use cached data (default True).

    Returns:
        DataFrame with index OHLCV, or None if fetch fails.
    """
    spec = get_universe_spec(symbol)
    return _fetch_single_ticker_data(
        ticker=spec.benchmark_index,
        period=period,
        universe=spec.symbol,
        display_name=spec.display_name,
        data_kind="benchmark_index_raw",
        use_cache=use_cache,
    )


def get_stock_data(
    tickers: list[str],
    period: str = "1y",
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Fetch historical OHLCV data for a list of tickers.

    Uses DataCache with a SHA-256-based custom key (no universe assumed).
    Pass universe/data_kind via DataCache directly if you need named keys.

    Args:
        tickers:   List of yfinance-normalised ticker symbols.
        period:    yfinance period string (default "1y").
        use_cache: Whether to use cached data (default True).

    Returns:
        DataFrame with OHLCV data, or None if fetch fails.
    """
    cache = None
    if use_cache:
        from .cache import DataCache
        cache = DataCache(cache_dir="data/cache")
        cached = cache.get(tickers, period, max_age_hours=24)
        if cached is not None:
            return cached

    try:
        import yfinance as yf
        logger.info(f"Fetching stock data for {len(tickers)} tickers (period: {period})")
        data = download_stock_data(tickers, period=period, yf=yf)
        if data is not None and not data.empty:
            logger.success(f"Fetched stock data: {data.shape}")
            if cache is not None:
                cache.save(data, tickers, period)
            return data
        logger.warning("Received empty stock data from yfinance")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch stock data: {e}")
        return None


# ---------------------------------------------------------------------------
# Back-compat wrappers
# ---------------------------------------------------------------------------

def get_sp500_tickers(use_cache: bool = True, max_age_hours: int = 24) -> list[str]:
    """Get S&P 500 market universe constituents (back-compat wrapper).

    Delegates to get_universe_tickers("SP500").
    """
    return get_universe_tickers("SP500", use_cache=use_cache, max_age_hours=max_age_hours)


def get_sp500_index_data(period: str = "1y", use_cache: bool = True) -> pd.DataFrame | None:
    """Fetch S&P 500 index (^GSPC) data (back-compat wrapper).

    Delegates to get_index_data("SP500").
    """
    return get_index_data("SP500", period=period, use_cache=use_cache)
