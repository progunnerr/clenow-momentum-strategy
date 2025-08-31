
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger


def _try_spy_holdings() -> list[str] | None:
    """Try to get S&P 500 tickers from SPY ETF holdings."""
    try:
        spy = yf.Ticker("SPY")
        holdings = spy.get_holdings()
        if holdings is not None and len(holdings) > 0:
            tickers = holdings.index.tolist()
            tickers = [ticker.split('.')[0] for ticker in tickers]
            logger.success(f"Successfully fetched {len(tickers)} S&P 500 tickers via SPY holdings")
            return tickers
    except Exception as e:
        logger.debug(f"get_holdings method failed: {e}")
    return None


def _try_spy_info() -> list[str] | None:
    """Try to get S&P 500 tickers from SPY ETF info."""
    try:
        spy = yf.Ticker("SPY")
        info = spy.info
        if 'holdings' in info:
            holdings = info['holdings']
            if holdings and len(holdings) > 0:
                tickers = [h.get('symbol', h.get('ticker', '')) for h in holdings if h.get('symbol') or h.get('ticker')]
                tickers = [t for t in tickers if t]  # Remove empty strings
                logger.success(f"Successfully fetched {len(tickers)} S&P 500 tickers via SPY info")
                return tickers
    except Exception as e:
        logger.debug(f"info holdings method failed: {e}")
    return None


def _get_major_sp500_tickers() -> list[str]:
    """Get a curated list of major S&P 500 companies."""
    major_sp500 = [
        # Technology giants
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AVGO", "ORCL", "CRM", "ADBE", "NFLX", "AMD", "INTC", "CSCO",

        # Healthcare & Pharma leaders
        "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY",

        # Financial powerhouses
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",

        # Consumer & Industrial leaders
        "HD", "PG", "KO", "PEP", "WMT", "DIS", "MCD", "NKE", "BA", "CAT",

        # Energy & Utilities
        "XOM", "CVX", "NEE", "DUK",

        # Communication & Media
        "VZ", "T", "TMUS", "CMCSA"
    ]

    # Verify these tickers exist and are tradeable
    verified_tickers = []
    logger.info(f"Verifying {len(major_sp500)} major S&P 500 tickers...")

    for ticker in major_sp500:
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            # Handle both attribute-style and dict-style access
            last_price = None
            if info:
                if hasattr(info, 'last_price'):
                    last_price = info.last_price
                elif isinstance(info, dict) and 'last_price' in info:
                    last_price = info['last_price']
            
            if last_price and last_price > 0:
                verified_tickers.append(ticker)
        except Exception as e:
            logger.debug(f"Could not verify ticker {ticker}: {e}")

    logger.success(f"Verified {len(verified_tickers)} major S&P 500 tickers")
    return verified_tickers


def get_sp500_tickers_yfinance() -> list[str]:
    """
    Fetch S&P 500 tickers using yfinance as fallback method.

    Returns:
        List of S&P 500 stock tickers, or empty list if fetch fails
    """
    logger.info("Attempting to fetch S&P 500 tickers via yfinance...")

    # Method 1: Try SPY holdings
    tickers = _try_spy_holdings()
    if tickers:
        return tickers

    # Method 2: Try SPY info
    tickers = _try_spy_info()
    if tickers:
        return tickers

    # Method 3: Use curated list as last resort
    try:
        logger.info("Using curated list of major S&P 500 companies...")
        return _get_major_sp500_tickers()
    except Exception as e:
        logger.error(f"All fallback methods failed: {e}")
        return []


def get_sp500_tickers() -> list[str]:
    """
    Scrapes the Wikipedia page for the list of S&P 500 companies and returns their tickers.
    Returns a list of tickers, or an empty list if fetching fails.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Add headers to avoid 403 Forbidden errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch S&P 500 tickers from Wikipedia: {e}")
        logger.info("Trying yfinance as fallback method...")
        return get_sp500_tickers_yfinance()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        logger.warning("Could not find the constituents table on Wikipedia page")
        logger.info("Trying yfinance as fallback method...")
        return get_sp500_tickers_yfinance()

    df = pd.read_html(StringIO(str(table)))[0]

    if "Symbol" not in df.columns:
        logger.warning("'Symbol' column not found in the constituents table")
        logger.info("Trying yfinance as fallback method...")
        return get_sp500_tickers_yfinance()

    tickers = df["Symbol"].tolist()
    logger.success(f"Successfully fetched {len(tickers)} S&P 500 tickers")
    return tickers


def get_stock_data(tickers: list[str], period: str = "1y") -> pd.DataFrame | None:
    """
    Fetch historical stock data for given tickers using yfinance.

    Args:
        tickers: List of stock tickers
        period: Period for data (1y, 2y, 5y, etc.)

    Returns:
        DataFrame with OHLCV data for all tickers, or None if fetch fails
    """
    try:
        logger.info(f"Fetching stock data for {len(tickers)} tickers (period: {period})")
        data = yf.download(tickers, period=period, group_by="ticker", auto_adjust=True)
        if data is not None and not data.empty:
            logger.success(f"Successfully fetched stock data: {data.shape}")
        else:
            logger.warning("Received empty stock data from yfinance")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch stock data: {e}")
        return None


def get_sp500_index_data(period: str = "1y") -> pd.DataFrame | None:
    """
    Fetch S&P 500 index data for market regime analysis.

    Args:
        period: Period for data (1y, 2y, 5y, etc.)

    Returns:
        DataFrame with S&P 500 OHLCV data, or None if fetch fails
    """
    try:
        logger.info(f"Fetching S&P 500 index data (period: {period})")
        data = yf.download("^GSPC", period=period, auto_adjust=True)
        if data is not None and not data.empty:
            logger.success(f"Successfully fetched S&P 500 index data: {data.shape}")
        else:
            logger.warning("Received empty S&P 500 index data from yfinance")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 index data: {e}")
        return None
