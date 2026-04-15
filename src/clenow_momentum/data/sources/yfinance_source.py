"""
Thin wrapper helpers for yfinance access used across the project.

These functions are deliberately small and accept optional dependency
injection so callers/tests can patch behaviors easily while keeping a
clean separation of concerns from orchestration code.
"""

from __future__ import annotations

from collections.abc import Iterable


def convert_ticker_for_yfinance(ticker: str) -> str:
    """
    Convert ticker symbols to yfinance format.

    yfinance uses hyphens instead of dots for certain tickers:
    - BRK.B -> BRK-B
    - BF.B  -> BF-B

    Exchange suffixes remain dot-separated:
    - SHOP.TO    -> SHOP.TO
    - BIP.UN.TO  -> BIP-UN.TO
    """
    exchange_suffixes = (".TO",)
    for suffix in exchange_suffixes:
        if ticker.endswith(suffix):
            base = ticker[: -len(suffix)]
            return f"{base.replace('.', '-')}{suffix}"

    return ticker.replace(".", "-")

def get_tickers_from_spy_holdings(*, yf=None) -> list[str] | None:
    """
    Try to load S&P 500 tickers via SPY ETF holdings.
    Returns a list or None if unavailable.
    """
    if yf is None:
        import yfinance as _yf  # type: ignore
        yf = _yf

    try:
        spy = yf.Ticker("SPY")
        holdings = spy.get_holdings()
        if holdings is None or len(holdings) == 0:
            return None
        # Strip suffixes like .B which often appear with class shares
        return [ticker.split(".")[0] for ticker in holdings.index.tolist()]
    except Exception:
        return None

def get_tickers_from_spy_info(*, yf=None) -> list[str] | None:
    """
    Fallback: attempt to read tickers from SPY.info["holdings"].
    Returns a list or None if unavailable.
    """
    if yf is None:
        import yfinance as _yf  # type: ignore
        yf = _yf

    try:
        spy = yf.Ticker("SPY")
        info = spy.info
        if "holdings" not in info:
            return None
        holdings = info["holdings"]
        if not holdings:
            return None
        tickers = [
            h.get("symbol", h.get("ticker", ""))
            for h in holdings
            if h.get("symbol") or h.get("ticker")
        ]
        return [t for t in tickers if t]
    except Exception:
        return None

def download_stock_data(
    tickers: Iterable[str], *, period: str = "1y", yf=None
):
    """
    Download OHLCV data for multiple tickers via yfinance.
    Returns a pandas DataFrame or None on error.
    """
    if yf is None:
        import yfinance as _yf  # type: ignore
        yf = _yf

    try:
        # Convert to yfinance format
        converted = [convert_ticker_for_yfinance(t) for t in tickers]
        return yf.download(converted, period=period, group_by="ticker", auto_adjust=True)
    except Exception:
        return None

def download_index_data(symbol: str, *, period: str = "1y", yf=None):
    """
    Download OHLCV data for a single index/ETF symbol via yfinance.
    Uses unadjusted Close values for broker-compatible technical MAs.
    Returns a pandas DataFrame or None on error.
    """
    if yf is None:
        import yfinance as _yf  # type: ignore
        yf = _yf

    try:
        return yf.download(symbol, period=period, auto_adjust=False)
    except Exception:
        return None
