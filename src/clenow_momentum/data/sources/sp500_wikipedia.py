"""
S&P 500 ticker fetching via Wikipedia.

Scrapes Wikipedia for the current list of S&P 500 constituents.
"""

from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger


def fetch_sp500_tickers_from_wikipedia(
    timeout: int = 10,
    headers: dict | None = None,
) -> list[str]:
    """
    Fetch the list of S&P 500 tickers from Wikipedia.

    Args:
        timeout: HTTP timeout in seconds (default 10)
        headers: Optional HTTP headers dict

    Returns:
        List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
        
    Raises:
        requests.RequestException: If HTTP request fails
        ValueError: If table parsing fails
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Default headers to avoid 403 blocks
    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    try:
        logger.debug(f"Fetching S&P 500 tickers from Wikipedia...")
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        
        if table is None:
            logger.error("Could not find constituents table on Wikipedia")
            raise ValueError("Could not find constituents table on Wikipedia")

        df_list = pd.read_html(StringIO(str(table)))
        if not df_list:
            logger.error("No tables found in HTML")
            raise ValueError("No tables found in HTML")
        
        df = df_list[0]

        if "Symbol" not in df.columns:
            logger.error(f"Symbol column not found. Available columns: {df.columns.tolist()}")
            raise ValueError("Symbol column not found in table")

        tickers = df["Symbol"].tolist()
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
        
    except requests.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing Wikipedia data: {e}")
        raise
