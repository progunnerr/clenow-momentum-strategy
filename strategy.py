import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_sp500_tickers():
    """
    Scrapes the Wikipedia page for the list of S&P 500 companies and returns their tickers.
    Returns a list of tickers, or an empty list if fetching fails.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.RequestException as e:
        print(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        print("Could not find the constituents table on the Wikipedia page.")
        return []

    # The first DataFrame from read_html should be the table we want.
    df = pd.read_html(str(table))[0]

    if "Symbol" not in df.columns:
        print("'Symbol' column not found in the table.")
        return []

    tickers = df["Symbol"].tolist()
    return tickers
