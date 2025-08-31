import sys
from strategy import get_sp500_tickers


def main():
    """
    Main function to run the momentum strategy.
    """
    print("Step 1: Fetching S&P 500 tickers for our stock universe...")
    tickers = get_sp500_tickers()

    if not tickers:
        print("Could not retrieve tickers. Exiting.")
        sys.exit(1)

    print(f"Successfully fetched {len(tickers)} tickers.")
    print("A few sample tickers:", tickers[:5])


if __name__ == "__main__":
    main()
