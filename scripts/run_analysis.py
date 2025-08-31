#!/usr/bin/env python3
"""
Run momentum analysis on S&P 500 stocks.

This script fetches S&P 500 data and displays the top momentum stocks
according to Clenow's momentum strategy.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Initialize logger configuration
from clenow_momentum.data.fetcher import get_sp500_tickers, get_stock_data
from clenow_momentum.indicators.momentum import (
    calculate_momentum_for_universe,
    get_top_momentum_stocks,
)


def main():
    """Main analysis function."""
    print("=" * 60)
    print("CLENOW MOMENTUM STRATEGY ANALYSIS")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    # Step 1: Get S&P 500 tickers
    print("Step 1: Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()

    if not tickers:
        print("❌ Could not retrieve tickers. Exiting.")
        return 1

    print(f"✅ Successfully fetched {len(tickers)} tickers")
    print("Sample tickers:", tickers[:10])
    print()

    # Step 2: Get stock data (last 6 months for 90-day momentum calculation)
    print("Step 2: Fetching stock data (6 months)...")
    print("⏳ This may take a moment...")

    # Use smaller subset for testing
    test_tickers = tickers[:50]  # First 50 stocks for faster testing
    print(f"🧪 Using first {len(test_tickers)} stocks for testing")

    stock_data = get_stock_data(test_tickers, period="6mo")

    if stock_data is None:
        print("❌ Could not retrieve stock data. Exiting.")
        return 1

    print("✅ Stock data retrieved successfully")
    print(f"Data shape: {stock_data.shape}")
    print(
        f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}"
    )
    print()

    # Step 3: Calculate momentum scores
    print("Step 3: Calculating momentum scores...")
    momentum_df = calculate_momentum_for_universe(stock_data, period=90)

    valid_scores = momentum_df.dropna(subset=["momentum_score"])
    print(f"✅ Calculated momentum for {len(valid_scores)} stocks")
    print()

    # Step 4: Get top 20% momentum stocks
    print("Step 4: Selecting top 20% momentum stocks...")
    top_stocks = get_top_momentum_stocks(momentum_df, top_pct=0.20)

    print(f"✅ Selected {len(top_stocks)} top momentum stocks")
    print()

    # Step 5: Display results
    print("=" * 60)
    print("TOP MOMENTUM STOCKS")
    print("=" * 60)

    # Prepare data for tabulate
    table_data = []
    for _, row in top_stocks.iterrows():
        table_data.append(
            [
                row["rank"],
                row["ticker"],
                f"{row['momentum_score']:.3f}",
                f"{row['annualized_slope']:.3f}",
                f"{row['r_squared']:.3f}",
                f"${row['current_price']:.2f}" if not pd.isna(row["current_price"]) else "N/A",
                f"{row['period_return_pct']:+.1f}%"
                if not pd.isna(row["period_return_pct"])
                else "N/A",
            ]
        )

    headers = ["Rank", "Ticker", "Momentum", "Ann. Slope", "R²", "Price", "90d Return"]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 30)
    print(f"Average Momentum Score: {top_stocks['momentum_score'].mean():.3f}")
    print(f"Average R-squared: {top_stocks['r_squared'].mean():.3f}")
    print(f"Average 90-day Return: {top_stocks['period_return_pct'].mean():.1f}%")
    print()

    print("✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    import pandas as pd  # Import pandas for the main function

    exit_code = main()
    sys.exit(exit_code)
