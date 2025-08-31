#!/usr/bin/env python3
"""
Run momentum analysis on S&P 500 stocks.

This script fetches S&P 500 data and displays the top momentum stocks
according to Clenow's momentum strategy.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Initialize logger configuration
from clenow_momentum.data.fetcher import get_sp500_tickers, get_stock_data
from clenow_momentum.indicators.filters import apply_all_filters
from clenow_momentum.indicators.momentum import (
    calculate_momentum_for_universe,
    get_top_momentum_stocks,
)
from clenow_momentum.strategy.market_regime import check_market_regime, should_trade_momentum


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

    # Use all S&P 500 tickers
    print(f"📈 Processing all {len(tickers)} S&P 500 stocks")

    stock_data = get_stock_data(tickers, period="6mo")

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

    # Step 4: Check market regime
    print("Step 4: Checking market regime...")
    market_regime = check_market_regime(period=200)
    trading_allowed, regime_reason = should_trade_momentum(market_regime)
    
    print(f"📊 Market Regime: {market_regime.get('regime', 'unknown').upper()}")
    print(f"📈 SPX: ${market_regime.get('current_price', 'N/A')} vs 200MA: ${market_regime.get('ma_value', 'N/A')}")
    print(f"🎯 Trading Status: {'✅ ALLOWED' if trading_allowed else '⛔ SUSPENDED'}")
    print(f"📝 Reason: {regime_reason}")
    print()

    # Step 5: Get top 20% momentum stocks
    print("Step 5: Selecting top 20% momentum stocks...")
    top_momentum_stocks = get_top_momentum_stocks(momentum_df, top_pct=0.20)

    print(f"✅ Selected {len(top_momentum_stocks)} top momentum stocks (before filters)")
    print()

    # Step 6: Apply trading filters
    print("Step 6: Applying trading filters...")
    
    if trading_allowed:
        print("🔍 Applying 100-day MA filter and gap exclusion filter...")
        filtered_stocks = apply_all_filters(
            top_momentum_stocks, 
            stock_data, 
            ma_period=100, 
            gap_threshold=0.15
        )
        
        print(f"✅ {len(filtered_stocks)} stocks passed all filters")
        final_stocks = filtered_stocks
    else:
        print("⛔ Market regime not favorable - skipping individual stock filters")
        print("📊 Showing momentum rankings for informational purposes only")
        final_stocks = top_momentum_stocks
        
    print()

    # Step 7: Display results
    print("=" * 60)
    if trading_allowed:
        print("TOP MOMENTUM STOCKS (FILTERED)")
        print("🎯 These stocks passed all Clenow strategy filters")
    else:
        print("TOP MOMENTUM STOCKS (UNFILTERED)")
        print("⚠️  For informational purposes only - market regime not favorable")
    print("=" * 60)

    if final_stocks.empty:
        print("❌ No stocks passed all filters")
        print()
    else:
        # Prepare data for tabulate
        table_data = []
        for i, (_, row) in enumerate(final_stocks.iterrows()):
            # Check if we have filter data (MA info)
            price_col = 'current_price'
            if 'latest_price' in row and not pd.isna(row['latest_price']):
                price_val = row['latest_price']
                ma_info = f" (vs MA: {row.get('price_vs_ma', 0):+.1%})" if 'price_vs_ma' in row else ""
            else:
                price_val = row.get('current_price', 0)
                ma_info = ""
            
            table_data.append([
                i + 1,  # Re-rank after filtering
                row["ticker"],
                f"{row['momentum_score']:.3f}",
                f"{row['annualized_slope']:.3f}",
                f"{row['r_squared']:.3f}",
                f"${price_val:.2f}{ma_info}" if not pd.isna(price_val) else "N/A",
                f"{row['period_return_pct']:+.1f}%" if not pd.isna(row["period_return_pct"]) else "N/A",
            ])

        headers = ["Rank", "Ticker", "Momentum", "Ann. Slope", "R²", "Price", "90d Return"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()

        # Enhanced summary statistics
        print("SUMMARY STATISTICS")
        print("-" * 50)
        print(f"📊 Stocks analyzed: {len(valid_scores)}")
        print(f"🎯 Top 20% selected: {len(top_momentum_stocks)}")
        if trading_allowed:
            print(f"✅ Passed all filters: {len(final_stocks)}")
            print(f"🔍 Filter success rate: {len(final_stocks)/len(top_momentum_stocks)*100:.1f}%")
        
        print(f"📈 Average Momentum Score: {final_stocks['momentum_score'].mean():.3f}")
        print(f"📐 Average R-squared: {final_stocks['r_squared'].mean():.3f}")
        print(f"📊 Average 90-day Return: {final_stocks['period_return_pct'].mean():.1f}%")
        
        if trading_allowed and len(final_stocks) > 0:
            print(f"🎯 Market Regime: {market_regime['regime'].upper()}")
            print(f"💼 Ready for position sizing and portfolio construction")
        elif not trading_allowed:
            print(f"⛔ Trading suspended due to {market_regime['regime']} market regime")
        print()

    print("✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
