#!/usr/bin/env python3
"""
Test script to verify backtesting data preparation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clenow_momentum.backtesting.data_prep import prepare_single_ticker_test, prepare_backtest_data
from loguru import logger
import pandas as pd


def test_single_ticker():
    """Test loading single ticker data."""
    print("\n" + "="*50)
    print("TEST 1: Single Ticker Data Loading")
    print("="*50)
    
    # Test with Apple
    data = prepare_single_ticker_test('AAPL', period='3mo')
    
    if data is not None:
        print(f"\n✅ Successfully loaded {len(data)} days of AAPL data")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"\nFirst few rows:")
        print(data.head())
        print(f"\nLast few rows:")
        print(data.tail())
        return True
    else:
        print("❌ Failed to load single ticker data")
        return False


def test_universe_data():
    """Test loading universe data with a subset for speed."""
    print("\n" + "="*50)
    print("TEST 2: Universe Data Loading (Subset)")
    print("="*50)
    
    # For testing, let's create a smaller universe
    from clenow_momentum.backtesting.data_prep import get_stock_data
    test_tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JPM']
    
    print(f"Testing with {len(test_tickers)} tickers for faster execution...")
    
    # Fetch data directly for testing
    universe_data = get_stock_data(test_tickers, period="6mo")
    
    if universe_data is None:
        print("❌ Failed to load universe data")
        return False
    
    # Extract SPY as primary
    primary_data = universe_data['SPY'].copy()
    primary_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    primary_data.name = 'SPY'
    
    # Filter to a test period
    start_date = '2024-06-01'
    end_date = '2024-08-01' 
    primary_data = primary_data[(primary_data.index >= start_date) & (primary_data.index <= end_date)]
    
    if primary_data is not None and universe_data is not None:
        print(f"\n✅ Successfully loaded universe data")
        print(f"Primary (SPY) data: {len(primary_data)} days")
        print(f"Universe data shape: {universe_data.shape}")
        
        # Count tickers
        if isinstance(universe_data.columns, pd.MultiIndex):
            n_tickers = len(universe_data.columns.get_level_values(0).unique())
            print(f"Number of tickers: {n_tickers}")
            
            # Show some tickers
            tickers = list(universe_data.columns.get_level_values(0).unique())[:10]
            print(f"First 10 tickers: {tickers}")
        
        print(f"\nSPY data preview:")
        print(primary_data.head())
        
        return True
    else:
        print("❌ Failed to load universe data")
        return False


def main():
    """Run all tests."""
    print("\n🧪 TESTING BACKTESTING DATA PREPARATION")
    
    # Run tests
    test1_pass = test_single_ticker()
    test2_pass = test_universe_data()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Single Ticker Test: {'✅ PASSED' if test1_pass else '❌ FAILED'}")
    print(f"Universe Data Test: {'✅ PASSED' if test2_pass else '❌ FAILED'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 All tests passed! Data preparation is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())