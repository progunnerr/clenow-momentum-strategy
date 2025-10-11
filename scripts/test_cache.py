#!/usr/bin/env python3
"""
Test data caching functionality.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clenow_momentum.data.cache import DataCache
from clenow_momentum.data import get_stock_data


def test_cache():
    """Test cache save and load."""
    print("\n🧪 TESTING DATA CACHE")
    print("="*50)
    
    # Initialize cache
    cache = DataCache(cache_dir="data/cache")
    
    # Test with a small set of tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    period = '3mo'
    
    print(f"\n1️⃣ Fetching fresh data for {test_tickers}...")
    start_time = time.time()
    
    # Fetch data
    data = get_stock_data(test_tickers, period=period)
    
    if data is None:
        print("❌ Failed to fetch data")
        return False
        
    fetch_time = time.time() - start_time
    print(f"✅ Fetched data in {fetch_time:.2f} seconds")
    print(f"   Shape: {data.shape}")
    
    # Save to cache
    print("\n2️⃣ Saving to cache...")
    cache.save(data, test_tickers, period)
    
    # Try loading from cache
    print("\n3️⃣ Loading from cache...")
    start_time = time.time()
    cached_data = cache.get(test_tickers, period)
    
    if cached_data is None:
        print("❌ Failed to load from cache")
        return False
        
    cache_time = time.time() - start_time
    print(f"✅ Loaded from cache in {cache_time:.2f} seconds")
    print(f"   Speed improvement: {fetch_time/cache_time:.1f}x faster")
    
    # Verify data matches
    print("\n4️⃣ Verifying data integrity...")
    if data.equals(cached_data):
        print("✅ Cached data matches original")
    else:
        print("⚠️  Cached data differs from original")
        
    # Show cache info
    print("\n5️⃣ Cache information:")
    info = cache.get_info()
    print(f"   Cache directory: {info['cache_dir']}")
    print(f"   Total cached entries: {info['total_cached']}")
    
    for entry in info['entries']:
        print(f"   - {entry['tickers_count']} tickers, "
              f"age: {entry['age_hours']:.1f}h, "
              f"size: {entry['file_size_mb']:.1f}MB")
    
    return True


def test_full_universe_cache():
    """Test caching with full S&P 500 universe."""
    print("\n\n🧪 TESTING FULL UNIVERSE CACHE")
    print("="*50)
    
    from clenow_momentum.backtesting.data_prep import prepare_backtest_data
    
    print("\n1️⃣ First call (should fetch from API)...")
    start_time = time.time()
    
    primary_data, universe_data = prepare_backtest_data(
        start_date='2024-01-01',
        end_date='2024-02-01',
        primary_ticker='SPY',
        use_cache=True
    )
    
    if primary_data is None:
        print("❌ Failed to prepare data")
        return False
        
    first_call_time = time.time() - start_time
    print(f"✅ First call completed in {first_call_time:.2f} seconds")
    
    print("\n2️⃣ Second call (should use cache)...")
    start_time = time.time()
    
    primary_data2, universe_data2 = prepare_backtest_data(
        start_date='2024-01-01',
        end_date='2024-02-01',
        primary_ticker='SPY',
        use_cache=True
    )
    
    if primary_data2 is None:
        print("❌ Failed to prepare data")
        return False
        
    second_call_time = time.time() - start_time
    print(f"✅ Second call completed in {second_call_time:.2f} seconds")
    print(f"   Speed improvement: {first_call_time/second_call_time:.1f}x faster")
    
    return True


def main():
    """Run cache tests."""
    
    # Test basic caching
    basic_test = test_cache()
    
    # Ask user if they want to test full universe (takes longer)
    print("\n" + "="*50)
    response = input("\n❓ Test full S&P 500 universe caching? This will take a few minutes on first run. (y/n): ")
    
    full_test = False
    if response.lower() == 'y':
        full_test = test_full_universe_cache()
    else:
        print("Skipping full universe test")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Basic cache test: {'✅ PASSED' if basic_test else '❌ FAILED'}")
    if response.lower() == 'y':
        print(f"Full universe test: {'✅ PASSED' if full_test else '❌ FAILED'}")
    
    print("\n💡 Tip: Cached data expires after 24 hours by default")
    print("   To clear cache, delete files in data/cache/")
    
    return 0 if basic_test else 1


if __name__ == "__main__":
    exit(main())