#!/usr/bin/env python3
"""
Warm the cache by pre-fetching S&P 500 data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clenow_momentum.data import get_universe_tickers, get_stock_data
from clenow_momentum.data.cache import DataCache
from clenow_momentum.utils.config import load_config
from loguru import logger


def main():
    """Pre-fetch and cache universe data (driven by MARKET_UNIVERSE env var)."""
    config = load_config()
    universe = config.get("universe", "SP500")

    print(f"\n🔥 WARMING CACHE - Fetching {universe} Data")
    print("="*50)

    # Initialize cache
    cache = DataCache(cache_dir="data/cache")

    # Get universe tickers
    print(f"Getting {universe} ticker list...")
    tickers = get_universe_tickers(universe)

    if not tickers:
        print(f"❌ Failed to get {universe} tickers")
        return 1

    print(f"Found {len(tickers)} tickers")
    
    # Periods to cache
    periods_to_cache = ['2y', '1y']  # Add more if needed
    
    for period in periods_to_cache:
        print(f"\n📊 Fetching {period} of data...")
        
        # Check if already cached
        existing = cache.get(tickers, period, max_age_hours=24)
        
        if existing is not None:
            print(f"✅ Already cached: {period}")
            continue
        
        # Fetch fresh data
        print(f"Fetching fresh data for {len(tickers)} tickers (this will take a few minutes)...")
        data = get_stock_data(tickers, period=period)
        
        if data is not None and not data.empty:
            # Save to cache
            cache.save(data, tickers, period)
            print(f"✅ Cached {period}: {data.shape}")
        else:
            print(f"❌ Failed to fetch {period} data")
    
    # Show cache status
    print("\n" + "="*50)
    print("CACHE STATUS")
    print("="*50)
    
    info = cache.get_info()
    print(f"Total cached entries: {info['total_cached']}")
    
    for entry in info['entries']:
        print(f"  • {entry['key']}: {entry['file_size_mb']:.1f}MB, age: {entry['age_hours']:.1f}h")
    
    print("\n✅ Cache warming complete!")
    print("You can now run backtests without waiting for data fetches.")
    
    return 0


if __name__ == "__main__":
    exit(main())