#!/usr/bin/env python3
"""
Cache management utility for backtesting data.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clenow_momentum.data.cache import DataCache


def main():
    """Main cache management function."""
    parser = argparse.ArgumentParser(description="Manage backtesting data cache")
    parser.add_argument('action', choices=['info', 'clear', 'clear-old'], 
                       help='Action to perform')
    parser.add_argument('--hours', type=int, default=24, 
                       help='For clear-old: remove cache older than this many hours')
    
    args = parser.parse_args()
    
    # Initialize cache
    cache = DataCache(cache_dir="data/cache")
    
    if args.action == 'info':
        # Show cache information
        info = cache.get_info()
        print("\n📦 CACHE INFORMATION")
        print("="*50)
        print(f"Cache directory: {info['cache_dir']}")
        print(f"Total cached entries: {info['total_cached']}")
        
        if info['entries']:
            print("\nCached datasets:")
            for entry in info['entries']:
                print(f"\n  • {entry['key']}")
                print(f"    Age: {entry['age_hours']:.1f} hours")
                print(f"    Size: {entry['file_size_mb']:.1f} MB")
                print(f"    Shape: {entry['shape']}")
                print(f"    Tickers: {entry['tickers_count']}")
        else:
            print("\nNo cached data found")
    
    elif args.action == 'clear':
        # Clear all cache
        response = input("⚠️  Clear ALL cached data? This cannot be undone. (y/n): ")
        if response.lower() == 'y':
            cache.clear()
            print("✅ Cache cleared")
        else:
            print("Cancelled")
    
    elif args.action == 'clear-old':
        # Clear old cache entries
        print(f"Clearing cache entries older than {args.hours} hours...")
        cache.clear(older_than_hours=args.hours)
        print("✅ Old cache entries cleared")
    
    return 0


if __name__ == "__main__":
    exit(main())