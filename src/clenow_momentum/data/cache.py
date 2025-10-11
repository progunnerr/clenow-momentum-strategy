"""
Data caching utilities for backtesting.

This module provides functionality to cache historical stock data to avoid
repeated API calls during backtesting iterations.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger


class DataCache:
    """
    Simple file-based cache for historical stock data.

    Stores data in pickle format with metadata about cache validity.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.pkl"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata if exists."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, tickers: list, period: str) -> str:
        """
        Generate cache key from parameters.

        Args:
            tickers: List of tickers
            period: Period string (e.g., '1y', '2y')

        Returns:
            Cache key string
        """
        # For S&P 500, use a simplified key that's more stable
        # This handles minor variations in ticker count (e.g., 503 vs 504)
        if 450 < len(tickers) < 550:  # S&P 500 range
            return f"sp500_universe_{period}"
        # For custom ticker lists, use hash
        sorted_tickers = sorted(tickers)
        ticker_hash = hash(tuple(sorted_tickers))
        return f"custom_{len(tickers)}tickers_{period}_{ticker_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def is_cache_valid(self, cache_key: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data exists and is still valid.

        Args:
            cache_key: Cache key
            max_age_hours: Maximum age of cache in hours (default 24)

        Returns:
            True if cache exists and is valid
        """
        if cache_key not in self.metadata:
            return False

        cache_info = self.metadata[cache_key]
        cache_age = datetime.now() - cache_info["timestamp"]

        if cache_age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {cache_key} (age: {cache_age})")
            return False

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            logger.warning(f"Cache metadata exists but file missing: {cache_path}")
            return False

        return True

    def get(
        self, tickers: list, period: str, max_age_hours: int = 24
    ) -> pd.DataFrame | None:
        """
        Get cached data if available and valid.

        Args:
            tickers: List of tickers
            period: Period string
            max_age_hours: Maximum cache age in hours

        Returns:
            Cached DataFrame or None if not available
        """
        cache_key = self._get_cache_key(tickers, period)

        # First try the exact key
        if self.is_cache_valid(cache_key, max_age_hours):
            cache_path = self._get_cache_path(cache_key)
        else:
            # For S&P 500, also check for any existing S&P 500 cache with similar ticker count
            if 450 < len(tickers) < 550:
                # Look for any S&P 500 cache with the same period
                found_key = None
                for key, info in self.metadata.items():
                    if (
                        info["period"] == period
                        and 450 < info["tickers_count"] < 550
                        and self.is_cache_valid(key, max_age_hours)
                    ):
                        found_key = key
                        logger.info(f"Found existing S&P 500 cache: {key}")
                        break

                if found_key:
                    cache_key = found_key
                    cache_path = self._get_cache_path(cache_key)
                else:
                    return None
            else:
                return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            cache_info = self.metadata[cache_key]
            logger.success(
                f"Loaded cached data: {cache_info['tickers_count']} tickers, "
                f"{len(data)} days, cached at {cache_info['timestamp'].strftime('%Y-%m-%d %H:%M')}"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            # Remove invalid cache entry
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            return None

    def save(self, data: pd.DataFrame, tickers: list, period: str):
        """
        Save data to cache.

        Args:
            data: DataFrame to cache
            tickers: List of tickers
            period: Period string
        """
        cache_key = self._get_cache_key(tickers, period)

        # For S&P 500, check if we already have a cache with different key
        # and remove old entries to avoid duplicates
        if 450 < len(tickers) < 550:
            keys_to_remove = []
            for key, info in self.metadata.items():
                if (
                    info["period"] == period
                    and 450 < info["tickers_count"] < 550
                    and key != cache_key
                ):
                    # Found old S&P 500 cache with same period
                    old_path = self._get_cache_path(key)
                    if old_path.exists():
                        old_path.unlink()
                        logger.info(f"Removing old cache: {key}")
                    keys_to_remove.append(key)

            # Clean up metadata
            for key in keys_to_remove:
                del self.metadata[key]

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # Update metadata
            self.metadata[cache_key] = {
                "timestamp": datetime.now(),
                "tickers_count": len(tickers),
                "period": period,
                "shape": data.shape,
                "date_range": (
                    (data.index[0], data.index[-1]) if not data.empty else None
                ),
            }
            self._save_metadata()

            logger.success(
                f"Cached data: {len(tickers)} tickers, {len(data)} days, "
                f"file: {cache_path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to cache data: {e}")

    def clear(self, older_than_hours: int | None = None):
        """
        Clear cache files.

        Args:
            older_than_hours: If specified, only clear files older than this
        """
        if older_than_hours is None:
            # Clear all cache files
            for file in self.cache_dir.glob("*.pkl"):
                if file != self.metadata_file:
                    file.unlink()
                    logger.info(f"Deleted cache file: {file.name}")

            self.metadata = {}
            self._save_metadata()
            logger.success("Cleared all cache files")

        else:
            # Clear only old files
            now = datetime.now()
            keys_to_remove = []

            for cache_key, info in self.metadata.items():
                age = now - info["timestamp"]
                if age > timedelta(hours=older_than_hours):
                    cache_path = self._get_cache_path(cache_key)
                    if cache_path.exists():
                        cache_path.unlink()
                        logger.info(f"Deleted old cache file: {cache_path.name}")
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                del self.metadata[key]

            if keys_to_remove:
                self._save_metadata()
                logger.success(f"Cleared {len(keys_to_remove)} old cache files")

    def get_info(self) -> dict:
        """
        Get information about cached data.

        Returns:
            Dictionary with cache information
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "total_cached": len(self.metadata),
            "entries": [],
        }

        for cache_key, metadata in self.metadata.items():
            cache_path = self._get_cache_path(cache_key)
            file_size = cache_path.stat().st_size if cache_path.exists() else 0

            info["entries"].append(
                {
                    "key": cache_key,
                    "timestamp": metadata["timestamp"],
                    "age_hours": (
                        datetime.now() - metadata["timestamp"]
                    ).total_seconds()
                    / 3600,
                    "tickers_count": metadata["tickers_count"],
                    "shape": metadata["shape"],
                    "file_size_mb": file_size / (1024 * 1024),
                }
            )

        return info
