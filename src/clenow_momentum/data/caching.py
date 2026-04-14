"""Caching mixins for data sources.

This module provides reusable caching functionality that can be mixed with
any data source implementation to add caching capabilities.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .interfaces import MarketDataSource


class CachedDataSource(MarketDataSource):
    """Base class for data sources with caching capabilities.

    This mixin provides common caching functionality that can be mixed with
    any concrete data source implementation. Use this by inheriting from both
    this class and your concrete data source.
    
    Example:
        class YFinanceCachedSource(CachedDataSource, YFinanceBase):
            def get_stock_data(self, tickers, period):
                cache_key = self._get_cache_key(tickers, period)
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data
                
                fresh_data = super().get_stock_data(tickers, period)
                self._save_to_cache(cache_key, fresh_data)
                return fresh_data
    """

    def __init__(self, cache_ttl_hours: int = 24):
        super().__init__()
        self.cache_ttl_hours = cache_ttl_hours
        self._cache: dict[str, dict[str, Any]] = {}

    def _get_cache_key(self, tickers: list[str], period: str) -> str:
        """Generate cache key for request parameters."""
        tickers_str = ",".join(sorted(tickers))
        return f"{tickers_str}_{period}"

    def _is_cache_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        if "timestamp" not in cache_entry:
            return False

        cache_age_hours = (datetime.now(UTC) - cache_entry["timestamp"]).total_seconds() / 3600
        return cache_age_hours < self.cache_ttl_hours

    def _get_from_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Get data from cache if available and valid."""
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry["data"]
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache with timestamp."""
        self._cache[cache_key] = {"data": data.copy(), "timestamp": datetime.now(UTC)}