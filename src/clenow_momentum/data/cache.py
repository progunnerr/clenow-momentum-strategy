"""Data caching utilities for backtesting.

Cache keys are universe- and data-kind-aware:
  Named path:  {SYMBOL}_{data_kind}_{period}  e.g. SP500_universe_1y
  Custom path: custom_{n}tickers_{period}_{sha256[:16]}

The magic 450<len<550 S&P 500 heuristic and the non-deterministic
hash(tuple(...)) key are both gone.
"""

import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger


class DataCache:
    """Simple file-based cache for historical stock data.

    Stores data in pickle format with metadata about cache validity.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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

    def _get_cache_key(
        self,
        tickers: list,
        period: str,
        universe: str | None = None,
        data_kind: str | None = None,
    ) -> str:
        """Generate a deterministic cache key.

        Named-universe path (when universe is given):
            {UNIVERSE}_{data_kind}_{period}   e.g. SP500_universe_1y
            data_kind defaults to "universe" when not provided.

        Custom path (no universe):
            custom_{n}tickers_{period}_{sha256[:16]}
            SHA-256 is computed over sorted tickers joined by newlines —
            stable across Python processes regardless of PYTHONHASHSEED.

        Args:
            tickers:   List of ticker symbols (used for custom key only).
            period:    Period string (e.g. '1y', '2y').
            universe:  IndexSymbol string ('SP500', 'RUSSELL1000', …) or None.
            data_kind: Semantic label for the data ('universe', 'benchmark_etf',
                       'benchmark_index'). Ignored when universe is None.

        Returns:
            Cache key string.
        """
        if universe is not None:
            kind = data_kind or "universe"
            return f"{universe.upper()}_{kind}_{period}"

        # Custom / ad-hoc ticker list — deterministic SHA-256
        digest = hashlib.sha256(
            "\n".join(sorted(tickers)).encode("utf-8")
        ).hexdigest()[:16]
        return f"custom_{len(tickers)}tickers_{period}_{digest}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def is_cache_valid(self, cache_key: str, max_age_hours: int = 24) -> bool:
        """Check if cached data exists and is still valid."""
        if cache_key not in self.metadata:
            return False

        cache_info = self.metadata[cache_key]
        now = datetime.now()  # noqa: DTZ005
        cache_age = now - cache_info["timestamp"]

        if cache_age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {cache_key} (age: {cache_age})")
            return False

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            logger.warning(f"Cache metadata exists but file missing: {cache_path}")
            return False

        return True

    def get(
        self,
        tickers: list,
        period: str,
        max_age_hours: int = 24,
        universe: str | None = None,
        data_kind: str | None = None,
    ) -> pd.DataFrame | None:
        """Get cached data if available and valid."""
        cache_key = self._get_cache_key(tickers, period, universe, data_kind)

        if not self.is_cache_valid(cache_key, max_age_hours):
            return None

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            cache_info = self.metadata[cache_key]
            logger.success(
                f"Loaded cached data: {cache_info['tickers_count']} tickers, "
                f"{len(data)} days, cached at "
                f"{cache_info['timestamp'].strftime('%Y-%m-%d %H:%M')}"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            return None

    def save(
        self,
        data: pd.DataFrame,
        tickers: list,
        period: str,
        universe: str | None = None,
        data_kind: str | None = None,
    ):
        """Save data to cache."""
        cache_key = self._get_cache_key(tickers, period, universe, data_kind)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            now = datetime.now()  # noqa: DTZ005
            self.metadata[cache_key] = {
                "timestamp": now,
                "tickers_count": len(tickers),
                "period": period,
                "universe": universe,
                "data_kind": data_kind,
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
        """Clear cache files."""
        if older_than_hours is None:
            for file in self.cache_dir.glob("*.pkl"):
                if file != self.metadata_file:
                    file.unlink()
                    logger.info(f"Deleted cache file: {file.name}")
            self.metadata = {}
            self._save_metadata()
            logger.success("Cleared all cache files")
        else:
            now = datetime.now()  # noqa: DTZ005
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
        """Get information about cached data."""
        info = {
            "cache_dir": str(self.cache_dir),
            "total_cached": len(self.metadata),
            "entries": [],
        }

        now = datetime.now()  # noqa: DTZ005
        for cache_key, metadata in self.metadata.items():
            cache_path = self._get_cache_path(cache_key)
            file_size = cache_path.stat().st_size if cache_path.exists() else 0
            info["entries"].append(
                {
                    "key": cache_key,
                    "timestamp": metadata["timestamp"],
                    "age_hours": (now - metadata["timestamp"]).total_seconds() / 3600,
                    "tickers_count": metadata["tickers_count"],
                    "universe": metadata.get("universe"),
                    "data_kind": metadata.get("data_kind"),
                    "shape": metadata["shape"],
                    "file_size_mb": file_size / (1024 * 1024),
                }
            )

        return info
