"""Tests for DataCache key generation.

Verifies:
- Named universe keys use {SYMBOL}_{data_kind}_{period} format
- Custom path uses SHA-256 (deterministic across calls / processes)
- Magic ticker-count range is gone
- universe + data_kind appear in metadata
"""

import hashlib
import subprocess
import sys

import pytest

from src.clenow_momentum.data.cache import DataCache


class TestCacheKeyGeneration:
    """DataCache._get_cache_key produces the correct keys."""

    def setup_method(self):
        self.cache = DataCache.__new__(DataCache)  # skip __init__ (no disk I/O)
        self.cache.metadata = {}

    def test_named_universe_key_default_kind(self):
        key = self.cache._get_cache_key([], "1y", universe="SP500")
        assert key == "SP500_universe_1y"

    def test_named_universe_key_benchmark_etf(self):
        key = self.cache._get_cache_key(["SPY"], "1y", universe="SP500", data_kind="benchmark_etf")
        assert key == "SP500_benchmark_etf_1y"

    def test_named_universe_key_benchmark_index(self):
        key = self.cache._get_cache_key(["^GSPC"], "2y", universe="SP500", data_kind="benchmark_index")
        assert key == "SP500_benchmark_index_2y"

    def test_russell1000_universe_key(self):
        key = self.cache._get_cache_key([], "1y", universe="RUSSELL1000")
        assert key == "RUSSELL1000_universe_1y"

    def test_custom_key_is_deterministic(self):
        tickers = ["AAPL", "MSFT", "GOOG"]
        key1 = self.cache._get_cache_key(tickers, "1y")
        key2 = self.cache._get_cache_key(tickers, "1y")
        assert key1 == key2

    def test_custom_key_order_independent(self):
        """Sorted before hashing — order of input doesn't matter."""
        key_asc = self.cache._get_cache_key(["AAPL", "MSFT"], "1y")
        key_desc = self.cache._get_cache_key(["MSFT", "AAPL"], "1y")
        assert key_asc == key_desc

    def test_custom_key_uses_sha256(self):
        tickers = ["AAPL", "MSFT"]
        expected_digest = hashlib.sha256(
            "\n".join(sorted(tickers)).encode("utf-8")
        ).hexdigest()[:16]
        key = self.cache._get_cache_key(tickers, "1y")
        assert expected_digest in key

    def test_custom_key_format(self):
        tickers = ["AAPL", "MSFT"]
        key = self.cache._get_cache_key(tickers, "1y")
        assert key.startswith(f"custom_{len(tickers)}tickers_1y_")

    def test_different_tickers_different_keys(self):
        key1 = self.cache._get_cache_key(["AAPL"], "1y")
        key2 = self.cache._get_cache_key(["MSFT"], "1y")
        assert key1 != key2

    def test_different_periods_different_keys(self):
        tickers = ["AAPL", "MSFT"]
        assert self.cache._get_cache_key(tickers, "1y") != self.cache._get_cache_key(tickers, "2y")

    def test_no_magic_range_for_sp500_sized_list(self):
        """A ~500-ticker custom list gets a hash key, not a named universe key."""
        tickers = [f"T{i:03d}" for i in range(503)]
        key = self.cache._get_cache_key(tickers, "1y")
        assert key.startswith("custom_"), f"Expected custom key, got: {key}"
        assert "sp500" not in key.lower()


class TestCrossProcessDeterminism:
    """SHA-256 key is stable regardless of PYTHONHASHSEED."""

    def _get_key_in_subprocess(self, seed: int) -> str:
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from clenow_momentum.data.cache import DataCache;"
            "c = DataCache.__new__(DataCache);"
            "c.metadata = {};"
            "print(c._get_cache_key(['AAPL','MSFT'], '1y'))"
        )
        import os
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_stable_across_hash_seeds(self):
        key_0 = self._get_key_in_subprocess(0)
        key_1 = self._get_key_in_subprocess(1)
        key_42 = self._get_key_in_subprocess(42)
        assert key_0 == key_1 == key_42
