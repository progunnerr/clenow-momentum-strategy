"""
Integration tests for S&P 500 ticker fetching from Wikipedia.

These tests make actual HTTP requests to Wikipedia and verify the scraper works correctly.

Optimization: Uses pytest fixtures with session scope to fetch tickers only ONCE for all tests.

Run with:
    pytest tests/integration/test_sp500_wikipedia_integration.py -v
    pytest tests/integration/test_sp500_wikipedia_integration.py -v -s  # with print output
    pytest tests/integration/test_sp500_wikipedia_integration.py -v -k "test_fetch_tickers_success"  # single test
"""

import pytest

from clenow_momentum.data.sources.sp500_wikipedia import (
    fetch_sp500_tickers_from_wikipedia,
)

# ============================================================================
# FIXTURES - Fetch data once and reuse across all tests
# ============================================================================


@pytest.fixture(scope="session")
def sp500_tickers():
    """Fetch S&P 500 tickers once per test session and reuse.

    This significantly speeds up tests by avoiding redundant HTTP requests.
    Scope='session' means this runs once for the entire test session.
    """
    print("\n[FIXTURE] Fetching S&P 500 tickers from Wikipedia (once per session)...")
    tickers = fetch_sp500_tickers_from_wikipedia()
    print(f"[FIXTURE] Fetched {len(tickers)} tickers successfully")
    return tickers


# ============================================================================
# TEST CLASS - Uses fixture for efficient testing
# ============================================================================


class TestWikipediaSP500Integration:
    """Integration tests for Wikipedia S&P 500 ticker scraper.

    Uses sp500_tickers fixture to fetch data once and reuse across all tests.
    """

    def test_fetch_tickers_success(self, sp500_tickers):
        """Test that we can successfully fetch S&P 500 tickers from Wikipedia."""
        # Assert - data comes from fixture
        assert sp500_tickers is not None, "Should return a list of tickers"
        assert isinstance(sp500_tickers, list), "Should return a list"
        assert len(sp500_tickers) > 0, "Should return at least one ticker"

    def test_ticker_count_reasonable(self, sp500_tickers):
        """Test that the number of tickers is reasonable (around 500-505)."""
        # Assert
        assert 480 <= len(sp500_tickers) <= 520, (
            f"Expected around 500-505 tickers, got {len(sp500_tickers)}. "
            "S&P 500 should have approximately 503 stocks."
        )

    def test_tickers_are_strings(self, sp500_tickers):
        """Test that all tickers are non-empty strings."""
        # Assert
        for ticker in sp500_tickers:
            assert isinstance(ticker, str), f"Ticker {ticker} is not a string"
            assert len(ticker) > 0, "Ticker should not be empty string"
            assert len(ticker) <= 5, (
                f"Ticker {ticker} is too long (>{5} chars). "
                "Most tickers are 1-5 characters."
            )

    def test_contains_known_tickers(self, sp500_tickers):
        """Test that the list contains well-known S&P 500 companies."""
        # Arrange - Known S&P 500 companies that should always be present
        known_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

        # Assert
        for known_ticker in known_tickers:
            assert known_ticker in sp500_tickers, (
                f"{known_ticker} should be in S&P 500 list. "
                f"Found tickers starting with {known_ticker[0]}: "
                f"{[t for t in sp500_tickers if t.startswith(known_ticker[0])]}"
            )

    def test_no_duplicate_tickers(self, sp500_tickers):
        """Test that there are no duplicate tickers in the list."""
        # Assert
        unique_tickers = set(sp500_tickers)
        assert len(sp500_tickers) == len(unique_tickers), (
            f"Found {len(sp500_tickers) - len(unique_tickers)} duplicate tickers. "
            f"Duplicates: {[t for t in sp500_tickers if sp500_tickers.count(t) > 1]}"
        )

    def test_tickers_are_uppercase(self, sp500_tickers):
        """Test that all ticker symbols are in uppercase."""
        # Assert
        non_uppercase = [t for t in sp500_tickers if t != t.upper()]
        assert (
            len(non_uppercase) == 0
        ), f"Found {len(non_uppercase)} non-uppercase tickers: {non_uppercase}"

    def test_no_special_characters_except_dot_and_dash(self, sp500_tickers):
        """Test that tickers only contain alphanumerics, dots, or dashes."""
        # Assert
        import re

        pattern = re.compile(r"^[A-Z0-9.-]+$")

        invalid_tickers = [t for t in sp500_tickers if not pattern.match(t)]
        assert (
            len(invalid_tickers) == 0
        ), f"Found tickers with invalid characters: {invalid_tickers}"

    def test_ticker_list_first_last(self, sp500_tickers):
        """Test and display first and last tickers (for manual verification)."""
        # Display
        print(f"\n{'='*60}")
        print(f"Total S&P 500 tickers fetched: {len(sp500_tickers)}")
        print(f"{'='*60}")
        print("\nFirst 10 tickers:")
        for i, ticker in enumerate(sp500_tickers[:10], 1):
            print(f"  {i:2d}. {ticker}")

        print("\nLast 10 tickers:")
        for i, ticker in enumerate(sp500_tickers[-10:], len(sp500_tickers) - 9):
            print(f"  {i:3d}. {ticker}")
        print(f"{'='*60}\n")

        # Always passes - this is just for informational output
        assert True


# ============================================================================
# TEST CLASS - Tests that need fresh fetches
# ============================================================================


class TestWikipediaFreshFetches:
    """Tests that require fresh fetches (not using cached fixture)."""

    def test_timeout_parameter(self):
        """Test that custom timeout parameter is accepted."""
        # Act - Direct fetch with custom timeout
        tickers = fetch_sp500_tickers_from_wikipedia(timeout=30)

        # Assert
        assert (
            len(tickers) > 400
        ), "Should fetch tickers successfully with custom timeout"

    @pytest.mark.slow
    def test_multiple_fetches_consistent(self):
        """Test that multiple fetches return the same results (within same day)."""
        # Act - Need to fetch twice to compare
        tickers1 = fetch_sp500_tickers_from_wikipedia()
        tickers2 = fetch_sp500_tickers_from_wikipedia()

        # Assert
        assert tickers1 == tickers2, (
            "Multiple fetches should return identical results. "
            f"Difference: {set(tickers1).symmetric_difference(set(tickers2))}"
        )

        # Always passes - this is just for informational output
        assert True


class TestWikipediaErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_timeout_raises_error(self):
        """Test that extremely short timeout raises appropriate error."""
        # Act & Assert
        with pytest.raises(Exception):  # Should raise RequestException
            fetch_sp500_tickers_from_wikipedia(timeout=0.001)

    def test_custom_headers_accepted(self):
        """Test that custom headers parameter is accepted."""
        # Arrange
        custom_headers = {
            "User-Agent": "TestBot/1.0",
            "Accept": "text/html",
        }

        # Act
        tickers = fetch_sp500_tickers_from_wikipedia(headers=custom_headers)

        # Assert
        assert len(tickers) > 400, "Should work with custom headers"
