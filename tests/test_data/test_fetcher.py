"""Tests for data provider functions."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.clenow_momentum.data import (
    get_universe_constituents,
    get_sp500_index_data,
    get_sp500_tickers,
    get_stock_data,
    get_universe_tickers,
)


# ---------------------------------------------------------------------------
# Back-compat: get_sp500_tickers still works
# ---------------------------------------------------------------------------

class TestGetSP500Tickers:
    """get_sp500_tickers() delegates to get_universe_tickers('SP500')."""

    @patch("src.clenow_momentum.data.provider.fetch_index_constituents_from_wikipedia")
    def test_successful_fetch(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(
            {
                "source_symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "company_name": ["Apple", "Microsoft", "Alphabet", "Amazon"],
                "sector": ["Technology", "Technology", "Communication Services", "Consumer Discretionary"],
            }
        )

        tickers = get_sp500_tickers(use_cache=False)

        assert tickers == ["AAPL", "MSFT", "GOOGL", "AMZN"]
        mock_fetch.assert_called_once()

    @patch("src.clenow_momentum.data.provider.fetch_index_constituents_from_wikipedia")
    def test_fetch_exception_raises_runtime_error(self, mock_fetch):
        mock_fetch.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Unable to fetch S&P 500 tickers"):
            get_sp500_tickers(use_cache=False)

    @patch("src.clenow_momentum.data.provider.fetch_index_constituents_from_wikipedia")
    def test_empty_fetch_raises_runtime_error(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(
            columns=["source_symbol", "company_name", "sector"]
        )

        with pytest.raises(RuntimeError, match="Unable to fetch S&P 500 tickers"):
            get_sp500_tickers(use_cache=False)


# ---------------------------------------------------------------------------
# Generic: get_universe_tickers works for SP500 and RUSSELL1000
# ---------------------------------------------------------------------------

class TestGetUniverseTickers:
    """get_universe_tickers() supports all registered universes."""

    @pytest.mark.parametrize("symbol", ["SP500", "RUSSELL1000"])
    @patch("src.clenow_momentum.data.provider.fetch_index_constituents_from_wikipedia")
    def test_successful_fetch(self, mock_fetch, symbol):
        mock_fetch.return_value = pd.DataFrame(
            {
                "source_symbol": ["AAPL", "MSFT"],
                "company_name": ["Apple", "Microsoft"],
                "sector": ["Technology", "Technology"],
            }
        )

        tickers = get_universe_tickers(symbol, use_cache=False)

        assert tickers == ["AAPL", "MSFT"]
        mock_fetch.assert_called_once()

    def test_unknown_universe_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe_tickers("INVALID", use_cache=False)

    @patch("src.clenow_momentum.data.provider.fetch_index_constituents_from_wikipedia")
    def test_russell1000_uses_correct_spec(self, mock_fetch):
        """Confirm the RUSSELL1000 spec is passed to the fetcher."""
        from src.clenow_momentum.data.universes import UNIVERSES
        mock_fetch.return_value = pd.DataFrame(
            {
                "source_symbol": ["AAPL"],
                "company_name": ["Apple"],
                "sector": ["Technology"],
            }
        )

        get_universe_tickers("RUSSELL1000", use_cache=False)

        called_spec = mock_fetch.call_args[0][0]
        assert called_spec == UNIVERSES["RUSSELL1000"]
        assert called_spec.benchmark_etf == "IWB"
        assert called_spec.benchmark_index == "^RUI"


class TestGetUniverseConstituents:
    """get_universe_constituents() returns normalized metadata."""

    @patch("src.clenow_momentum.data.provider.fetch_index_constituents_from_wikipedia")
    def test_returns_normalized_constituent_metadata(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame(
            {
                "source_symbol": ["BRK.B", "MSFT"],
                "company_name": ["Berkshire Hathaway", "Microsoft"],
                "sector": ["Financials", "Technology"],
            }
        )

        constituents = get_universe_constituents("SP500", use_cache=False)

        assert constituents["ticker"].tolist() == ["BRK-B", "MSFT"]
        assert constituents["source_symbol"].tolist() == ["BRK.B", "MSFT"]
        assert constituents["company_name"].tolist() == ["Berkshire Hathaway", "Microsoft"]
        assert constituents["sector"].tolist() == ["Financials", "Technology"]


# ---------------------------------------------------------------------------
# Stock data
# ---------------------------------------------------------------------------

class TestGetStockData:
    """Test stock data retrieval."""

    @patch("src.clenow_momentum.data.provider.download_stock_data")
    def test_successful_data_fetch(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_data = pd.DataFrame(
            {
                ("AAPL", "Close"): [150.0] * 5,
                ("AAPL", "Volume"): [1_000_000] * 5,
            },
            index=dates,
        )
        mock_download.return_value = mock_data

        data = get_stock_data(["AAPL", "MSFT"], period="1y", use_cache=False)

        assert data is not None
        assert not data.empty
        mock_download.assert_called_once()

    @patch("src.clenow_momentum.data.provider.download_stock_data")
    def test_data_fetch_exception_returns_none(self, mock_download):
        mock_download.side_effect = Exception("API error")

        data = get_stock_data(["AAPL"], period="1y", use_cache=False)

        assert data is None


# ---------------------------------------------------------------------------
# SP500 index data (back-compat wrapper)
# ---------------------------------------------------------------------------

class TestGetSP500IndexData:
    """get_sp500_index_data() delegates to get_index_data('SP500')."""

    @patch("src.clenow_momentum.data.provider.download_index_data")
    def test_successful_sp500_fetch(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_data = pd.DataFrame(
            {"Close": [4500.0] * 5, "Volume": [1_000_000] * 5},
            index=dates,
        )
        mock_download.return_value = mock_data

        data = get_sp500_index_data(period="1y", use_cache=False)

        assert data is not None
        assert not data.empty
        mock_download.assert_called_once()

    @patch("src.clenow_momentum.data.provider.download_index_data")
    def test_sp500_fetch_exception_returns_none(self, mock_download):
        mock_download.side_effect = Exception("API error")

        data = get_sp500_index_data(period="1y", use_cache=False)

        assert data is None
