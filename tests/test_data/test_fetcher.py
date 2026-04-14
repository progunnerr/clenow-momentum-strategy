"""Tests for data provider functions."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.clenow_momentum.data import get_sp500_index_data, get_sp500_tickers, get_stock_data


class TestGetSP500Tickers:
    """Test S&P 500 ticker retrieval."""

    @patch("src.clenow_momentum.data.provider.fetch_sp500_tickers_from_wikipedia")
    def test_successful_fetch(self, mock_fetch):
        mock_fetch.return_value = ["AAPL", "MSFT", "GOOGL", "AMZN"]

        tickers = get_sp500_tickers(use_cache=False)

        assert tickers == ["AAPL", "MSFT", "GOOGL", "AMZN"]
        mock_fetch.assert_called_once_with(timeout=10)

    @patch("src.clenow_momentum.data.provider.fetch_sp500_tickers_from_wikipedia")
    def test_fetch_exception_raises_runtime_error(self, mock_fetch):
        mock_fetch.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Unable to fetch S&P 500 tickers"):
            get_sp500_tickers(use_cache=False)

    @patch("src.clenow_momentum.data.provider.fetch_sp500_tickers_from_wikipedia")
    def test_empty_fetch_raises_runtime_error(self, mock_fetch):
        mock_fetch.return_value = []

        with pytest.raises(RuntimeError, match="Unable to fetch S&P 500 tickers"):
            get_sp500_tickers(use_cache=False)


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

        tickers = ["AAPL", "MSFT"]
        data = get_stock_data(tickers, period="1y", use_cache=False)

        assert data is not None
        assert not data.empty
        mock_download.assert_called_once()

    @patch("src.clenow_momentum.data.provider.download_stock_data")
    def test_data_fetch_exception(self, mock_download):
        mock_download.side_effect = Exception("API error")

        data = get_stock_data(["AAPL"], period="1y", use_cache=False)

        assert data is None


class TestGetSP500IndexData:
    """Test S&P 500 index data retrieval."""

    @patch("src.clenow_momentum.data.provider.download_index_data")
    def test_successful_sp500_fetch(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_data = pd.DataFrame(
            {
                "Close": [4500.0] * 5,
                "Volume": [1_000_000] * 5,
            },
            index=dates,
        )
        mock_download.return_value = mock_data

        data = get_sp500_index_data(period="1y", use_cache=False)

        assert data is not None
        assert not data.empty
        mock_download.assert_called_once()

    @patch("src.clenow_momentum.data.provider.download_index_data")
    def test_sp500_fetch_exception(self, mock_download):
        mock_download.side_effect = Exception("API error")

        data = get_sp500_index_data(period="1y", use_cache=False)

        assert data is None
