from unittest.mock import MagicMock, patch

import pandas as pd

from src.clenow_momentum.data.fetcher import get_sp500_index_data, get_sp500_tickers, get_stock_data


class TestGetSP500Tickers:
    """Test the S&P 500 ticker fetching function."""

    @patch("src.clenow_momentum.data.fetcher.requests.get")
    @patch("src.clenow_momentum.data.fetcher.pd.read_html")
    def test_successful_fetch(self, mock_read_html, mock_requests):
        """Test successful ticker fetching."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html><table id='constituents'></table></html>"
        mock_requests.return_value = mock_response

        # Mock pandas read_html to return test data
        test_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"]})
        mock_read_html.return_value = [test_df]

        # Call function
        tickers = get_sp500_tickers()

        # Assertions
        assert tickers == ["AAPL", "MSFT", "GOOGL", "AMZN"]
        # Check that the request was made with headers
        mock_requests.assert_called_once()
        call_args = mock_requests.call_args
        assert call_args[0][0] == "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        assert "headers" in call_args[1]
        assert call_args[1]["timeout"] == 10

    @patch("src.clenow_momentum.data.fetcher.get_sp500_tickers_yfinance")
    @patch("src.clenow_momentum.data.fetcher.requests.get")
    def test_request_exception(self, mock_requests, mock_yfinance):
        """Test handling of request exceptions - should fallback to yfinance."""
        import requests
        mock_requests.side_effect = requests.RequestException("Network error")
        mock_yfinance.return_value = ["AAPL", "MSFT"]  # Fallback returns some tickers

        tickers = get_sp500_tickers()

        assert tickers == ["AAPL", "MSFT"]  # Should get fallback results
        mock_yfinance.assert_called_once()

    @patch("src.clenow_momentum.data.fetcher.get_sp500_tickers_yfinance")
    @patch("src.clenow_momentum.data.fetcher.requests.get")
    @patch("src.clenow_momentum.data.fetcher.BeautifulSoup")
    def test_missing_table(self, mock_soup, mock_requests, mock_yfinance):
        """Test handling when constituents table is not found - should fallback."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html></html>"
        mock_requests.return_value = mock_response

        # Mock BeautifulSoup to return None for table
        mock_soup_instance = MagicMock()
        mock_soup_instance.find.return_value = None
        mock_soup.return_value = mock_soup_instance
        
        # Mock yfinance fallback
        mock_yfinance.return_value = ["AAPL", "MSFT"]

        tickers = get_sp500_tickers()

        assert tickers == ["AAPL", "MSFT"]  # Should get fallback results
        mock_yfinance.assert_called_once()

    @patch("src.clenow_momentum.data.fetcher.get_sp500_tickers_yfinance")
    @patch("src.clenow_momentum.data.fetcher.requests.get")
    @patch("src.clenow_momentum.data.fetcher.pd.read_html")
    def test_missing_symbol_column(self, mock_read_html, mock_requests, mock_yfinance):
        """Test handling when Symbol column is missing - should fallback."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html><table id='constituents'></table></html>"
        mock_requests.return_value = mock_response

        # Mock pandas read_html to return data without Symbol column
        test_df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})  # Wrong column name
        mock_read_html.return_value = [test_df]
        
        # Mock yfinance fallback
        mock_yfinance.return_value = ["AAPL", "MSFT"]

        tickers = get_sp500_tickers()

        assert tickers == ["AAPL", "MSFT"]  # Should get fallback results
        mock_yfinance.assert_called_once()


class TestGetStockData:
    """Test stock data fetching function."""

    @patch("src.clenow_momentum.data.fetcher.yf.download")
    def test_successful_data_fetch(self, mock_download):
        """Test successful stock data fetching."""
        # Create mock data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_data = pd.DataFrame(
            {"AAPL": {"Close": [150.0] * 100, "Volume": [1000000] * 100}}, index=dates
        )
        mock_download.return_value = mock_data

        tickers = ["AAPL", "MSFT"]
        data = get_stock_data(tickers, period="1y")

        assert data is not None
        mock_download.assert_called_once_with(
            tickers, period="1y", group_by="ticker", auto_adjust=True
        )

    @patch("src.clenow_momentum.data.fetcher.yf.download")
    def test_data_fetch_exception(self, mock_download):
        """Test handling of yfinance exceptions."""
        mock_download.side_effect = Exception("API error")

        data = get_stock_data(["AAPL"], period="1y")

        assert data is None


class TestGetSP500IndexData:
    """Test S&P 500 index data fetching."""

    @patch("src.clenow_momentum.data.fetcher.yf.download")
    def test_successful_sp500_fetch(self, mock_download):
        """Test successful S&P 500 data fetching."""
        # Create mock S&P 500 data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        mock_data = pd.DataFrame({"Close": [4500.0] * 100, "Volume": [1000000] * 100}, index=dates)
        mock_download.return_value = mock_data

        data = get_sp500_index_data(period="1y")

        assert data is not None
        mock_download.assert_called_once_with("^GSPC", period="1y", auto_adjust=True)

    @patch("src.clenow_momentum.data.fetcher.yf.download")
    def test_sp500_fetch_exception(self, mock_download):
        """Test handling of S&P 500 fetch exceptions."""
        mock_download.side_effect = Exception("API error")

        data = get_sp500_index_data(period="1y")

        assert data is None
