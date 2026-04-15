"""Tests for the generic Wikipedia constituent fetcher.

Uses HTML fixtures instead of live network calls so tests are fast and
deterministic. The strict table-selection logic (filter by symbol column
+ expected_row_range) is the main focus.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.clenow_momentum.data.sources.sp500_wikipedia import (
    fetch_index_constituents_from_wikipedia,
    fetch_index_tickers_from_wikipedia,
)
from src.clenow_momentum.data.universes import UNIVERSES


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

def _make_sp500_html(tickers: list[str]) -> str:
    """Minimal S&P 500 Wikipedia page with id='constituents' table."""
    rows = "\n".join(
        f"<tr><td>{t}</td><td>Company {t}</td><td>Technology</td></tr>"
        for t in tickers
    )
    return f"""
    <html><body>
    <table id="constituents">
      <thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </body></html>
    """


def _make_russell_html(tickers: list[str], col_name: str = "Symbol") -> str:
    """Minimal Russell 1000 page with a matching table (no stable id)."""
    rows = "\n".join(
        f"<tr><td>{t}</td><td>Company {t}</td><td>Industrials</td></tr>"
        for t in tickers
    )
    return f"""
    <html><body>
    <p>Some intro text</p>
    <table>
      <thead><tr><th>{col_name}</th><th>Company</th><th>GICS Sector</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </body></html>
    """


def _make_ambiguous_html(tickers: list[str]) -> str:
    """Page with TWO matching tables — should trigger strict-selection error."""
    rows = "\n".join(f"<tr><td>{t}</td></tr>" for t in tickers)
    table = f"""
    <table>
      <thead><tr><th>Symbol</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """
    return f"<html><body>{table}{table}</body></html>"


def _make_no_match_html() -> str:
    """Page with a table that has wrong columns and wrong row count."""
    return """
    <html><body>
    <table>
      <thead><tr><th>Name</th><th>Industry</th></tr></thead>
      <tbody><tr><td>Foo</td><td>Tech</td></tr></tbody>
    </table>
    </body></html>
    """


def _mock_response(html: str) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# S&P 500 (id-based selection)
# ---------------------------------------------------------------------------

class TestSP500Fetcher:
    SP500 = UNIVERSES["SP500"]

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_returns_tickers(self, mock_get):
        tickers = [f"T{i:03d}" for i in range(503)]
        mock_get.return_value = _mock_response(_make_sp500_html(tickers))

        result = fetch_index_tickers_from_wikipedia(self.SP500)

        assert len(result) == 503
        assert result[0] == "T000"

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_returns_constituent_metadata(self, mock_get):
        tickers = [f"T{i:03d}" for i in range(503)]
        mock_get.return_value = _mock_response(_make_sp500_html(tickers))

        result = fetch_index_constituents_from_wikipedia(self.SP500)

        assert list(result.columns) == ["source_symbol", "company_name", "sector"]
        assert result.iloc[0].to_dict() == {
            "source_symbol": "T000",
            "company_name": "Company T000",
            "sector": "Technology",
        }

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_missing_table_id_raises(self, mock_get):
        mock_get.return_value = _mock_response("<html><body><p>no table</p></body></html>")

        with pytest.raises(ValueError, match="constituents"):
            fetch_index_tickers_from_wikipedia(self.SP500)

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_missing_symbol_column_raises(self, mock_get):
        # Table exists but has wrong column name
        html = """<html><body>
        <table id="constituents">
          <thead><tr><th>Ticker</th></tr></thead>
          <tbody><tr><td>AAPL</td></tr></tbody>
        </table></body></html>"""
        mock_get.return_value = _mock_response(html)

        with pytest.raises(ValueError, match="Symbol"):
            fetch_index_tickers_from_wikipedia(self.SP500)

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_network_error_propagates(self, mock_get):
        import requests as req
        mock_get.side_effect = req.RequestException("timeout")

        with pytest.raises(req.RequestException):
            fetch_index_tickers_from_wikipedia(self.SP500)


# ---------------------------------------------------------------------------
# Russell 1000 (filter-based selection)
# ---------------------------------------------------------------------------

class TestRussell1000Fetcher:
    R1K = UNIVERSES["RUSSELL1000"]

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_returns_tickers_symbol_column(self, mock_get):
        tickers = [f"T{i:04d}" for i in range(1000)]
        mock_get.return_value = _mock_response(_make_russell_html(tickers, col_name="Symbol"))

        result = fetch_index_tickers_from_wikipedia(self.R1K)

        assert len(result) == 1000

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_returns_constituent_metadata(self, mock_get):
        tickers = [f"T{i:04d}" for i in range(1000)]
        mock_get.return_value = _mock_response(_make_russell_html(tickers, col_name="Symbol"))

        result = fetch_index_constituents_from_wikipedia(self.R1K)

        assert result.loc[0, "source_symbol"] == "T0000"
        assert result.loc[0, "company_name"] == "Company T0000"
        assert result.loc[0, "sector"] == "Industrials"

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_returns_tickers_ticker_column(self, mock_get):
        """Falls back to 'Ticker' column name."""
        tickers = [f"T{i:04d}" for i in range(1000)]
        mock_get.return_value = _mock_response(_make_russell_html(tickers, col_name="Ticker"))

        result = fetch_index_tickers_from_wikipedia(self.R1K)

        assert len(result) == 1000

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_ambiguous_tables_raises(self, mock_get):
        """Two matching tables → strict filter should raise."""
        tickers = [f"T{i:04d}" for i in range(1000)]
        mock_get.return_value = _mock_response(_make_ambiguous_html(tickers))

        with pytest.raises(ValueError, match="Ambiguous"):
            fetch_index_tickers_from_wikipedia(self.R1K)

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_no_matching_table_raises(self, mock_get):
        mock_get.return_value = _mock_response(_make_no_match_html())

        with pytest.raises(ValueError, match="No table matched"):
            fetch_index_tickers_from_wikipedia(self.R1K)

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_row_count_outside_range_raises(self, mock_get):
        """Only 5 rows — outside expected_row_range → no match."""
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
        mock_get.return_value = _mock_response(_make_russell_html(tickers, col_name="Symbol"))

        with pytest.raises(ValueError, match="No table matched"):
            fetch_index_tickers_from_wikipedia(self.R1K)


# ---------------------------------------------------------------------------
# Back-compat aliases
# ---------------------------------------------------------------------------

class TestBackCompatAliases:
    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_fetch_sp500_tickers_from_wikipedia(self, mock_get):
        from src.clenow_momentum.data.sources.sp500_wikipedia import (
            fetch_sp500_tickers_from_wikipedia,
        )
        tickers = [f"T{i:03d}" for i in range(503)]
        mock_get.return_value = _mock_response(_make_sp500_html(tickers))

        result = fetch_sp500_tickers_from_wikipedia()

        assert len(result) == 503

    @patch("src.clenow_momentum.data.sources.sp500_wikipedia.requests.get")
    def test_get_sp500_tickers_wikipedia(self, mock_get):
        from src.clenow_momentum.data.sources.sp500_wikipedia import get_sp500_tickers_wikipedia
        tickers = [f"T{i:03d}" for i in range(503)]
        mock_get.return_value = _mock_response(_make_sp500_html(tickers))

        result = get_sp500_tickers_wikipedia()

        assert len(result) == 503
