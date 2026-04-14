"""
Tests for market regime detection components in the new market_analysis architecture.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from clenow_momentum.data.interfaces import MarketDataSource, TickerSource
from clenow_momentum.market_analysis import MarketAnalysisFacade
from clenow_momentum.market_analysis.regime_detector import MarketRegimeDetector, RegimeStatus


def create_mock_market_data_source(spy_data: pd.DataFrame | None) -> Mock:
    """Create a mocked MarketDataSource returning provided benchmark data.

    The regime detector now calls get_market_data(period, benchmark_ticker)
    rather than get_index_data directly, so we mock that method.
    """
    market_data_source = Mock(spec=MarketDataSource)
    market_data_source.get_market_data.return_value = spy_data
    market_data_source.get_index_data.return_value = spy_data  # keep for other callers
    return market_data_source


def create_mock_ticker_source() -> Mock:
    """Create a mocked TickerSource for facade construction."""
    ticker_source = Mock(spec=TickerSource)
    ticker_source.get_tickers_for_index.return_value = ["AAPL", "MSFT"]
    ticker_source.get_supported_indices.return_value = ["SP500"]
    ticker_source.is_available.return_value = True
    return ticker_source


def create_bullish_spy_data(periods: int = 250) -> pd.DataFrame:
    """Create bullish SPY data where latest price is above long-term MA."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    closes = list(range(400, 400 + periods))
    return pd.DataFrame({"Close": closes}, index=dates)


def create_bearish_spy_data(periods: int = 250) -> pd.DataFrame:
    """Create bearish SPY data where latest price is below long-term MA."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    closes = list(range(400 + periods, 400, -1))
    return pd.DataFrame({"Close": closes}, index=dates)


class TestMarketRegimeDetector:
    """Test market regime detector behavior."""

    def test_check_regime_bullish(self):
        market_data_source = create_mock_market_data_source(create_bullish_spy_data())
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime(ma_period=200)

        assert result.regime == "bullish"
        assert result.trading_allowed is True
        assert result.current_price is not None
        assert result.ma_value is not None
        assert result.price_vs_ma_pct is not None
        assert result.price_vs_ma_pct > 0
        assert result.error is None

    def test_check_regime_bearish(self):
        market_data_source = create_mock_market_data_source(create_bearish_spy_data())
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime(ma_period=200)

        assert result.regime == "bearish"
        assert result.trading_allowed is False
        assert result.current_price is not None
        assert result.ma_value is not None
        assert result.price_vs_ma_pct is not None
        assert result.price_vs_ma_pct < 0
        assert result.error is None

    def test_check_regime_no_data(self):
        market_data_source = create_mock_market_data_source(None)
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime(ma_period=200)

        assert result.regime == "unknown"
        assert result.trading_allowed is False
        assert result.current_price is None
        assert result.ma_value is None
        assert result.error == "Could not fetch market data"

    def test_check_regime_insufficient_data(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        short_data = pd.DataFrame({"Close": range(400, 450)}, index=dates)
        market_data_source = create_mock_market_data_source(short_data)
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime(ma_period=200)

        assert result.regime == "unknown"
        assert result.trading_allowed is False
        assert result.ma_value is None
        assert "Insufficient data" in (result.error or "")

    def test_check_regime_missing_close_column(self):
        dates = pd.date_range("2024-01-01", periods=250, freq="D")
        invalid_data = pd.DataFrame({"Open": range(250), "High": range(250)}, index=dates)
        market_data_source = create_mock_market_data_source(invalid_data)
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime(ma_period=200)

        assert result.regime == "unknown"
        assert result.trading_allowed is False
        assert "must contain 'Close' column" in (result.error or "")


class TestShouldTradeMomentum:
    """Test trading decision behavior based on regime status."""

    def test_should_trade_momentum_bullish(self):
        detector = MarketRegimeDetector(create_mock_market_data_source(create_bullish_spy_data()))
        regime = RegimeStatus(
            regime="bullish",
            current_price=500.0,
            ma_value=450.0,
            price_vs_ma_pct=11.11,
            trading_allowed=True,
            latest_date="2024-09-01",
            ma_period=200,
            error=None,
        )

        should_trade, reason = detector.should_trade_momentum(regime_status=regime)

        assert should_trade is True
        assert "bullish" in reason.lower()
        assert "200MA" in reason

    def test_should_trade_momentum_bearish(self):
        detector = MarketRegimeDetector(create_mock_market_data_source(create_bearish_spy_data()))
        regime = RegimeStatus(
            regime="bearish",
            current_price=400.0,
            ma_value=450.0,
            price_vs_ma_pct=-11.11,
            trading_allowed=False,
            latest_date="2024-09-01",
            ma_period=200,
            error=None,
        )

        should_trade, reason = detector.should_trade_momentum(regime_status=regime)

        assert should_trade is False
        assert "bearish" in reason.lower()
        assert "suspended" in reason.lower()

    def test_should_trade_momentum_error(self):
        detector = MarketRegimeDetector(create_mock_market_data_source(None))
        regime = RegimeStatus(
            regime="unknown",
            current_price=None,
            ma_value=None,
            price_vs_ma_pct=None,
            trading_allowed=False,
            latest_date=None,
            ma_period=200,
            error="Network error",
        )

        should_trade, reason = detector.should_trade_momentum(regime_status=regime)

        assert should_trade is False
        assert "error" in reason.lower()

    @patch("clenow_momentum.market_analysis.regime_detector.MarketRegimeDetector.check_regime")
    def test_should_trade_momentum_no_regime_provided(self, mock_check_regime):
        detector = MarketRegimeDetector(create_mock_market_data_source(create_bullish_spy_data()))
        mock_check_regime.return_value = RegimeStatus(
            regime="bullish",
            current_price=500.0,
            ma_value=450.0,
            price_vs_ma_pct=11.11,
            trading_allowed=True,
            latest_date="2024-09-01",
            ma_period=200,
            error=None,
        )

        should_trade, _ = detector.should_trade_momentum()

        assert should_trade is True
        mock_check_regime.assert_called_once()


class TestMarketAnalysisFacade:
    """Test facade behavior replacing legacy helper functions."""

    def test_check_regime(self):
        market_data_source = create_mock_market_data_source(create_bullish_spy_data())
        facade = MarketAnalysisFacade(market_data_source, create_mock_ticker_source())

        result = facade.check_regime(period=200)

        assert result.regime == "bullish"
        assert result.trading_allowed is True
        assert result.error is None

    def test_get_detailed_status_current_behavior(self):
        market_data_source = create_mock_market_data_source(create_bullish_spy_data())
        facade = MarketAnalysisFacade(market_data_source, create_mock_ticker_source())

        result = facade.get_detailed_status(period=50)

        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "Extended regime metrics not yet implemented"

    def test_should_trade_momentum_via_facade(self):
        market_data_source = create_mock_market_data_source(create_bullish_spy_data())
        facade = MarketAnalysisFacade(market_data_source, create_mock_ticker_source())

        should_trade, reason = facade.should_trade_momentum(period=200)

        assert should_trade is True
        assert "bullish" in reason.lower()


class TestErrorHandling:
    """Test error handling paths in regime APIs."""

    def test_detector_handles_datasource_exception(self):
        market_data_source = Mock(spec=MarketDataSource)
        market_data_source.get_index_data.side_effect = RuntimeError("Data source down")
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime()

        assert result.regime == "unknown"
        assert result.trading_allowed is False
        assert result.error == "Could not fetch market data"

    def test_facade_returns_false_when_regime_has_error(self):
        market_data_source = create_mock_market_data_source(None)
        facade = MarketAnalysisFacade(market_data_source, create_mock_ticker_source())

        should_trade, reason = facade.should_trade_momentum(period=200)

        assert should_trade is False
        assert "market data error" in reason.lower()


class TestInputValidation:
    """Test input validation and deterministic calculation characteristics."""

    def test_detector_uses_requested_ma_period(self):
        market_data_source = create_mock_market_data_source(create_bullish_spy_data(periods=260))
        detector = MarketRegimeDetector(market_data_source)

        result_50 = detector.check_regime(ma_period=50)
        result_200 = detector.check_regime(ma_period=200)

        assert result_50.ma_period == 50
        assert result_200.ma_period == 200
        assert result_50.ma_value is not None
        assert result_200.ma_value is not None

    def test_detector_returns_dataclass(self):
        market_data_source = create_mock_market_data_source(create_bullish_spy_data())
        detector = MarketRegimeDetector(market_data_source)

        result = detector.check_regime()

        assert isinstance(result, RegimeStatus)
        assert result.regime in {"bullish", "bearish", "unknown"}
        assert isinstance(result.trading_allowed, bool)
