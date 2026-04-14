"""Tests that MarketRegimeDetector resolves benchmark from UniverseSpec, not hardcoded SPY."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from clenow_momentum.market_analysis.regime_detector import MarketRegimeDetector


def _make_price_df(price: float = 500.0, days: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    return pd.DataFrame({"Close": [price] * days}, index=dates)


class TestRegimeBenchmarkTicker:
    """The benchmark_ticker kwarg drives which symbol is fetched."""

    def test_default_benchmark_is_spy(self):
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source)
        assert detector.benchmark_ticker == "SPY"

    def test_russell1000_benchmark_is_iwb(self):
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source, benchmark_ticker="IWB")
        assert detector.benchmark_ticker == "IWB"

    def test_get_market_data_called_with_correct_ticker(self):
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source, benchmark_ticker="IWB")
        detector.check_regime(ma_period=200)

        source.get_market_data.assert_called_once_with(period="1y", benchmark_ticker="IWB")

    def test_spy_detector_calls_spy(self):
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source, benchmark_ticker="SPY")
        detector.check_regime()

        source.get_market_data.assert_called_once_with(period="1y", benchmark_ticker="SPY")

    def test_get_regime_benchmark_data(self):
        """_get_regime_benchmark_data() fetches via market_data_source."""
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source)
        result = detector._get_regime_benchmark_data()

        assert result is not None
        source.get_market_data.assert_called_once()


class TestRegimeFromUniverseSpec:
    """Wire UniverseSpec.benchmark_etf into the detector at construction time."""

    def test_sp500_spec_produces_spy_detector(self):
        from clenow_momentum.data.universes import UNIVERSES
        spec = UNIVERSES["SP500"]
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source, benchmark_ticker=spec.benchmark_etf)
        detector.check_regime()

        call_kwargs = source.get_market_data.call_args[1]
        assert call_kwargs["benchmark_ticker"] == "SPY"

    def test_russell1000_spec_produces_iwb_detector(self):
        from clenow_momentum.data.universes import UNIVERSES
        spec = UNIVERSES["RUSSELL1000"]
        source = MagicMock()
        source.get_market_data.return_value = _make_price_df()

        detector = MarketRegimeDetector(source, benchmark_ticker=spec.benchmark_etf)
        detector.check_regime()

        call_kwargs = source.get_market_data.call_args[1]
        assert call_kwargs["benchmark_ticker"] == "IWB"
