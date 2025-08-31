"""
Tests for market regime detection module.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from clenow_momentum.strategy.market_regime import (
    calculate_market_ma,
    check_market_regime,
    get_sp500_ma_status,
    should_trade_momentum,
)


class TestMarketMA:
    """Test market moving average calculations."""

    def create_sample_spy_data(self):
        """Create sample SPY data for testing."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')

        # Create trending upward data
        base_prices = list(range(400, 450))  # 50 SPY-like prices
        extension_prices = list(range(450, 650))  # 200 more prices
        prices = base_prices + extension_prices  # Total 250 prices

        return pd.DataFrame({
            'Open': prices,
            'High': [p + 2 for p in prices],
            'Low': [p - 2 for p in prices],
            'Close': prices,
            'Volume': [1000000] * len(dates)
        }, index=dates)

    def test_calculate_market_ma(self):
        """Test market moving average calculation."""
        spy_data = self.create_sample_spy_data()

        # Calculate 50-day MA
        ma = calculate_market_ma(spy_data, period=50)

        assert isinstance(ma, pd.Series)
        assert len(ma) == len(spy_data)

        # First 49 values should be NaN
        assert pd.isna(ma.iloc[:49]).all()

        # 50th value should be average of first 50 closes
        expected_ma50 = spy_data['Close'].iloc[:50].mean()
        assert abs(ma.iloc[49] - expected_ma50) < 0.001

    def test_calculate_market_ma_missing_close(self):
        """Test MA calculation with missing Close column."""
        spy_data = self.create_sample_spy_data()
        spy_data = spy_data.drop('Close', axis=1)

        with pytest.raises(ValueError, match="SPY data must contain 'Close' column"):
            calculate_market_ma(spy_data, period=50)


class TestMarketRegime:
    """Test market regime detection."""

    def create_bullish_spy_data(self):
        """Create bullish SPY data (above MA)."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')

        # Uptrending prices - current price above MA
        base_prices = list(range(400, 500))  # Strong uptrend

        return pd.DataFrame({
            'Close': base_prices + [510] * (250 - len(base_prices))  # Current price high
        }, index=dates)

    def create_bearish_spy_data(self):
        """Create bearish SPY data (below MA)."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')

        # Downtrending prices - current price below MA
        base_prices = list(range(500, 400, -1))  # Strong downtrend

        return pd.DataFrame({
            'Close': base_prices + [390] * (250 - len(base_prices))  # Current price low
        }, index=dates)

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_check_market_regime_bullish(self, mock_get_data):
        """Test bullish market regime detection."""
        mock_get_data.return_value = self.create_bullish_spy_data()

        result = check_market_regime(period=200)

        assert result['regime'] == 'bullish'
        assert result['trading_allowed'] is True
        assert result['current_price'] is not None
        assert result['ma_value'] is not None
        assert result['price_vs_ma'] > 0  # Price above MA

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_check_market_regime_bearish(self, mock_get_data):
        """Test bearish market regime detection."""
        mock_get_data.return_value = self.create_bearish_spy_data()

        result = check_market_regime(period=200)

        assert result['regime'] == 'bearish'
        assert result['trading_allowed'] is False
        assert result['current_price'] is not None
        assert result['ma_value'] is not None
        assert result['price_vs_ma'] < 0  # Price below MA

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_check_market_regime_no_data(self, mock_get_data):
        """Test market regime check with no data."""
        mock_get_data.return_value = None

        result = check_market_regime(period=200)

        assert result['regime'] == 'unknown'
        assert result['trading_allowed'] is False
        assert 'error' in result
        assert result['current_price'] is None

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_check_market_regime_insufficient_data(self, mock_get_data):
        """Test market regime with insufficient data for MA."""
        # Create data with only 50 days (insufficient for 200-day MA)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        short_data = pd.DataFrame({
            'Close': range(400, 450)
        }, index=dates)

        mock_get_data.return_value = short_data

        result = check_market_regime(period=200)

        assert result['regime'] == 'unknown'
        assert result['trading_allowed'] is False
        assert 'error' in result


class TestSP500MAStatus:
    """Test detailed S&P 500 MA status."""

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_get_sp500_ma_status(self, mock_get_data):
        """Test getting detailed MA status."""
        spy_data = self.create_sample_spy_data()
        mock_get_data.return_value = spy_data

        result = get_sp500_ma_status(period=50)

        assert 'current_price' in result
        assert 'ma_value' in result
        assert 'above_ma' in result
        assert 'ma_trend' in result
        assert 'consecutive_days_current_regime' in result
        assert 'recent_30d_above_ma' in result
        assert 'recent_30d_below_ma' in result
        assert result['ma_period'] == 50

    def create_sample_spy_data(self):
        """Create sample SPY data with trend."""
        dates = pd.date_range('2024-01-01', periods=300, freq='D')

        # Create uptrending data
        base_prices = list(range(400, 500)) + list(range(500, 550))
        prices = base_prices + [550] * (300 - len(base_prices))

        return pd.DataFrame({
            'Close': prices
        }, index=dates)

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_get_sp500_ma_status_no_data(self, mock_get_data):
        """Test MA status with no data."""
        mock_get_data.return_value = None

        result = get_sp500_ma_status(period=50)

        assert 'error' in result


class TestShouldTradeMomentum:
    """Test trading decision logic."""

    def test_should_trade_momentum_bullish(self):
        """Test trading decision with bullish regime."""
        market_regime = {
            'regime': 'bullish',
            'trading_allowed': True,
            'ma_period': 200
        }

        should_trade, reason = should_trade_momentum(market_regime)

        assert should_trade is True
        assert 'bullish' in reason.lower()
        assert '200MA' in reason

    def test_should_trade_momentum_bearish(self):
        """Test trading decision with bearish regime."""
        market_regime = {
            'regime': 'bearish',
            'trading_allowed': False,
            'ma_period': 200
        }

        should_trade, reason = should_trade_momentum(market_regime)

        assert should_trade is False
        assert 'bearish' in reason.lower()
        assert 'suspended' in reason.lower()

    def test_should_trade_momentum_error(self):
        """Test trading decision with error."""
        market_regime = {
            'regime': 'unknown',
            'trading_allowed': False,
            'error': 'Network error'
        }

        should_trade, reason = should_trade_momentum(market_regime)

        assert should_trade is False
        assert 'error' in reason.lower()

    @patch('clenow_momentum.strategy.market_regime.check_market_regime')
    def test_should_trade_momentum_no_regime_provided(self, mock_check_regime):
        """Test trading decision when no regime provided."""
        mock_check_regime.return_value = {
            'regime': 'bullish',
            'trading_allowed': True,
            'ma_period': 200
        }

        should_trade, reason = should_trade_momentum()

        assert should_trade is True
        mock_check_regime.assert_called_once()


class TestErrorHandling:
    """Test error handling in market regime functions."""

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_check_market_regime_exception(self, mock_get_data):
        """Test exception handling in market regime check."""
        # Return None to simulate failed data fetch
        mock_get_data.return_value = None

        result = check_market_regime()

        assert result['regime'] == 'unknown'
        assert result['trading_allowed'] is False
        assert 'error' in result

    @patch('clenow_momentum.strategy.market_regime.get_sp500_data')
    def test_get_sp500_ma_status_exception(self, mock_get_data):
        """Test exception handling in MA status."""
        # Return None to simulate failed data fetch
        mock_get_data.return_value = None

        result = get_sp500_ma_status()

        assert 'error' in result
