"""
Tests for risk management module.
"""

import numpy as np
import pandas as pd
import pytest

from clenow_momentum.indicators.risk import (
    apply_risk_limits,
    build_portfolio,
    calculate_atr,
    calculate_atr_for_universe,
    calculate_position_size,
    calculate_true_range,
)


class TestTrueRange:
    """Test True Range calculations."""

    def test_calculate_true_range(self):
        """Test basic True Range calculation."""
        # Create sample OHLC data
        high = pd.Series([105, 110, 115, 120, 118])
        low = pd.Series([100, 105, 110, 115, 112])
        close = pd.Series([102, 108, 113, 117, 115])

        tr = calculate_true_range(high, low, close)

        # First value should be High - Low (no previous close)
        assert tr.iloc[0] == 5  # 105 - 100

        # Second value should consider previous close
        # Options: 110-105=5, |110-102|=8, |105-102|=3 → max=8
        assert tr.iloc[1] == 8

    def test_true_range_with_gaps(self):
        """Test True Range with price gaps."""
        high = pd.Series([105, 150, 145])  # Large gap up
        low = pd.Series([100, 145, 140])
        close = pd.Series([102, 148, 142])

        tr = calculate_true_range(high, low, close)

        # Second period has large gap
        # Options: 150-145=5, |150-102|=48, |145-102|=43 → max=48
        assert tr.iloc[1] == 48


class TestATR:
    """Test ATR calculations."""

    def create_sample_ohlc_data(self, periods=30):
        """Create sample OHLC data for testing."""
        dates = pd.date_range('2024-01-01', periods=periods, freq='D')

        # Create trending data with volatility
        close_prices = np.linspace(100, 120, periods) + np.random.normal(0, 2, periods)
        high_prices = close_prices + np.random.uniform(1, 3, periods)
        low_prices = close_prices - np.random.uniform(1, 3, periods)
        open_prices = close_prices + np.random.normal(0, 1, periods)

        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 2000000, periods)
        }, index=dates)

    def test_calculate_atr(self):
        """Test ATR calculation."""
        data = self.create_sample_ohlc_data(30)

        atr = calculate_atr(data, period=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(data)

        # First values should be NaN until we have enough data
        assert pd.isna(atr.iloc[0])

        # Later values should be positive
        assert atr.iloc[-1] > 0

    def test_atr_missing_columns(self):
        """Test ATR calculation with missing required columns."""
        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103]
            # Missing High and Low
        })

        with pytest.raises(ValueError, match="Data must contain 'High' column"):
            calculate_atr(data)


class TestATRForUniverse:
    """Test ATR calculation for multiple stocks."""

    def create_sample_multiindex_data(self):
        """Create sample MultiIndex stock data."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        data_dict = {}

        # Stock A: Normal volatility
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            base_prices = np.linspace(100, 120, len(dates))
            if col == 'Volume':
                data_dict[('STOCK_A', col)] = np.random.randint(1000000, 2000000, len(dates))
            elif col == 'High':
                data_dict[('STOCK_A', col)] = base_prices + 2
            elif col == 'Low':
                data_dict[('STOCK_A', col)] = base_prices - 2
            else:
                data_dict[('STOCK_A', col)] = base_prices + np.random.normal(0, 0.5, len(dates))

        # Stock B: Higher volatility
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            base_prices = np.linspace(200, 180, len(dates))
            if col == 'Volume':
                data_dict[('STOCK_B', col)] = np.random.randint(1000000, 2000000, len(dates))
            elif col == 'High':
                data_dict[('STOCK_B', col)] = base_prices + 5  # Higher volatility
            elif col == 'Low':
                data_dict[('STOCK_B', col)] = base_prices - 5
            else:
                data_dict[('STOCK_B', col)] = base_prices + np.random.normal(0, 2, len(dates))

        df = pd.DataFrame(data_dict, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Ticker', 'Type'])

        return df

    def test_calculate_atr_for_universe(self):
        """Test ATR calculation for multiple stocks."""
        data = self.create_sample_multiindex_data()

        result = calculate_atr_for_universe(data, period=14)

        assert not result.empty
        assert 'ticker' in result.columns
        assert 'atr' in result.columns
        assert len(result) == 2  # Two stocks

        # Stock B should have higher ATR (more volatile)
        stock_a_atr = result[result['ticker'] == 'STOCK_A']['atr'].iloc[0]
        stock_b_atr = result[result['ticker'] == 'STOCK_B']['atr'].iloc[0]

        assert stock_b_atr > stock_a_atr


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        account_value = 1000000  # $1M
        risk_per_trade = 0.001   # 0.1%
        stock_price = 100
        atr = 2  # $2 ATR

        result = calculate_position_size(
            account_value=account_value,
            risk_per_trade=risk_per_trade,
            stock_price=stock_price,
            atr=atr
        )

        # Risk amount = $1M * 0.1% = $1,000
        # Shares = $1,000 / $2 ATR = 500 shares
        assert result['shares'] == 500
        assert result['investment_amount'] == 50000  # 500 * $100
        assert result['actual_risk'] == 1000  # 500 * $2
        assert result['limited_by'] == 'risk_limit'

    def test_position_size_limited_by_max_position(self):
        """Test position sizing limited by maximum position size."""
        account_value = 1000000
        risk_per_trade = 0.001
        stock_price = 50  # Lower price
        atr = 0.5  # Low ATR
        max_position_pct = 0.05  # 5% max position

        result = calculate_position_size(
            account_value=account_value,
            risk_per_trade=risk_per_trade,
            stock_price=stock_price,
            atr=atr,
            max_position_pct=max_position_pct
        )

        # Risk-based: $1,000 / $0.5 = 2,000 shares = $100,000
        # Position limit: 5% of $1M = $50,000 = 1,000 shares
        assert result['shares'] == 1000  # Limited by position size
        assert result['investment_amount'] == 50000
        assert result['limited_by'] == 'position_limit'

    def test_position_size_invalid_inputs(self):
        """Test position sizing with invalid inputs."""
        with pytest.raises(ValueError, match="ATR must be positive"):
            calculate_position_size(1000000, 0.001, 100, 0)  # Zero ATR

        with pytest.raises(ValueError, match="Stock price must be positive"):
            calculate_position_size(1000000, 0.001, 0, 2)  # Zero price


class TestBuildPortfolio:
    """Test portfolio construction."""

    def create_sample_filtered_stocks(self):
        """Create sample filtered stocks data."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'momentum_score': [0.8, 0.7, 0.6],
            'current_price': [150, 300, 2500],
            'annualized_slope': [1.2, 1.0, 0.8],
            'r_squared': [0.85, 0.80, 0.75]
        })

    def create_sample_stock_data_for_portfolio(self):
        """Create sample stock data for portfolio building."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')

        data_dict = {}
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            base_price = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500}[ticker]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col == 'Volume':
                    data_dict[(ticker, col)] = np.random.randint(1000000, 2000000, len(dates))
                elif col == 'High':
                    data_dict[(ticker, col)] = [base_price + 2] * len(dates)
                elif col == 'Low':
                    data_dict[(ticker, col)] = [base_price - 2] * len(dates)
                else:
                    data_dict[(ticker, col)] = [base_price] * len(dates)

        df = pd.DataFrame(data_dict, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Ticker', 'Type'])

        return df

    def test_build_portfolio(self):
        """Test complete portfolio building."""
        filtered_stocks = self.create_sample_filtered_stocks()
        stock_data = self.create_sample_stock_data_for_portfolio()

        portfolio = build_portfolio(
            filtered_stocks=filtered_stocks,
            stock_data=stock_data,
            account_value=1000000,
            risk_per_trade=0.001
        )

        assert not portfolio.empty
        assert len(portfolio) == 3  # Three stocks

        # Check required columns
        required_cols = ['ticker', 'shares', 'investment', 'atr', 'actual_risk']
        for col in required_cols:
            assert col in portfolio.columns

        # All positions should have positive shares
        assert (portfolio['shares'] > 0).all()

        # Portfolio should be sorted by investment amount
        assert portfolio['investment'].is_monotonic_decreasing

    def test_build_portfolio_empty_stocks(self):
        """Test portfolio building with empty stock list."""
        empty_stocks = pd.DataFrame()
        stock_data = self.create_sample_stock_data_for_portfolio()

        portfolio = build_portfolio(
            filtered_stocks=empty_stocks,
            stock_data=stock_data
        )

        assert portfolio.empty


class TestApplyRiskLimits:
    """Test risk limit application."""

    def create_sample_portfolio(self):
        """Create sample portfolio for testing."""
        return pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D', 'E'],
            'shares': [100, 200, 300, 50, 150],
            'investment': [50000, 40000, 30000, 5000, 25000],  # One below min
            'atr': [2.5, 2.0, 1.5, 1.0, 3.0],
            'actual_risk': [250, 400, 450, 50, 450]
        })

    def test_apply_risk_limits_max_positions(self):
        """Test maximum position limit."""
        portfolio = self.create_sample_portfolio()

        limited_portfolio = apply_risk_limits(
            portfolio_df=portfolio,
            max_positions=3,
            min_position_value=0
        )

        assert len(limited_portfolio) == 3
        # Should keep the 3 largest positions
        assert limited_portfolio['investment'].tolist() == [50000, 40000, 30000]

    def test_apply_risk_limits_min_position_value(self):
        """Test minimum position value filter."""
        portfolio = self.create_sample_portfolio()

        limited_portfolio = apply_risk_limits(
            portfolio_df=portfolio,
            max_positions=10,
            min_position_value=10000
        )

        # Should exclude position with $5,000 investment
        assert len(limited_portfolio) == 4
        assert 5000 not in limited_portfolio['investment'].tolist()

    def test_apply_risk_limits_empty_portfolio(self):
        """Test risk limits with empty portfolio."""
        empty_portfolio = pd.DataFrame()

        result = apply_risk_limits(empty_portfolio)

        assert result.empty


class TestIntegration:
    """Integration tests for complete risk management workflow."""

    def test_complete_workflow(self):
        """Test complete risk management workflow."""
        # Create realistic test data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        # Create stock data with different volatilities
        stocks = ['AAPL', 'MSFT', 'GOOGL']
        data_dict = {}

        for i, ticker in enumerate(stocks):
            base_price = 100 + i * 50  # Different price levels
            volatility = 1 + i * 0.5   # Different volatilities

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col == 'Volume':
                    data_dict[(ticker, col)] = np.random.randint(1000000, 2000000, len(dates))
                elif col == 'High':
                    data_dict[(ticker, col)] = [base_price + volatility] * len(dates)
                elif col == 'Low':
                    data_dict[(ticker, col)] = [base_price - volatility] * len(dates)
                else:
                    data_dict[(ticker, col)] = [base_price] * len(dates)

        stock_data = pd.DataFrame(data_dict, index=dates)
        stock_data.columns = pd.MultiIndex.from_tuples(stock_data.columns)

        # Create filtered stocks
        filtered_stocks = pd.DataFrame({
            'ticker': stocks,
            'momentum_score': [0.8, 0.7, 0.6],
            'current_price': [100, 150, 200]
        })

        # Build portfolio
        portfolio = build_portfolio(
            filtered_stocks=filtered_stocks,
            stock_data=stock_data,
            account_value=1000000,
            risk_per_trade=0.001
        )

        # Apply risk limits
        final_portfolio = apply_risk_limits(portfolio, max_positions=3)

        # Verify results
        assert not final_portfolio.empty
        assert len(final_portfolio) <= 3

        # Each position should risk approximately the same amount
        risks = final_portfolio['actual_risk'].tolist()
        risk_target = 1000  # 0.1% of $1M

        for risk in risks:
            # Risk should be close to target (within reasonable tolerance)
            assert abs(risk - risk_target) < risk_target * 0.5  # 50% tolerance


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_atr(self):
        """Test handling of zero ATR values."""
        with pytest.raises(ValueError):
            calculate_position_size(1000000, 0.001, 100, 0)

    def test_very_high_volatility(self):
        """Test very high volatility stocks."""
        # Stock with very high ATR should result in very small position
        result = calculate_position_size(
            account_value=1000000,
            risk_per_trade=0.001,
            stock_price=100,
            atr=50  # Very high ATR
        )

        # Should result in small position
        assert result['shares'] == 20  # $1000 / $50 ATR
        assert result['investment_amount'] == 2000
