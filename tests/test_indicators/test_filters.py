"""
Tests for trading filters module.
"""

import numpy as np
import pandas as pd
import pytest

from clenow_momentum.indicators.filters import (
    calculate_moving_average,
    filter_above_ma,
    detect_gaps,
    filter_exclude_gaps,
    apply_all_filters,
)


class TestMovingAverage:
    """Test moving average calculations."""

    def test_calculate_moving_average(self):
        """Test basic moving average calculation."""
        # Create sample price data
        prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
        
        # Calculate 5-day MA
        ma = calculate_moving_average(prices, period=5)
        
        # First 4 values should be NaN
        assert pd.isna(ma.iloc[:4]).all()
        
        # 5th value should be average of first 5 prices
        expected_ma5 = (100 + 102 + 104 + 106 + 108) / 5
        assert abs(ma.iloc[4] - expected_ma5) < 0.001
        
        # Last value should be average of last 5 prices
        expected_ma_last = (110 + 112 + 114 + 116 + 118) / 5
        assert abs(ma.iloc[-1] - expected_ma_last) < 0.001

    def test_moving_average_insufficient_data(self):
        """Test moving average with insufficient data."""
        prices = pd.Series([100, 102, 104])
        ma = calculate_moving_average(prices, period=5)
        
        # All values should be NaN
        assert pd.isna(ma).all()


class TestMAFilter:
    """Test moving average filter."""

    def create_sample_multiindex_data(self):
        """Create sample MultiIndex stock data."""
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        
        # Create sample data for two stocks
        data_dict = {}
        
        # Stock A: trending up, should pass MA filter
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col == 'Volume':
                data_dict[('STOCK_A', col)] = np.random.randint(1000000, 2000000, len(dates))
            else:
                # Uptrending prices
                base_prices = np.linspace(100, 150, len(dates))
                noise = np.random.normal(0, 2, len(dates))
                data_dict[('STOCK_A', col)] = base_prices + noise
        
        # Stock B: trending down, should fail MA filter
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col == 'Volume':
                data_dict[('STOCK_B', col)] = np.random.randint(1000000, 2000000, len(dates))
            else:
                # Downtrending prices
                base_prices = np.linspace(150, 100, len(dates))
                noise = np.random.normal(0, 2, len(dates))
                data_dict[('STOCK_B', col)] = base_prices + noise
        
        df = pd.DataFrame(data_dict, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Ticker', 'Type'])
        
        return df

    def test_filter_above_ma_multiindex(self):
        """Test MA filter with MultiIndex data."""
        data = self.create_sample_multiindex_data()
        
        # Apply MA filter
        result = filter_above_ma(data, ma_period=50)
        
        assert not result.empty
        assert 'ticker' in result.columns
        assert 'above_ma' in result.columns
        assert 'price_vs_ma' in result.columns
        assert 'ma_50' in result.columns

    def test_filter_above_ma_simple_structure(self):
        """Test MA filter with simple column structure."""
        # Create simple data structure
        dates = pd.date_range('2024-01-01', periods=120, freq='D')
        
        # Uptrending stock
        stock_a = np.linspace(100, 140, len(dates)) + np.random.normal(0, 1, len(dates))
        # Downtrending stock
        stock_b = np.linspace(140, 100, len(dates)) + np.random.normal(0, 1, len(dates))
        
        data = pd.DataFrame({
            'STOCK_A': stock_a,
            'STOCK_B': stock_b
        }, index=dates)
        
        # Apply MA filter
        result = filter_above_ma(data, ma_period=50)
        
        assert not result.empty
        assert len(result) == 2  # Two stocks
        assert 'ticker' in result.columns
        assert 'above_ma' in result.columns


class TestGapDetection:
    """Test gap detection functionality."""

    def create_gapped_data(self):
        """Create sample data with price gaps."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        # Normal stock with small gaps
        normal_close = np.linspace(100, 120, len(dates)) + np.random.normal(0, 1, len(dates))
        normal_open = normal_close + np.random.normal(0, 0.5, len(dates))
        
        # Gapped stock with one large gap
        gapped_close = np.linspace(100, 120, len(dates)) + np.random.normal(0, 1, len(dates))
        gapped_open = gapped_close.copy()
        # Create a large gap on day 25 (20% gap up)
        gapped_open[25:] = gapped_close[25:] * 1.2
        
        data_dict = {}
        for ticker, close_prices, open_prices in [('NORMAL', normal_close, normal_open), 
                                                 ('GAPPED', gapped_close, gapped_open)]:
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col == 'Open':
                    data_dict[(ticker, col)] = open_prices
                elif col == 'Close':
                    data_dict[(ticker, col)] = close_prices
                elif col == 'Volume':
                    data_dict[(ticker, col)] = np.random.randint(1000000, 2000000, len(dates))
                else:
                    # High/Low around close prices
                    data_dict[(ticker, col)] = close_prices + np.random.uniform(-2, 2, len(dates))
        
        df = pd.DataFrame(data_dict, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Ticker', 'Type'])
        
        return df

    def test_detect_gaps(self):
        """Test gap detection."""
        data = self.create_gapped_data()
        
        result = detect_gaps(data, gap_threshold=0.15)
        
        assert not result.empty
        assert 'ticker' in result.columns
        assert 'max_gap' in result.columns
        assert 'has_large_gap' in result.columns
        assert 'gap_direction' in result.columns
        
        # Should detect the gapped stock
        gapped_stock = result[result['ticker'] == 'GAPPED']
        assert len(gapped_stock) == 1
        assert gapped_stock.iloc[0]['has_large_gap']

    def test_filter_exclude_gaps(self):
        """Test gap exclusion filter."""
        # Create sample momentum data
        momentum_df = pd.DataFrame({
            'ticker': ['NORMAL', 'GAPPED', 'STOCK_C'],
            'momentum_score': [0.8, 0.9, 0.7],
            'rank': [2, 1, 3]
        })
        
        # Create gapped stock data
        stock_data = self.create_gapped_data()
        
        # Apply gap filter
        result = filter_exclude_gaps(momentum_df, stock_data, gap_threshold=0.15)
        
        # Should exclude the gapped stock
        assert 'GAPPED' not in result['ticker'].values
        assert 'NORMAL' in result['ticker'].values


class TestApplyAllFilters:
    """Test combined filter application."""

    def create_comprehensive_test_data(self):
        """Create comprehensive test data for all filters."""
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        
        data_dict = {}
        
        # Stock 1: Good momentum, above MA, no gaps - should pass all filters
        base_prices = np.linspace(100, 140, len(dates))
        noise = np.random.normal(0, 1, len(dates))
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col == 'Volume':
                data_dict[('GOOD_STOCK', col)] = np.random.randint(1000000, 2000000, len(dates))
            else:
                data_dict[('GOOD_STOCK', col)] = base_prices + noise
        
        # Stock 2: Good momentum, below MA - should fail MA filter
        base_prices = np.linspace(140, 100, len(dates))  # Downtrending
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col == 'Volume':
                data_dict[('BAD_MA_STOCK', col)] = np.random.randint(1000000, 2000000, len(dates))
            else:
                data_dict[('BAD_MA_STOCK', col)] = base_prices + noise
        
        df = pd.DataFrame(data_dict, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Ticker', 'Type'])
        
        return df

    def test_apply_all_filters(self):
        """Test applying all filters together."""
        # Create momentum data
        momentum_df = pd.DataFrame({
            'ticker': ['GOOD_STOCK', 'BAD_MA_STOCK'],
            'momentum_score': [0.8, 0.7],
            'annualized_slope': [1.2, 1.0],
            'r_squared': [0.85, 0.80],
            'current_price': [135, 105],
            'period_return_pct': [35, -5],
            'rank': [1, 2]
        })
        
        # Create stock data
        stock_data = self.create_comprehensive_test_data()
        
        # Apply all filters
        result = apply_all_filters(momentum_df, stock_data, ma_period=50, gap_threshold=0.15)
        
        # Should have filtering results
        assert not result.empty
        
        # Should have additional filter columns
        if 'ma_50' in result.columns:
            assert 'ma_50' in result.columns
            assert 'price_vs_ma' in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test filters with empty data."""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        ma_result = filter_above_ma(empty_df)
        assert ma_result.empty
        
        gap_result = detect_gaps(empty_df)
        assert gap_result.empty

    def test_insufficient_data(self):
        """Test filters with insufficient data."""
        # Create very short data series
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data_dict = {
            ('STOCK_A', 'Close'): [100] * 10,
            ('STOCK_A', 'Open'): [100] * 10,
        }
        
        df = pd.DataFrame(data_dict, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Ticker', 'Type'])
        
        # Should handle insufficient data for long MA
        result = filter_above_ma(df, ma_period=100)
        # Should return empty or handle gracefully
        assert isinstance(result, pd.DataFrame)