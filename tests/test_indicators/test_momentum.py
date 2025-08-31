import numpy as np
import pandas as pd

from clenow_momentum.indicators.momentum import (
    calculate_exponential_regression_slope,
    calculate_momentum_for_universe,
    calculate_momentum_score,
    get_top_momentum_stocks,
)


class TestExponentialRegressionSlope:
    """Test exponential regression slope calculation."""

    def test_perfect_uptrend(self):
        """Test with perfect exponential uptrend."""
        # Create perfect exponential growth: price doubles every 30 days
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series([100 * (1.023**i) for i in range(90)], index=dates)

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should have positive slope and high R-squared
        assert slope > 0
        assert r_squared > 0.95  # Very strong trend

    def test_perfect_downtrend(self):
        """Test with perfect exponential downtrend."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series([100 * (0.99**i) for i in range(90)], index=dates)

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should have negative slope and high R-squared
        assert slope < 0
        assert r_squared > 0.95

    def test_sideways_trend(self):
        """Test with sideways/flat price action."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series([100] * 90, index=dates)  # Flat prices

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should have near-zero slope
        assert abs(slope) < 0.01  # Very small slope
        # For a perfectly flat line, R² is mathematically undefined (0/0)
        # In practice, it will be 0 or NaN since there's no variance to explain
        assert r_squared >= 0.0 and r_squared <= 1.0  # Valid R² range

    def test_noisy_data(self):
        """Test with noisy/random data."""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series(100 + np.random.normal(0, 5, 90), index=dates)

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should return valid numbers, low R-squared
        assert not np.isnan(slope)
        assert not np.isnan(r_squared)
        assert r_squared < 0.3  # Low R-squared for random data

    def test_insufficient_data(self):
        """Test with insufficient data points."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series([100] * 50, index=dates)

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should return NaN for insufficient data
        assert np.isnan(slope)
        assert np.isnan(r_squared)

    def test_with_nan_values(self):
        """Test handling of NaN values in price series."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series([100 + i for i in range(90)], index=dates)
        prices.iloc[20:30] = np.nan  # Insert some NaN values

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should handle NaNs and still calculate
        assert not np.isnan(slope)
        assert not np.isnan(r_squared)

    def test_too_many_nans(self):
        """Test when too many NaN values exist."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series([np.nan] * 90, index=dates)
        prices.iloc[:5] = [100, 101, 102, 103, 104]  # Only 5 valid values

        slope, r_squared = calculate_exponential_regression_slope(prices, period=90)

        # Should return NaN when insufficient clean data
        assert np.isnan(slope)
        assert np.isnan(r_squared)


class TestMomentumScore:
    """Test momentum score calculation."""

    def test_strong_momentum(self):
        """Test momentum score for strong trend."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series([100 * (1.01**i) for i in range(90)], index=dates)

        score = calculate_momentum_score(prices, period=90)

        # Should have positive momentum score
        assert score > 0
        assert not np.isnan(score)

    def test_weak_momentum(self):
        """Test momentum score for weak/noisy trend."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        prices = pd.Series(100 + np.random.normal(0, 10, 90), index=dates)

        score = calculate_momentum_score(prices, period=90)

        # Should have low momentum score due to low R-squared
        assert not np.isnan(score)
        assert abs(score) < 1.0  # Low score for random data

    def test_insufficient_data(self):
        """Test momentum score with insufficient data."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = pd.Series([100] * 30, index=dates)

        score = calculate_momentum_score(prices, period=90)

        assert np.isnan(score)


class TestCalculateMomentumForUniverse:
    """Test momentum calculation for stock universe."""

    def create_mock_stock_data(self):
        """Create mock multi-stock data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        # Create different trend types
        stocks = {
            "STRONG_UP": [100 * (1.01**i) for i in range(100)],  # Strong uptrend
            "WEAK_UP": [100 * (1.002**i) for i in range(100)],  # Weak uptrend
            "SIDEWAYS": [100 + np.sin(i / 10) for i in range(100)],  # Sideways
            "DOWN": [100 * (0.999**i) for i in range(100)],  # Downtrend
        }

        # Create DataFrame with yfinance-like structure
        data = pd.DataFrame(index=dates)
        for stock, prices in stocks.items():
            data[(stock, "Close")] = prices

        return data

    def test_momentum_calculation_universe(self):
        """Test momentum calculation for multiple stocks."""
        data = self.create_mock_stock_data()

        momentum_df = calculate_momentum_for_universe(data, period=90)

        # Should have results for all stocks
        assert len(momentum_df) == 4
        assert "ticker" in momentum_df.columns
        assert "momentum_score" in momentum_df.columns
        assert "annualized_slope" in momentum_df.columns
        assert "r_squared" in momentum_df.columns

        # Should be sorted by momentum score (descending)
        scores = momentum_df["momentum_score"].dropna()
        assert all(scores.iloc[i] >= scores.iloc[i + 1] for i in range(len(scores) - 1))

    def test_simple_column_structure(self):
        """Test with simple column structure (not multi-level)."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "AAPL": [100 * (1.01**i) for i in range(100)],
                "MSFT": [200 * (1.005**i) for i in range(100)],
            },
            index=dates,
        )

        momentum_df = calculate_momentum_for_universe(data, period=90)

        assert len(momentum_df) == 2
        assert "AAPL" in momentum_df["ticker"].values
        assert "MSFT" in momentum_df["ticker"].values


class TestGetTopMomentumStocks:
    """Test top momentum stock selection."""

    def create_mock_momentum_df(self):
        """Create mock momentum DataFrame."""
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "momentum_score": [0.8, 0.6, 0.4, 0.2, np.nan],
                "annualized_slope": [0.9, 0.7, 0.5, 0.3, np.nan],
                "r_squared": [0.9, 0.8, 0.7, 0.6, np.nan],
                "current_price": [150, 250, 2500, 3000, 200],
                "period_return_pct": [15, 10, 8, 5, np.nan],
            }
        )

    def test_top_20_percent_selection(self):
        """Test selecting top 20% of stocks."""
        momentum_df = self.create_mock_momentum_df()

        top_stocks = get_top_momentum_stocks(momentum_df, top_pct=0.20)

        # Should select 20% of valid stocks (1 out of 4 valid)
        assert len(top_stocks) == 1
        assert top_stocks.iloc[0]["ticker"] == "AAPL"  # Highest momentum
        assert "rank" in top_stocks.columns
        assert top_stocks.iloc[0]["rank"] == 1

    def test_top_50_percent_selection(self):
        """Test selecting top 50% of stocks."""
        momentum_df = self.create_mock_momentum_df()

        top_stocks = get_top_momentum_stocks(momentum_df, top_pct=0.50)

        # Should select 50% of valid stocks (2 out of 4 valid)
        assert len(top_stocks) == 2
        assert list(top_stocks["ticker"]) == ["AAPL", "MSFT"]
        assert list(top_stocks["rank"]) == [1, 2]

    def test_handles_nan_values(self):
        """Test that NaN momentum scores are properly filtered."""
        momentum_df = self.create_mock_momentum_df()

        top_stocks = get_top_momentum_stocks(momentum_df, top_pct=1.0)  # Select all

        # Should only include stocks with valid momentum scores
        assert len(top_stocks) == 4  # TSLA with NaN should be excluded
        assert "TSLA" not in top_stocks["ticker"].values

    def test_minimum_one_stock(self):
        """Test that at least one stock is always selected."""
        momentum_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "momentum_score": [0.1],
                "annualized_slope": [0.1],
                "r_squared": [0.1],
                "current_price": [150],
                "period_return_pct": [1],
            }
        )

        top_stocks = get_top_momentum_stocks(momentum_df, top_pct=0.01)  # Very small percentage

        # Should still select at least 1 stock
        assert len(top_stocks) == 1
