"""Tests for order generation system."""

import pandas as pd
import pytest
from decimal import Decimal
from unittest.mock import Mock

# Import directly to avoid dependency issues
from clenow_momentum.trading.order_generation import OrderGenerationService, OrderGenerationResult
from clenow_momentum.interfaces import MarketDataProvider


class TestOrderGenerationService:
    """Test the complete order generation system."""

    @pytest.fixture
    def mock_market_data(self):
        """Mock market data provider."""
        mock_provider = Mock(spec=MarketDataProvider)
        mock_provider.get_current_price.side_effect = lambda ticker: {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0,
            "TSLA": 800.0,
            "AMZN": 3200.0
        }.get(ticker, 100.0)
        return mock_provider

    @pytest.fixture
    def service(self, mock_market_data):
        """Order generation service with mock data provider."""
        return OrderGenerationService(mock_market_data)

    def test_service_initialization(self, service):
        """Test that the service initializes properly."""
        assert service.market_data_provider is not None
        assert service.exit_strategy is not None
        assert service.adjust_strategy is not None
        assert service.entry_strategy is not None
        assert len(service.strategies) == 3

    def test_generate_orders_empty_portfolios(self, service):
        """Test order generation with empty current portfolio."""
        current_portfolio = pd.DataFrame()
        target_portfolio = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'shares': [10, 5],
            'investment': [1500.0, 14000.0],
            'current_price': [150.0, 2800.0]
        })
        
        result = service.generate_orders(
            current_portfolio=current_portfolio,
            target_portfolio=target_portfolio,
            available_cash=Decimal("20000")
        )
        
        # Should generate entry orders for new positions
        assert len(result.orders) == 2
        assert len(result.entry_orders) == 2
        assert len(result.exit_orders) == 0
        assert len(result.adjust_orders) == 0
        
        # Orders should be for AAPL and GOOGL
        tickers = [order.ticker for order in result.orders]
        assert "AAPL" in tickers
        assert "GOOGL" in tickers

    def test_generate_orders_with_exits(self, service):
        """Test order generation with positions to exit."""
        current_portfolio = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'shares': [10, 20],
            'current_price': [150.0, 300.0],
            'market_value': [1500.0, 6000.0]
        })
        
        # Target only has AAPL, so MSFT should be exited
        target_portfolio = pd.DataFrame({
            'ticker': ['AAPL'],
            'shares': [15],
            'investment': [2250.0],
            'current_price': [150.0]
        })
        
        result = service.generate_orders(
            current_portfolio=current_portfolio,
            target_portfolio=target_portfolio,
            available_cash=Decimal("10000")
        )
        
        # Should have exit order for MSFT and adjust order for AAPL
        assert len(result.exit_orders) == 1
        assert result.exit_orders[0].ticker == "MSFT"
        
        assert len(result.adjust_orders) == 1
        assert result.adjust_orders[0].ticker == "AAPL"

    def test_cash_constraints(self, service):
        """Test that cash constraints are respected."""
        current_portfolio = pd.DataFrame()
        target_portfolio = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'shares': [10, 5],
            'investment': [1500.0, 14000.0],
            'current_price': [150.0, 2800.0]
        })
        
        # Only have enough cash for AAPL
        result = service.generate_orders(
            current_portfolio=current_portfolio,
            target_portfolio=target_portfolio,
            available_cash=Decimal("2000")  # With buffer, only ~$1000 available
        )
        
        # Should only generate order for AAPL (cheaper position)
        assert len(result.orders) == 1
        assert result.orders[0].ticker == "AAPL"

    def test_input_validation(self, service):
        """Test input validation."""
        current_portfolio = pd.DataFrame()
        
        # Empty target portfolio should raise error
        with pytest.raises(ValueError, match="Target portfolio cannot be empty"):
            service.generate_orders(
                current_portfolio=current_portfolio,
                target_portfolio=pd.DataFrame(),
                available_cash=Decimal("10000")
            )
        
        # Negative cash should raise error
        with pytest.raises(ValueError, match="Available cash cannot be negative"):
            target_portfolio = pd.DataFrame({
                'ticker': ['AAPL'],
                'shares': [10],
                'investment': [1500.0],
                'current_price': [150.0]
            })
            
            service.generate_orders(
                current_portfolio=current_portfolio,
                target_portfolio=target_portfolio,
                available_cash=Decimal("-1000")
            )

    def test_order_generation_result_structure(self, service):
        """Test that OrderGenerationResult has expected structure."""
        current_portfolio = pd.DataFrame()
        target_portfolio = pd.DataFrame({
            'ticker': ['AAPL'],
            'shares': [10],
            'investment': [1500.0],
            'current_price': [150.0]
        })
        
        result = service.generate_orders(
            current_portfolio=current_portfolio,
            target_portfolio=target_portfolio,
            available_cash=Decimal("5000")
        )
        
        # Check all expected fields are present
        assert hasattr(result, 'orders')
        assert hasattr(result, 'exit_orders')
        assert hasattr(result, 'adjust_orders')
        assert hasattr(result, 'entry_orders')
        assert hasattr(result, 'total_order_value')
        assert hasattr(result, 'cash_after_orders')
        assert hasattr(result, 'skipped_positions')
        assert hasattr(result, 'generation_summary')
        
        # Check types
        assert isinstance(result.total_order_value, Decimal)
        assert isinstance(result.cash_after_orders, Decimal)
        assert isinstance(result.generation_summary, str)