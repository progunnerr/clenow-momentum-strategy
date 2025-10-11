"""Direct tests for order generation without full module imports."""

import sys
import os
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock

# Add src to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Direct imports
from clenow_momentum.trading.order_generation import OrderGenerationService, OrderGenerationResult
from clenow_momentum.trading.order_strategies.base import OrderContext
from clenow_momentum.trading.order_strategies.exit_strategy import ExitStrategy
from clenow_momentum.trading.order_strategies.entry_strategy import EntryStrategy
from clenow_momentum.interfaces import MarketDataSource


def test_order_context_creation():
    """Test that OrderContext can be created and provides expected interface."""
    mock_provider = Mock(spec=MarketDataSource)
    mock_provider.get_current_price.return_value = 150.0
    
    current_portfolio = pd.DataFrame({
        'ticker': ['AAPL'],
        'shares': [10],
        'current_price': [150.0],
        'market_value': [1500.0]
    })
    
    target_portfolio = pd.DataFrame({
        'ticker': ['AAPL'],
        'shares': [15],
        'investment': [2250.0],
        'current_price': [150.0]
    })
    
    context = OrderContext(
        current_portfolio=current_portfolio,
        target_portfolio=target_portfolio,
        available_cash=Decimal("5000"),
        market_data_provider=mock_provider
    )
    
    # Test basic functionality
    assert context.get_current_tickers() == {'AAPL'}
    assert context.get_target_tickers() == {'AAPL'}
    assert context.get_current_price('AAPL') == 150.0
    assert context.available_cash == Decimal("5000")
    
    print("✓ OrderContext creation and basic functionality working")


def test_exit_strategy_basic():
    """Test basic exit strategy functionality."""
    mock_provider = Mock(spec=MarketDataSource)
    mock_provider.get_current_price.return_value = 150.0
    
    # Current portfolio has AAPL, target portfolio is empty -> should exit AAPL
    current_portfolio = pd.DataFrame({
        'ticker': ['AAPL'],
        'shares': [10],
        'current_price': [150.0],
        'market_value': [1500.0]
    })
    
    target_portfolio = pd.DataFrame({
        'ticker': [],
        'shares': [],
        'investment': [],
        'current_price': []
    })
    
    context = OrderContext(
        current_portfolio=current_portfolio,
        target_portfolio=target_portfolio,
        available_cash=Decimal("5000"),
        market_data_provider=mock_provider
    )
    
    strategy = ExitStrategy()
    orders = strategy.generate_orders(context)
    
    assert len(orders) == 1
    assert orders[0].ticker == 'AAPL'
    assert orders[0].shares == 10  # Sell all shares
    
    print("✓ Exit strategy basic functionality working")


def test_entry_strategy_basic():
    """Test basic entry strategy functionality."""
    mock_provider = Mock(spec=MarketDataSource)
    mock_provider.get_current_price.return_value = 150.0
    
    # Empty current portfolio, target has AAPL -> should enter AAPL
    current_portfolio = pd.DataFrame({
        'ticker': [],
        'shares': [],
        'current_price': [],
        'market_value': []
    })
    
    target_portfolio = pd.DataFrame({
        'ticker': ['AAPL'],
        'shares': [10],
        'investment': [1500.0],
        'current_price': [150.0]
    })
    
    context = OrderContext(
        current_portfolio=current_portfolio,
        target_portfolio=target_portfolio,
        available_cash=Decimal("5000"),
        market_data_provider=mock_provider
    )
    
    strategy = EntryStrategy()
    orders = strategy.generate_orders(context)
    
    assert len(orders) == 1
    assert orders[0].ticker == 'AAPL'
    assert orders[0].shares == 10  # Buy target shares
    
    print("✓ Entry strategy basic functionality working")


def test_order_generation_service():
    """Test the complete order generation service."""
    mock_provider = Mock(spec=MarketDataSource)
    mock_provider.get_current_price.side_effect = lambda ticker: {
        'AAPL': 150.0,
        'GOOGL': 2800.0
    }.get(ticker, 100.0)
    
    service = OrderGenerationService(mock_provider)
    
    # Test with entry orders (empty current, target has positions)
    current_portfolio = pd.DataFrame({
        'ticker': [],
        'shares': [],
        'current_price': [],
        'market_value': []
    })
    
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
    
    # Should have created OrderGenerationResult
    assert isinstance(result, OrderGenerationResult)
    assert len(result.orders) == 2
    assert len(result.entry_orders) == 2
    assert len(result.exit_orders) == 0
    assert isinstance(result.total_order_value, Decimal)
    assert isinstance(result.generation_summary, str)
    
    print("✓ OrderGenerationService basic functionality working")


if __name__ == "__main__":
    """Run tests directly."""
    try:
        test_order_context_creation()
        test_exit_strategy_basic()
        test_entry_strategy_basic()
        test_order_generation_service()
        
        print("\n✅ All order generation system tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)