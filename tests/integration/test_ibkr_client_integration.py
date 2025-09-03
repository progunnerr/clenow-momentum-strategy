"""
Integration tests for IBKRClient with real TWS connection.

These tests require TWS or IB Gateway running on port 7497 (paper trading).
Run with: uv run pytest tests/integration/ -v
"""

import os
import time

import pytest
from loguru import logger

from clenow_momentum.data_sources import IBKRClient, IBKRError


# Skip all tests in this file if TWS is not available
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "").lower() == "true",
    reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=false to run)",
)


class TestIBKRClientIntegration:
    """Integration tests requiring real TWS connection."""

    @pytest.fixture
    def client(self):
        """Create and connect client for tests."""
        client = IBKRClient(port=7497, client_id=99, account="DU9041643")
        client.connect()
        yield client
        client.disconnect()

    def test_connection_lifecycle(self):
        """Test connect, reconnect, and disconnect."""
        client = IBKRClient(port=7497, client_id=99)
        
        # Connect
        client.connect()
        assert client.is_connected()
        
        # Disconnect
        client.disconnect()
        assert not client.is_connected()
        
        # Reconnect (should create new IB instance)
        client.connect()
        assert client.is_connected()
        
        # Clean up
        client.disconnect()

    def test_context_manager(self):
        """Test context manager."""
        with IBKRClient(port=7497, client_id=99, account="DU9041643") as client:
            assert client.is_connected()
            
            # Should be able to get account info
            summary = client.get_account_summary()
            assert summary.net_liquidation > 0
        
        # Should be disconnected after context
        assert not client.is_connected()

    def test_account_data(self, client):
        """Test retrieving account information."""
        # Account summary
        summary = client.get_account_summary()
        assert summary.net_liquidation > 0
        assert summary.total_cash >= 0
        assert summary.buying_power >= 0
        
        logger.info(f"Account: Net Liq=${summary.net_liquidation:,.2f}, Cash=${summary.total_cash:,.2f}")

    def test_positions(self, client):
        """Test position retrieval."""
        positions = client.get_positions()
        
        # Should return a list (empty or with positions)
        assert isinstance(positions, list)
        
        # If there are positions, verify structure
        if positions:
            pos = positions[0]
            assert hasattr(pos, "symbol")
            assert hasattr(pos, "quantity")
            assert hasattr(pos, "avg_cost")
            assert hasattr(pos, "market_value")
            assert hasattr(pos, "unrealized_pnl")
            
            logger.info(f"Sample position: {pos.symbol} x {pos.quantity}")

    def test_open_orders(self, client):
        """Test retrieving open orders."""
        orders = client.get_open_orders()
        
        # Should return a list (empty or with orders)
        assert isinstance(orders, list)
        
        # If there are orders, verify they have expected attributes
        if orders:
            order = orders[0]
            assert hasattr(order, "order")
            assert hasattr(order.order, "orderId")
            logger.info(f"Found {len(orders)} open orders")

    def test_multiple_reconnections(self):
        """Test that multiple reconnection cycles work."""
        client = IBKRClient(port=7497, client_id=99)
        
        for i in range(3):
            # Connect
            client.connect()
            assert client.is_connected()
            
            # Verify we can do operations
            positions = client.get_positions()
            assert isinstance(positions, list)
            
            # Disconnect
            client.disconnect()
            assert not client.is_connected()
            
            # Small delay between cycles
            time.sleep(0.5)

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        client = IBKRClient(port=7497, client_id=99)
        
        # Should raise error when not connected
        with pytest.raises(IBKRError, match="Not connected"):
            client.get_positions()
        
        # Connect for other tests
        client.connect()
        
        # Should work now
        positions = client.get_positions()
        assert isinstance(positions, list)
        
        client.disconnect()

    @pytest.mark.skip(reason="Only run if you want to test order placement")
    def test_order_placement(self, client):
        """
        Test actual order placement (PAPER TRADING ONLY).
        
        Skip by default to avoid accidental orders.
        To run: pytest tests/integration/ -k test_order_placement -v -s
        """
        # Place a small test order
        symbol = "AAPL"
        quantity = 1
        
        # Place buy order
        trade = client.buy(symbol, quantity)
        assert trade.order.orderId > 0
        
        logger.warning(f"Order test placed order {trade.order.orderId} - check your paper account")


def test_connection_timeout():
    """Test connection timeout handling."""
    # Use wrong port to force timeout
    client = IBKRClient(port=9999)
    
    with pytest.raises(IBKRError, match="Connection failed"):
        client.connect()
    
    assert not client.is_connected()