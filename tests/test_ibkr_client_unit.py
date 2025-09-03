"""
Unit tests for IBKRClient.

Tests the clean IBKR client implementation with proper mocking.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from ib_async import Trade

from clenow_momentum.data_sources import AccountSummary, IBKRClient, IBKRError, Position


class TestIBKRClient:
    """Test suite for IBKRClient."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return IBKRClient(host="127.0.0.1", port=7497, client_id=99)

    @pytest.fixture
    def mock_ib(self):
        """Create a mock IB instance."""
        mock = MagicMock()
        mock.isConnected.return_value = True
        mock.connectAsync = AsyncMock()
        mock.disconnect = Mock()
        mock.managedAccounts.return_value = ["DU123456"]
        mock.placeOrder = Mock()
        mock.positions.return_value = []
        mock.accountSummary.return_value = []
        mock.openTrades.return_value = []
        mock.cancelOrder = Mock()
        mock.waitOnUpdate = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_connect_success(self, client, mock_ib):
        """Test successful connection."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            await client.connect()

            assert client.is_connected()
            assert client._account == "DU123456"
            mock_ib.connectAsync.assert_called_once_with(
                host="127.0.0.1",
                port=7497,
                clientId=99,
            )

    @pytest.mark.asyncio
    async def test_connect_timeout(self, client):
        """Test connection timeout."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB") as mock_ib_class:
            mock_ib = mock_ib_class.return_value
            mock_ib.connectAsync = AsyncMock(side_effect=asyncio.TimeoutError())

            with pytest.raises(IBKRError, match="Connection timeout"):
                await client.connect()

            assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_ib):
        """Test disconnection."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            await client.connect()
            await client.disconnect()

            mock_ib.disconnect.assert_called_once()
            assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_buy_order(self, client, mock_ib):
        """Test placing a buy order."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            # Setup mock trade
            mock_trade = Mock(spec=Trade)
            mock_trade.order.orderId = 12345
            mock_ib.placeOrder.return_value = mock_trade

            await client.connect()
            trade = await client.buy("AAPL", 100)

            assert trade.order.orderId == 12345
            mock_ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_sell_order(self, client, mock_ib):
        """Test placing a sell order."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_trade = Mock(spec=Trade)
            mock_trade.order.orderId = 12346
            mock_ib.placeOrder.return_value = mock_trade

            await client.connect()
            trade = await client.sell("AAPL", 50)

            assert trade.order.orderId == 12346
            mock_ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_buy_limit_order(self, client, mock_ib):
        """Test placing a limit buy order."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_trade = Mock(spec=Trade)
            mock_trade.order.orderId = 12347
            mock_ib.placeOrder.return_value = mock_trade

            await client.connect()
            trade = await client.buy_limit("AAPL", 100, 150.50)

            assert trade.order.orderId == 12347
            mock_ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_quantity(self, client, mock_ib):
        """Test order with invalid quantity."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            await client.connect()

            with pytest.raises(ValueError, match="Quantity must be positive"):
                await client.buy("AAPL", -10)

    @pytest.mark.asyncio
    async def test_not_connected_error(self, client):
        """Test operations when not connected."""
        with pytest.raises(IBKRError, match="Not connected"):
            await client.buy("AAPL", 100)

    @pytest.mark.asyncio
    async def test_get_positions(self, client, mock_ib):
        """Test getting positions."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            # Setup mock positions
            mock_pos = Mock()
            mock_pos.contract.symbol = "AAPL"
            mock_pos.position = 100
            mock_pos.avgCost = 150.25
            mock_pos.marketValue = 15500.00
            mock_pos.unrealizedPNL = 475.00
            mock_ib.positions.return_value = [mock_pos]

            await client.connect()
            positions = await client.get_positions()

            assert len(positions) == 1
            assert positions[0].symbol == "AAPL"
            assert positions[0].quantity == 100
            assert positions[0].avg_cost == 150.25

    @pytest.mark.asyncio
    async def test_get_account_summary(self, client, mock_ib):
        """Test getting account summary."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            # Setup mock account summary
            mock_items = [
                Mock(tag="NetLiquidation", value="100000.00"),
                Mock(tag="BuyingPower", value="50000.00"),
                Mock(tag="TotalCashValue", value="25000.00"),
                Mock(tag="ExcessLiquidity", value="45000.00"),
            ]
            mock_ib.accountSummary.return_value = mock_items

            await client.connect()
            summary = await client.get_account_summary()

            assert summary.net_liquidation == 100000.00
            assert summary.buying_power == 50000.00
            assert summary.total_cash == 25000.00
            assert summary.excess_liquidity == 45000.00

    @pytest.mark.asyncio
    async def test_get_position_single(self, client, mock_ib):
        """Test getting a specific position."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_pos = Mock()
            mock_pos.contract.symbol = "AAPL"
            mock_pos.position = 100
            mock_pos.avgCost = 150.25
            mock_pos.marketValue = 15500.00
            mock_pos.unrealizedPNL = 475.00
            mock_ib.positions.return_value = [mock_pos]

            await client.connect()
            position = await client.get_position("AAPL")

            assert position is not None
            assert position.symbol == "AAPL"
            assert position.quantity == 100

    @pytest.mark.asyncio
    async def test_get_position_not_found(self, client, mock_ib):
        """Test getting a position that doesn't exist."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_ib.positions.return_value = []

            await client.connect()
            position = await client.get_position("TSLA")

            assert position is None

    @pytest.mark.asyncio
    async def test_cancel_order(self, client, mock_ib):
        """Test cancelling an order."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_trade = Mock(spec=Trade)
            mock_trade.order.orderId = 12348

            await client.connect()
            await client.cancel_order(mock_trade)

            mock_ib.cancelOrder.assert_called_once_with(mock_trade.order)

    @pytest.mark.asyncio
    async def test_get_open_orders(self, client, mock_ib):
        """Test getting open orders."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_trades = [Mock(spec=Trade), Mock(spec=Trade)]
            mock_ib.openTrades.return_value = mock_trades

            await client.connect()
            orders = await client.get_open_orders()

            assert len(orders) == 2
            mock_ib.openTrades.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_order(self, client, mock_ib):
        """Test waiting for an order to complete."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_trade = Mock(spec=Trade)
            mock_ib.waitOnUpdate.return_value = mock_trade

            await client.connect()
            result = await client.wait_for_order(mock_trade, timeout=30)

            assert result == mock_trade
            mock_ib.waitOnUpdate.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_ib):
        """Test using client as context manager."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            async with IBKRClient() as client:
                assert client.is_connected()
                mock_ib.connectAsync.assert_called_once()

            # Should be disconnected after context exit
            mock_ib.disconnect.assert_called_once()

    def test_repr(self, client):
        """Test string representation."""
        repr_str = repr(client)
        assert "IBKRClient" in repr_str
        assert "127.0.0.1" in repr_str
        assert "7497" in repr_str
        assert "disconnected" in repr_str

    @pytest.mark.asyncio
    async def test_reconnection_creates_new_ib_instance(self, client):
        """Test that reconnection creates a new IB instance."""
        with patch("clenow_momentum.data_sources.ibkr_client.IB") as mock_ib_class:
            mock_ib1 = MagicMock()
            mock_ib2 = MagicMock()
            mock_ib_class.side_effect = [mock_ib1, mock_ib2]

            mock_ib1.isConnected.return_value = True
            mock_ib1.connectAsync = AsyncMock()
            mock_ib1.managedAccounts.return_value = ["DU123456"]

            mock_ib2.isConnected.return_value = True
            mock_ib2.connectAsync = AsyncMock()
            mock_ib2.managedAccounts.return_value = ["DU123456"]

            # First connection
            await client.connect()
            assert client._ib is mock_ib1

            # Disconnect
            await client.disconnect()

            # Reconnect - should create new IB instance
            await client.connect()
            assert client._ib is mock_ib2
            assert client._ib is not mock_ib1