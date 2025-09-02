"""
Tests for IBKR connector module.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ib_async import MarketOrder

from clenow_momentum.trading.ibkr_connector import (
    ConnectionStatus,
    IBKRConnectionError,
    IBKRConnector,
    IBKROrderError,
)


class TestIBKRConnector:
    """Test IBKRConnector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.connector = IBKRConnector(
            host="127.0.0.1", port=7497, client_id=1, account_id="DU9041643", timeout=30
        )

    def test_connector_initialization(self):
        """Test connector initialization."""
        assert self.connector.host == "127.0.0.1"
        assert self.connector.port == 7497
        assert self.connector.client_id == 1
        assert self.connector.account_id == "DU9041643"
        assert self.connector.timeout == 30
        assert self.connector.status == ConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.ibkr_connector.IB")
    async def test_successful_connection(self, mock_ib_class):
        """Test successful connection to IBKR."""
        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ["TEST123", "TEST456"]
        mock_ib.reqAccountUpdates = MagicMock()
        mock_ib_class.return_value = mock_ib
        self.connector.ib = mock_ib

        result = await self.connector.connect()

        assert result is True
        assert self.connector.status == ConnectionStatus.CONNECTED
        mock_ib.connectAsync.assert_called_once_with(
            host="127.0.0.1", port=7497, clientId=1, timeout=30
        )

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.ibkr_connector.IB")
    async def test_connection_timeout(self, mock_ib_class):
        """Test connection timeout."""
        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock(side_effect=TimeoutError())
        mock_ib_class.return_value = mock_ib
        self.connector.ib = mock_ib

        result = await self.connector.connect()

        assert result is False
        assert self.connector.status == ConnectionStatus.FAILED

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.ibkr_connector.IB")
    async def test_connection_failure(self, mock_ib_class):
        """Test connection failure."""
        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock(side_effect=Exception("Connection failed"))
        mock_ib_class.return_value = mock_ib
        self.connector.ib = mock_ib

        result = await self.connector.connect()

        assert result is False
        assert self.connector.status == ConnectionStatus.FAILED

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        self.connector.ib = mock_ib
        self.connector.status = ConnectionStatus.CONNECTED

        await self.connector.disconnect()

        mock_ib.disconnect.assert_called_once()
        assert self.connector.status == ConnectionStatus.DISCONNECTED

    def test_is_connected(self):
        """Test connection status check."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        self.connector.ib = mock_ib
        self.connector.status = ConnectionStatus.CONNECTED

        assert self.connector.is_connected() is True

        mock_ib.isConnected.return_value = False
        assert self.connector.is_connected() is False

        self.connector.status = ConnectionStatus.FAILED
        mock_ib.isConnected.return_value = True
        assert self.connector.is_connected() is False

    @pytest.mark.asyncio
    async def test_get_account_info_not_connected(self):
        """Test get account info when not connected."""
        self.connector.status = ConnectionStatus.DISCONNECTED

        with pytest.raises(IBKRConnectionError):
            await self.connector.get_account_info()

    @pytest.mark.asyncio
    async def test_get_account_info_success(self):
        """Test successful account info retrieval."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        self.connector.ib = mock_ib
        self.connector.status = ConnectionStatus.CONNECTED

        # Mock account values
        mock_account_values = [
            MagicMock(tag="TotalCashValue", value="50000.0"),
            MagicMock(tag="NetLiquidation", value="100000.0"),
            MagicMock(tag="BuyingPower", value="200000.0"),
        ]

        # Mock positions
        mock_positions = [
            MagicMock(
                contract=MagicMock(secType="STK", symbol="AAPL"),
                position=100,
                avgCost=150.0,
                marketValue=16000.0,
                unrealizedPNL=1000.0,
            ),
            MagicMock(
                contract=MagicMock(secType="STK", symbol="MSFT"),
                position=50,
                avgCost=300.0,
                marketValue=16000.0,
                unrealizedPNL=1000.0,
            ),
        ]

        mock_ib.accountValues.return_value = mock_account_values
        mock_ib.positions.return_value = mock_positions

        result = await self.connector.get_account_info()

        assert result["account_id"] == "DU9041643"
        assert result["total_cash"] == 50000.0
        assert result["net_liquidation"] == 100000.0
        assert result["buying_power"] == 200000.0
        assert len(result["positions"]) == 2
        assert result["positions"]["AAPL"]["shares"] == 100
        assert result["positions"]["MSFT"]["shares"] == 50

    @pytest.mark.asyncio
    async def test_create_market_order(self):
        """Test creating market order."""
        order = await self.connector.create_stock_order("AAPL", 100, "MKT")

        assert order.action == "BUY"
        assert order.totalQuantity == 100
        assert order.orderType == "MKT"

    @pytest.mark.asyncio
    async def test_create_limit_order(self):
        """Test creating limit order."""
        order = await self.connector.create_stock_order("AAPL", -50, "LMT", 150.0)

        assert order.action == "SELL"
        assert order.totalQuantity == 50
        assert order.orderType == "LMT"
        assert order.lmtPrice == 150.0

    @pytest.mark.asyncio
    async def test_create_limit_order_no_price(self):
        """Test creating limit order without price."""
        with pytest.raises(IBKROrderError):
            await self.connector.create_stock_order("AAPL", 100, "LMT")

    @pytest.mark.asyncio
    async def test_submit_order_not_connected(self):
        """Test submitting order when not connected."""
        self.connector.status = ConnectionStatus.DISCONNECTED
        order = MarketOrder("BUY", 100)

        with pytest.raises(IBKRConnectionError):
            await self.connector.submit_order("AAPL", order)

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.ibkr_connector.Stock")
    async def test_submit_order_success(self, mock_stock):
        """Test successful order submission."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        self.connector.ib = mock_ib
        self.connector.status = ConnectionStatus.CONNECTED

        mock_contract = MagicMock()
        mock_stock.return_value = mock_contract

        mock_trade = MagicMock()
        mock_trade.order.orderId = 12345
        mock_ib.placeOrder.return_value = mock_trade

        order = MarketOrder("BUY", 100)
        result = await self.connector.submit_order("AAPL", order)

        assert result == mock_trade
        mock_stock.assert_called_once_with("AAPL", "SMART", "USD")
        mock_ib.placeOrder.assert_called_once_with(mock_contract, order)

    @pytest.mark.asyncio
    async def test_cancel_order_not_connected(self):
        """Test canceling order when not connected."""
        self.connector.status = ConnectionStatus.DISCONNECTED
        mock_trade = MagicMock()

        with pytest.raises(IBKRConnectionError):
            await self.connector.cancel_order(mock_trade)

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        """Test successful order cancellation."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        self.connector.ib = mock_ib
        self.connector.status = ConnectionStatus.CONNECTED

        mock_trade = MagicMock()
        mock_trade.order.orderId = 12345

        result = await self.connector.cancel_order(mock_trade)

        assert result is True
        mock_ib.cancelOrder.assert_called_once_with(mock_trade.order)

    @pytest.mark.asyncio
    async def test_get_open_orders_not_connected(self):
        """Test getting open orders when not connected."""
        self.connector.status = ConnectionStatus.DISCONNECTED

        with pytest.raises(IBKRConnectionError):
            await self.connector.get_open_orders()

    @pytest.mark.asyncio
    async def test_get_open_orders_success(self):
        """Test successful open orders retrieval."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        self.connector.ib = mock_ib
        self.connector.status = ConnectionStatus.CONNECTED

        mock_trades = [MagicMock(), MagicMock()]
        mock_ib.openTrades.return_value = mock_trades

        result = await self.connector.get_open_orders()

        assert result == mock_trades

    def test_event_handlers(self):
        """Test event handler setup."""
        mock_ib = MagicMock()
        mock_disconnected = MagicMock()
        mock_error = MagicMock()
        mock_order_status = MagicMock()

        mock_ib.disconnectedEvent = mock_disconnected
        mock_ib.errorEvent = mock_error
        mock_ib.orderStatusEvent = mock_order_status

        self.connector.ib = mock_ib

        self.connector._setup_event_handlers()

        # Verify event handlers were added
        mock_disconnected.__iadd__.assert_called_once()
        mock_error.__iadd__.assert_called_once()
        mock_order_status.__iadd__.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_disconnected_event(self):
        """Test disconnected event handler."""
        self.connector.status = ConnectionStatus.CONNECTED
        callback_called = False

        def disconnect_callback():
            nonlocal callback_called
            callback_called = True

        self.connector.on_disconnected = disconnect_callback

        # Run in async context to handle the reconnect task
        self.connector._on_disconnected()

        assert self.connector.status == ConnectionStatus.DISCONNECTED
        assert callback_called

        # Cancel the reconnect task if it was created
        if self.connector._reconnect_task:
            self.connector._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.connector._reconnect_task

    def test_on_error_event(self):
        """Test error event handler."""
        callback_called = False
        error_info = None

        def error_callback(error):
            nonlocal callback_called, error_info
            callback_called = True
            error_info = error

        self.connector.on_error = error_callback

        self.connector._on_error(123, 502, "Connection lost", None)

        assert callback_called
        assert error_info["code"] == 502
        assert error_info["message"] == "Connection lost"
        assert self.connector.status == ConnectionStatus.FAILED

    def test_on_order_status_event(self):
        """Test order status event handler."""
        callback_called = False
        trade_info = None

        def order_callback(trade):
            nonlocal callback_called, trade_info
            callback_called = True
            trade_info = trade

        self.connector.on_order_status = order_callback

        mock_trade = MagicMock()
        mock_trade.order.orderId = 12345
        mock_trade.orderStatus.status = "Filled"

        self.connector._on_order_status(mock_trade)

        assert callback_called
        assert trade_info == mock_trade

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with (
            patch.object(self.connector, "connect") as mock_connect,
            patch.object(self.connector, "disconnect") as mock_disconnect,
        ):
            mock_connect.return_value = True

            async with self.connector as conn:
                assert conn == self.connector

            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()
