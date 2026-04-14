"""Unit tests for the current synchronous IBKR client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clenow_momentum.data_sources import AccountSummary, IBKRClient, IBKRError, Position


class TestIBKRClient:
    """Test suite for IBKRClient."""

    @pytest.fixture
    def client(self):
        return IBKRClient(host="127.0.0.1", port=7497, client_id=99)

    @pytest.fixture
    def mock_ib(self):
        mock = MagicMock()
        mock.isConnected.return_value = True
        mock.disconnect.return_value = None
        mock.placeOrder.return_value = SimpleNamespace(order=SimpleNamespace(orderId=12345))
        mock.positions.return_value = []
        mock.accountValues.return_value = []
        mock.openTrades.return_value = []
        mock.sleep.return_value = None
        return mock

    def test_connect_success(self, client, mock_ib):
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            client.connect()

            assert client.is_connected()
            mock_ib.connect.assert_called_once_with("127.0.0.1", 7497, clientId=99)

    def test_connect_failure(self, client):
        with patch("clenow_momentum.data_sources.ibkr_client.IB") as mock_ib_class:
            mock_ib = mock_ib_class.return_value
            mock_ib.connect.side_effect = RuntimeError("boom")

            with pytest.raises(IBKRError, match="Connection failed"):
                client.connect()

            assert not client.is_connected()

    def test_disconnect(self, client, mock_ib):
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            client.connect()
            client.disconnect()

            mock_ib.disconnect.assert_called_once()
            assert not client.is_connected()

    def test_buy_order(self, client, mock_ib):
        mock_trade = SimpleNamespace(order=SimpleNamespace(orderId=12345))
        mock_ib.placeOrder.return_value = mock_trade

        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            client.connect()
            trade = client.buy("AAPL", 100)

            assert trade.order.orderId == 12345
            mock_ib.placeOrder.assert_called_once()
            mock_ib.sleep.assert_called_once_with(0.1)

    def test_sell_order(self, client, mock_ib):
        mock_trade = SimpleNamespace(order=SimpleNamespace(orderId=12346))
        mock_ib.placeOrder.return_value = mock_trade

        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            client.connect()
            trade = client.sell("AAPL", 50)

            assert trade.order.orderId == 12346
            mock_ib.placeOrder.assert_called_once()
            mock_ib.sleep.assert_called_once_with(0.1)

    def test_not_connected_error(self, client):
        with pytest.raises(IBKRError, match="Not connected"):
            client.buy("AAPL", 100)

    def test_get_positions(self, client, mock_ib):
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_pos = SimpleNamespace(
                contract=SimpleNamespace(symbol="AAPL"),
                position=100,
                avgCost=150.25,
                marketValue=15500.00,
                unrealizedPNL=475.00,
            )
            mock_ib.positions.return_value = [mock_pos]

            client.connect()
            positions = client.get_positions()

            assert len(positions) == 1
            assert positions[0] == Position(
                symbol="AAPL",
                quantity=100.0,
                avg_cost=150.25,
                market_value=15500.0,
                unrealized_pnl=475.0,
            )

    def test_get_account_summary(self, client, mock_ib):
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_items = [
                SimpleNamespace(tag="NetLiquidation", value="100000.00"),
                SimpleNamespace(tag="BuyingPower", value="50000.00"),
                SimpleNamespace(tag="TotalCashValue", value="25000.00"),
                SimpleNamespace(tag="ExcessLiquidity", value="45000.00"),
            ]
            mock_ib.accountValues.return_value = mock_items

            client.connect()
            summary = client.get_account_summary()

            assert summary == AccountSummary(
                net_liquidation=100000.0,
                buying_power=50000.0,
                total_cash=25000.0,
                excess_liquidity=45000.0,
            )

    def test_get_open_orders(self, client, mock_ib):
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            mock_trades = [MagicMock(), MagicMock()]
            mock_ib.openTrades.return_value = mock_trades

            client.connect()
            orders = client.get_open_orders()

            assert len(orders) == 2
            mock_ib.openTrades.assert_called_once()

    def test_context_manager(self, mock_ib):
        with patch("clenow_momentum.data_sources.ibkr_client.IB", return_value=mock_ib):
            with IBKRClient() as client:
                assert client.is_connected()
                mock_ib.connect.assert_called_once()

            mock_ib.disconnect.assert_called_once()

    def test_repr(self, client):
        repr_str = repr(client)
        assert "IBKRClient" in repr_str
        assert "127.0.0.1" in repr_str
        assert "7497" in repr_str
        assert "disconnected" in repr_str

    def test_reconnection_creates_new_ib_instance(self, client):
        with patch("clenow_momentum.data_sources.ibkr_client.IB") as mock_ib_class:
            mock_ib1 = MagicMock()
            mock_ib2 = MagicMock()
            mock_ib1.isConnected.return_value = True
            mock_ib2.isConnected.return_value = True
            mock_ib_class.side_effect = [mock_ib1, mock_ib2]

            client.connect()
            assert client._ib is mock_ib1

            client.disconnect()

            client.connect()
            assert client._ib is mock_ib2
            assert client._ib is not mock_ib1
