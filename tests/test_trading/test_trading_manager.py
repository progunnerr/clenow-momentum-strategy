"""Tests for the current synchronous trading manager API."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clenow_momentum.data.sources.ibkr_client import AccountSummary
from clenow_momentum.strategy.rebalancing import OrderType, Portfolio, RebalancingOrder
from clenow_momentum.trading.trading_manager import TradingManager, TradingManagerError


class TestTradingManager:
    """Test TradingManager class."""

    def setup_method(self):
        self.mock_config = {
            "strategy_allocation": 100000,
            "portfolio_state_file": "test_portfolio.json",
            "cash_buffer": 0.02,
            "max_positions": 20,
            "max_position_pct": 0.05,
            "risk_per_trade": 0.001,
            "ibkr": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1,
                "account_id": "TEST123",
                "timeout": 30,
                "auto_reconnect": True,
            },
            "ibkr_host": "127.0.0.1",
            "ibkr_port": 7497,
            "ibkr_client_id": 1,
            "ibkr_account_id": "TEST123",
            "ibkr_timeout": 30,
            "ibkr_auto_reconnect": True,
        }

    def test_initialization(self):
        trading_manager = TradingManager(self.mock_config)

        assert trading_manager.config == self.mock_config
        assert trading_manager.is_connected is False
        assert trading_manager.trading_session_active is False

    @patch("clenow_momentum.trading.trading_manager.SyncTradingExecutionEngine")
    @patch("clenow_momentum.trading.trading_manager.PortfolioSynchronizer")
    @patch("clenow_momentum.trading.trading_manager.IBKRClient")
    def test_successful_initialization(
        self, mock_ibkr_client_cls, mock_portfolio_sync_cls, mock_execution_engine_cls
    ):
        mock_ibkr_client = MagicMock()
        mock_ibkr_client_cls.return_value = mock_ibkr_client

        trading_manager = TradingManager(self.mock_config)
        result = trading_manager.initialize()

        assert result is True
        assert trading_manager.is_connected is True
        mock_ibkr_client.connect.assert_called_once()
        mock_portfolio_sync_cls.assert_called_once_with(self.mock_config)
        mock_execution_engine_cls.assert_called_once_with(mock_ibkr_client)

    def test_initialization_with_missing_ibkr_config(self):
        invalid_config = self.mock_config.copy()
        invalid_config.pop("ibkr")
        trading_manager = TradingManager(invalid_config)

        with pytest.raises(TradingManagerError, match="IBKR configuration not found"):
            trading_manager.initialize()

    @patch("clenow_momentum.trading.trading_manager.IBKRClient")
    def test_initialization_connection_failure(self, mock_ibkr_client_cls):
        mock_ibkr_client = MagicMock()
        mock_ibkr_client.connect.side_effect = RuntimeError("Connection failed")
        mock_ibkr_client_cls.return_value = mock_ibkr_client

        trading_manager = TradingManager(self.mock_config)

        with pytest.raises(TradingManagerError, match="Initialization failed"):
            trading_manager.initialize()

    def test_not_initialized_operations(self):
        trading_manager = TradingManager(self.mock_config)

        with pytest.raises(TradingManagerError):
            trading_manager.execute_rebalancing([])

        with pytest.raises(TradingManagerError):
            trading_manager.sync_portfolio_only()

        with pytest.raises(TradingManagerError):
            trading_manager.get_account_status()

    def test_execute_rebalancing_dry_run(self):
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True

        portfolio = Portfolio(cash=10000.0)
        trading_manager.ibkr_client = MagicMock()
        trading_manager.ibkr_client.get_account_summary.return_value = AccountSummary(
            net_liquidation=100000.0,
            buying_power=200000.0,
            total_cash=10000.0,
            excess_liquidity=90000.0,
        )
        trading_manager.risk_control_system = MagicMock()
        trading_manager.risk_control_system.pre_trade_validation.return_value = []
        trading_manager.risk_control_system.can_proceed_with_trading.return_value = (
            True,
            "All checks passed",
        )
        trading_manager.risk_control_system.post_trade_monitoring.return_value = []
        trading_manager.execution_engine = MagicMock()
        trading_manager.execution_engine.execute_rebalancing_orders.return_value = (
            [],
            {"successful": 1, "failed": 0, "partially_filled": 0, "total_value_traded": 15000.0},
        )

        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=100,
                current_price=150.0,
                order_value=15000.0,
                reason="New position",
                priority=1,
            )
        ]

        result = trading_manager.execute_rebalancing(
            orders, dry_run=True, pre_synced_portfolio=portfolio
        )

        assert result["status"] == "completed_successfully"
        assert result["orders_executed"] == 1
        assert result["dry_run"] is True
        assert result["portfolio_synced"] is True
        trading_manager.execution_engine.execute_rebalancing_orders.assert_called_once()

    def test_execute_rebalancing_risk_blocked(self):
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True

        portfolio = Portfolio(cash=10000.0)
        trading_manager.ibkr_client = MagicMock()
        trading_manager.ibkr_client.get_account_summary.return_value = AccountSummary(
            net_liquidation=100000.0,
            buying_power=200000.0,
            total_cash=10000.0,
            excess_liquidity=90000.0,
        )
        trading_manager.risk_control_system = MagicMock()
        trading_manager.risk_control_system.pre_trade_validation.return_value = []
        trading_manager.risk_control_system.can_proceed_with_trading.return_value = (
            False,
            "Risk check failed",
        )

        result = trading_manager.execute_rebalancing([], pre_synced_portfolio=portfolio)

        assert result["status"] == "blocked_by_risk_controls"
        assert result["risk_checks_passed"] is False

    @patch("clenow_momentum.trading.trading_manager.save_ibkr_portfolio")
    @patch("clenow_momentum.trading.trading_manager.load_portfolio_state")
    def test_sync_portfolio_only(self, mock_load_portfolio, mock_save_portfolio):
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True
        trading_manager.ibkr_client = MagicMock()
        trading_manager.ibkr_client.get_positions.return_value = [
            SimpleNamespace(symbol="AAPL", quantity=100, avg_cost=150.0, market_value=16000.0),
            SimpleNamespace(symbol="MSFT", quantity=50, avg_cost=300.0, market_value=15500.0),
        ]
        trading_manager.ibkr_client.get_account_summary.return_value = AccountSummary(
            net_liquidation=41500.0,
            buying_power=100000.0,
            total_cash=10000.0,
            excess_liquidity=90000.0,
        )
        mock_load_portfolio.return_value = Portfolio()

        result = trading_manager.sync_portfolio_only()

        assert result.cash == 10000.0
        assert result.num_positions == 2
        assert result.positions["AAPL"].shares == 100
        assert result.positions["MSFT"].shares == 50
        mock_save_portfolio.assert_called_once()

    def test_get_account_status(self):
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True
        trading_manager.last_sync_time = datetime.now(UTC)
        trading_manager.ibkr_client = MagicMock()
        trading_manager.ibkr_client.get_account_summary.return_value = AccountSummary(
            net_liquidation=100000.0,
            buying_power=200000.0,
            total_cash=50000.0,
            excess_liquidity=150000.0,
        )
        trading_manager.risk_control_system = MagicMock()
        trading_manager.risk_control_system.get_risk_status.return_value = {
            "circuit_breaker": False
        }
        trading_manager.execution_engine = MagicMock()
        trading_manager.execution_engine.get_active_orders.return_value = []

        result = trading_manager.get_account_status()

        assert result["net_liquidation"] == 100000.0
        assert result["total_cash"] == 50000.0
        assert result["trading_manager"]["is_connected"] is True
        assert result["trading_manager"]["trading_mode"] == "paper"

    def test_emergency_stop(self):
        trading_manager = TradingManager(self.mock_config)
        trading_manager.trading_session_active = True
        trading_manager.execution_engine = MagicMock()
        trading_manager.execution_engine.get_active_orders.return_value = ["AAPL"]
        trading_manager.execution_engine.cancel_order.return_value = True

        trading_manager.emergency_stop("Test emergency")

        assert trading_manager.risk_control_system.circuit_breaker.is_tripped is True
        trading_manager.execution_engine.cancel_order.assert_called_once_with("AAPL")
        assert trading_manager.trading_session_active is False

    def test_disconnect(self):
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True
        trading_manager.trading_session_active = True
        mock_ibkr_client = MagicMock()
        trading_manager.ibkr_client = mock_ibkr_client

        trading_manager.disconnect()

        mock_ibkr_client.disconnect.assert_called_once()
        assert trading_manager.is_connected is False
        assert trading_manager.trading_session_active is False

    def test_get_status_summary(self):
        trading_manager = TradingManager(self.mock_config)

        summary = trading_manager.get_status_summary()
        assert "not connected" in summary

        trading_manager.is_connected = True
        trading_manager.trading_session_active = True
        trading_manager.last_sync_time = datetime.now(UTC)

        summary = trading_manager.get_status_summary()
        assert "Connected" in summary
        assert "PAPER" in summary
        assert "ACTIVE" in summary

    def test_context_manager(self):
        with (
            patch.object(TradingManager, "initialize") as mock_init,
            patch.object(TradingManager, "disconnect") as mock_disconnect,
        ):
            mock_init.return_value = True

            with TradingManager(self.mock_config) as manager:
                assert isinstance(manager, TradingManager)

            mock_init.assert_called_once()
            mock_disconnect.assert_called_once()
