"""Integration-style tests for the current trading manager workflow."""

from unittest.mock import MagicMock, patch

import pytest

from clenow_momentum.data.sources.ibkr_client import AccountSummary
from clenow_momentum.strategy.rebalancing import OrderType, Portfolio, RebalancingOrder
from clenow_momentum.trading import validate_ibkr_config
from clenow_momentum.trading.trading_manager import TradingManager, TradingManagerError


class TestIBKRIntegration:
    """Test complete trading-manager workflows using mocked broker dependencies."""

    def setup_method(self):
        self.test_config = {
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

    def test_risk_controls_integration(self):
        risky_orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=1000,
                current_price=150.0,
                order_value=150000.0,
                reason="Dangerous position",
            )
        ]
        trading_manager = TradingManager(self.test_config)
        trading_manager.is_connected = True
        trading_manager.ibkr_client = MagicMock()
        trading_manager.ibkr_client.get_account_summary.return_value = AccountSummary(
            net_liquidation=100000.0,
            buying_power=1000.0,
            total_cash=1000.0,
            excess_liquidity=1000.0,
        )

        portfolio = Portfolio(cash=1000.0)
        execution_results = trading_manager.execute_rebalancing(
            risky_orders, pre_synced_portfolio=portfolio
        )

        assert execution_results["status"] == "blocked_by_risk_controls"
        assert execution_results["risk_checks_passed"] is False
        assert "risk" in execution_results["risk_reason"].lower()

    @patch("clenow_momentum.trading.trading_manager.SyncTradingExecutionEngine")
    @patch("clenow_momentum.trading.trading_manager.PortfolioSynchronizer")
    @patch("clenow_momentum.trading.trading_manager.IBKRClient")
    def test_error_handling_and_recovery(
        self, mock_ibkr_client_cls, mock_portfolio_sync_cls, mock_execution_engine_cls
    ):
        mock_ibkr_client = MagicMock()
        mock_ibkr_client.connect.side_effect = RuntimeError("Connection failure")
        mock_ibkr_client_cls.return_value = mock_ibkr_client

        trading_manager = TradingManager(self.test_config)

        with pytest.raises(TradingManagerError):
            trading_manager.initialize()

        mock_ibkr_client.connect.side_effect = None
        assert trading_manager.initialize() is True
        assert trading_manager.is_connected is True

    def test_emergency_stop_scenario(self):
        trading_manager = TradingManager(self.test_config)
        trading_manager.trading_session_active = True
        trading_manager.execution_engine = MagicMock()
        trading_manager.execution_engine.get_active_orders.return_value = ["AAPL", "GOOGL"]
        trading_manager.execution_engine.cancel_order.return_value = True

        trading_manager.emergency_stop("Market crash detected")

        assert trading_manager.trading_session_active is False
        assert trading_manager.risk_control_system.circuit_breaker.is_tripped is True
        assert trading_manager.execution_engine.cancel_order.call_count == 2

    @patch("clenow_momentum.trading.trading_manager.save_ibkr_portfolio")
    @patch("clenow_momentum.trading.trading_manager.load_portfolio_state")
    def test_portfolio_discrepancy_detection(self, mock_load_portfolio, mock_save_portfolio):
        trading_manager = TradingManager(self.test_config)
        trading_manager.is_connected = True
        trading_manager.ibkr_client = MagicMock()
        trading_manager.ibkr_client.get_positions.return_value = [
            MagicMock(symbol="AAPL", quantity=100, avg_cost=150.0, market_value=15000.0),
            MagicMock(symbol="TSLA", quantity=50, avg_cost=800.0, market_value=40000.0),
        ]
        trading_manager.ibkr_client.get_account_summary.return_value = AccountSummary(
            net_liquidation=100000.0,
            buying_power=200000.0,
            total_cash=15000.0,
            excess_liquidity=160000.0,
        )
        mock_load_portfolio.return_value = Portfolio(cash=10000.0)

        synced_portfolio = trading_manager.sync_portfolio_only()

        assert synced_portfolio.cash == 15000.0
        assert "AAPL" in synced_portfolio.positions
        assert "TSLA" in synced_portfolio.positions
        mock_save_portfolio.assert_called_once()

    def test_configuration_validation(self):
        issues = validate_ibkr_config(self.test_config)
        critical_issues = [issue for issue in issues if "❌" in issue]
        assert len(critical_issues) == 0

        invalid_config = {
            **self.test_config,
            "ibkr": {
                **self.test_config["ibkr"],
                "host": "",
                "port": 0,
            },
        }
        issues = validate_ibkr_config(invalid_config)
        critical_issues = [issue for issue in issues if "❌" in issue]
        assert len(critical_issues) > 0

    def test_trading_manager_context_manager(self):
        with (
            patch.object(TradingManager, "initialize") as mock_init,
            patch.object(TradingManager, "disconnect") as mock_disconnect,
        ):
            mock_init.return_value = True

            with TradingManager(self.test_config) as trading_manager:
                assert isinstance(trading_manager, TradingManager)

            mock_init.assert_called_once()
            mock_disconnect.assert_called_once()
