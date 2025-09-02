"""
Integration tests for IBKR trading system.

These tests validate the complete workflow from analysis to execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clenow_momentum.strategy.rebalancing import OrderType, RebalancingOrder
from clenow_momentum.trading.trading_manager import TradingManager


class TestIBKRIntegration:
    """Test complete IBKR integration workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            "strategy_allocation": 100000,
            "ibkr_host": "127.0.0.1",
            "ibkr_port": 7497,  # TWS Paper
            "ibkr_client_id": 1,
            "ibkr_account_id": "TEST123",
            "ibkr_timeout": 30,
            "ibkr_auto_reconnect": True,
            "portfolio_state_file": "test_portfolio.json",
            "cash_buffer": 0.02,
            "max_positions": 20,
            "max_position_pct": 0.05,
            "risk_per_trade": 0.001,
        }

        self.sample_orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.SELL,
                shares=50,
                current_price=150.0,
                order_value=7500.0,
                reason="Reduce position",
                priority=1,
            ),
            RebalancingOrder(
                ticker="GOOGL",
                order_type=OrderType.BUY,
                shares=10,
                current_price=2500.0,
                order_value=25000.0,
                reason="New position",
                priority=3,
            ),
            RebalancingOrder(
                ticker="MSFT",
                order_type=OrderType.BUY,
                shares=25,
                current_price=300.0,
                order_value=7500.0,
                reason="Increase position",
                priority=4,
            ),
        ]

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    async def test_risk_controls_integration(self, mock_validate, mock_create_connector):
        """Test risk controls integration in complete workflow."""
        mock_validate.return_value = []

        mock_connector = AsyncMock()
        mock_connector.connect.return_value = True
        mock_connector.get_account_info.return_value = {
            "net_liquidation": 100000,
            "total_cash": 1000,  # Very low cash
            "positions": {},
        }
        mock_create_connector.return_value = mock_connector

        # Create risky orders (too large positions)
        risky_orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=1000,
                current_price=150.0,
                order_value=150000.0,  # 150% of account - way too large
                reason="Dangerous position",
            )
        ]

        with patch("clenow_momentum.trading.trading_manager.load_portfolio_state") as mock_load:
            mock_portfolio = MagicMock()
            mock_portfolio.cash = 1000
            mock_portfolio.total_market_value = 99000
            mock_portfolio.positions = {}
            mock_load.return_value = mock_portfolio

            async with TradingManager(self.test_config) as trading_manager:
                execution_results = await trading_manager.execute_rebalancing(risky_orders)

                # Should be blocked by risk controls
                assert execution_results["status"] == "blocked_by_risk_controls"
                assert execution_results["risk_checks_passed"] is False
                assert "risk" in execution_results["risk_reason"].lower()

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    async def test_error_handling_and_recovery(self, mock_validate, mock_create_connector):
        """Test error handling and recovery scenarios."""
        mock_validate.return_value = []

        # Test connection failure
        mock_connector = AsyncMock()
        mock_connector.connect.return_value = False
        mock_create_connector.return_value = mock_connector

        trading_manager = TradingManager(self.test_config)

        with pytest.raises(Exception):  # Should raise TradingManagerError  # noqa: B017
            await trading_manager.initialize()

        # Test successful recovery after fixing connection
        mock_connector.connect.return_value = True
        await trading_manager.initialize()
        assert trading_manager.is_connected is True

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    async def test_emergency_stop_scenario(self, mock_validate, mock_create_connector):
        """Test emergency stop functionality."""
        mock_validate.return_value = []

        mock_connector = AsyncMock()
        mock_connector.connect.return_value = True
        mock_connector.get_account_info.return_value = {
            "net_liquidation": 100000,
            "total_cash": 10000,
            "positions": {},
        }
        mock_create_connector.return_value = mock_connector

        async with TradingManager(self.test_config) as trading_manager:
            # Simulate active trading session
            trading_manager.trading_session_active = True

            # Mock active orders
            if trading_manager.execution_engine:
                trading_manager.execution_engine.active_orders = {
                    "AAPL": MagicMock(),
                    "GOOGL": MagicMock(),
                }

            # Trigger emergency stop
            await trading_manager.emergency_stop("Market crash detected")

            # Verify emergency stop effects
            assert trading_manager.trading_session_active is False
            assert trading_manager.risk_control_system.circuit_breaker.is_tripped is True

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    async def test_portfolio_discrepancy_detection(self, mock_validate, mock_create_connector):
        """Test portfolio discrepancy detection and reconciliation."""
        mock_validate.return_value = []

        mock_connector = AsyncMock()
        mock_connector.connect.return_value = True
        mock_create_connector.return_value = mock_connector

        # Mock discrepant account info
        mock_connector.get_account_info.return_value = {
            "total_cash": 15000,  # Different from portfolio
            "net_liquidation": 100000,
            "positions": {
                "AAPL": {"shares": 100, "avg_cost": 150, "market_value": 15000, "unrealized_pnl": 0},
                "TSLA": {"shares": 50, "avg_cost": 800, "market_value": 40000, "unrealized_pnl": 0},  # Not in portfolio
            },
        }

        with patch("clenow_momentum.trading.trading_manager.load_portfolio_state") as mock_load:
            from datetime import UTC, datetime

            from clenow_momentum.strategy.rebalancing import Portfolio, Position

            mock_portfolio = Portfolio(cash=10000)  # Different from IBKR
            # Add AAPL position (TSLA is missing)
            position = Position(
                ticker="AAPL",
                shares=100,
                entry_price=150.0,
                current_price=150.0,
                entry_date=datetime.now(UTC),
                atr=5.0
            )
            mock_portfolio.add_position(position)
            mock_load.return_value = mock_portfolio

            async with TradingManager(self.test_config) as trading_manager:
                # This should detect and handle discrepancies
                synced_portfolio = await trading_manager.sync_portfolio_only()

                # Portfolio should be updated with IBKR data
                assert synced_portfolio is not None

    def test_configuration_validation(self):
        """Test configuration validation."""
        from clenow_momentum.trading import validate_ibkr_config

        # Test valid configuration
        valid_config = self.test_config.copy()
        issues = validate_ibkr_config(valid_config)
        critical_issues = [issue for issue in issues if "❌" in issue]
        assert len(critical_issues) == 0

        # Test invalid configuration
        invalid_config = valid_config.copy()
        invalid_config["ibkr_host"] = ""
        invalid_config["ibkr_port"] = 0

        issues = validate_ibkr_config(invalid_config)
        critical_issues = [issue for issue in issues if "❌" in issue]
        assert len(critical_issues) > 0

    @pytest.mark.asyncio
    async def test_trading_manager_context_manager(self):
        """Test trading manager as context manager."""
        with (
            patch("clenow_momentum.trading.trading_manager.create_ibkr_connector") as mock_create,
            patch("clenow_momentum.trading.trading_manager.validate_ibkr_config") as mock_validate,
        ):
            mock_validate.return_value = []
            mock_connector = AsyncMock()
            mock_connector.connect.return_value = True
            mock_create.return_value = mock_connector

            # Should connect on entry and disconnect on exit
            async with TradingManager(self.test_config) as trading_manager:
                assert trading_manager.is_connected is True

            # After exit, should be disconnected
            assert trading_manager.is_connected is False
