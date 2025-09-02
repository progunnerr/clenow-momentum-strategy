"""
Tests for trading manager module.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clenow_momentum.strategy.rebalancing import OrderType, RebalancingOrder
from clenow_momentum.trading.trading_manager import TradingManager, TradingManagerError


class TestTradingManager:
    """Test TradingManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "strategy_allocation": 100000,
            "ibkr_host": "127.0.0.1",
            "ibkr_port": 7497,  # TWS Paper
            "ibkr_client_id": 1,
            "ibkr_account_id": "TEST123",
            "ibkr_timeout": 30,
            "ibkr_auto_reconnect": True,
            "portfolio_state_file": "test_portfolio.json",
            "cash_buffer": 0.02,
        }

    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    def test_initialization(self, mock_validate, mock_create_connector):
        """Test trading manager initialization."""
        mock_validate.return_value = []
        trading_manager = TradingManager(self.mock_config)

        assert trading_manager.config == self.mock_config
        assert trading_manager.is_connected is False
        assert trading_manager.trading_session_active is False

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    @patch("clenow_momentum.trading.trading_manager.PortfolioSynchronizer")
    @patch("clenow_momentum.trading.trading_manager.TradingExecutionEngine")
    async def test_successful_initialization(
        self, mock_execution_engine, mock_portfolio_sync, mock_validate, mock_create_connector
    ):
        """Test successful initialization."""
        mock_validate.return_value = []
        mock_connector = AsyncMock()
        mock_connector.connect.return_value = True
        mock_create_connector.return_value = mock_connector

        trading_manager = TradingManager(self.mock_config)
        result = await trading_manager.initialize()

        assert result is True
        assert trading_manager.is_connected is True
        mock_connector.connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    async def test_initialization_with_config_errors(self, mock_validate, mock_create_connector):
        """Test initialization with configuration errors."""
        mock_validate.return_value = ["❌ Critical error", "⚠️ Warning"]

        trading_manager = TradingManager(self.mock_config)

        with pytest.raises(TradingManagerError):
            await trading_manager.initialize()

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.create_ibkr_connector")
    @patch("clenow_momentum.trading.trading_manager.validate_ibkr_config")
    async def test_initialization_connection_failure(self, mock_validate, mock_create_connector):
        """Test initialization with connection failure."""
        mock_validate.return_value = []
        mock_connector = AsyncMock()
        mock_connector.connect.return_value = False
        mock_create_connector.return_value = mock_connector

        trading_manager = TradingManager(self.mock_config)

        with pytest.raises(TradingManagerError):
            await trading_manager.initialize()

    @pytest.mark.asyncio
    async def test_not_initialized_operations(self):
        """Test operations when not initialized."""
        trading_manager = TradingManager(self.mock_config)

        with pytest.raises(TradingManagerError):
            await trading_manager.execute_rebalancing([])

        with pytest.raises(TradingManagerError):
            await trading_manager.sync_portfolio_only()

        with pytest.raises(TradingManagerError):
            await trading_manager.get_account_status()

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.load_portfolio_state")
    @patch("clenow_momentum.trading.trading_manager.save_portfolio_state")
    async def test_execute_rebalancing_dry_run(self, mock_save_portfolio, mock_load_portfolio):
        """Test execute rebalancing in dry run mode."""
        # Set up mocks
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True

        mock_portfolio = MagicMock()
        mock_portfolio.num_positions = 5
        mock_portfolio.cash = 10000
        mock_portfolio.total_market_value = 90000
        mock_load_portfolio.return_value = mock_portfolio

        mock_portfolio_sync = AsyncMock()
        mock_portfolio_sync.sync_portfolio_from_ibkr.return_value = mock_portfolio
        trading_manager.portfolio_sync = mock_portfolio_sync

        mock_connector = AsyncMock()
        mock_connector.get_account_info.return_value = {
            "net_liquidation": 100000,
            "total_cash": 10000,
            "positions": {},
        }
        trading_manager.ibkr_connector = mock_connector

        mock_risk_control = MagicMock()
        mock_risk_control.pre_trade_validation.return_value = []
        mock_risk_control.can_proceed_with_trading.return_value = (True, "All checks passed")
        trading_manager.risk_control_system = mock_risk_control

        mock_execution_engine = AsyncMock()
        mock_execution_engine.execute_rebalancing_orders.return_value = (
            [],
            {"successful": 5, "failed": 0, "total_value_traded": 50000},
        )
        trading_manager.execution_engine = mock_execution_engine

        # Create test orders
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

        result = await trading_manager.execute_rebalancing(orders, dry_run=True)

        assert result["status"] == "completed_successfully"
        assert result["orders_executed"] == 5
        assert result["dry_run"] is True
        mock_execution_engine.execute_rebalancing_orders.assert_called_once()

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.load_portfolio_state")
    async def test_execute_rebalancing_risk_blocked(self, mock_load_portfolio):
        """Test execute rebalancing blocked by risk controls."""
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True

        mock_portfolio = MagicMock()
        mock_load_portfolio.return_value = mock_portfolio

        mock_portfolio_sync = AsyncMock()
        mock_portfolio_sync.sync_portfolio_from_ibkr.return_value = mock_portfolio
        trading_manager.portfolio_sync = mock_portfolio_sync

        mock_connector = AsyncMock()
        mock_connector.get_account_info.return_value = {"net_liquidation": 100000}
        trading_manager.ibkr_connector = mock_connector

        mock_risk_control = MagicMock()
        mock_risk_control.pre_trade_validation.return_value = []
        mock_risk_control.can_proceed_with_trading.return_value = (False, "Risk check failed")
        trading_manager.risk_control_system = mock_risk_control

        result = await trading_manager.execute_rebalancing([])

        assert result["status"] == "blocked_by_risk_controls"
        assert result["risk_checks_passed"] is False

    @pytest.mark.asyncio
    @patch("clenow_momentum.trading.trading_manager.load_portfolio_state")
    @patch("clenow_momentum.trading.trading_manager.save_portfolio_state")
    async def test_sync_portfolio_only(self, mock_save_portfolio, mock_load_portfolio):
        """Test portfolio sync only."""
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True

        mock_portfolio = MagicMock()
        mock_portfolio.num_positions = 10
        mock_load_portfolio.return_value = mock_portfolio

        mock_portfolio_sync = AsyncMock()
        mock_portfolio_sync.sync_portfolio_from_ibkr.return_value = mock_portfolio
        trading_manager.portfolio_sync = mock_portfolio_sync

        result = await trading_manager.sync_portfolio_only()

        assert result == mock_portfolio
        mock_portfolio_sync.sync_portfolio_from_ibkr.assert_called_once()
        mock_save_portfolio.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_account_status(self):
        """Test get account status."""
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True
        trading_manager.last_sync_time = datetime.now(UTC)

        mock_connector = AsyncMock()
        mock_connector.get_account_info.return_value = {
            "account_id": "TEST123",
            "total_cash": 50000,
            "net_liquidation": 100000,
        }
        trading_manager.ibkr_connector = mock_connector

        mock_risk_control = MagicMock()
        mock_risk_control.get_risk_status.return_value = {"circuit_breaker": False}
        trading_manager.risk_control_system = mock_risk_control

        mock_execution_engine = MagicMock()
        mock_execution_engine.get_active_orders.return_value = {}
        trading_manager.execution_engine = mock_execution_engine

        result = await trading_manager.get_account_status()

        assert result["account_id"] == "TEST123"
        assert result["total_cash"] == 50000
        assert result["trading_manager"]["is_connected"] is True
        assert result["trading_manager"]["trading_mode"] == "paper"

    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Test emergency stop."""
        trading_manager = TradingManager(self.mock_config)
        trading_manager.trading_session_active = True

        mock_risk_control = MagicMock()
        trading_manager.risk_control_system = mock_risk_control

        mock_execution_engine = MagicMock()  # Changed from AsyncMock to MagicMock
        mock_execution_engine.get_active_orders.return_value = {"AAPL": MagicMock()}
        mock_execution_engine.cancel_order = AsyncMock(return_value=True)  # Only cancel_order is async
        trading_manager.execution_engine = mock_execution_engine

        await trading_manager.emergency_stop("Test emergency")

        mock_risk_control.emergency_stop.assert_called_once_with("Test emergency")
        mock_execution_engine.cancel_order.assert_called_once_with("AAPL")
        assert trading_manager.trading_session_active is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect."""
        trading_manager = TradingManager(self.mock_config)
        trading_manager.is_connected = True
        trading_manager.trading_session_active = True

        mock_connector = AsyncMock()
        trading_manager.ibkr_connector = mock_connector

        await trading_manager.disconnect()

        mock_connector.disconnect.assert_called_once()
        assert trading_manager.is_connected is False
        assert trading_manager.trading_session_active is False

    def test_get_status_summary(self):
        """Test status summary."""
        trading_manager = TradingManager(self.mock_config)

        # Not connected
        summary = trading_manager.get_status_summary()
        assert "not connected" in summary

        # Connected
        trading_manager.is_connected = True
        trading_manager.trading_session_active = True
        trading_manager.last_sync_time = datetime.now(UTC)

        summary = trading_manager.get_status_summary()
        assert "Connected" in summary
        assert "PAPER" in summary
        assert "ACTIVE" in summary

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with (
            patch.object(TradingManager, "initialize") as mock_init,
            patch.object(TradingManager, "disconnect") as mock_disconnect,
        ):
            mock_init.return_value = True

            async with TradingManager(self.mock_config) as manager:
                assert isinstance(manager, TradingManager)

            mock_init.assert_called_once()
            mock_disconnect.assert_called_once()
