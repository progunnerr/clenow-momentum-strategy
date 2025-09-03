"""
High-level trading manager for IBKR integration.

This module provides a unified interface for all trading operations,
integrating the IBKR connector, execution engine, portfolio sync, and risk controls.
"""

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from ..data_sources import IBKRClient
from ..data_sources.ibkr_client import get_trading_mode
from ..strategy.rebalancing import (
    Portfolio,
    RebalancingOrder,
    load_portfolio_state,
    save_portfolio_state,
)
from ..utils.config import load_config
from .execution_engine_sync import SyncTradingExecutionEngine
from .portfolio_sync import PortfolioSynchronizer
from .risk_controls import RiskCheckResult, RiskControlSystem


class TradingManagerError(Exception):
    """Base exception for trading manager errors."""

    pass


class TradingManager:
    """
    High-level trading manager for IBKR integration.

    This class provides a unified interface for:
    - Connecting to IBKR
    - Synchronizing portfolio state
    - Executing rebalancing orders
    - Managing risk controls
    - Updating portfolio persistence
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize trading manager.

        Args:
            config: Configuration dictionary (loads from env if None)
        """
        self.config = config or load_config()

        # Initialize components (but don't connect yet)
        self.ibkr_client: IBKRClient | None = None
        self.portfolio_sync = None
        self.execution_engine = None
        self.risk_control_system = RiskControlSystem(self.config)

        # State tracking
        self.is_connected = False
        self.last_sync_time: datetime | None = None
        self.trading_session_active = False

        logger.info("Trading manager initialized")

    def initialize(self) -> bool:
        """
        Initialize and connect all trading components.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing trading manager...")

            # Validate configuration
            ibkr_config = self.config.get("ibkr", {})
            if not ibkr_config:
                raise TradingManagerError("IBKR configuration not found")

            # Create and connect IBKR client (now synchronous)
            logger.info("Creating IBKR client...")
            self.ibkr_client = IBKRClient(
                host=ibkr_config.get("host", "127.0.0.1"),
                port=ibkr_config.get("port", 7497),
                client_id=ibkr_config.get("client_id", 1),
                account=ibkr_config.get("account_id", ""),
            )

            logger.info("Connecting to IBKR...")
            self.ibkr_client.connect()

            # Initialize other components
            # Using the new synchronous execution engine
            self.portfolio_sync = PortfolioSynchronizer(self.ibkr_client)
            self.execution_engine = SyncTradingExecutionEngine(self.ibkr_client)

            self.is_connected = True
            logger.success("Trading manager initialized successfully")

            # Log trading mode based on port
            port = ibkr_config.get("port", 7497)
            trading_mode = "PAPER" if port in [7497, 4002] else "LIVE"
            logger.info(f"Trading mode: {trading_mode} TRADING")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading manager: {e}")
            self.is_connected = False
            raise TradingManagerError(f"Initialization failed: {e}") from e

    def execute_rebalancing(
        self,
        rebalancing_orders: list[RebalancingOrder],
        portfolio_file: Path | None = None,
        dry_run: bool = None,
        force_execution: bool = False,
        pre_synced_portfolio: Portfolio | None = None,
    ) -> dict:
        """
        Execute complete rebalancing process.

        Args:
            rebalancing_orders: Orders to execute
            portfolio_file: Path to portfolio state file
            dry_run: Override dry run mode (auto-detects from port if None)
            force_execution: If True, skip all confirmation prompts
            pre_synced_portfolio: If provided, skip portfolio sync (already synced)

        Returns:
            Dictionary with execution results
        """
        if not self.is_connected:
            raise TradingManagerError("Trading manager not initialized")

        # Determine dry run mode based on port
        if dry_run is None:
            port = self.config.get("ibkr", {}).get("port", 7497)
            dry_run = port in [7497, 4002]  # Paper trading ports

        logger.info(f"Starting rebalancing execution (dry_run={dry_run})")

        execution_results = {
            "start_time": datetime.now(UTC),
            "orders_submitted": len(rebalancing_orders),
            "orders_executed": 0,
            "orders_failed": 0,
            "total_value_traded": 0.0,
            "risk_checks_passed": False,
            "portfolio_synced": False,
            "dry_run": dry_run,
        }

        try:
            # 1. Load or use pre-synced portfolio
            if pre_synced_portfolio is not None:
                # Portfolio already synced, use it directly
                logger.info("Using pre-synced portfolio from IBKR...")
                portfolio = pre_synced_portfolio
                execution_results["portfolio_synced"] = True
                self.last_sync_time = datetime.now(UTC)
            else:
                # Load and sync portfolio
                logger.info("Loading current portfolio state...")
                if portfolio_file is None:
                    portfolio_file = Path(self.config["portfolio_state_file"])

                portfolio = load_portfolio_state(portfolio_file)

                # 2. Sync portfolio with IBKR
                logger.info("Synchronizing portfolio with IBKR...")
                portfolio = self.sync_portfolio_only(portfolio_file)
                execution_results["portfolio_synced"] = True
                self.last_sync_time = datetime.now(UTC)

            # 3. Get account information and check cash availability
            account_summary = self.ibkr_client.get_account_summary()
            ibkr_available_cash = account_summary.total_cash
            strategy_allocation = self.config["strategy_allocation"]

            # Use the smaller of strategy allocation or available cash
            effective_allocation = min(strategy_allocation, ibkr_available_cash)

            # Log cash availability status
            if effective_allocation < strategy_allocation:
                logger.warning(
                    f"Strategy allocation limited by available cash: "
                    f"${effective_allocation:,.0f} available vs ${strategy_allocation:,.0f} allocated"
                )
                execution_results["cash_limited"] = True
                execution_results["available_cash"] = ibkr_available_cash
                execution_results["requested_allocation"] = strategy_allocation
                execution_results["effective_allocation"] = effective_allocation
            else:
                logger.info(f"Sufficient cash available: ${ibkr_available_cash:,.0f} for ${strategy_allocation:,.0f} allocation")
                execution_results["cash_limited"] = False

            # Use total portfolio value for risk calculations
            # Account value = current portfolio value (positions + cash)
            account_value = portfolio.total_market_value + portfolio.cash

            # 4. Run pre-trade risk checks
            logger.info("Running pre-trade risk validation...")
            risk_checks = self.risk_control_system.pre_trade_validation(
                rebalancing_orders, portfolio, account_value, ibkr_cash=ibkr_available_cash
            )

            # Check if trading can proceed
            can_proceed, risk_reason = self.risk_control_system.can_proceed_with_trading(risk_checks)
            execution_results["risk_checks_passed"] = can_proceed
            execution_results["risk_reason"] = risk_reason

            if not can_proceed:
                logger.error(f"Trading blocked by risk controls: {risk_reason}")
                execution_results["status"] = "blocked_by_risk_controls"
                return execution_results

            logger.success("Risk checks passed - proceeding with execution")

            # 5. Scale orders if cash-limited
            if effective_allocation < strategy_allocation:
                rebalancing_orders = self._scale_orders_for_cash_limit(
                    rebalancing_orders, strategy_allocation, effective_allocation
                )
                logger.info(f"Scaled {len(rebalancing_orders)} orders to fit available cash")

            # 6. Execute orders
            self.trading_session_active = True
            executed_orders, exec_summary = self.execution_engine.execute_rebalancing_orders(
                rebalancing_orders, portfolio, dry_run=dry_run, force_execution=force_execution
            )

            # Update execution results
            execution_results.update({
                "orders_executed": exec_summary.get("successful", 0),
                "orders_failed": exec_summary.get("failed", 0),
                "orders_partially_filled": exec_summary.get("partially_filled", 0),
                "total_value_traded": exec_summary.get("total_value_traded", 0.0),
                "total_commission": exec_summary.get("total_commission", 0.0),
                "execution_duration_minutes": exec_summary.get("duration_minutes", 0.0),
            })

            # 6. Final portfolio sync and validation
            if not dry_run:
                logger.info("Final portfolio synchronization...")
                portfolio = self.sync_portfolio_only(portfolio_file)

                # TODO: Implement discrepancy detection
                execution_results["discrepancies"] = 0

            # 7. Save updated portfolio state
            if not dry_run:
                portfolio.last_rebalance_date = datetime.now(UTC)
                save_portfolio_state(portfolio, portfolio_file)
                logger.info(f"Portfolio state saved to {portfolio_file}")

            # 8. Post-trade risk monitoring
            alerts = self.risk_control_system.post_trade_monitoring(portfolio, account_value)
            if alerts:
                execution_results["post_trade_alerts"] = len(alerts)
                for alert in alerts:
                    if alert.result == RiskCheckResult.CRITICAL_FAIL:
                        logger.critical(f"🚨 Critical alert: {alert.message}")

            execution_results["status"] = "completed_successfully"
            execution_results["end_time"] = datetime.now(UTC)

            logger.success(f"Rebalancing execution completed: {execution_results['orders_executed']} orders executed")
            return execution_results

        except Exception as e:
            logger.error(f"Rebalancing execution failed: {e}")
            execution_results["status"] = "failed"
            execution_results["error"] = str(e)
            execution_results["end_time"] = datetime.now(UTC)
            raise TradingManagerError(f"Rebalancing execution failed: {e}") from e

        finally:
            self.trading_session_active = False

    def sync_portfolio_only(self, portfolio_file: Path | None = None) -> Portfolio:
        """
        Sync portfolio with IBKR without executing trades.

        Args:
            portfolio_file: Path to portfolio state file

        Returns:
            Updated portfolio
        """
        if not self.is_connected:
            raise TradingManagerError("Trading manager not initialized")

        try:
            logger.info("Syncing portfolio with IBKR...")

            if portfolio_file is None:
                portfolio_file = Path(self.config["portfolio_state_file"])

            portfolio = load_portfolio_state(portfolio_file)

            # Get positions directly from IBKR
            positions = self.ibkr_client.get_positions()
            account_summary = self.ibkr_client.get_account_summary()

            # Update portfolio with IBKR data
            portfolio.cash = account_summary.total_cash

            # Clear and update positions
            portfolio.positions.clear()
            for pos in positions:
                from ..strategy.rebalancing import Position as PortfolioPosition
                portfolio.positions[pos.symbol] = PortfolioPosition(
                    ticker=pos.symbol,
                    shares=int(pos.quantity),
                    entry_price=pos.avg_cost,
                    current_price=pos.market_value / pos.quantity if pos.quantity else 0,
                    entry_date=datetime.now(UTC),
                    avg_cost_basis=pos.avg_cost
                )

            # Save updated state
            save_portfolio_state(portfolio, portfolio_file)
            self.last_sync_time = datetime.now(UTC)

            logger.success(f"Portfolio synced: {portfolio.num_positions} positions")
            return portfolio

        except Exception as e:
            logger.error(f"Portfolio sync failed: {e}")
            raise TradingManagerError(f"Portfolio sync failed: {e}") from e

    def get_account_status(self) -> dict:
        """
        Get comprehensive account status.

        Returns:
            Dictionary with account information
        """
        if not self.is_connected:
            raise TradingManagerError("Trading manager not initialized")

        try:
            account_summary = self.ibkr_client.get_account_summary()
            account_info = {
                "net_liquidation": account_summary.net_liquidation,
                "buying_power": account_summary.buying_power,
                "total_cash": account_summary.total_cash,
                "excess_liquidity": account_summary.excess_liquidity
            }

            # Add additional status information
            return {
                **account_info,
                "trading_manager": {
                    "is_connected": self.is_connected,
                    "last_sync_time": self.last_sync_time,
                    "trading_session_active": self.trading_session_active,
                    "trading_mode": get_trading_mode(self.config),
                },
                "risk_controls": self.risk_control_system.get_risk_status(),
                "active_orders": len(self.execution_engine.get_active_orders()) if self.execution_engine else 0,
            }


        except Exception as e:
            logger.error(f"Failed to get account status: {e}")
            raise TradingManagerError(f"Failed to get account status: {e}") from e

    def emergency_stop(self, reason: str) -> None:
        """
        Emergency stop all trading operations.

        Args:
            reason: Reason for emergency stop
        """
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")

        try:
            # Trip circuit breaker
            self.risk_control_system.emergency_stop(reason)

            # Cancel all active orders if possible
            if self.execution_engine:
                active_orders = self.execution_engine.get_active_orders()
                for ticker in active_orders:
                    try:
                        self.execution_engine.cancel_order(ticker)
                        logger.info(f"Cancelled order for {ticker}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order for {ticker}: {e}")

            self.trading_session_active = False
            logger.critical("Emergency stop completed")

        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    def disconnect(self) -> None:
        """Disconnect from IBKR and clean up resources."""
        try:
            if self.ibkr_client:
                self.ibkr_client.disconnect()
                self.ibkr_client = None

            self.is_connected = False
            self.trading_session_active = False

            logger.info("Trading manager disconnected")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def get_status_summary(self) -> str:
        """Get a brief status summary string."""
        if not self.is_connected:
            return "❌ Trading manager not connected"

        detected_mode = get_trading_mode(self.config)
        trading_mode = "📄 PAPER" if detected_mode == "paper" else "💰 LIVE" if detected_mode == "live" else "❓ UNKNOWN"
        session_status = "🔄 ACTIVE" if self.trading_session_active else "⏸️ IDLE"

        sync_status = ""
        if self.last_sync_time:
            minutes_ago = int((datetime.now(UTC) - self.last_sync_time).total_seconds() / 60)
            sync_status = f" | Last sync: {minutes_ago}m ago"

        return f"✅ Connected | {trading_mode} | {session_status}{sync_status}"

    def _scale_orders_for_cash_limit(
        self,
        orders: list[RebalancingOrder],
        original_allocation: float,
        effective_allocation: float
    ) -> list[RebalancingOrder]:
        """
        Scale down orders when available cash is less than strategy allocation.

        Args:
            orders: Original rebalancing orders
            original_allocation: Original strategy allocation amount
            effective_allocation: Available cash amount

        Returns:
            Scaled orders that fit within available cash
        """
        if effective_allocation >= original_allocation:
            return orders  # No scaling needed

        scale_factor = effective_allocation / original_allocation
        logger.info(f"Scaling orders by {scale_factor:.3f} due to cash limitation")

        scaled_orders = []
        total_buy_value = 0

        # First pass: Scale all orders
        for order in orders:
            scaled_order = RebalancingOrder(
                ticker=order.ticker,
                order_type=order.order_type,
                shares=int(order.shares * scale_factor) if order.order_type.value == "BUY" else order.shares,
                current_price=order.current_price,
                order_value=0,  # Will recalculate
                reason=f"{order.reason} (scaled for cash limit)",
                priority=order.priority,
                status=order.status,
                limit_price=order.limit_price,
            )

            # Recalculate order value
            scaled_order.order_value = scaled_order.shares * scaled_order.current_price

            # Track buy orders total
            if scaled_order.order_type.value == "BUY":
                total_buy_value += scaled_order.order_value

            scaled_orders.append(scaled_order)

        # Second pass: Further adjust if still over cash limit
        if total_buy_value > effective_allocation:
            additional_scale = effective_allocation / total_buy_value
            logger.warning(f"Additional scaling required: {additional_scale:.3f}")

            for order in scaled_orders:
                if order.order_type.value == "BUY":
                    order.shares = max(1, int(order.shares * additional_scale))
                    order.order_value = order.shares * order.current_price
                    order.reason += " (further scaled)"

        # Log scaling results
        original_buy_value = sum(o.order_value for o in orders if o.order_type.value == "BUY")
        final_buy_value = sum(o.order_value for o in scaled_orders if o.order_type.value == "BUY")

        logger.info(
            f"Order scaling complete: ${original_buy_value:,.0f} -> ${final_buy_value:,.0f} "
            f"({final_buy_value/original_buy_value:.1%} of original)"
        )

        return scaled_orders
