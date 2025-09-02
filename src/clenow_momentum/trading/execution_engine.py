"""
Trading execution engine for IBKR integration.

This module handles the execution of rebalancing orders through IBKR,
including order management, status tracking, and error handling.
"""

import asyncio
import sys
from datetime import UTC, datetime

from loguru import logger

from ..strategy.rebalancing import OrderStatus, OrderType, Portfolio, RebalancingOrder
from .ibkr_connector import IBKRConnector
from .portfolio_sync import PortfolioSynchronizer


class ExecutionError(Exception):
    """Base exception for execution engine errors."""

    pass


class TradingExecutionEngine:
    """
    Handles execution of trading orders through IBKR.

    This engine:
    - Executes rebalancing orders in priority order
    - Monitors order status and handles partial fills
    - Updates portfolio state with execution results
    - Provides execution reporting and logging
    """

    def __init__(self, ibkr_connector: IBKRConnector, portfolio_sync: PortfolioSynchronizer):
        """
        Initialize trading execution engine.

        Args:
            ibkr_connector: Connected IBKR connector
            portfolio_sync: Portfolio synchronizer instance
        """
        self.ibkr = ibkr_connector
        self.portfolio_sync = portfolio_sync
        self.active_orders: dict[str, RebalancingOrder] = {}  # ticker -> order
        self.execution_log: list[dict] = []

        # Execution settings
        self.order_timeout = 300  # 5 minutes
        self.max_retry_attempts = 3
        self.retry_delay = 10  # 10 seconds

        # Confirmation settings - always require by default
        self.skip_confirmations = False  # Can be set to skip all confirmations

        logger.info("Trading execution engine initialized")

    async def execute_rebalancing_orders(
        self,
        orders: list[RebalancingOrder],
        portfolio: Portfolio,
        dry_run: bool = False,
        max_concurrent_orders: int = 5,
        force_execution: bool = False,
    ) -> tuple[list[RebalancingOrder], dict]:
        """
        Execute a list of rebalancing orders.

        Args:
            orders: List of orders to execute
            portfolio: Current portfolio to update
            dry_run: If True, simulate execution without placing real orders
            max_concurrent_orders: Maximum number of concurrent orders
            force_execution: If True, skip all confirmation prompts

        Returns:
            Tuple of (executed_orders, execution_summary)
        """
        # Set skip confirmations based on force flag
        self.skip_confirmations = force_execution

        logger.info(f"Starting execution of {len(orders)} orders (dry_run={dry_run}, force={force_execution})")

        if dry_run:
            return await self._simulate_execution(orders)

        # Sort orders by priority (sell orders first, then buy orders)
        # For orders with same priority, use a secondary sort key:
        # - For buy orders, use momentum_rank if available to ensure proper order
        # - This preserves the momentum ranking within each priority group
        def sort_key(order):
            # Primary sort by priority
            primary = order.priority
            # Secondary sort: for buy orders with momentum_rank, use it; otherwise use 999
            secondary = getattr(order, 'momentum_rank', 999) if order.order_type == OrderType.BUY else 0
            return (primary, secondary)

        sorted_orders = sorted(orders, key=sort_key)

        executed_orders = []
        execution_summary = {
            "start_time": datetime.now(UTC),
            "total_orders": len(orders),
            "successful": 0,
            "failed": 0,
            "partially_filled": 0,
            "total_value_traded": 0.0,
            "total_commission": 0.0,
        }

        try:
            # Execute SELL orders first (priority 1-2)
            sell_orders = [o for o in sorted_orders if o.order_type == OrderType.SELL]
            if sell_orders:
                logger.info(f"Executing {len(sell_orders)} SELL orders first...")
                sell_results = await self._execute_order_batch(
                    sell_orders, portfolio, max_concurrent_orders
                )
                executed_orders.extend(sell_results)

            # Wait for all sell orders to complete before buying
            await self._wait_for_order_completion(sell_orders, timeout=self.order_timeout)

            # Sync portfolio to get updated cash position
            logger.info("Syncing portfolio after sell orders...")
            await self.portfolio_sync.sync_portfolio_from_ibkr(portfolio)

            # Execute BUY orders (priority 3-4)
            buy_orders = [o for o in sorted_orders if o.order_type == OrderType.BUY]
            if buy_orders:
                logger.info(f"Executing {len(buy_orders)} BUY orders...")
                buy_results = await self._execute_order_batch(
                    buy_orders, portfolio, max_concurrent_orders
                )
                logger.info(f"Buy batch returned {len(buy_results)} executed orders")
                executed_orders.extend(buy_results)

            # Wait for all orders to complete
            await self._wait_for_order_completion(buy_orders, timeout=self.order_timeout)

            # Final portfolio sync
            logger.info("Final portfolio synchronization...")
            await self.portfolio_sync.sync_portfolio_from_ibkr(portfolio)

            # Update execution summary
            for order in executed_orders:
                if order.status == OrderStatus.EXECUTED:
                    execution_summary["successful"] += 1
                    execution_summary["total_value_traded"] += order.order_value
                    if order.commission:
                        execution_summary["total_commission"] += order.commission
                elif order.status == OrderStatus.FAILED:
                    execution_summary["failed"] += 1
                elif order.is_partially_filled:
                    execution_summary["partially_filled"] += 1

            execution_summary["end_time"] = datetime.now(UTC)
            execution_summary["duration_minutes"] = (
                execution_summary["end_time"] - execution_summary["start_time"]
            ).total_seconds() / 60

            logger.success(
                f"Execution completed: {execution_summary['successful']} successful, "
                f"{execution_summary['failed']} failed, "
                f"{execution_summary['partially_filled']} partially filled"
            )

            return executed_orders, execution_summary

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            execution_summary["error"] = str(e)
            execution_summary["end_time"] = datetime.now(UTC)
            raise ExecutionError(f"Order execution failed: {e}") from e

    async def _execute_order_batch(
        self, orders: list[RebalancingOrder], portfolio: Portfolio, max_concurrent: int
    ) -> list[RebalancingOrder]:
        """
        Execute a batch of orders sequentially to ensure confirmations work properly.

        Args:
            orders: Orders to execute
            portfolio: Current portfolio
            max_concurrent: Maximum concurrent executions (ignored when confirmations needed)

        Returns:
            List of executed orders with updated status
        """
        executed_orders = []

        # Execute orders sequentially when confirmations are needed
        # This ensures the user can see and respond to each confirmation
        for order in orders:
            try:
                result = await self._execute_single_order(order)
                if result:
                    logger.debug(f"Order {order.ticker} executed successfully")
                    executed_orders.append(result)
            except Exception as e:
                logger.error(f"Order {order.ticker} failed: {e}")
                order.status = OrderStatus.FAILED
                order.error_message = str(e)
                # Don't stop on individual order failure unless user cancelled all
                if "User cancelled all orders" in str(e):
                    raise

        return executed_orders

    async def _get_order_confirmation(self, order: RebalancingOrder) -> bool:
        """
        Display order details in checkout style and get user confirmation.
        
        Args:
            order: Order to confirm
            
        Returns:
            True if user confirms, False otherwise
        """
        print("\n" + "="*60)
        print("📋 ORDER CONFIRMATION REQUIRED")
        print("="*60)

        # Order type and symbol
        action_emoji = "📉" if order.order_type.value == "SELL" else "📈"
        action_color = "\033[91m" if order.order_type.value == "SELL" else "\033[92m"  # Red for sell, green for buy
        reset_color = "\033[0m"

        print(f"\n{action_emoji} {action_color}{order.order_type.value} ORDER{reset_color}")
        print(f"Symbol: {order.ticker}")
        print(f"Shares: {order.shares:,}")
        print(f"Current Price: ${order.current_price:.2f}")
        print(f"Total Value: ${order.order_value:,.2f}")

        # Show detailed reasoning
        print("\n📊 REASON:")
        # Parse the reason for better display
        reason_lines = order.reason.split(". ")
        for line in reason_lines:
            if line:
                print(f"  • {line}")

        # For BUY orders, show additional metrics if available
        if order.order_type.value == "BUY":
            print("\n📈 TRADE METRICS:")
            if hasattr(order, 'momentum_rank'):
                print(f"  • Momentum Rank: #{order.momentum_rank}")
            if hasattr(order, 'momentum_score'):
                print(f"  • Momentum Score: {order.momentum_score:.3f}")
            if hasattr(order, 'r_squared'):
                print(f"  • R-squared: {order.r_squared:.3f}")
            # Show target weight percentage
            if hasattr(order, 'target_weight'):
                weight_pct = order.target_weight * 100
                print(f"  • Target Weight: {weight_pct:.1f}% of portfolio")
            elif hasattr(order, 'account_value'):
                # Fallback: calculate position percentage
                position_pct = (order.order_value / order.account_value) * 100
                print(f"  • Position Size: {position_pct:.1f}% of portfolio")
            
            # Show volatility/risk metrics
            print("\n📊 RISK METRICS:")
            if hasattr(order, 'atr'):
                print(f"  • ATR (20-day): ${order.atr:.2f}")
            if hasattr(order, 'volatility_pct'):
                print(f"  • Volatility: {order.volatility_pct:.2f}% of price")
            if hasattr(order, 'stop_loss'):
                print(f"  • Stop Loss: ${order.stop_loss:.2f}")
                # Calculate stop loss percentage
                stop_loss_pct = ((order.current_price - order.stop_loss) / order.current_price) * 100
                print(f"  • Stop Distance: -{stop_loss_pct:.1f}%")
            if hasattr(order, 'risk_amount'):
                print(f"  • $ Risk: ${order.risk_amount:,.0f}")
            
            # Show filters passed
            if hasattr(order, 'filters_passed') and order.filters_passed:
                print("\n✅ FILTERS PASSED:")
                for filter_item in order.filters_passed:
                    print(f"  • {filter_item}")

        # For SELL orders, show exit reasoning
        if order.order_type.value == "SELL":
            print("\n📉 EXIT ANALYSIS:")
            if "momentum" in order.reason.lower():
                print("  • Stock dropped out of top momentum rankings")
            if "filter" in order.reason.lower():
                print("  • Failed one or more strategy filters")
            if "gap" in order.reason.lower():
                print("  • Large gap detected (>15%)")
            if "moving average" in order.reason.lower() or "ma" in order.reason.lower():
                print("  • Price fell below moving average")
            if "rebalance" in order.reason.lower():
                print("  • Regular portfolio rebalancing")
            if "position" in order.reason.lower():
                print("  • Position no longer in target portfolio")

        # Warning based on order type
        print(f"\n⚠️  This will {order.order_type.value.lower()} {order.shares:,} shares of {order.ticker}")
        print(f"    Impact: ${order.order_value:,.2f}")

        # Get user input
        print("\nOptions:")
        print("  [y] Execute this order")
        print("  [n] Skip this order")
        print("  [a] Approve all remaining orders")
        print("  [q] Cancel all orders and quit")
        sys.stdout.flush()  # Ensure output is displayed

        # Use asyncio to run input in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, input, "\nYour choice: ")
        response = response.strip().lower()

        if response == 'a':
            self.skip_confirmations = True
            print("✅ All remaining orders will be executed automatically")
            return True
        if response == 'q':
            print("❌ Cancelling all orders and exiting...")
            raise ExecutionError("User cancelled all orders")
        if response in ['y', 'yes']:
            print("✅ Order confirmed")
            return True
        print("⏭️  Skipping this order")
        return False


    async def _execute_single_order(self, order: RebalancingOrder) -> RebalancingOrder:
        """
        Execute a single rebalancing order.

        Args:
            order: Order to execute

        Returns:
            Updated order with execution status
        """
        logger.debug(f"Starting execution for order: {order.ticker} ({order.order_type.value})")
        try:
            # Check if we need confirmation (not in dry run, not skipping)
            if not self.skip_confirmations:
                logger.info(f"Requesting confirmation for {order.ticker} order...")
                if not await self._get_order_confirmation(order):
                    logger.info(f"Order for {order.ticker} skipped by user")
                    order.status = OrderStatus.CANCELLED
                    order.error_message = "Cancelled by user"
                    return order

            logger.info(
                f"Executing {order.order_type.value} order: {order.ticker} "
                f"({order.shares} shares @ ${order.current_price:.2f})"
            )

            # Create IBKR order
            quantity = order.shares if order.order_type == OrderType.BUY else -order.shares
            ibkr_order = await self.ibkr.create_stock_order(
                symbol=order.ticker,
                quantity=quantity,
                order_type="MKT",  # Use market orders for now
            )

            # Add order reference
            order.order_ref = f"{order.ticker}_{order.order_type.value}_{datetime.now(UTC).strftime('%H%M%S')}"

            # Submit order to IBKR
            trade = await self.ibkr.submit_order(order.ticker, ibkr_order)
            order.ibkr_order_id = trade.order.orderId
            order.ibkr_trade_id = str(trade.contract.conId) if hasattr(trade, 'contract') else None

            # Track active order
            self.active_orders[order.ticker] = order

            # Log execution attempt
            self.execution_log.append({
                "timestamp": datetime.now(UTC),
                "action": "order_submitted",
                "ticker": order.ticker,
                "order_type": order.order_type.value,
                "shares": order.shares,
                "ibkr_order_id": order.ibkr_order_id,
            })

            logger.success(
                f"Order submitted: {order.ticker} (IBKR ID: {order.ibkr_order_id})"
            )

            return order

        except Exception as e:
            logger.error(f"Failed to execute order {order.ticker}: {e}")
            order.status = OrderStatus.FAILED
            order.error_message = str(e)

            self.execution_log.append({
                "timestamp": datetime.now(UTC),
                "action": "order_failed",
                "ticker": order.ticker,
                "error": str(e),
            })

            raise ExecutionError(f"Failed to execute order {order.ticker}: {e}") from e

    async def _wait_for_order_completion(
        self, orders: list[RebalancingOrder], timeout: int = 300
    ) -> None:
        """
        Wait for orders to complete execution.

        Args:
            orders: Orders to monitor
            timeout: Maximum wait time in seconds
        """
        start_time = datetime.now(UTC)
        pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]

        logger.info(f"Waiting for {len(pending_orders)} orders to complete...")

        while pending_orders and (datetime.now(UTC) - start_time).total_seconds() < timeout:
            # Check order status updates
            for order in pending_orders[:]:  # Create copy for safe iteration
                try:
                    # Get updated trade status from IBKR
                    open_trades = await self.ibkr.get_open_orders()

                    # Find matching trade
                    matching_trade = None
                    for trade in open_trades:
                        if trade.order.orderId == order.ibkr_order_id:
                            matching_trade = trade
                            break

                    if matching_trade:
                        # Update order status from trade
                        order.update_from_ibkr_trade(matching_trade)

                        if order.status in [OrderStatus.EXECUTED, OrderStatus.FAILED, OrderStatus.CANCELLED]:
                            pending_orders.remove(order)

                            self.execution_log.append({
                                "timestamp": datetime.now(UTC),
                                "action": "order_completed",
                                "ticker": order.ticker,
                                "status": order.status.value,
                                "filled_quantity": order.filled_quantity,
                                "avg_fill_price": order.avg_fill_price,
                            })

                            logger.info(
                                f"Order completed: {order.ticker} - {order.status.value} "
                                f"({order.filled_quantity}/{order.shares} filled)"
                            )

                except Exception as e:
                    logger.error(f"Error checking order status for {order.ticker}: {e}")

            if pending_orders:
                await asyncio.sleep(2)  # Wait 2 seconds before next check

        # Handle timeout
        if pending_orders:
            logger.warning(f"{len(pending_orders)} orders did not complete within timeout")
            for order in pending_orders:
                order.status = OrderStatus.FAILED
                order.error_message = "Order timeout"

    async def _simulate_execution(
        self, orders: list[RebalancingOrder]
    ) -> tuple[list[RebalancingOrder], dict]:
        """
        Simulate order execution for dry run mode.

        Args:
            orders: Orders to simulate

        Returns:
            Tuple of (simulated_orders, execution_summary)
        """
        logger.info("Simulating order execution (dry run mode)")

        for order in orders:
            # Simulate successful execution
            order.status = OrderStatus.EXECUTED
            order.execution_price = order.current_price
            order.execution_time = datetime.now(UTC)
            order.filled_quantity = order.shares
            order.avg_fill_price = order.current_price
            order.commission = 1.0  # Simulate $1 commission

        execution_summary = {
            "start_time": datetime.now(UTC),
            "end_time": datetime.now(UTC),
            "total_orders": len(orders),
            "successful": len(orders),
            "failed": 0,
            "partially_filled": 0,
            "total_value_traded": sum(o.order_value for o in orders),
            "total_commission": len(orders) * 1.0,
            "simulation": True,
        }

        logger.success(f"Simulated execution of {len(orders)} orders")
        return orders, execution_summary

    def get_active_orders(self) -> dict[str, RebalancingOrder]:
        """Get currently active orders."""
        return self.active_orders.copy()

    def get_execution_log(self) -> list[dict]:
        """Get execution log."""
        return self.execution_log.copy()

    async def cancel_order(self, ticker: str) -> bool:
        """
        Cancel an active order.

        Args:
            ticker: Ticker symbol of order to cancel

        Returns:
            True if cancellation successful
        """
        if ticker not in self.active_orders:
            logger.warning(f"No active order found for {ticker}")
            return False

        order = self.active_orders[ticker]

        try:
            # Find the trade to cancel
            open_trades = await self.ibkr.get_open_orders()
            matching_trade = None

            for trade in open_trades:
                if trade.order.orderId == order.ibkr_order_id:
                    matching_trade = trade
                    break

            if matching_trade:
                success = await self.ibkr.cancel_order(matching_trade)
                if success:
                    order.status = OrderStatus.CANCELLED
                    del self.active_orders[ticker]

                    self.execution_log.append({
                        "timestamp": datetime.now(UTC),
                        "action": "order_cancelled",
                        "ticker": ticker,
                        "ibkr_order_id": order.ibkr_order_id,
                    })

                    logger.info(f"Successfully cancelled order for {ticker}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to cancel order for {ticker}: {e}")
            return False

    def clear_completed_orders(self) -> int:
        """
        Clear completed orders from active tracking.

        Returns:
            Number of orders cleared
        """
        completed_tickers = []

        for ticker, order in self.active_orders.items():
            if order.status in [OrderStatus.EXECUTED, OrderStatus.FAILED, OrderStatus.CANCELLED]:
                completed_tickers.append(ticker)

        for ticker in completed_tickers:
            del self.active_orders[ticker]

        logger.info(f"Cleared {len(completed_tickers)} completed orders")
        return len(completed_tickers)
