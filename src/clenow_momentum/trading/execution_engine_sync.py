"""
Synchronous trading execution engine for IBKR integration.

This module handles the execution of rebalancing orders through IBKR,
using the synchronous IBKRClient.
"""

import time
from datetime import UTC, datetime

from loguru import logger

from ..data_sources import IBKRClient
from ..strategy.rebalancing import OrderStatus, OrderType, Portfolio, RebalancingOrder


class ExecutionError(Exception):
    """Base exception for execution engine errors."""
    pass


class SyncTradingExecutionEngine:
    """
    Handles synchronous execution of trading orders through IBKR.
    
    This engine:
    - Executes rebalancing orders in priority order
    - Updates portfolio state with execution results
    - Provides execution reporting and logging
    """

    def __init__(self, ibkr_client: IBKRClient):
        """
        Initialize synchronous trading execution engine.
        
        Args:
            ibkr_client: Connected IBKR client
        """
        self.ibkr = ibkr_client
        self.active_orders: dict[str, RebalancingOrder] = {}  # ticker -> order
        self.execution_log: list[dict] = []

        # Confirmation settings
        self.skip_confirmations = False  # Can be set to skip all confirmations

        logger.info("Synchronous trading execution engine initialized")

    def execute_rebalancing_orders(
        self,
        orders: list[RebalancingOrder],
        portfolio: Portfolio,
        dry_run: bool = False,
        force_execution: bool = False,
    ) -> tuple[list[RebalancingOrder], dict]:
        """
        Execute a list of rebalancing orders.
        
        Args:
            orders: List of orders to execute
            portfolio: Current portfolio to update
            dry_run: If True, simulate execution without placing real orders
            force_execution: If True, skip all confirmation prompts
            
        Returns:
            Tuple of (executed_orders, execution_summary)
        """
        # Set skip confirmations based on force flag
        self.skip_confirmations = force_execution

        logger.info(f"Starting execution of {len(orders)} orders (dry_run={dry_run}, force={force_execution})")

        if dry_run:
            return self._simulate_execution(orders)

        # Sort orders by priority (sell orders first, then buy orders)
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
                for order in sell_orders:
                    result = self._execute_single_order(order)
                    if result:
                        executed_orders.append(result)

                # Brief pause after sells to let cash settle
                time.sleep(2)

            # Execute BUY orders (priority 3-4)
            buy_orders = [o for o in sorted_orders if o.order_type == OrderType.BUY]
            if buy_orders:
                logger.info(f"Executing {len(buy_orders)} BUY orders...")
                for i, order in enumerate(buy_orders, 1):
                    logger.info(f"Processing BUY order {i}/{len(buy_orders)}: {order.ticker}")
                    result = self._execute_single_order(order)
                    if result:
                        executed_orders.append(result)

                    # Check how many orders are actually in TWS
                    if hasattr(self.ibkr, 'get_open_orders'):
                        open_orders = self.ibkr.get_open_orders()
                        logger.info(f"Open orders in TWS after {i} submissions: {len(open_orders)}")

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

    def _execute_single_order(self, order: RebalancingOrder) -> RebalancingOrder | None:
        """
        Execute a single order through IBKR.
        
        Args:
            order: Order to execute
            
        Returns:
            Executed order with updated status, or None if failed
        """
        try:
            # Get confirmation if not skipping
            if not self.skip_confirmations:
                confirmed = self._get_order_confirmation(order)
                if not confirmed:
                    logger.info(f"User skipped order for {order.ticker}")
                    order.status = OrderStatus.CANCELLED
                    return order
                logger.info(f"User confirmed order for {order.ticker}")

            logger.info(f"Executing {order.order_type.value} order for {order.ticker}: {order.shares} shares @ ${order.current_price:.2f}")

            # Place the order through IBKR
            logger.info(f"Submitting {order.order_type.value} order to IBKR: {order.ticker} x {order.shares}")

            try:
                if order.order_type == OrderType.BUY:
                    trade = self.ibkr.buy(order.ticker, order.shares)
                else:
                    trade = self.ibkr.sell(order.ticker, order.shares)

                # Check if we got a trade object back
                if trade:
                    order_id = trade.order.orderId if trade.order else None
                    logger.info(f"IBKR returned trade object with order ID: {order_id}")

                    # Give IBKR a moment to process
                    time.sleep(0.5)
                else:
                    logger.warning(f"No trade object returned from IBKR for {order.ticker}")

                # Update order status
                order.status = OrderStatus.EXECUTED
                order.executed_at = datetime.now(UTC)
                order.executed_price = order.current_price  # Will be updated with actual fill price later

                # Log execution
                self.execution_log.append({
                    "ticker": order.ticker,
                    "order_type": order.order_type.value,
                    "shares": order.shares,
                    "price": order.current_price,
                    "value": order.order_value,
                    "time": order.executed_at,
                    "trade_id": trade.order.orderId if trade and trade.order else None
                })

                logger.success(f"✅ {order.order_type.value} order executed for {order.ticker} (Order ID: {trade.order.orderId if trade and trade.order else 'Unknown'})")
                return order

            except Exception as e:
                logger.error(f"IBKR order submission failed for {order.ticker}: {e}")
                raise

        except Exception as e:
            logger.error(f"Failed to execute order for {order.ticker}: {e}")
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            return order

    def _get_order_confirmation(self, order: RebalancingOrder) -> bool:
        """
        Display order details and get user confirmation.
        
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

        print("\n" + "-"*60)

        # Get user confirmation
        while True:
            print("\n⚡ Execute this order?")
            print("  [Y] Yes - Execute order")
            print("  [N] No - Skip this order")
            print("  [A] Auto - Execute all remaining orders without confirmation")
            print("  [C] Cancel - Stop all order execution")

            response = input("\nYour choice (Y/N/A/C): ").strip().upper()

            if response == 'Y':
                print("✅ Order confirmed")
                return True
            if response == 'N':
                print("⏭️ Order skipped")
                return False
            if response == 'A':
                print("🚀 Executing all remaining orders without confirmation")
                self.skip_confirmations = True
                return True
            if response == 'C':
                print("🛑 Cancelling all order execution")
                raise ExecutionError("User cancelled all orders")
            print("❌ Invalid input. Please enter Y, N, A, or C.")

    def _simulate_execution(self, orders: list[RebalancingOrder]) -> tuple[list[RebalancingOrder], dict]:
        """
        Simulate order execution for dry run mode.
        
        Args:
            orders: Orders to simulate
            
        Returns:
            Tuple of (simulated_orders, execution_summary)
        """
        logger.info("DRY RUN MODE - Simulating order execution...")

        executed_orders = []
        execution_summary = {
            "start_time": datetime.now(UTC),
            "total_orders": len(orders),
            "successful": 0,
            "failed": 0,
            "partially_filled": 0,
            "total_value_traded": 0.0,
            "total_commission": 0.0,
            "dry_run": True
        }

        for order in orders:
            # Simulate getting confirmation
            if not self.skip_confirmations:
                print(f"\n[DRY RUN] Would execute {order.order_type.value} order:")
                print(f"  Symbol: {order.ticker}")
                print(f"  Shares: {order.shares:,}")
                print(f"  Price: ${order.current_price:.2f}")
                print(f"  Value: ${order.order_value:,.2f}")

            # Mark as simulated
            order.status = OrderStatus.EXECUTED
            order.executed_at = datetime.now(UTC)
            order.executed_price = order.current_price
            order.commission = order.order_value * 0.001  # Simulate 0.1% commission

            executed_orders.append(order)
            execution_summary["successful"] += 1
            execution_summary["total_value_traded"] += order.order_value
            execution_summary["total_commission"] += order.commission

        execution_summary["end_time"] = datetime.now(UTC)
        execution_summary["duration_minutes"] = (
            execution_summary["end_time"] - execution_summary["start_time"]
        ).total_seconds() / 60

        print(f"\n[DRY RUN] Simulated {len(executed_orders)} orders successfully")

        return executed_orders, execution_summary

    def get_active_orders(self) -> list[str]:
        """Get list of tickers with active orders."""
        return list(self.active_orders.keys())

    def cancel_order(self, ticker: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            ticker: Ticker symbol to cancel
            
        Returns:
            True if cancelled successfully
        """
        if ticker in self.active_orders:
            logger.info(f"Cancelling order for {ticker}")
            # In a real implementation, would call IBKR to cancel
            # For now, just remove from tracking
            del self.active_orders[ticker]
            return True
        return False
