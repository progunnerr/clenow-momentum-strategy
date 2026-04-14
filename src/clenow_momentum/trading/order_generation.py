"""Order generation service for portfolio rebalancing.

This module provides the main interface for generating trading orders
by coordinating all order strategies (exit, adjust, entry).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from ..data.interfaces import MarketDataSource
from .order_strategies import AdjustStrategy, EntryStrategy, ExitStrategy, OrderContext

if TYPE_CHECKING:
    from ..domain import RebalancingOrder


@dataclass
class OrderGenerationResult:
    """Result of order generation including orders and metadata."""
    orders: list["RebalancingOrder"]
    exit_orders: list["RebalancingOrder"]
    adjust_orders: list["RebalancingOrder"]
    entry_orders: list["RebalancingOrder"]
    total_order_value: Decimal
    cash_after_orders: Decimal
    skipped_positions: int
    generation_summary: str


class OrderGenerationService:
    """Service for generating trading orders for portfolio rebalancing.

    This service coordinates multiple order strategies to generate
    a complete set of rebalancing orders:

    1. Exit orders - close positions not in target portfolio
    2. Adjust orders - modify existing positions to match target weights
    3. Entry orders - create new positions from target portfolio

    The strategies are executed in priority order to ensure proper
    cash flow management.
    """

    def __init__(self, market_data_provider: MarketDataSource):
        """Initialize the order generation service.

        Args:
            market_data_provider: Service for getting current market prices
        """
        self.market_data_provider = market_data_provider

        # Initialize order strategies
        self.exit_strategy = ExitStrategy()
        self.adjust_strategy = AdjustStrategy()
        self.entry_strategy = EntryStrategy()

        # All strategies in priority order
        self.strategies = [
            self.exit_strategy,
            self.adjust_strategy,
            self.entry_strategy
        ]

        logger.debug("Order generation service initialized")

    def generate_orders(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        available_cash: Decimal,
        cash_buffer: Decimal = Decimal("1000")
    ) -> OrderGenerationResult:
        """Generate all rebalancing orders using coordinated strategies.

        Args:
            current_portfolio: Current portfolio positions with columns:
                - ticker: stock symbol
                - shares: current shares owned
                - current_price: current market price
                - market_value: current position value
            target_portfolio: Target portfolio with columns:
                - ticker: stock symbol
                - shares: target shares to own
                - investment: target dollar investment
                - current_price: current market price
                - momentum_score: momentum ranking score
            available_cash: Cash available for trading
            cash_buffer: Minimum cash to maintain (default $1000)

        Returns:
            OrderGenerationResult with all generated orders and metadata

        Raises:
            ValueError: If portfolio data is invalid or inconsistent
        """
        logger.info("Starting order generation for portfolio rebalancing")

        # Validate inputs
        self._validate_inputs(current_portfolio, target_portfolio, available_cash)

        # Adjust available cash for buffer
        effective_cash = max(Decimal("0"), available_cash - cash_buffer)

        # Create order context
        context = OrderContext(
            current_portfolio=current_portfolio,
            target_portfolio=target_portfolio,
            available_cash=effective_cash,
            market_data_provider=self.market_data_provider
        )

        # Generate orders using each strategy in priority order
        all_orders = []
        exit_orders = []
        adjust_orders = []
        entry_orders = []

        for strategy in self.strategies:
            try:
                logger.debug(f"Executing strategy: {strategy.__class__.__name__}")
                strategy_orders = strategy.generate_orders(context)

                # Track orders by strategy type
                if isinstance(strategy, ExitStrategy):
                    exit_orders.extend(strategy_orders)
                elif isinstance(strategy, AdjustStrategy):
                    adjust_orders.extend(strategy_orders)
                elif isinstance(strategy, EntryStrategy):
                    entry_orders.extend(strategy_orders)

                all_orders.extend(strategy_orders)

                # Update context cash for next strategy
                context = self._update_context_cash(context, strategy_orders)

                logger.debug(f"Generated {len(strategy_orders)} orders from {strategy.__class__.__name__}")

            except Exception as e:
                logger.error(f"Error in {strategy.__class__.__name__}: {e}")
                # Continue with other strategies rather than failing completely
                continue

        # Sort all orders by priority and calculate totals
        all_orders.sort(key=lambda x: x.priority)
        total_order_value = sum(abs(order.order_value) for order in all_orders)
        cash_after_orders = context.available_cash

        # Build summary
        summary = self._build_generation_summary(
            exit_orders, adjust_orders, entry_orders, total_order_value
        )

        logger.info(f"Order generation complete: {len(all_orders)} total orders")
        logger.info(summary)

        return OrderGenerationResult(
            orders=all_orders,
            exit_orders=exit_orders,
            adjust_orders=adjust_orders,
            entry_orders=entry_orders,
            total_order_value=Decimal(str(total_order_value)),
            cash_after_orders=Decimal(str(cash_after_orders)),
            skipped_positions=0,  # TODO: Track this across strategies
            generation_summary=summary
        )

    def _validate_inputs(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        available_cash: Decimal
    ) -> None:
        """Validate input parameters for order generation.

        Args:
            current_portfolio: Current portfolio DataFrame
            target_portfolio: Target portfolio DataFrame
            available_cash: Available cash amount

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate current portfolio
        if current_portfolio.empty:
            logger.warning("Current portfolio is empty - generating entry orders only")
        else:
            required_current_cols = ["ticker", "shares", "current_price", "market_value"]
            missing_cols = [col for col in required_current_cols if col not in current_portfolio.columns]
            if missing_cols:
                raise ValueError(f"Current portfolio missing required columns: {missing_cols}")

        # Validate target portfolio
        if target_portfolio.empty:
            raise ValueError("Target portfolio cannot be empty")

        required_target_cols = ["ticker", "shares", "investment", "current_price"]
        missing_cols = [col for col in required_target_cols if col not in target_portfolio.columns]
        if missing_cols:
            raise ValueError(f"Target portfolio missing required columns: {missing_cols}")

        # Validate cash
        if available_cash < 0:
            raise ValueError(f"Available cash cannot be negative: {available_cash}")

        logger.debug("Input validation passed")

    def _update_context_cash(
        self,
        context: OrderContext,
        new_orders: list["RebalancingOrder"]
    ) -> OrderContext:
        """Update order context with cash changes from new orders.

        Args:
            context: Current order context
            new_orders: Orders that will affect available cash

        Returns:
            Updated OrderContext with new cash balance
        """
        # Calculate net cash change (SELL orders add cash, BUY orders subtract)
        from ..strategy.rebalancing import OrderType

        cash_change = Decimal("0")
        for order in new_orders:
            if order.order_type == OrderType.SELL:
                cash_change += Decimal(str(order.order_value))
            else:  # BUY
                cash_change -= Decimal(str(order.order_value))

        new_available_cash = context.available_cash + cash_change

        # Create new context with updated cash
        return OrderContext(
            current_portfolio=context.current_portfolio,
            target_portfolio=context.target_portfolio,
            available_cash=new_available_cash,
            market_data_provider=context.market_data_provider
        )

    def _build_generation_summary(
        self,
        exit_orders: list["RebalancingOrder"],
        adjust_orders: list["RebalancingOrder"],
        entry_orders: list["RebalancingOrder"],
        total_order_value: float
    ) -> str:
        """Build a human-readable summary of order generation.

        Args:
            exit_orders: List of exit orders
            adjust_orders: List of adjustment orders
            entry_orders: List of entry orders
            total_order_value: Total value of all orders

        Returns:
            Formatted summary string
        """
        from ..strategy.rebalancing import OrderType

        summary_parts = []

        # Exit orders summary
        if exit_orders:
            exit_value = sum(order.order_value for order in exit_orders)
            summary_parts.append(f"{len(exit_orders)} exit orders (${exit_value:,.0f})")

        # Adjust orders summary
        if adjust_orders:
            buys = [o for o in adjust_orders if o.order_type == OrderType.BUY]
            sells = [o for o in adjust_orders if o.order_type == OrderType.SELL]

            if buys and sells:
                buy_value = sum(order.order_value for order in buys)
                sell_value = sum(order.order_value for order in sells)
                summary_parts.append(
                    f"{len(adjust_orders)} adjust orders "
                    f"({len(buys)} buys ${buy_value:,.0f}, {len(sells)} sells ${sell_value:,.0f})"
                )
            elif buys:
                buy_value = sum(order.order_value for order in buys)
                summary_parts.append(f"{len(adjust_orders)} adjust buy orders (${buy_value:,.0f})")
            elif sells:
                sell_value = sum(order.order_value for order in sells)
                summary_parts.append(f"{len(adjust_orders)} adjust sell orders (${sell_value:,.0f})")

        # Entry orders summary
        if entry_orders:
            entry_value = sum(order.order_value for order in entry_orders)
            summary_parts.append(f"{len(entry_orders)} entry orders (${entry_value:,.0f})")

        if not summary_parts:
            return "No orders generated"

        return f"Generated: {', '.join(summary_parts)}. Total order value: ${total_order_value:,.0f}"
