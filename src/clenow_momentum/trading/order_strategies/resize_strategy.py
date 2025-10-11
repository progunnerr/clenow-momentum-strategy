"""Resize strategy for generating orders to adjust position sizes.

This strategy handles generating SELL orders for positions that need to be
reduced in size (but not completely exited) to match target allocations.
"""

from typing import TYPE_CHECKING

from loguru import logger

from .base import OrderContext, OrderStrategy

if TYPE_CHECKING:
    from ...domain import RebalancingOrder


class ResizeStrategy(OrderStrategy):
    """Strategy for generating resize orders to adjust position sizes.

    This strategy identifies positions that exist in both current and target
    portfolios but need to be reduced in size, and generates SELL orders
    for the excess shares.
    """

    @property
    def priority(self) -> int:
        """Resize orders have medium priority (after exits, before entries)."""
        return 2

    @property
    def description(self) -> str:
        """Human-readable description of this strategy."""
        return "Generate SELL orders to reduce position sizes to target allocations"

    def generate_orders(self, context: OrderContext) -> list["RebalancingOrder"]:
        """Generate resize orders for positions that need to be reduced.

        Args:
            context: OrderContext with current and target portfolios

        Returns:
            List of SELL orders for position size reductions
        """

        orders = []
        current_tickers = context.get_current_tickers()
        target_tickers = context.get_target_tickers()

        # Find positions that exist in both portfolios (candidates for resizing)
        common_tickers = current_tickers & target_tickers

        if not common_tickers:
            logger.debug("No common positions to potentially resize")
            return orders

        resize_count = 0

        for ticker in common_tickers:
            try:
                order = self._create_resize_order(ticker, context)
                if order is not None:  # Only add if resizing is needed
                    orders.append(order)
                    resize_count += 1
                    logger.debug(f"RESIZE order: {ticker} - {order.shares} shares")

            except Exception as e:
                logger.warning(f"Failed to create resize order for {ticker}: {e}")
                continue

        if resize_count > 0:
            logger.info(f"Generating resize orders for {resize_count} positions")
        else:
            logger.debug("No positions need resizing")

        return orders

    def _create_resize_order(self, ticker: str, context: OrderContext) -> "RebalancingOrder | None":
        """Create a SELL order to reduce a position to target size.

        Args:
            ticker: Stock ticker to potentially resize
            context: Order context with portfolio and market data

        Returns:
            RebalancingOrder for position reduction, or None if no resize needed

        Raises:
            ValueError: If position or target data is invalid
        """
        from ...strategy.rebalancing import OrderType, RebalancingOrder

        # Get current position
        position = context.current_portfolio.positions.get(ticker)
        if not position:
            raise ValueError(f"Position for {ticker} not found in current portfolio")

        if position.shares <= 0:
            raise ValueError(f"Invalid current share count for {ticker}: {position.shares}")

        # Get target information
        target_info = context.get_target_info(ticker)
        target_value = target_info.get("investment")
        if target_value is None or target_value <= 0:
            raise ValueError(f"Invalid target investment value for {ticker}: {target_value}")

        # Calculate target shares based on current price
        current_price = context.get_current_price(ticker)
        target_shares = int(target_value / current_price)

        # Check if we need to reduce the position
        if target_shares >= position.shares:
            # No reduction needed
            return None

        # Calculate shares to sell
        shares_to_sell = int(position.shares) - target_shares
        if shares_to_sell <= 0:
            return None

        # Build detailed resize reason
        resize_reason = self._build_resize_reason(
            ticker, int(position.shares), target_shares, target_value
        )

        # Calculate order value
        order_value = shares_to_sell * current_price

        return RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.SELL,
            shares=shares_to_sell,
            current_price=current_price,
            order_value=order_value,
            reason=resize_reason,
            priority=self.priority
        )

    def _build_resize_reason(
        self,
        ticker: str,
        current_shares: int,
        target_shares: int,
        target_value: float
    ) -> str:
        """Build a detailed reason for resizing this position.

        Args:
            ticker: Stock ticker being resized
            current_shares: Current number of shares
            target_shares: Target number of shares
            target_value: Target investment value

        Returns:
            Detailed resize reason string
        """
        return (
            f"Position size adjustment. Reducing from {current_shares} to {target_shares} shares "
            f"(target value: ${target_value:,.0f}) for optimal portfolio balance"
        )
