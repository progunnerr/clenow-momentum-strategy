"""Exit strategy for generating orders to completely sell positions.

This strategy handles generating SELL orders for positions that need to be
completely exited from the portfolio (not in target portfolio anymore).
"""

from typing import TYPE_CHECKING

from loguru import logger

from .base import OrderContext, OrderStrategy

if TYPE_CHECKING:
    from ...domain import RebalancingOrder


class ExitStrategy(OrderStrategy):
    """Strategy for generating exit orders to completely sell positions.

    This strategy identifies positions that are in the current portfolio
    but not in the target portfolio, and generates SELL orders to exit
    them completely.
    """

    @property
    def priority(self) -> int:
        """Exit orders have highest priority (executed first)."""
        return 1

    @property
    def description(self) -> str:
        """Human-readable description of this strategy."""
        return "Generate SELL orders to completely exit positions no longer in target portfolio"

    def generate_orders(self, context: OrderContext) -> list["RebalancingOrder"]:
        """Generate exit orders for positions to be completely sold.

        Args:
            context: OrderContext with current and target portfolios

        Returns:
            List of SELL orders for complete position exits
        """

        orders = []
        current_tickers = context.get_current_tickers()
        target_tickers = context.get_target_tickers()

        # Find positions to completely exit
        tickers_to_exit = current_tickers - target_tickers

        if not tickers_to_exit:
            logger.debug("No positions to exit")
            return orders

        logger.info(f"Generating exit orders for {len(tickers_to_exit)} positions")

        for ticker in tickers_to_exit:
            try:
                order = self._create_exit_order(ticker, context)
                orders.append(order)
                logger.debug(f"EXIT order: {ticker} - {order.shares} shares")

            except Exception as e:
                logger.warning(f"Failed to create exit order for {ticker}: {e}")
                continue

        return orders

    def _create_exit_order(self, ticker: str, context: OrderContext) -> "RebalancingOrder":
        """Create a SELL order to completely exit a position.

        Args:
            ticker: Stock ticker to exit
            context: Order context with portfolio and market data

        Returns:
            RebalancingOrder for complete position exit

        Raises:
            ValueError: If position data is invalid or incomplete
        """
        from ...strategy.rebalancing import OrderType, RebalancingOrder

        # Get current position
        position = context.current_portfolio.positions.get(ticker)
        if not position:
            raise ValueError(f"Position for {ticker} not found in current portfolio")

        if position.shares <= 0:
            raise ValueError(f"Invalid share count for {ticker}: {position.shares}")

        # Get current price
        current_price = context.get_current_price(ticker)

        # Build detailed exit reason
        exit_reason = self._build_exit_reason(ticker, context)

        # Calculate order value
        order_value = position.shares * current_price

        return RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.SELL,
            shares=int(position.shares),
            current_price=current_price,
            order_value=order_value,
            reason=exit_reason,
            priority=self.priority
        )

    def _build_exit_reason(self, ticker: str, context: OrderContext) -> str:
        """Build a detailed reason for exiting this position.

        Args:
            ticker: Stock ticker being exited
            context: Order context for additional analysis

        Returns:
            Detailed exit reason string
        """
        reasons = []

        # Check if we can analyze why it was dropped
        if not context.stock_data.empty:
            try:
                if ticker in context.stock_data.columns.get_level_values(1):
                    ticker_data = context.stock_data.xs(ticker, level=1, axis=1)
                    if not ticker_data.empty:
                        reasons.append("Position no longer meets momentum criteria")
                    else:
                        reasons.append("No current market data available")
                else:
                    reasons.append("Stock data not available for analysis")
            except (KeyError, IndexError):
                reasons.append("Unable to analyze current market conditions")
        else:
            reasons.append("Not in target portfolio")

        # Add standard exit reason
        reasons.append("Full position exit required for rebalancing")

        return ". ".join(reasons)
