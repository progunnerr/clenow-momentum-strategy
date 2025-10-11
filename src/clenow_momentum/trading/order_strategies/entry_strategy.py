"""Entry strategy for generating orders to create new positions.

This strategy handles generating BUY orders for new positions that are
in the target portfolio but not in the current portfolio.
"""

from typing import TYPE_CHECKING

from loguru import logger

from .base import OrderContext, OrderStrategy

if TYPE_CHECKING:
    from ...domain import RebalancingOrder


class EntryStrategy(OrderStrategy):
    """Strategy for generating entry orders to create new positions.

    This strategy identifies stocks that are in the target portfolio
    but not currently held, and generates BUY orders to establish
    new positions (subject to available cash).
    """

    @property
    def priority(self) -> int:
        """Entry orders have lowest priority (executed last)."""
        return 3

    @property
    def description(self) -> str:
        """Human-readable description of this strategy."""
        return "Generate BUY orders to establish new positions from target portfolio"

    def generate_orders(self, context: OrderContext) -> list["RebalancingOrder"]:
        """Generate entry orders for new positions to be created.

        Orders are generated in the order they appear in the target portfolio
        (which should be momentum-ranked) until cash runs out.

        Args:
            context: OrderContext with current and target portfolios

        Returns:
            List of BUY orders for new position entries
        """

        orders = []
        current_tickers = context.get_current_tickers()
        target_tickers = context.get_target_tickers()

        # Find new positions to create
        new_tickers = target_tickers - current_tickers

        if not new_tickers:
            logger.debug("No new positions to create")
            return orders

        # Track available cash as we generate orders
        available_cash = context.available_cash
        skipped_due_to_cash = []
        entry_count = 0

        # Process target portfolio in order (should be momentum-ranked)
        for _, target_row in context.target_portfolio.iterrows():
            ticker = target_row["ticker"]

            # Skip if not a new position
            if ticker not in new_tickers:
                continue

            try:
                # Check if we have enough cash for this position
                investment_needed = target_row.get("investment", 0)

                if investment_needed > available_cash:
                    skipped_due_to_cash.append({
                        "ticker": ticker,
                        "needed": investment_needed,
                        "available": available_cash,
                        "rank": target_row.get("portfolio_rank", target_row.get("momentum_rank", "N/A"))
                    })
                    logger.debug(
                        f"Insufficient cash for {ticker}: need ${investment_needed:,.2f}, "
                        f"have ${available_cash:,.2f}"
                    )
                    continue

                # Create the entry order
                order = self._create_entry_order(ticker, target_row, context)
                orders.append(order)
                entry_count += 1

                # Update available cash
                available_cash -= order.order_value

                logger.debug(f"ENTRY order: {ticker} - {order.shares} shares (${order.order_value:,.0f})")

            except Exception as e:
                logger.warning(f"Failed to create entry order for {ticker}: {e}")
                continue

        # Log results
        if entry_count > 0:
            logger.info(f"Generating entry orders for {entry_count} new positions")

        if skipped_due_to_cash:
            logger.info(f"Skipped {len(skipped_due_to_cash)} positions due to insufficient cash")
            for skipped in skipped_due_to_cash[:3]:  # Show first 3
                logger.debug(
                    f"Skipped {skipped['ticker']} (rank #{skipped['rank']}): "
                    f"needed ${skipped['needed']:,.0f}"
                )

        return orders

    def _create_entry_order(self, ticker: str, target_row, context: OrderContext) -> "RebalancingOrder":
        """Create a BUY order to establish a new position.

        Args:
            ticker: Stock ticker for new position
            target_row: Row from target portfolio with position details
            context: Order context with market data

        Returns:
            RebalancingOrder for new position entry

        Raises:
            ValueError: If target data is invalid or incomplete
        """
        from ...strategy.rebalancing import OrderType, RebalancingOrder

        # Get target investment amount and shares
        investment_amount = target_row.get("investment")
        target_shares = target_row.get("shares")

        if investment_amount is None or investment_amount <= 0:
            raise ValueError(f"Invalid investment amount for {ticker}: {investment_amount}")

        if target_shares is None or target_shares <= 0:
            raise ValueError(f"Invalid target shares for {ticker}: {target_shares}")

        # Get current price
        current_price = target_row.get("current_price")
        if current_price is None or current_price <= 0:
            # Fallback to context price lookup
            current_price = context.get_current_price(ticker)

        # Build detailed entry reason
        entry_reason = self._build_entry_reason(ticker, target_row)

        return RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.BUY,
            shares=int(target_shares),
            current_price=current_price,
            order_value=investment_amount,
            reason=entry_reason,
            priority=self.priority
        )

    def _build_entry_reason(self, ticker: str, target_row) -> str:
        """Build a detailed reason for entering this position.

        Args:
            ticker: Stock ticker for new position
            target_row: Row from target portfolio with metrics

        Returns:
            Detailed entry reason string
        """
        reasons = []

        # Add momentum metrics if available
        momentum_score = target_row.get("momentum_score")
        if momentum_score is not None:
            reasons.append(f"Momentum score: {momentum_score:.3f}")

        # Add ranking information
        rank = target_row.get("portfolio_rank", target_row.get("momentum_rank"))
        if rank is not None:
            reasons.append(f"Rank #{rank}")

        # Add standard entry reason
        reasons.append("New position entry based on strong momentum signal")

        return ". ".join(reasons) if reasons else "New position - entering based on momentum signal"
