"""Base classes and context for order generation strategies.

This module defines the abstract OrderStrategy interface and the OrderContext
value object that carries all necessary information for order generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

from ...data.interfaces import MarketDataSource

if TYPE_CHECKING:
    from ...domain import RebalancingOrder


@dataclass(frozen=True)
class OrderContext:
    """Immutable context containing all data needed for order generation.

    This value object encapsulates all the parameters that order strategies need,
    following the principle of passing complete context rather than individual parameters.
    """

    current_portfolio: pd.DataFrame
    target_portfolio: pd.DataFrame
    available_cash: Decimal
    market_data_provider: MarketDataSource

    def get_current_tickers(self) -> set[str]:
        """Get set of tickers currently in portfolio."""
        if self.current_portfolio.empty:
            return set()
        return set(self.current_portfolio["ticker"].values)

    def get_target_tickers(self) -> set[str]:
        """Get set of tickers in target portfolio."""
        if self.target_portfolio.empty:
            return set()
        return set(self.target_portfolio["ticker"].values)

    def get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker from market data provider.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current stock price

        Raises:
            ValueError: If price cannot be determined
        """
        # First try to get price from current portfolio
        if not self.current_portfolio.empty:
            current_rows = self.current_portfolio[self.current_portfolio["ticker"] == ticker]
            if not current_rows.empty and "current_price" in current_rows.columns:
                price = current_rows.iloc[0]["current_price"]
                if price and price > 0:
                    return float(price)

        # Try target portfolio
        if not self.target_portfolio.empty:
            target_rows = self.target_portfolio[self.target_portfolio["ticker"] == ticker]
            if not target_rows.empty and "current_price" in target_rows.columns:
                price = target_rows.iloc[0]["current_price"]
                if price and price > 0:
                    return float(price)

        # Fallback to market data provider
        try:
            price = self.market_data_provider.get_current_price(ticker)
            if price and price > 0:
                return float(price)
        except Exception:
            pass

        raise ValueError(f"Cannot determine current price for {ticker}")

    def get_current_position_info(self, ticker: str) -> dict:
        """Get current portfolio information for a ticker."""
        if self.current_portfolio.empty:
            raise ValueError(f"Current portfolio is empty - {ticker} not found")

        current_rows = self.current_portfolio[self.current_portfolio["ticker"] == ticker]
        if current_rows.empty:
            raise ValueError(f"Ticker {ticker} not found in current portfolio")

        return current_rows.iloc[0].to_dict()

    def get_target_info(self, ticker: str) -> dict:
        """Get target portfolio information for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with target information

        Raises:
            ValueError: If ticker not in target portfolio
        """
        if self.target_portfolio.empty:
            raise ValueError(f"Target portfolio is empty - {ticker} not found")

        target_rows = self.target_portfolio[self.target_portfolio["ticker"] == ticker]
        if target_rows.empty:
            raise ValueError(f"Ticker {ticker} not found in target portfolio")

        return target_rows.iloc[0].to_dict()


class OrderStrategy(ABC):
    """Abstract base class for order generation strategies.

    Each concrete strategy implements a specific type of order generation
    (exits, position adjustments, new entries) following the Strategy pattern.
    """

    @abstractmethod
    def generate_orders(self, context: OrderContext) -> list["RebalancingOrder"]:
        """Generate orders based on the given context.

        Args:
            context: OrderContext with all necessary data

        Returns:
            List of rebalancing orders to execute
        """
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority level for this strategy's orders.

        Lower numbers indicate higher priority (executed first).
        Typical priorities:
        - 1: Exit orders (sell everything)
        - 2: Resize orders (partial sells)
        - 3: Entry orders (new buys)

        Returns:
            Priority level as integer
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy.

        Returns:
            Strategy description
        """
        pass
