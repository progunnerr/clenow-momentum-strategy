"""Broker adapter interfaces for trading integration.

These interfaces define contracts for interacting with different brokers
(IBKR, TD Ameritrade, etc.) without coupling business logic to specific
broker implementations.

Co-located with trading module for easier navigation and maintenance.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum


class ConnectionStatus(Enum):
    """Broker connection status."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class BrokerAdapter(ABC):
    """Abstract interface for broker integrations.

    Provides standardized access to broker functionality regardless of
    the underlying broker API (IBKR, TD Ameritrade, etc.).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker.

        Returns:
            True if connection successful

        Raises:
            BrokerConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to broker.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    def get_connection_status(self) -> ConnectionStatus:
        """Get current connection status.

        Returns:
            Current connection status
        """
        pass

    @abstractmethod
    def get_account_summary(self) -> "AccountSummary":
        """Get account summary information.

        Returns:
            Account summary with cash, equity, etc.

        Raises:
            BrokerError: If account info cannot be retrieved
        """
        pass

    @abstractmethod
    def get_positions(self) -> list["BrokerPosition"]:
        """Get current portfolio positions.

        Returns:
            List of current positions

        Raises:
            BrokerError: If positions cannot be retrieved
        """
        pass

    @abstractmethod
    def place_order(self, order: "BrokerOrder") -> str:
        """Place a trading order.

        Args:
            order: Order to place

        Returns:
            Broker order ID

        Raises:
            BrokerError: If order placement fails
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Broker order ID to cancel

        Returns:
            True if cancellation successful

        Raises:
            BrokerError: If cancellation fails
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> "OrderStatusInfo":
        """Get status of a specific order.

        Args:
            order_id: Broker order ID

        Returns:
            Order status information

        Raises:
            BrokerError: If status cannot be retrieved
        """
        pass

    @abstractmethod
    def get_executions(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list["ExecutionReport"]:
        """Get trade executions.

        Args:
            start_date: Filter from this date (inclusive)
            end_date: Filter to this date (inclusive)

        Returns:
            List of execution reports

        Raises:
            BrokerError: If executions cannot be retrieved
        """
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if market is open
        """
        pass

    @abstractmethod
    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get current market prices for market universe securities.

        Args:
            tickers: List of symbols from the market universe to get prices for

        Returns:
            Dictionary mapping ticker to current price

        Raises:
            BrokerError: If prices cannot be retrieved
        """
        pass


class BrokerError(Exception):
    """Exception raised when broker operations fail."""

    def __init__(self, message: str, broker: str = "", original_error: Exception = None):
        self.message = message
        self.broker = broker
        self.original_error = original_error
        super().__init__(f"{broker}: {message}" if broker else message)


class BrokerConnectionError(BrokerError):
    """Exception raised when broker connection fails."""

    pass


# Data classes for broker integration (to be implemented in domain layer)
class AccountSummary:
    """Account summary information from broker."""
    
    def __init__(
        self,
        total_cash: float = 0.0,
        net_liquidation: float = 0.0,
        buying_power: float = 0.0,
        excess_liquidity: float = 0.0,
    ):
        self.total_cash = total_cash
        self.net_liquidation = net_liquidation
        self.buying_power = buying_power
        self.excess_liquidity = excess_liquidity


class BrokerPosition:
    """Position information from broker."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        market_value: float,
        unrealized_pnl: float = 0.0,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.market_value = market_value
        self.unrealized_pnl = unrealized_pnl


class BrokerOrder:
    """Order specification for broker."""
    
    def __init__(
        self,
        symbol: str,
        action: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MKT",  # "MKT", "LMT", etc.
        limit_price: float | None = None,
    ):
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price


class OrderStatusInfo:
    """Order status information from broker."""
    
    def __init__(
        self,
        order_id: str,
        status: OrderStatus,
        filled: float = 0.0,
        remaining: float = 0.0,
        avg_fill_price: float = 0.0,
    ):
        self.order_id = order_id
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avg_fill_price = avg_fill_price


class ExecutionReport:
    """Trade execution report from broker."""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.commission = commission