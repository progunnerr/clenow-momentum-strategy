"""Broker adapter interfaces for trading integration.

These interfaces define contracts for interacting with different brokers
(IBKR, TD Ameritrade, etc.) without coupling business logic to specific
broker implementations.
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
        """Get current market prices for tickers.

        Args:
            tickers: List of symbols to get prices for

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

    pass


class BrokerPosition:
    """Position information from broker."""

    pass


class BrokerOrder:
    """Order to be sent to broker."""

    pass


class OrderStatusInfo:
    """Order status information from broker."""

    pass


class ExecutionReport:
    """Trade execution report from broker."""

    pass
