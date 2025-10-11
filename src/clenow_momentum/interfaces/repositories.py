"""Repository interfaces for data persistence.

These interfaces define contracts for persisting and retrieving domain objects
without coupling business logic to specific storage implementations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..domain import Portfolio, RebalancingOrder


class PortfolioRepository(ABC):
    """Abstract interface for portfolio persistence.

    Handles saving and loading portfolio state without coupling
    to specific storage mechanisms (files, databases, etc.).
    """

    @abstractmethod
    def save_portfolio(self, portfolio: "Portfolio") -> bool:
        """Save portfolio state.

        Args:
            portfolio: Portfolio domain object to save

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def load_portfolio(self) -> Optional["Portfolio"]:
        """Load current portfolio state.

        Returns:
            Portfolio domain object or None if not found

        Raises:
            RepositoryError: If load operation fails
        """
        pass

    @abstractmethod
    def save_rebalancing_history(
        self, rebalancing_date: datetime, orders: list["RebalancingOrder"], results: dict[str, Any]
    ) -> bool:
        """Save rebalancing operation history.

        Args:
            rebalancing_date: Date of rebalancing
            orders: List of orders that were executed
            results: Results of the rebalancing operation

        Returns:
            True if save successful

        Raises:
            RepositoryError: If save operation fails
        """
        pass

    @abstractmethod
    def get_rebalancing_history(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get historical rebalancing operations.

        Args:
            start_date: Filter from this date (inclusive)
            end_date: Filter to this date (inclusive)

        Returns:
            List of rebalancing operation records

        Raises:
            RepositoryError: If load operation fails
        """
        pass

    @abstractmethod
    def backup_portfolio(self, backup_suffix: str = "") -> bool:
        """Create a backup of current portfolio state.

        Args:
            backup_suffix: Optional suffix for backup filename

        Returns:
            True if backup successful

        Raises:
            RepositoryError: If backup operation fails
        """
        pass


class RepositoryError(Exception):
    """Exception raised when repository operations fail."""

    def __init__(self, message: str, operation: str = "", original_error: Exception = None):
        self.message = message
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"{operation}: {message}" if operation else message)
