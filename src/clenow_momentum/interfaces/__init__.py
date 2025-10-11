"""Core interfaces and abstractions for the Clenow Momentum Strategy.

This module defines the fundamental contracts that decouple business logic
from infrastructure concerns, following the Dependency Inversion Principle.
"""

from .brokers import BrokerAdapter
from .data_sources import MarketDataSource, TickerSource
from .repositories import PortfolioRepository

__all__ = [
    "MarketDataSource",
    "TickerSource",
    "PortfolioRepository",
    "BrokerAdapter",
]
