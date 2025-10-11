"""Order generation strategies for portfolio rebalancing.

This package implements the Strategy pattern to break down the complex order generation
logic into focused, single-responsibility classes. Each strategy handles one type of
order generation (exits, resizing, entries) with clean separation of concerns.
"""

from .base import OrderContext, OrderStrategy
from .entry_strategy import EntryStrategy
from .exit_strategy import ExitStrategy
from .resize_strategy import ResizeStrategy

# Import AdjustStrategy alias
AdjustStrategy = ResizeStrategy

__all__ = [
    "OrderStrategy",
    "OrderContext",
    "ExitStrategy",
    "ResizeStrategy",
    "AdjustStrategy",
    "EntryStrategy",
]
