"""Trading execution and broker connectivity."""

from clenow_momentum.data.sources.ibkr_client import get_trading_mode
from clenow_momentum.utils.config import validate_ibkr_config
from .execution_engine_sync import ExecutionError, SyncTradingExecutionEngine
from .portfolio_sync import PortfolioSyncError, PortfolioSynchronizer
from .risk_controls import (
    CircuitBreaker,
    RiskCheckOutput,
    RiskCheckResult,
    RiskControlSystem,
    RiskLevel,
)
from .trading_manager import TradingManager, TradingManagerError
from .order_strategies import (
    OrderStrategy,
    OrderContext, 
    ExitStrategy,
    AdjustStrategy,
    EntryStrategy
)
from .order_generation import OrderGenerationService, OrderGenerationResult

__all__ = [
    # Portfolio Synchronization
    "PortfolioSynchronizer",
    "PortfolioSyncError",
    # Execution Engine
    "SyncTradingExecutionEngine",
    "ExecutionError",
    # Risk Controls
    "RiskControlSystem",
    "CircuitBreaker",
    "RiskCheckOutput",
    "RiskCheckResult",
    "RiskLevel",
    # Trading Manager
    "TradingManager",
    "TradingManagerError",
    # Order Generation
    "OrderStrategy",
    "OrderContext",
    "ExitStrategy", 
    "AdjustStrategy",
    "EntryStrategy",
    "OrderGenerationService",
    "OrderGenerationResult",
    # Utility functions
    "get_trading_mode",
    "validate_ibkr_config",
]
