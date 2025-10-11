"""Market analysis modules for Clenow momentum strategy.

This package contains focused modules for different aspects of market analysis,
replacing the monolithic market_regime.py file with clean, single-responsibility modules.
"""

from .debug_utils import DebugDataManager
from .facade import CompleteMarketAnalysis, MarketAnalysisFacade
from .ma_analyzer import MAPosition, MovingAverageAnalyzer, ShortTermMAs
from .market_breadth import BreadthMetrics, MarketBreadthAnalyzer
from .momentum_analyzer import AbsoluteMomentumAnalyzer, DailyPerformance, MomentumMetrics
from .regime_detector import MarketRegimeDetector, RegimeStatus

__all__ = [
    # Facade (primary interface)
    "MarketAnalysisFacade",
    "CompleteMarketAnalysis",
    # Individual analyzers (for advanced usage)
    "MarketRegimeDetector",
    "RegimeStatus",
    "MarketBreadthAnalyzer",
    "BreadthMetrics",
    "AbsoluteMomentumAnalyzer",
    "MomentumMetrics",
    "DailyPerformance",
    "MovingAverageAnalyzer",
    "MAPosition",
    "ShortTermMAs",
    "DebugDataManager",
]
