"""Market Analysis Facade - Unified interface for all market analysis.

This facade provides a simplified API for accessing all market analysis functionality,
replacing the procedural functions from the legacy market_regime.py module.
"""

from dataclasses import dataclass

from loguru import logger

from ..data.interfaces import MarketDataSource, TickerSource
from .debug_utils import DebugDataManager
from .ma_analyzer import MAPosition, MovingAverageAnalyzer, ShortTermMAs
from .market_breadth import BreadthMetrics, MarketBreadthAnalyzer
from .momentum_analyzer import AbsoluteMomentumAnalyzer, DailyPerformance, MomentumMetrics
from .regime_detector import MarketRegimeDetector, RegimeStatus


@dataclass(frozen=True)
class CompleteMarketAnalysis:
    """Complete market analysis results from all analyzers."""

    regime: RegimeStatus
    breadth: BreadthMetrics
    momentum: MomentumMetrics
    daily_performance: DailyPerformance
    short_term_mas: ShortTermMAs
    ma_position: MAPosition


class MarketAnalysisFacade:
    """Unified interface for all market analysis functionality.
    
    This facade orchestrates all market analysis components and provides
    a clean API that replaces the legacy procedural functions.
    
    Example usage:
        facade = MarketAnalysisFacade(market_data_source, ticker_source)
        analysis = facade.get_complete_analysis()
        
        # Or use individual methods
        regime = facade.check_regime()
        breadth = facade.calculate_breadth()
    """

    def __init__(
        self,
        market_data_source: MarketDataSource,
        ticker_source: TickerSource,
        debug_manager: DebugDataManager | None = None,
        benchmark_ticker: str = "SPY",
    ):
        """Initialize facade with data sources.

        Args:
            market_data_source: Source for market benchmark data.
            ticker_source:      Source for universe ticker list.
            debug_manager:      Optional debug data manager.
            benchmark_ticker:   ETF symbol used for regime detection.
                                Defaults to "SPY". Pass the active
                                universe's benchmark_etf (e.g. "IWB"
                                for Russell 1000).
        """
        self.market_data_source = market_data_source
        self.ticker_source = ticker_source
        self.debug_manager = debug_manager or DebugDataManager()
        self.benchmark_ticker = benchmark_ticker

        # Initialize all analyzers (no data sources - just pure logic)
        self.regime_detector = MarketRegimeDetector(
            market_data_source, self.debug_manager, benchmark_ticker=benchmark_ticker
        )
        self.breadth_analyzer = MarketBreadthAnalyzer(
            market_data_source, ticker_source, self.debug_manager
        )
        self.momentum_analyzer = AbsoluteMomentumAnalyzer(self.debug_manager)
        self.ma_analyzer = MovingAverageAnalyzer(self.debug_manager)

    def check_regime(self, period: int = 200) -> RegimeStatus:
        """Check current market regime (SPX vs MA).
        
        Args:
            period: Moving average period (default 200)
            
        Returns:
            RegimeStatus with regime information
        """
        return self.regime_detector.check_regime(ma_period=period)

    def should_trade_momentum(self, regime: RegimeStatus | None = None, period: int = 200) -> tuple[bool, str]:
        """Determine if momentum trading should be active.
        
        Args:
            regime: Optional pre-calculated regime status
            period: MA period if regime not provided
            
        Returns:
            Tuple of (should_trade, reason)
        """
        if regime is None:
            regime = self.check_regime(period=period)
            
        if regime.error:
            return False, f"Market data error: {regime.error}"
            
        if regime.trading_allowed:
            return True, f"Market regime is {regime.regime} ({self.benchmark_ticker} above {period}MA)"
        else:
            return False, f"Market regime is {regime.regime} ({self.benchmark_ticker} below {period}MA) - momentum trading suspended"

    def get_detailed_status(self, period: int = 200) -> dict:
        """Get detailed market regime status with history.
        
        This is equivalent to the old get_sp500_ma_status() function.
        Note: This currently returns basic regime status. Extended metrics
        (crossover_date, days_since_crossover, etc.) not yet implemented.
        
        Args:
            period: Moving average period
            
        Returns:
            Dict with regime status (basic version for now)
        """
        # For now, return basic regime status until we implement extended metrics
        regime = self.regime_detector.check_regime(ma_period=period)
        # Return as dict with 'error' key to indicate limited functionality
        return {"error": "Extended regime metrics not yet implemented"}

    def calculate_breadth(
        self, period: int = 200, stock_data=None, tickers: list[str] | None = None
    ) -> BreadthMetrics:
        """Calculate market breadth (% stocks above MA).
        
        Args:
            period: Moving average period
            stock_data: Optional pre-fetched stock data
            tickers: Optional ticker list (will fetch if not provided)
            
        Returns:
            BreadthMetrics with breadth statistics
        """
        return self.breadth_analyzer.calculate_breadth(
            ma_period=period, stock_data=stock_data, tickers=tickers
        )

    def calculate_absolute_momentum(self, period_months: int = 12) -> MomentumMetrics:
        """Calculate absolute momentum of S&P 500.
        
        Args:
            period_months: Number of months for return calculation
            
        Returns:
            MomentumMetrics with momentum values
        """
        # Fetch data once for this calculation
        period_str = f"{period_months + 1}mo"
        try:
            spy_data = self.market_data_source.get_market_data(period=period_str, benchmark_ticker=self.benchmark_ticker)
            return self.momentum_analyzer.calculate_momentum(spy_data, period_months=period_months)
        except Exception as e:
            logger.error(f"Error fetching data for momentum calculation: {e}")
            return MomentumMetrics(
                period_return=0.0,
                period_months=period_months,
                current_price=0.0,
                past_price=0.0,
                momentum_strength="Unknown",
                all_returns={},
                bullish=False,
                error=str(e),
            )

    def calculate_daily_performance(self) -> DailyPerformance:
        """Calculate S&P 500 daily performance metrics.
        
        Returns:
            DailyPerformance with daily metrics
        """
        # Fetch recent data
        try:
            spy_data = self.market_data_source.get_market_data(period="1mo", benchmark_ticker=self.benchmark_ticker)
            return self.momentum_analyzer.calculate_daily_performance(spy_data)
        except Exception as e:
            logger.error(f"Error fetching data for daily performance: {e}")
            return DailyPerformance(
                current_price=0.0,
                previous_close=0.0,
                daily_change=0.0,
                daily_change_pct=0.0,
                daily_trend="Unknown",
                error=str(e),
            )

    def calculate_short_term_mas(self) -> ShortTermMAs:
        """Calculate short-term moving averages (10 EMA, 20 SMA, 50 SMA).
        
        Returns:
            ShortTermMAs with calculated values
        """
        # Fetch data for MA calculations
        try:
            spy_data = self.market_data_source.get_market_data(period="3mo", benchmark_ticker=self.benchmark_ticker)
            return self.ma_analyzer.calculate_short_term_mas(spy_data)
        except Exception as e:
            logger.error(f"Error fetching data for short-term MAs: {e}")
            return ShortTermMAs(current_price=0.0, error=str(e))

    def analyze_ma_position(
        self, long_term_ma: float | None = None, short_term_mas: ShortTermMAs | None = None
    ) -> MAPosition:
        """Analyze S&P 500 position relative to all moving averages.
        
        Args:
            long_term_ma: Optional 200-day MA value
            short_term_mas: Optional pre-calculated short-term MAs
            
        Returns:
            MAPosition with analysis results
        """
        # If short_term_mas not provided, fetch data and calculate
        if short_term_mas is None:
            try:
                spy_data = self.market_data_source.get_market_data(period="3mo", benchmark_ticker=self.benchmark_ticker)
                return self.ma_analyzer.analyze_position(
                    long_term_ma=long_term_ma, spy_data=spy_data
                )
            except Exception as e:
                logger.error(f"Error fetching data for MA position analysis: {e}")
                return MAPosition(
                    current_price=0.0,
                    mas_above=0,
                    mas_below=0,
                    total_mas=0,
                    market_structure="Unknown",
                    ma_positions=[],
                    mas_aligned=False,
                    error=str(e),
                )
        
        return self.ma_analyzer.analyze_position(
            long_term_ma=long_term_ma, short_term_mas=short_term_mas
        )

    def get_complete_analysis(
        self, period: int = 200, stock_data=None, tickers: list[str] | None = None
    ) -> CompleteMarketAnalysis:
        """Get complete market analysis from all components.
        
        This is a convenience method that runs all analyzers and returns
        a unified result object.
        
        Args:
            period: Moving average period for regime/breadth
            stock_data: Optional pre-fetched stock data for breadth
            tickers: Optional ticker list for breadth
            
        Returns:
            CompleteMarketAnalysis with all results
        """
        logger.info("Running complete market analysis...")
        
        # Fetch all necessary market data once (orchestrator pattern)
        try:
            spy_data_3mo = self.market_data_source.get_market_data(period="3mo", benchmark_ticker=self.benchmark_ticker)
            spy_data_13mo = self.market_data_source.get_market_data(period="13mo", benchmark_ticker=self.benchmark_ticker)
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            # Return error state for all components
            regime = self.check_regime(period=period)
            return CompleteMarketAnalysis(
                regime=regime,
                breadth=BreadthMetrics(0.0, 0, 0, 0, 0, "Unknown", period, error=str(e)),
                momentum=MomentumMetrics(0.0, 12, 0.0, 0.0, "Unknown", {}, False, error=str(e)),
                daily_performance=DailyPerformance(0.0, 0.0, 0.0, 0.0, "Unknown", error=str(e)),
                short_term_mas=ShortTermMAs(0.0, error=str(e)),
                ma_position=MAPosition(0.0, 0, 0, 0, "Unknown", [], False, error=str(e)),
            )
        
        # Run all analyses with pre-fetched data
        regime = self.check_regime(period=period)
        breadth = self.calculate_breadth(period=period, stock_data=stock_data, tickers=tickers)
        momentum = self.momentum_analyzer.calculate_momentum(spy_data_13mo, period_months=12)
        daily_perf = self.momentum_analyzer.calculate_daily_performance(spy_data_3mo)
        short_mas = self.ma_analyzer.calculate_short_term_mas(spy_data_3mo)
        
        # Use regime MA value for position analysis
        long_term_ma = regime.ma_value if regime.ma_value else None
        ma_position = self.ma_analyzer.analyze_position(
            long_term_ma=long_term_ma, short_term_mas=short_mas
        )
        
        return CompleteMarketAnalysis(
            regime=regime,
            breadth=breadth,
            momentum=momentum,
            daily_performance=daily_perf,
            short_term_mas=short_mas,
            ma_position=ma_position,
        )
