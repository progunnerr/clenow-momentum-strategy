"""
Safety and risk controls for IBKR trading integration.

This module implements multiple layers of risk management and safety checks
to prevent dangerous trading situations and protect capital.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from loguru import logger

from ..strategy.rebalancing import Portfolio, RebalancingOrder


class RiskLevel(Enum):
    """Risk level enumeration."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskCheckResult(Enum):
    """Risk check result enumeration."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    CRITICAL_FAIL = "CRITICAL_FAIL"


@dataclass
class RiskCheckOutput:
    """Output from a risk check."""

    result: RiskCheckResult
    message: str
    risk_level: RiskLevel
    details: dict = None
    suggested_action: str = ""


class CircuitBreaker:
    """
    Circuit breaker to halt trading in emergency situations.

    Monitors for:
    - Excessive losses
    - Unusual market conditions
    - System errors
    - Manual intervention requests
    """

    def __init__(self, max_daily_loss_pct: float = 0.02, max_position_loss_pct: float = 0.10):
        """
        Initialize circuit breaker.

        Args:
            max_daily_loss_pct: Maximum daily loss percentage (2% default)
            max_position_loss_pct: Maximum single position loss percentage (10% default)
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_loss_pct = max_position_loss_pct
        self.is_tripped = False
        self.trip_reason = ""
        self.trip_time: datetime | None = None
        self.daily_pnl_start = 0.0

    def check_circuit_breaker(self, portfolio: Portfolio, account_value: float) -> RiskCheckOutput:
        """
        Check if circuit breaker should be triggered.

        Args:
            portfolio: Current portfolio
            account_value: Total account value

        Returns:
            Risk check output
        """
        if self.is_tripped:
            return RiskCheckOutput(
                result=RiskCheckResult.CRITICAL_FAIL,
                message=f"Circuit breaker already tripped: {self.trip_reason}",
                risk_level=RiskLevel.CRITICAL,
                details={"trip_time": self.trip_time},
                suggested_action="Manual intervention required to reset circuit breaker"
            )

        # Check daily P&L
        current_pnl = portfolio.total_market_value + portfolio.cash - account_value
        daily_loss_pct = abs(current_pnl) / account_value if account_value > 0 else 0

        if daily_loss_pct > self.max_daily_loss_pct:
            self._trip_breaker(f"Daily loss exceeded {self.max_daily_loss_pct:.1%}: {daily_loss_pct:.1%}")
            return RiskCheckOutput(
                result=RiskCheckResult.CRITICAL_FAIL,
                message=f"CIRCUIT BREAKER TRIPPED: Daily loss {daily_loss_pct:.1%} exceeds limit {self.max_daily_loss_pct:.1%}",
                risk_level=RiskLevel.CRITICAL,
                details={"daily_loss_pct": daily_loss_pct, "limit": self.max_daily_loss_pct},
                suggested_action="STOP ALL TRADING - Review positions and market conditions"
            )

        # Check individual position losses
        for ticker, position in portfolio.positions.items():
            position_loss_pct = abs(position.unrealized_pnl_pct)
            if position_loss_pct > self.max_position_loss_pct:
                self._trip_breaker(f"Position {ticker} loss exceeded {self.max_position_loss_pct:.1%}: {position_loss_pct:.1%}")
                return RiskCheckOutput(
                    result=RiskCheckResult.CRITICAL_FAIL,
                    message=f"CIRCUIT BREAKER TRIPPED: Position {ticker} loss {position_loss_pct:.1%} exceeds limit",
                    risk_level=RiskLevel.CRITICAL,
                    details={"ticker": ticker, "position_loss_pct": position_loss_pct},
                    suggested_action="STOP ALL TRADING - Review position and consider emergency exit"
                )

        return RiskCheckOutput(
            result=RiskCheckResult.PASS,
            message="Circuit breaker checks passed",
            risk_level=RiskLevel.LOW
        )

    def _trip_breaker(self, reason: str):
        """Trip the circuit breaker."""
        self.is_tripped = True
        self.trip_reason = reason
        self.trip_time = datetime.now(UTC)
        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")

    def reset_breaker(self, manual_override: bool = False) -> bool:
        """
        Reset the circuit breaker.

        Args:
            manual_override: Whether this is a manual override

        Returns:
            True if reset successful
        """
        if not manual_override:
            logger.warning("Circuit breaker reset requires manual override")
            return False

        self.is_tripped = False
        self.trip_reason = ""
        self.trip_time = None
        logger.warning("Circuit breaker manually reset")
        return True


class RiskControlSystem:
    """
    Comprehensive risk control system for trading operations.

    Implements multiple layers of safety checks and risk management controls.
    """

    def __init__(self, config: dict):
        """
        Initialize risk control system.

        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=config.get("max_daily_loss_pct", 0.02),
            max_position_loss_pct=config.get("max_position_loss_pct", 0.10)
        )

        # Risk limits
        self.max_position_pct = config.get("max_position_pct", 0.05)
        self.max_positions = config.get("max_positions", 20)
        self.min_position_value = config.get("min_position_value", 5000)
        self.max_order_value = config.get("max_order_value", 50000)
        self.max_daily_trades = config.get("max_daily_trades", 50)

        # Trading session tracking
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now(UTC).date()

        logger.info("Risk control system initialized")

    def pre_trade_validation(
        self,
        orders: list[RebalancingOrder],
        portfolio: Portfolio,
        account_value: float,
        ibkr_cash: float | None = None
    ) -> list[RiskCheckOutput]:
        """
        Comprehensive pre-trade validation.

        Args:
            orders: Orders to validate
            portfolio: Current portfolio
            account_value: Strategy allocation amount
            ibkr_cash: Available cash in IBKR account (optional)

        Returns:
            List of risk check results
        """
        logger.info(f"Running pre-trade validation for {len(orders)} orders...")

        checks = []

        # Reset daily counters if new day
        self._reset_daily_counters()

        # 1. Circuit breaker check
        checks.append(self.circuit_breaker.check_circuit_breaker(portfolio, account_value))

        # 2. Account balance checks
        checks.append(self._check_account_balance(portfolio, account_value))

        # 3. Position limit checks
        checks.append(self._check_position_limits(orders, portfolio, account_value))

        # 4. Order validation checks
        checks.append(self._check_order_validity(orders))

        # 5. Market conditions checks
        checks.append(self._check_market_conditions())

        # 6. Daily trade limits
        checks.append(self._check_daily_trade_limits(orders))

        # 7. Concentration risk checks
        checks.append(self._check_concentration_risk(orders, portfolio, account_value))

        # 8. Cash sufficiency check (if IBKR cash provided)
        if ibkr_cash is not None:
            checks.append(self._check_cash_sufficiency(orders, ibkr_cash, account_value))

        # Log results
        critical_fails = [c for c in checks if c.result == RiskCheckResult.CRITICAL_FAIL]
        fails = [c for c in checks if c.result == RiskCheckResult.FAIL]
        warnings = [c for c in checks if c.result == RiskCheckResult.WARNING]

        if critical_fails:
            logger.critical(f"🚨 {len(critical_fails)} CRITICAL risk check failures")
        if fails:
            logger.error(f"❌ {len(fails)} risk check failures")
        if warnings:
            logger.warning(f"⚠️ {len(warnings)} risk warnings")

        logger.info(f"Pre-trade validation completed: {len(checks)} checks run")
        return checks

    def can_proceed_with_trading(self, risk_checks: list[RiskCheckOutput]) -> tuple[bool, str]:
        """
        Determine if trading can proceed based on risk checks.

        Args:
            risk_checks: Results from pre-trade validation

        Returns:
            Tuple of (can_proceed, reason)
        """
        # Any critical failure blocks trading
        critical_fails = [c for c in risk_checks if c.result == RiskCheckResult.CRITICAL_FAIL]
        if critical_fails:
            return False, f"Critical risk failures: {'; '.join(c.message for c in critical_fails)}"

        # Multiple failures block trading
        fails = [c for c in risk_checks if c.result == RiskCheckResult.FAIL]
        if len(fails) > 2:
            return False, f"Too many risk failures ({len(fails)}): trading suspended"

        # High-risk failures block trading
        high_risk_fails = [c for c in fails if c.risk_level == RiskLevel.HIGH]
        if high_risk_fails:
            return False, f"High-risk failures: {'; '.join(c.message for c in high_risk_fails)}"

        return True, "Risk checks passed - trading can proceed"

    def _check_account_balance(self, portfolio: Portfolio, account_value: float) -> RiskCheckOutput:
        """Check account balance and cash position."""
        try:
            cash_pct = portfolio.cash / account_value if account_value > 0 else 0

            if portfolio.cash < 0:
                return RiskCheckOutput(
                    result=RiskCheckResult.CRITICAL_FAIL,
                    message="Negative cash balance detected",
                    risk_level=RiskLevel.CRITICAL,
                    details={"cash": portfolio.cash},
                    suggested_action="Account may be in margin call - do not trade"
                )

            if cash_pct < 0.01:  # Less than 1% cash
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message=f"Low cash balance: {cash_pct:.1%}",
                    risk_level=RiskLevel.MEDIUM,
                    details={"cash_pct": cash_pct},
                    suggested_action="Consider reducing position sizes"
                )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message="Account balance checks passed",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Account balance check failed: {e}",
                risk_level=RiskLevel.HIGH
            )

    def _check_position_limits(
        self, orders: list[RebalancingOrder], portfolio: Portfolio, account_value: float
    ) -> RiskCheckOutput:
        """Check position size and count limits."""
        try:
            # Count total positions after orders
            buy_orders = {o.ticker for o in orders if o.order_type.value == "BUY"}
            sell_orders = {o.ticker for o in orders if o.order_type.value == "SELL"}

            projected_positions = len(portfolio.positions) + len(buy_orders) - len(sell_orders)

            if projected_positions > self.max_positions:
                return RiskCheckOutput(
                    result=RiskCheckResult.FAIL,
                    message=f"Too many positions: {projected_positions} > {self.max_positions}",
                    risk_level=RiskLevel.HIGH,
                    details={"projected_positions": projected_positions, "limit": self.max_positions},
                    suggested_action="Reduce number of target positions"
                )

            # Check individual position sizes
            for order in orders:
                position_pct = order.order_value / account_value if account_value > 0 else 0
                if position_pct > self.max_position_pct:
                    return RiskCheckOutput(
                        result=RiskCheckResult.FAIL,
                        message=f"Position {order.ticker} too large: {position_pct:.1%} > {self.max_position_pct:.1%}",
                        risk_level=RiskLevel.HIGH,
                        details={"ticker": order.ticker, "position_pct": position_pct},
                        suggested_action="Reduce position size"
                    )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message="Position limit checks passed",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Position limit check failed: {e}",
                risk_level=RiskLevel.HIGH
            )

    def _check_order_validity(self, orders: list[RebalancingOrder]) -> RiskCheckOutput:
        """Check order validity and sanity."""
        try:
            issues = []

            for order in orders:
                # Check for zero or negative quantities
                if order.shares <= 0:
                    issues.append(f"{order.ticker}: Invalid share quantity {order.shares}")

                # Check for unreasonable prices
                if order.current_price <= 0 or order.current_price > 10000:
                    issues.append(f"{order.ticker}: Suspicious price ${order.current_price}")

                # Check for huge orders
                if order.order_value > self.max_order_value:
                    issues.append(f"{order.ticker}: Order too large ${order.order_value:,.0f}")

                # Check for penny stocks (potential manipulation risk)
                if order.current_price < 1.0:
                    issues.append(f"{order.ticker}: Penny stock risk at ${order.current_price:.2f}")

            if issues:
                return RiskCheckOutput(
                    result=RiskCheckResult.FAIL if len(issues) > 3 else RiskCheckResult.WARNING,
                    message=f"Order validity issues: {'; '.join(issues[:3])}",
                    risk_level=RiskLevel.HIGH if len(issues) > 3 else RiskLevel.MEDIUM,
                    details={"issues": issues},
                    suggested_action="Review and fix order issues"
                )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message="Order validity checks passed",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Order validity check failed: {e}",
                risk_level=RiskLevel.HIGH
            )

    def _check_market_conditions(self) -> RiskCheckOutput:
        """Check for dangerous market conditions."""
        try:
            current_time = datetime.now(UTC)

            # Check trading hours (9:30 AM - 4:00 PM ET)
            # This is a simplified check - in production you'd use proper market calendar
            hour = current_time.hour
            if hour < 14 or hour >= 21:  # Rough UTC equivalent of US market hours
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message="Trading outside normal market hours",
                    risk_level=RiskLevel.MEDIUM,
                    suggested_action="Consider waiting for market open"
                )

            # Check for market holidays (simplified - you'd use a proper calendar)
            if current_time.weekday() >= 5:  # Weekend
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message="Weekend trading detected",
                    risk_level=RiskLevel.MEDIUM,
                    suggested_action="Verify market is open"
                )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message="Market conditions acceptable",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Market conditions check failed: {e}",
                risk_level=RiskLevel.MEDIUM
            )

    def _check_daily_trade_limits(self, orders: list[RebalancingOrder]) -> RiskCheckOutput:
        """Check daily trading limits."""
        try:
            projected_trades = self.daily_trade_count + len(orders)

            if projected_trades > self.max_daily_trades:
                return RiskCheckOutput(
                    result=RiskCheckResult.FAIL,
                    message=f"Daily trade limit exceeded: {projected_trades} > {self.max_daily_trades}",
                    risk_level=RiskLevel.HIGH,
                    details={"daily_trades": self.daily_trade_count, "new_orders": len(orders)},
                    suggested_action="Reduce number of trades or wait until tomorrow"
                )

            if projected_trades > self.max_daily_trades * 0.8:
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message=f"Approaching daily trade limit: {projected_trades}/{self.max_daily_trades}",
                    risk_level=RiskLevel.MEDIUM,
                    suggested_action="Monitor trade count"
                )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message="Daily trade limit checks passed",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Daily trade limit check failed: {e}",
                risk_level=RiskLevel.MEDIUM
            )

    def _check_concentration_risk(
        self, orders: list[RebalancingOrder], portfolio: Portfolio, account_value: float
    ) -> RiskCheckOutput:
        """Check for concentration risk."""
        try:
            # Calculate sector/size concentration (simplified - you'd use proper sector data)
            total_order_value = sum(o.order_value for o in orders)

            if total_order_value > account_value * 0.5:  # More than 50% turnover
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message=f"High portfolio turnover: ${total_order_value:,.0f} ({total_order_value/account_value:.1%})",
                    risk_level=RiskLevel.MEDIUM,
                    details={"turnover_pct": total_order_value/account_value},
                    suggested_action="Consider phasing trades over multiple days"
                )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message="Concentration risk checks passed",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Concentration risk check failed: {e}",
                risk_level=RiskLevel.MEDIUM
            )

    def _reset_daily_counters(self):
        """Reset daily counters if new day."""
        current_date = datetime.now(UTC).date()
        if current_date > self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            logger.info("Daily trade counters reset")

    def post_trade_monitoring(self, portfolio: Portfolio, account_value: float) -> list[RiskCheckOutput]:
        """
        Monitor portfolio after trading.

        Args:
            portfolio: Updated portfolio
            account_value: Current account value

        Returns:
            List of monitoring alerts
        """
        alerts = []

        # Check circuit breaker conditions
        cb_check = self.circuit_breaker.check_circuit_breaker(portfolio, account_value)
        if cb_check.result != RiskCheckResult.PASS:
            alerts.append(cb_check)

        # Check for large position movements
        for ticker, position in portfolio.positions.items():
            if abs(position.unrealized_pnl_pct) > 0.05:  # 5% move
                alerts.append(RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message=f"Large position movement: {ticker} {position.unrealized_pnl_pct:+.1%}",
                    risk_level=RiskLevel.MEDIUM,
                    details={"ticker": ticker, "pnl_pct": position.unrealized_pnl_pct},
                    suggested_action="Monitor position closely"
                ))

        return alerts

    def emergency_stop(self, reason: str) -> None:
        """
        Emergency stop - trip circuit breaker immediately.

        Args:
            reason: Reason for emergency stop
        """
        self.circuit_breaker._trip_breaker(f"EMERGENCY STOP: {reason}")
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")

    def get_risk_status(self) -> dict:
        """Get current risk system status."""
        return {
            "circuit_breaker_status": {
                "is_tripped": self.circuit_breaker.is_tripped,
                "trip_reason": self.circuit_breaker.trip_reason,
                "trip_time": self.circuit_breaker.trip_time,
            },
            "daily_trades": {
                "count": self.daily_trade_count,
                "limit": self.max_daily_trades,
                "remaining": self.max_daily_trades - self.daily_trade_count,
            },
            "limits": {
                "max_positions": self.max_positions,
                "max_position_pct": self.max_position_pct,
                "max_order_value": self.max_order_value,
            },
            "last_reset": self.last_reset_date,
        }

    def _check_cash_sufficiency(
        self, orders: list[RebalancingOrder], ibkr_cash: float, strategy_allocation: float
    ) -> RiskCheckOutput:
        """Check if sufficient cash is available for orders."""
        try:
            # Calculate total buy order value
            buy_orders = [o for o in orders if o.order_type.value == "BUY"]
            total_buy_value = sum(o.order_value for o in buy_orders)

            # Check against IBKR available cash
            if total_buy_value > ibkr_cash:
                return RiskCheckOutput(
                    result=RiskCheckResult.FAIL,
                    message=f"Insufficient IBKR cash: ${total_buy_value:,.0f} needed, ${ibkr_cash:,.0f} available",
                    risk_level=RiskLevel.HIGH,
                    details={
                        "buy_orders_value": total_buy_value,
                        "available_cash": ibkr_cash,
                        "shortfall": total_buy_value - ibkr_cash
                    },
                    suggested_action="Reduce position sizes or add cash to account"
                )

            # Check against strategy allocation limit
            if total_buy_value > strategy_allocation:
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message=f"Buy orders exceed strategy allocation: ${total_buy_value:,.0f} > ${strategy_allocation:,.0f}",
                    risk_level=RiskLevel.MEDIUM,
                    details={
                        "buy_orders_value": total_buy_value,
                        "strategy_allocation": strategy_allocation,
                        "excess": total_buy_value - strategy_allocation
                    },
                    suggested_action="Consider reducing position sizes"
                )

            # Warning if using most of available cash
            cash_utilization = total_buy_value / ibkr_cash if ibkr_cash > 0 else 0
            if cash_utilization > 0.9:  # Using >90% of cash
                return RiskCheckOutput(
                    result=RiskCheckResult.WARNING,
                    message=f"High cash utilization: {cash_utilization:.1%} of available cash",
                    risk_level=RiskLevel.MEDIUM,
                    details={"cash_utilization": cash_utilization},
                    suggested_action="Consider keeping more cash buffer"
                )

            return RiskCheckOutput(
                result=RiskCheckResult.PASS,
                message=f"Cash sufficiency check passed: ${total_buy_value:,.0f} needed, ${ibkr_cash:,.0f} available",
                risk_level=RiskLevel.LOW
            )

        except Exception as e:
            return RiskCheckOutput(
                result=RiskCheckResult.FAIL,
                message=f"Cash sufficiency check failed: {e}",
                risk_level=RiskLevel.HIGH
            )
