"""
Tests for risk control system.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from clenow_momentum.strategy.rebalancing import OrderType, Portfolio, Position, RebalancingOrder
from clenow_momentum.trading.risk_controls import (
    CircuitBreaker,
    RiskCheckResult,
    RiskControlSystem,
    RiskLevel,
)


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(max_daily_loss_pct=0.02, max_position_loss_pct=0.10)

    def test_initialization(self):
        """Test circuit breaker initialization."""
        assert self.circuit_breaker.max_daily_loss_pct == 0.02
        assert self.circuit_breaker.max_position_loss_pct == 0.10
        assert self.circuit_breaker.is_tripped is False
        assert self.circuit_breaker.trip_reason == ""

    def test_circuit_breaker_normal_conditions(self):
        """Test circuit breaker under normal conditions."""
        portfolio = Portfolio(cash=85000)

        # Add a position with small loss (less than 2% of total account value)
        position = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=148.0,  # $200 loss
            entry_date=datetime.now(UTC),
            atr=5.0,
        )
        portfolio.add_position(position)

        # Total portfolio: 85000 cash + 14800 position = 99800
        # Account value: 100000
        # Loss: 200 / 100000 = 0.2% (less than 2% threshold)
        result = self.circuit_breaker.check_circuit_breaker(portfolio, 100000)

        assert result.result == RiskCheckResult.PASS
        assert self.circuit_breaker.is_tripped is False

    def test_circuit_breaker_daily_loss_limit(self):
        """Test circuit breaker trips on excessive daily loss."""
        portfolio = Portfolio(cash=10000)

        # Add position with large loss (> 2% of account)
        position = Position(
            ticker="AAPL",
            shares=1000,
            entry_price=150.0,
            current_price=120.0,  # $30k loss on $100k account = 30%
            entry_date=datetime.now(UTC),
            atr=5.0,
        )
        portfolio.add_position(position)

        result = self.circuit_breaker.check_circuit_breaker(portfolio, 100000)

        assert result.result == RiskCheckResult.CRITICAL_FAIL
        assert self.circuit_breaker.is_tripped is True
        assert "Daily loss" in self.circuit_breaker.trip_reason

    def test_circuit_breaker_position_loss_limit(self):
        """Test circuit breaker trips on excessive position loss."""
        portfolio = Portfolio(cash=87000)

        # Add position with > 10% loss
        position = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=130.0,  # 13.3% loss, $2000 loss
            entry_date=datetime.now(UTC),
            atr=5.0,
        )
        portfolio.add_position(position)

        # Total portfolio: 87000 cash + 13000 position = 100000
        # But this represents a 2000 loss on original position (87000 + 15000 = 102000)
        # Daily loss: 2000/100000 = 2% - should not trip on daily loss
        # Position loss: 20/150 = 13.3% - should trip on position loss
        result = self.circuit_breaker.check_circuit_breaker(portfolio, 100000)

        assert result.result == RiskCheckResult.CRITICAL_FAIL
        assert self.circuit_breaker.is_tripped is True
        # Either daily loss or position loss might trigger first
        assert "loss" in self.circuit_breaker.trip_reason

    def test_circuit_breaker_already_tripped(self):
        """Test circuit breaker when already tripped."""
        self.circuit_breaker.is_tripped = True
        self.circuit_breaker.trip_reason = "Previous failure"

        portfolio = Portfolio(cash=100000)
        result = self.circuit_breaker.check_circuit_breaker(portfolio, 100000)

        assert result.result == RiskCheckResult.CRITICAL_FAIL
        assert "already tripped" in result.message

    def test_reset_breaker_no_override(self):
        """Test reset without manual override."""
        self.circuit_breaker.is_tripped = True
        result = self.circuit_breaker.reset_breaker(manual_override=False)

        assert result is False
        assert self.circuit_breaker.is_tripped is True

    def test_reset_breaker_with_override(self):
        """Test reset with manual override."""
        self.circuit_breaker.is_tripped = True
        result = self.circuit_breaker.reset_breaker(manual_override=True)

        assert result is True
        assert self.circuit_breaker.is_tripped is False


class TestRiskControlSystem:
    """Test RiskControlSystem class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "max_daily_loss_pct": 0.02,
            "max_position_loss_pct": 0.10,
            "max_position_pct": 0.05,
            "max_positions": 20,
            "min_position_value": 5000,
            "max_order_value": 50000,
            "max_daily_trades": 50,
        }
        self.risk_system = RiskControlSystem(self.config)

    def test_initialization(self):
        """Test risk control system initialization."""
        assert self.risk_system.max_positions == 20
        assert self.risk_system.max_order_value == 50000
        assert self.risk_system.max_daily_trades == 50

    def test_account_balance_checks_negative_cash(self):
        """Test account balance check with negative cash."""
        portfolio = Portfolio(cash=-1000)
        result = self.risk_system._check_account_balance(portfolio, 100000)

        assert result.result == RiskCheckResult.CRITICAL_FAIL
        assert "Negative cash balance" in result.message

    def test_account_balance_checks_low_cash(self):
        """Test account balance check with low cash."""
        portfolio = Portfolio(cash=500)  # 0.5% of $100k
        result = self.risk_system._check_account_balance(portfolio, 100000)

        assert result.result == RiskCheckResult.WARNING
        assert "Low cash balance" in result.message

    def test_account_balance_checks_normal(self):
        """Test account balance check with normal cash."""
        portfolio = Portfolio(cash=5000)  # 5% of $100k
        result = self.risk_system._check_account_balance(portfolio, 100000)

        assert result.result == RiskCheckResult.PASS

    def test_position_limit_checks_too_many_positions(self):
        """Test position limit check with too many positions."""
        portfolio = Portfolio()

        # Create orders that would result in too many positions
        orders = []
        for i in range(25):  # Exceeds limit of 20
            orders.append(
                RebalancingOrder(
                    ticker=f"STOCK{i}",
                    order_type=OrderType.BUY,
                    shares=100,
                    current_price=100.0,
                    order_value=10000.0,
                    reason="Test order",
                )
            )

        result = self.risk_system._check_position_limits(orders, portfolio, 1000000)

        assert result.result == RiskCheckResult.FAIL
        assert "Too many positions" in result.message

    def test_position_limit_checks_position_too_large(self):
        """Test position limit check with position too large."""
        portfolio = Portfolio()

        # Create order for position > 5% of account
        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=1000,
                current_price=100.0,
                order_value=100000.0,  # 10% of $1M account
                reason="Large position",
            )
        ]

        result = self.risk_system._check_position_limits(orders, portfolio, 1000000)

        assert result.result == RiskCheckResult.FAIL
        assert "Position AAPL too large" in result.message

    def test_order_validity_checks_invalid_shares(self):
        """Test order validity check with invalid share quantity."""
        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=0,  # Invalid
                current_price=100.0,
                order_value=0.0,
                reason="Invalid order",
            )
        ]

        result = self.risk_system._check_order_validity(orders)

        # Single issue returns WARNING, not FAIL
        assert result.result == RiskCheckResult.WARNING
        assert "Invalid share quantity" in result.message

    def test_order_validity_checks_suspicious_price(self):
        """Test order validity check with suspicious price."""
        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=100,
                current_price=15000.0,  # Suspiciously high
                order_value=1500000.0,
                reason="Suspicious price",
            )
        ]

        result = self.risk_system._check_order_validity(orders)

        # Two issues (suspicious price + order too large) returns WARNING
        assert result.result == RiskCheckResult.WARNING
        assert "Suspicious price" in result.message

    def test_order_validity_checks_penny_stock(self):
        """Test order validity check with penny stock."""
        orders = [
            RebalancingOrder(
                ticker="PENNY",
                order_type=OrderType.BUY,
                shares=1000,
                current_price=0.50,  # Penny stock
                order_value=500.0,
                reason="Penny stock",
            )
        ]

        result = self.risk_system._check_order_validity(orders)

        assert result.result == RiskCheckResult.WARNING
        assert "Penny stock risk" in result.message

    def test_daily_trade_limits_exceeded(self):
        """Test daily trade limit exceeded."""
        self.risk_system.daily_trade_count = 45

        orders = []
        for i in range(10):  # Would bring total to 55, exceeding limit of 50
            orders.append(
                RebalancingOrder(
                    ticker=f"STOCK{i}",
                    order_type=OrderType.BUY,
                    shares=100,
                    current_price=100.0,
                    order_value=10000.0,
                    reason="Test order",
                )
            )

        result = self.risk_system._check_daily_trade_limits(orders)

        assert result.result == RiskCheckResult.FAIL
        assert "Daily trade limit exceeded" in result.message

    def test_daily_trade_limits_warning(self):
        """Test daily trade limit warning."""
        self.risk_system.daily_trade_count = 35

        orders = []
        for i in range(6):  # Would bring total to 41, > 80% of 50
            orders.append(
                RebalancingOrder(
                    ticker=f"STOCK{i}",
                    order_type=OrderType.BUY,
                    shares=100,
                    current_price=100.0,
                    order_value=10000.0,
                    reason="Test order",
                )
            )

        result = self.risk_system._check_daily_trade_limits(orders)

        assert result.result == RiskCheckResult.WARNING
        assert "Approaching daily trade limit" in result.message

    def test_pre_trade_validation(self):
        """Test complete pre-trade validation."""
        portfolio = Portfolio(cash=100000)  # Match account value to avoid false circuit breaker trip
        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.BUY,
                shares=100,
                current_price=100.0,
                order_value=10000.0,
                reason="Test order",
            )
        ]

        checks = self.risk_system.pre_trade_validation(orders, portfolio, 100000)

        # Should have multiple checks
        assert len(checks) >= 6

        # All should pass under normal conditions
        critical_fails = [c for c in checks if c.result == RiskCheckResult.CRITICAL_FAIL]
        assert len(critical_fails) == 0

    def test_can_proceed_with_trading_critical_fail(self):
        """Test trading blocked by critical failure."""
        checks = [
            MagicMock(result=RiskCheckResult.CRITICAL_FAIL, message="Critical error"),
            MagicMock(result=RiskCheckResult.PASS, message="OK"),
        ]

        can_proceed, reason = self.risk_system.can_proceed_with_trading(checks)

        assert can_proceed is False
        assert "Critical risk failures" in reason

    def test_can_proceed_with_trading_too_many_fails(self):
        """Test trading blocked by too many failures."""
        checks = [
            MagicMock(result=RiskCheckResult.FAIL, message="Error 1", risk_level=RiskLevel.MEDIUM),
            MagicMock(result=RiskCheckResult.FAIL, message="Error 2", risk_level=RiskLevel.MEDIUM),
            MagicMock(result=RiskCheckResult.FAIL, message="Error 3", risk_level=RiskLevel.MEDIUM),
        ]

        can_proceed, reason = self.risk_system.can_proceed_with_trading(checks)

        assert can_proceed is False
        assert "Too many risk failures" in reason

    def test_can_proceed_with_trading_high_risk_fail(self):
        """Test trading blocked by high-risk failure."""
        checks = [
            MagicMock(result=RiskCheckResult.FAIL, message="High risk error", risk_level=RiskLevel.HIGH),
            MagicMock(result=RiskCheckResult.PASS, message="OK"),
        ]

        can_proceed, reason = self.risk_system.can_proceed_with_trading(checks)

        assert can_proceed is False
        assert "High-risk failures" in reason

    def test_can_proceed_with_trading_success(self):
        """Test trading allowed."""
        checks = [
            MagicMock(result=RiskCheckResult.PASS, message="OK"),
            MagicMock(result=RiskCheckResult.WARNING, message="Warning", risk_level=RiskLevel.LOW),
        ]

        can_proceed, reason = self.risk_system.can_proceed_with_trading(checks)

        assert can_proceed is True
        assert "Risk checks passed" in reason

    def test_post_trade_monitoring(self):
        """Test post-trade monitoring."""
        portfolio = Portfolio(cash=10000)

        # Add position with large movement
        position = Position(
            ticker="AAPL",
            shares=100,
            entry_price=100.0,
            current_price=110.0,  # 10% gain
            entry_date=datetime.now(UTC),
            atr=5.0,
        )
        portfolio.add_position(position)

        alerts = self.risk_system.post_trade_monitoring(portfolio, 100000)

        # Should have alerts for large position movement
        movement_alerts = [a for a in alerts if "Large position movement" in a.message]
        assert len(movement_alerts) > 0

    def test_emergency_stop(self):
        """Test emergency stop."""
        self.risk_system.emergency_stop("Test emergency")

        assert self.risk_system.circuit_breaker.is_tripped is True
        assert "EMERGENCY STOP" in self.risk_system.circuit_breaker.trip_reason

    def test_get_risk_status(self):
        """Test get risk status."""
        status = self.risk_system.get_risk_status()

        assert "circuit_breaker_status" in status
        assert "daily_trades" in status
        assert "limits" in status
        assert status["limits"]["max_positions"] == 20
