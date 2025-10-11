"""Value objects for the Clenow Momentum Strategy domain.

Value objects are immutable objects that represent concepts with no identity.
They encapsulate business logic and ensure data integrity through validation.
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Money:
    """Represents a monetary amount with currency awareness.

    Immutable value object that handles monetary calculations
    with proper precision and validation.
    """

    amount: Decimal
    currency: str = "USD"

    def __post_init__(self):
        """Validate money object after initialization."""
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))

        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")

        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be a valid 3-letter code")

    @classmethod
    def from_float(cls, amount: float, currency: str = "USD") -> "Money":
        """Create Money from float value.

        Args:
            amount: Float amount
            currency: Currency code

        Returns:
            Money instance
        """
        return cls(Decimal(str(amount)), currency)

    @classmethod
    def zero(cls, currency: str = "USD") -> "Money":
        """Create zero money amount.

        Args:
            currency: Currency code

        Returns:
            Money instance with zero amount
        """
        return cls(Decimal("0"), currency)

    def __add__(self, other: "Money") -> "Money":
        """Add two money amounts."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        """Subtract two money amounts."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
        result_amount = self.amount - other.amount
        if result_amount < 0:
            raise ValueError("Subtraction would result in negative money")
        return Money(result_amount, self.currency)

    def __mul__(self, multiplier: int | float | Decimal) -> "Money":
        """Multiply money by a number."""
        if not isinstance(multiplier, int | float | Decimal):
            raise TypeError("Money can only be multiplied by numbers")

        if isinstance(multiplier, int | float):
            multiplier = Decimal(str(multiplier))

        return Money(self.amount * multiplier, self.currency)

    def __truediv__(self, divisor: int | float | Decimal) -> "Money":
        """Divide money by a number."""
        if not isinstance(divisor, int | float | Decimal):
            raise TypeError("Money can only be divided by numbers")

        if divisor == 0:
            raise ValueError("Cannot divide by zero")

        if isinstance(divisor, int | float):
            divisor = Decimal(str(divisor))

        return Money(self.amount / divisor, self.currency)

    def __lt__(self, other: "Money") -> bool:
        """Less than comparison."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount < other.amount

    def __le__(self, other: "Money") -> bool:
        """Less than or equal comparison."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount <= other.amount

    def __gt__(self, other: "Money") -> bool:
        """Greater than comparison."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount > other.amount

    def __ge__(self, other: "Money") -> bool:
        """Greater than or equal comparison."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount >= other.amount

    def to_float(self) -> float:
        """Convert to float for external APIs."""
        return float(self.amount)

    def __str__(self) -> str:
        """String representation."""
        return f"${self.amount:.2f}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Money({self.amount}, '{self.currency}')"


@dataclass(frozen=True)
class Percentage:
    """Represents a percentage value with proper validation.

    Immutable value object for percentage calculations,
    ensuring values are within valid ranges.
    """

    value: Decimal  # Stored as decimal (e.g., 0.05 for 5%)

    def __post_init__(self):
        """Validate percentage after initialization."""
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, "value", Decimal(str(self.value)))

    @classmethod
    def from_percent(cls, percent: float | int) -> "Percentage":
        """Create percentage from percent value (e.g., 5.0 for 5%).

        Args:
            percent: Percentage value (e.g., 5.0 for 5%)

        Returns:
            Percentage instance
        """
        return cls(Decimal(str(percent)) / Decimal("100"))

    @classmethod
    def from_decimal(cls, decimal: float | Decimal) -> "Percentage":
        """Create percentage from decimal value (e.g., 0.05 for 5%).

        Args:
            decimal: Decimal value (e.g., 0.05 for 5%)

        Returns:
            Percentage instance
        """
        return cls(Decimal(str(decimal)))

    def to_percent(self) -> float:
        """Convert to percentage (e.g., 5.0 for 5%)."""
        return float(self.value * 100)

    def to_decimal(self) -> float:
        """Convert to decimal (e.g., 0.05 for 5%)."""
        return float(self.value)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.to_percent():.2f}%"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Percentage({self.value})"


@dataclass(frozen=True)
class RiskMetrics:
    """Risk calculation results for position sizing.

    Immutable value object containing ATR-based risk calculations
    used in Clenow's position sizing methodology.
    """

    atr: Decimal
    stop_loss_multiplier: Decimal  # Typically 3.0 per Clenow
    stop_loss_distance: Decimal  # atr * stop_loss_multiplier
    risk_amount: Money  # Dollar amount at risk

    def __post_init__(self):
        """Validate risk metrics after initialization."""
        # Convert to Decimal if needed
        for field in ["atr", "stop_loss_multiplier", "stop_loss_distance"]:
            value = getattr(self, field)
            if not isinstance(value, Decimal):
                object.__setattr__(self, field, Decimal(str(value)))

        # Validation
        if self.atr <= 0:
            raise ValueError("ATR must be positive")
        if self.stop_loss_multiplier <= 0:
            raise ValueError("Stop loss multiplier must be positive")
        if self.stop_loss_distance <= 0:
            raise ValueError("Stop loss distance must be positive")
        if self.risk_amount.amount <= 0:
            raise ValueError("Risk amount must be positive")

    @classmethod
    def calculate(
        cls,
        atr: float | Decimal,
        stop_loss_multiplier: float | Decimal,
        account_value: Money,
        risk_per_trade: Percentage,
    ) -> "RiskMetrics":
        """Calculate risk metrics from input parameters.

        Args:
            atr: Average True Range value
            stop_loss_multiplier: Multiplier for stop loss (typically 3.0)
            account_value: Total account value
            risk_per_trade: Risk percentage per trade

        Returns:
            RiskMetrics instance with calculated values
        """
        atr_decimal = Decimal(str(atr))
        multiplier_decimal = Decimal(str(stop_loss_multiplier))
        stop_loss_distance = atr_decimal * multiplier_decimal
        risk_amount = account_value * risk_per_trade.to_decimal()

        return cls(
            atr=atr_decimal,
            stop_loss_multiplier=multiplier_decimal,
            stop_loss_distance=stop_loss_distance,
            risk_amount=risk_amount,
        )

    def calculate_position_shares(self, stock_price: Money) -> int:
        """Calculate number of shares based on risk metrics.

        Args:
            stock_price: Current stock price

        Returns:
            Number of shares to buy
        """
        if stock_price.amount <= 0:
            raise ValueError("Stock price must be positive")

        # Position size = Risk Amount / Stop Loss Distance
        shares_decimal = self.risk_amount.amount / self.stop_loss_distance
        return max(0, int(shares_decimal))


@dataclass(frozen=True)
class PositionSize:
    """Position sizing calculation results.

    Immutable value object containing the results of position sizing
    calculations with information about limiting factors.
    """

    shares: int
    investment_value: Money
    limiting_factor: str  # "risk", "max_position", or "min_investment"
    risk_metrics: RiskMetrics
    max_position_value: Money

    def __post_init__(self):
        """Validate position size after initialization."""
        if self.shares < 0:
            raise ValueError("Shares cannot be negative")

        if self.limiting_factor not in ["risk", "max_position", "min_investment"]:
            raise ValueError(f"Invalid limiting factor: {self.limiting_factor}")

    @classmethod
    def calculate(
        cls,
        risk_metrics: RiskMetrics,
        stock_price: Money,
        max_position_pct: Percentage,
        account_value: Money,
        min_investment: Money | None = None,
    ) -> "PositionSize":
        """Calculate position size from risk metrics and constraints.

        Args:
            risk_metrics: Risk calculation results
            stock_price: Current stock price
            max_position_pct: Maximum position as percentage of account
            account_value: Total account value
            min_investment: Minimum investment amount

        Returns:
            PositionSize with calculated values and limiting factor
        """
        # Calculate shares based on risk
        shares_from_risk = risk_metrics.calculate_position_shares(stock_price)

        # Calculate maximum shares based on position limit
        max_position_value = account_value * max_position_pct.to_decimal()
        max_shares = int(max_position_value.amount / stock_price.amount)

        # Determine limiting factor and final shares
        if shares_from_risk <= max_shares:
            final_shares = shares_from_risk
            limiting_factor = "risk"
        else:
            final_shares = max_shares
            limiting_factor = "max_position"

        # Check minimum investment constraint
        investment_value = Money(stock_price.amount * final_shares, stock_price.currency)
        if min_investment is not None and investment_value < min_investment:
            final_shares = 0
            investment_value = Money.zero(stock_price.currency)
            limiting_factor = "min_investment"

        return cls(
            shares=final_shares,
            investment_value=investment_value,
            limiting_factor=limiting_factor,
            risk_metrics=risk_metrics,
            max_position_value=max_position_value,
        )

    @property
    def is_limited_by_risk(self) -> bool:
        """Check if position was limited by risk calculation."""
        return self.limiting_factor == "risk"

    @property
    def is_limited_by_max_position(self) -> bool:
        """Check if position was limited by maximum position size."""
        return self.limiting_factor == "max_position"

    @property
    def is_limited_by_min_investment(self) -> bool:
        """Check if position was rejected due to minimum investment."""
        return self.limiting_factor == "min_investment"
