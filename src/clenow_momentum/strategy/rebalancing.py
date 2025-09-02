"""
Portfolio rebalancing logic for Clenow momentum strategy.

This module handles the generation of rebalancing orders by comparing
current portfolio positions with target allocations.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import pandas as pd
from loguru import logger


class OrderType(Enum):
    """Order type enumeration."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Position:
    """Represents a portfolio position with IBKR integration support."""

    ticker: str
    shares: int
    entry_price: float
    current_price: float
    entry_date: datetime
    atr: float
    stop_loss: float | None = None
    # IBKR-specific fields
    ibkr_position_id: str | None = None
    avg_cost_basis: float | None = None
    realized_pnl: float = 0.0
    last_updated: datetime | None = None

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def update_from_ibkr_position(self, ibkr_position) -> None:
        """
        Update position from IBKR position data.

        Args:
            ibkr_position: IBKR position object
        """
        if hasattr(ibkr_position, 'position'):
            self.shares = int(ibkr_position.position)

        if hasattr(ibkr_position, 'avgCost'):
            self.avg_cost_basis = ibkr_position.avgCost
            # Update entry price if we have better data
            if self.avg_cost_basis and self.avg_cost_basis > 0:
                self.entry_price = self.avg_cost_basis

        if hasattr(ibkr_position, 'marketPrice'):
            self.current_price = ibkr_position.marketPrice

        if hasattr(ibkr_position, 'realizedPNL'):
            self.realized_pnl = ibkr_position.realizedPNL

        self.last_updated = datetime.now(UTC)


@dataclass
class RebalancingOrder:
    """Represents a rebalancing order with IBKR integration support."""

    ticker: str
    order_type: OrderType
    shares: int
    current_price: float
    order_value: float
    reason: str
    priority: int = 0  # Lower number = higher priority
    status: OrderStatus = OrderStatus.PENDING
    execution_price: float | None = None
    execution_time: datetime | None = None
    # IBKR-specific fields
    ibkr_order_id: int | None = None
    ibkr_trade_id: str | None = None
    limit_price: float | None = None
    order_ref: str | None = None
    filled_quantity: int = 0
    avg_fill_price: float | None = None
    commission: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            "ticker": self.ticker,
            "order_type": self.order_type.value,
            "shares": self.shares,
            "current_price": self.current_price,
            "order_value": self.order_value,
            "reason": self.reason,
            "priority": self.priority,
            "status": self.status.value,
            "execution_price": self.execution_price,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            "ibkr_order_id": self.ibkr_order_id,
            "ibkr_trade_id": self.ibkr_trade_id,
            "limit_price": self.limit_price,
            "order_ref": self.order_ref,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "error_message": self.error_message,
        }

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.filled_quantity >= self.shares and self.status == OrderStatus.EXECUTED

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.shares

    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to be filled."""
        return max(0, self.shares - self.filled_quantity)

    def update_from_ibkr_trade(self, trade) -> None:
        """
        Update order status from IBKR Trade object.

        Args:
            trade: IBKR Trade object from ib_async
        """
        if hasattr(trade, 'order') and hasattr(trade, 'orderStatus'):
            order = trade.order
            status = trade.orderStatus

            # Update basic fields
            self.ibkr_order_id = order.orderId
            self.filled_quantity = status.filled
            self.avg_fill_price = status.avgFillPrice if status.avgFillPrice > 0 else None

            # Update status based on IBKR status
            ibkr_status = status.status.upper()
            if ibkr_status in ['FILLED']:
                self.status = OrderStatus.EXECUTED
                self.execution_price = status.avgFillPrice
                self.execution_time = datetime.now(UTC)
            elif ibkr_status in ['CANCELLED']:
                self.status = OrderStatus.CANCELLED
            elif ibkr_status in ['SUBMITTED', 'PRESUBMITTED']:
                self.status = OrderStatus.PENDING
            elif 'ERROR' in ibkr_status:
                self.status = OrderStatus.FAILED

        # Update commission if available
        if hasattr(trade, 'fills') and trade.fills:
            total_commission = sum(fill.commissionReport.commission
                                 for fill in trade.fills
                                 if fill.commissionReport)
            if total_commission:
                self.commission = total_commission


@dataclass
class Portfolio:
    """Represents the current portfolio state."""

    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    last_rebalance_date: datetime | None = None

    @property
    def total_market_value(self) -> float:
        """Calculate total portfolio market value."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value (positions + cash)."""
        return self.total_market_value + self.cash

    @property
    def num_positions(self) -> int:
        """Get number of positions."""
        return len(self.positions)

    def add_position(self, position: Position):
        """Add a position to the portfolio."""
        self.positions[position.ticker] = position
        logger.debug(f"Added position: {position.ticker} ({position.shares} shares)")

    def remove_position(self, ticker: str):
        """Remove a position from the portfolio."""
        if ticker in self.positions:
            del self.positions[ticker]
            logger.debug(f"Removed position: {ticker}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to DataFrame."""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for ticker, pos in self.positions.items():
            data.append(
                {
                    "ticker": ticker,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "entry_date": pos.entry_date,
                }
            )

        return pd.DataFrame(data)


def save_portfolio_state(portfolio: Portfolio, filepath: Path | None = None) -> Path:
    """
    Save portfolio state to JSON file.

    Args:
        portfolio: Portfolio to save
        filepath: Path to save file (defaults to data/portfolio_state.json)

    Returns:
        Path to saved file
    """
    if filepath is None:
        # Default location in project data directory
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        filepath = data_dir / "portfolio_state.json"

    state = {
        "cash": portfolio.cash,
        "last_rebalance_date": portfolio.last_rebalance_date.isoformat()
        if portfolio.last_rebalance_date
        else None,
        "positions": {},
    }

    for ticker, pos in portfolio.positions.items():
        state["positions"][ticker] = {
            "shares": pos.shares,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "entry_date": pos.entry_date.isoformat(),
            "atr": pos.atr,
            "stop_loss": pos.stop_loss,
        }

    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Saved portfolio state to {filepath}")
    return filepath


def load_portfolio_state(filepath: Path | None = None) -> Portfolio:
    """
    Load portfolio state from JSON file.

    Args:
        filepath: Path to state file (defaults to data/portfolio_state.json)

    Returns:
        Loaded portfolio or empty portfolio if file doesn't exist
    """
    if filepath is None:
        project_root = Path(__file__).parent.parent.parent.parent
        filepath = project_root / "data" / "portfolio_state.json"

    if not filepath.exists():
        logger.warning(f"Portfolio state file not found: {filepath}")
        return Portfolio()

    try:
        with open(filepath) as f:
            state = json.load(f)

        portfolio = Portfolio(cash=state.get("cash", 0.0))

        if state.get("last_rebalance_date"):
            portfolio.last_rebalance_date = datetime.fromisoformat(state["last_rebalance_date"])

        for ticker, pos_data in state.get("positions", {}).items():
            position = Position(
                ticker=ticker,
                shares=pos_data["shares"],
                entry_price=pos_data["entry_price"],
                current_price=pos_data["current_price"],
                entry_date=datetime.fromisoformat(pos_data["entry_date"]),
                atr=pos_data["atr"],
                stop_loss=pos_data.get("stop_loss"),
            )
            portfolio.add_position(position)

        logger.info(
            f"Loaded portfolio with {portfolio.num_positions} positions and ${portfolio.cash:,.2f} cash"
        )
        return portfolio

    except Exception as e:
        logger.error(f"Error loading portfolio state: {e}")
        return Portfolio()


def calculate_target_weights(
    momentum_stocks: pd.DataFrame, max_positions: int = 20
) -> pd.DataFrame:
    """
    Calculate target portfolio weights for momentum stocks.

    Equal-weight allocation among selected stocks.

    Args:
        momentum_stocks: DataFrame with momentum stocks
        max_positions: Maximum number of positions

    Returns:
        DataFrame with target weights
    """
    if momentum_stocks.empty:
        return pd.DataFrame()

    # Take top stocks up to max_positions
    target_stocks = momentum_stocks.head(max_positions).copy()

    # Equal weight allocation
    num_positions = len(target_stocks)
    target_weight = 1.0 / num_positions

    target_stocks["target_weight"] = target_weight
    target_stocks["target_value_pct"] = target_weight * 100

    logger.debug(
        f"Calculated equal weights for {num_positions} positions ({target_weight:.2%} each)"
    )

    return target_stocks


def generate_rebalancing_orders(
    current_portfolio: Portfolio,
    target_portfolio: pd.DataFrame,
    stock_data: pd.DataFrame,
    account_value: float,
    cash_buffer: float = 0.02,
) -> list[RebalancingOrder]:
    """
    Generate orders to rebalance from current to target portfolio.

    Args:
        current_portfolio: Current portfolio state
        target_portfolio: Target portfolio DataFrame with positions and sizes
        stock_data: Current stock data with prices
        account_value: Total account value for position sizing
        cash_buffer: Cash buffer to maintain (default 2%)

    Returns:
        List of rebalancing orders (sells first, then buys)
    """
    orders = []
    available_cash = current_portfolio.cash

    current_tickers = set(current_portfolio.positions.keys())
    target_tickers = set(target_portfolio["ticker"].values) if not target_portfolio.empty else set()

    # 1. Generate SELL orders for positions to exit
    tickers_to_sell = current_tickers - target_tickers

    for ticker in tickers_to_sell:
        position = current_portfolio.positions[ticker]

        # Build detailed exit reason
        reasons = []

        # Check if stock data is available for more context
        if not stock_data.empty and ticker in stock_data.columns.get_level_values(1):
            ticker_data = stock_data.xs(ticker, level=1, axis=1)
            if not ticker_data.empty:
                # Check for filter failures or momentum drop
                reasons.append("Position no longer meets momentum criteria")
        else:
            reasons.append("Not in target portfolio")

        reasons.append("Full position exit required for rebalancing")

        order = RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.SELL,
            shares=position.shares,
            current_price=position.current_price,
            order_value=position.market_value,
            reason=". ".join(reasons),
            priority=1,  # Sells have highest priority
        )
        orders.append(order)
        available_cash += position.market_value
        logger.debug(f"SELL order: {ticker} - {position.shares} shares (exit position)")

    # 2. Generate SELL orders for positions to reduce
    for ticker in current_tickers & target_tickers:
        current_pos = current_portfolio.positions[ticker]
        target_row = target_portfolio[target_portfolio["ticker"] == ticker].iloc[0]

        # Calculate target shares based on position sizing
        target_value = target_row["investment"]
        target_shares = int(target_value / current_pos.current_price)

        if target_shares < current_pos.shares:
            shares_to_sell = current_pos.shares - target_shares
            order_value = shares_to_sell * current_pos.current_price

            order = RebalancingOrder(
                ticker=ticker,
                order_type=OrderType.SELL,
                shares=shares_to_sell,
                current_price=current_pos.current_price,
                order_value=order_value,
                reason=f"Position size adjustment. Reducing from {current_pos.shares} to {target_shares} shares for optimal portfolio balance",
                priority=2,  # Partial sells have second priority
            )
            orders.append(order)
            available_cash += order_value
            logger.debug(f"SELL order: {ticker} - {shares_to_sell} shares (reduce position)")

    # 3. Generate BUY orders for new positions (in momentum rank order as provided)
    # Process target portfolio in order - it should already be sorted by momentum
    for _, target_row in target_portfolio.iterrows():
        ticker = target_row["ticker"]

        # Skip if not a new position
        if ticker in current_tickers:
            continue

        # Check if we have enough cash
        if target_row["investment"] > available_cash:
            logger.warning(
                f"Insufficient cash for {ticker}: need ${target_row['investment']:,.2f}, have ${available_cash:,.2f}"
            )
            continue

        # Build detailed buy reason with metrics
        reasons = []
        if "momentum_score" in target_row:
            reasons.append(f"Momentum score: {target_row['momentum_score']:.3f}")
        # Use portfolio_rank as the display rank (portfolio is already sorted by momentum)
        if "portfolio_rank" in target_row:
            reasons.append(f"Rank #{target_row['portfolio_rank']}")
        elif "momentum_rank" in target_row:
            reasons.append(f"Rank #{target_row['momentum_rank']}")
        reasons.append("New position entry based on strong momentum signal")

        order = RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.BUY,
            shares=int(target_row["shares"]),
            current_price=target_row["current_price"],
            order_value=target_row["investment"],
            reason=". ".join(reasons) if reasons else "New position - entering based on momentum signal",
            priority=3,  # New buys have third priority
        )

        # Add momentum metrics as attributes for display
        if "momentum_score" in target_row:
            order.momentum_score = target_row["momentum_score"]
        # Use portfolio_rank as momentum_rank (portfolio is already sorted by momentum)
        if "portfolio_rank" in target_row:
            order.momentum_rank = target_row["portfolio_rank"]
        elif "momentum_rank" in target_row:
            order.momentum_rank = target_row["momentum_rank"]
        if "r_squared" in target_row:
            order.r_squared = target_row["r_squared"]

        # Add weight percentage and filter information
        if "target_weight" in target_row:
            order.target_weight = target_row["target_weight"]
        elif "target_value_pct" in target_row:
            order.target_weight = target_row["target_value_pct"] / 100
        else:
            # Calculate if not present
            order.target_weight = target_row["investment"] / account_value

        # Add filter information
        order.filters_passed = []
        if "above_ma" in target_row and target_row.get("above_ma", False):
            order.filters_passed.append("Above 100-day MA")
        if "price_vs_ma" in target_row:
            pct_above_ma = target_row["price_vs_ma"] * 100
            order.filters_passed.append(f"Price {pct_above_ma:+.1f}% vs MA")
        if "latest_price" in target_row and "ma_100" in target_row:
            order.filters_passed.append(f"MA: ${target_row['ma_100']:.2f}")
        if "gap_detected" in target_row and not target_row.get("gap_detected", False):
            order.filters_passed.append("No significant gaps")
        if "index_member" in target_row and target_row.get("index_member", False):
            order.filters_passed.append("S&P 500 member")
        if "high_above_low" in target_row:
            pct_above_low = (target_row["high_above_low"] - 1) * 100
            order.filters_passed.append(f"Within {pct_above_low:.1f}% of 52w high")
        
        # Add volatility/ATR information
        if "atr" in target_row:
            order.atr = target_row["atr"]
            order.volatility_pct = (target_row["atr"] / target_row["current_price"]) * 100
        if "stop_loss" in target_row:
            order.stop_loss = target_row["stop_loss"]
        if "actual_risk" in target_row:
            order.risk_amount = target_row["actual_risk"]
        orders.append(order)
        available_cash -= target_row["investment"]
        logger.debug(f"BUY order: {ticker} - {int(target_row['shares'])} shares (new position)")

    # 4. Generate BUY orders for positions to increase
    for ticker in current_tickers & target_tickers:
        current_pos = current_portfolio.positions[ticker]
        target_row = target_portfolio[target_portfolio["ticker"] == ticker].iloc[0]

        target_shares = int(target_row["shares"])

        if target_shares > current_pos.shares:
            shares_to_buy = target_shares - current_pos.shares
            order_value = shares_to_buy * target_row["current_price"]

            # Check if we have enough cash
            if order_value > available_cash:
                logger.warning(
                    f"Insufficient cash to increase {ticker}: need ${order_value:,.2f}, have ${available_cash:,.2f}"
                )
                continue

            order = RebalancingOrder(
                ticker=ticker,
                order_type=OrderType.BUY,
                shares=shares_to_buy,
                current_price=target_row["current_price"],
                order_value=order_value,
                reason=f"Position size increase. Adding {shares_to_buy} shares to reach target allocation of {target_shares} total shares",
                priority=4,  # Increases have lowest priority
            )

            # Add momentum metrics if available
            if "momentum_score" in target_row:
                order.momentum_score = target_row["momentum_score"]
            # Use portfolio_rank as momentum_rank (portfolio is already sorted by momentum)
            if "portfolio_rank" in target_row:
                order.momentum_rank = target_row["portfolio_rank"]
            elif "momentum_rank" in target_row:
                order.momentum_rank = target_row["momentum_rank"]

            # Add weight percentage
            if "target_weight" in target_row:
                order.target_weight = target_row["target_weight"]
            elif "target_value_pct" in target_row:
                order.target_weight = target_row["target_value_pct"] / 100
            else:
                order.target_weight = target_row["investment"] / account_value
            
            # Add volatility/ATR information
            if "atr" in target_row:
                order.atr = target_row["atr"]
                order.volatility_pct = (target_row["atr"] / target_row["current_price"]) * 100
            if "stop_loss" in target_row:
                order.stop_loss = target_row["stop_loss"]
            if "actual_risk" in target_row:
                order.risk_amount = target_row["actual_risk"]
            orders.append(order)
            available_cash -= order_value
            logger.debug(f"BUY order: {ticker} - {shares_to_buy} shares (increase position)")

    # Sort orders by priority (sells first, then buys)
    orders.sort(key=lambda x: x.priority)

    # Log summary
    total_sells = sum(o.order_value for o in orders if o.order_type == OrderType.SELL)
    total_buys = sum(o.order_value for o in orders if o.order_type == OrderType.BUY)
    num_sells = len([o for o in orders if o.order_type == OrderType.SELL])
    num_buys = len([o for o in orders if o.order_type == OrderType.BUY])

    logger.info(f"Generated {len(orders)} rebalancing orders:")
    logger.info(f"  - {num_sells} SELL orders: ${total_sells:,.2f}")
    logger.info(f"  - {num_buys} BUY orders: ${total_buys:,.2f}")
    logger.info(f"  - Net cash flow: ${total_sells - total_buys:,.2f}")
    logger.info(f"  - Final cash position: ${available_cash:,.2f}")

    return orders


def create_rebalancing_summary(
    current_portfolio: Portfolio, orders: list[RebalancingOrder], target_portfolio: pd.DataFrame
) -> dict:
    """
    Create a summary of the rebalancing operation.

    Args:
        current_portfolio: Current portfolio state
        orders: List of rebalancing orders
        target_portfolio: Target portfolio configuration

    Returns:
        Dictionary with rebalancing summary
    """
    summary = {
        "current_positions": current_portfolio.num_positions,
        "target_positions": len(target_portfolio) if not target_portfolio.empty else 0,
        "current_value": current_portfolio.total_value,
        "current_cash": current_portfolio.cash,
        "num_orders": len(orders),
        "num_sells": len([o for o in orders if o.order_type == OrderType.SELL]),
        "num_buys": len([o for o in orders if o.order_type == OrderType.BUY]),
        "total_sell_value": sum(o.order_value for o in orders if o.order_type == OrderType.SELL),
        "total_buy_value": sum(o.order_value for o in orders if o.order_type == OrderType.BUY),
    }

    # Calculate portfolio turnover
    if current_portfolio.total_market_value > 0:
        sell_value = summary["total_sell_value"]
        portfolio_value = current_portfolio.total_market_value
        summary["turnover_pct"] = (sell_value / portfolio_value) * 100
    else:
        summary["turnover_pct"] = 100.0 if summary["num_buys"] > 0 else 0.0

    # Expected cash after rebalancing
    summary["expected_cash"] = (
        current_portfolio.cash + summary["total_sell_value"] - summary["total_buy_value"]
    )

    # Positions being added/removed
    current_tickers = set(current_portfolio.positions.keys())
    target_tickers = set(target_portfolio["ticker"].values) if not target_portfolio.empty else set()

    summary["positions_to_add"] = list(target_tickers - current_tickers)
    summary["positions_to_remove"] = list(current_tickers - target_tickers)
    summary["positions_to_keep"] = list(current_tickers & target_tickers)

    return summary


def simulate_rebalancing_execution(
    portfolio: Portfolio, orders: list[RebalancingOrder], stock_data: pd.DataFrame
) -> Portfolio:
    """
    Simulate the execution of rebalancing orders.

    This creates a new portfolio state after applying all orders.

    Args:
        portfolio: Current portfolio
        orders: Rebalancing orders to execute
        stock_data: Current stock data

    Returns:
        New portfolio state after rebalancing
    """
    # Create a copy of the portfolio
    new_portfolio = Portfolio(cash=portfolio.cash, last_rebalance_date=datetime.now(UTC))

    # Copy existing positions
    for _ticker, pos in portfolio.positions.items():
        new_portfolio.add_position(pos)

    # Execute orders
    for order in orders:
        if order.order_type == OrderType.SELL:
            if order.ticker in new_portfolio.positions:
                pos = new_portfolio.positions[order.ticker]
                if order.shares >= pos.shares:
                    # Full exit
                    new_portfolio.cash += pos.market_value
                    new_portfolio.remove_position(order.ticker)
                else:
                    # Partial sell
                    new_portfolio.cash += order.order_value
                    pos.shares -= order.shares

        elif order.order_type == OrderType.BUY:
            if order.ticker in new_portfolio.positions:
                # Add to existing position
                pos = new_portfolio.positions[order.ticker]
                # Calculate new average price
                total_value = (pos.shares * pos.entry_price) + order.order_value
                total_shares = pos.shares + order.shares
                pos.entry_price = total_value / total_shares
                pos.shares = total_shares
                pos.current_price = order.current_price
            else:
                # New position
                position = Position(
                    ticker=order.ticker,
                    shares=order.shares,
                    entry_price=order.current_price,
                    current_price=order.current_price,
                    entry_date=datetime.now(UTC),
                    atr=0.0,  # Would be calculated from stock data in production
                )
                new_portfolio.add_position(position)

            new_portfolio.cash -= order.order_value

    logger.info(
        f"Simulated rebalancing: {new_portfolio.num_positions} positions, ${new_portfolio.cash:,.2f} cash"
    )

    return new_portfolio
