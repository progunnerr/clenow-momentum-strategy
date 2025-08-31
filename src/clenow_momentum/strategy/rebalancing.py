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
    """Represents a portfolio position."""
    ticker: str
    shares: int
    entry_price: float
    current_price: float
    entry_date: datetime
    atr: float
    stop_loss: float | None = None

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


@dataclass
class RebalancingOrder:
    """Represents a rebalancing order."""
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

    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            'ticker': self.ticker,
            'order_type': self.order_type.value,
            'shares': self.shares,
            'current_price': self.current_price,
            'order_value': self.order_value,
            'reason': self.reason,
            'priority': self.priority,
            'status': self.status.value,
        }


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
            data.append({
                'ticker': ticker,
                'shares': pos.shares,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'entry_date': pos.entry_date,
            })

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
        'cash': portfolio.cash,
        'last_rebalance_date': portfolio.last_rebalance_date.isoformat() if portfolio.last_rebalance_date else None,
        'positions': {}
    }

    for ticker, pos in portfolio.positions.items():
        state['positions'][ticker] = {
            'shares': pos.shares,
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'entry_date': pos.entry_date.isoformat(),
            'atr': pos.atr,
            'stop_loss': pos.stop_loss,
        }

    with open(filepath, 'w') as f:
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

        portfolio = Portfolio(
            cash=state.get('cash', 0.0)
        )

        if state.get('last_rebalance_date'):
            portfolio.last_rebalance_date = datetime.fromisoformat(state['last_rebalance_date'])

        for ticker, pos_data in state.get('positions', {}).items():
            position = Position(
                ticker=ticker,
                shares=pos_data['shares'],
                entry_price=pos_data['entry_price'],
                current_price=pos_data['current_price'],
                entry_date=datetime.fromisoformat(pos_data['entry_date']),
                atr=pos_data['atr'],
                stop_loss=pos_data.get('stop_loss')
            )
            portfolio.add_position(position)

        logger.info(f"Loaded portfolio with {portfolio.num_positions} positions and ${portfolio.cash:,.2f} cash")
        return portfolio

    except Exception as e:
        logger.error(f"Error loading portfolio state: {e}")
        return Portfolio()


def calculate_target_weights(
    momentum_stocks: pd.DataFrame,
    max_positions: int = 20
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

    target_stocks['target_weight'] = target_weight
    target_stocks['target_value_pct'] = target_weight * 100

    logger.debug(f"Calculated equal weights for {num_positions} positions ({target_weight:.2%} each)")

    return target_stocks


def generate_rebalancing_orders(
    current_portfolio: Portfolio,
    target_portfolio: pd.DataFrame,
    stock_data: pd.DataFrame,
    account_value: float,
    cash_buffer: float = 0.02
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
    account_value * (1 - cash_buffer)

    current_tickers = set(current_portfolio.positions.keys())
    target_tickers = set(target_portfolio['ticker'].values) if not target_portfolio.empty else set()

    # 1. Generate SELL orders for positions to exit
    tickers_to_sell = current_tickers - target_tickers

    for ticker in tickers_to_sell:
        position = current_portfolio.positions[ticker]
        order = RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.SELL,
            shares=position.shares,
            current_price=position.current_price,
            order_value=position.market_value,
            reason="Not in target portfolio - full exit",
            priority=1  # Sells have highest priority
        )
        orders.append(order)
        available_cash += position.market_value
        logger.debug(f"SELL order: {ticker} - {position.shares} shares (exit position)")

    # 2. Generate SELL orders for positions to reduce
    for ticker in current_tickers & target_tickers:
        current_pos = current_portfolio.positions[ticker]
        target_row = target_portfolio[target_portfolio['ticker'] == ticker].iloc[0]

        # Calculate target shares based on position sizing
        target_value = target_row['investment']
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
                reason=f"Reduce position from {current_pos.shares} to {target_shares} shares",
                priority=2  # Partial sells have second priority
            )
            orders.append(order)
            available_cash += order_value
            logger.debug(f"SELL order: {ticker} - {shares_to_sell} shares (reduce position)")

    # 3. Generate BUY orders for new positions
    tickers_to_buy = target_tickers - current_tickers

    for ticker in tickers_to_buy:
        target_row = target_portfolio[target_portfolio['ticker'] == ticker].iloc[0]

        # Check if we have enough cash
        if target_row['investment'] > available_cash:
            logger.warning(f"Insufficient cash for {ticker}: need ${target_row['investment']:,.2f}, have ${available_cash:,.2f}")
            continue

        order = RebalancingOrder(
            ticker=ticker,
            order_type=OrderType.BUY,
            shares=int(target_row['shares']),
            current_price=target_row['current_price'],
            order_value=target_row['investment'],
            reason="New position - entering based on momentum signal",
            priority=3  # New buys have third priority
        )
        orders.append(order)
        available_cash -= target_row['investment']
        logger.debug(f"BUY order: {ticker} - {int(target_row['shares'])} shares (new position)")

    # 4. Generate BUY orders for positions to increase
    for ticker in current_tickers & target_tickers:
        current_pos = current_portfolio.positions[ticker]
        target_row = target_portfolio[target_portfolio['ticker'] == ticker].iloc[0]

        target_shares = int(target_row['shares'])

        if target_shares > current_pos.shares:
            shares_to_buy = target_shares - current_pos.shares
            order_value = shares_to_buy * target_row['current_price']

            # Check if we have enough cash
            if order_value > available_cash:
                logger.warning(f"Insufficient cash to increase {ticker}: need ${order_value:,.2f}, have ${available_cash:,.2f}")
                continue

            order = RebalancingOrder(
                ticker=ticker,
                order_type=OrderType.BUY,
                shares=shares_to_buy,
                current_price=target_row['current_price'],
                order_value=order_value,
                reason=f"Increase position from {current_pos.shares} to {target_shares} shares",
                priority=4  # Increases have lowest priority
            )
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
    current_portfolio: Portfolio,
    orders: list[RebalancingOrder],
    target_portfolio: pd.DataFrame
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
        'current_positions': current_portfolio.num_positions,
        'target_positions': len(target_portfolio) if not target_portfolio.empty else 0,
        'current_value': current_portfolio.total_value,
        'current_cash': current_portfolio.cash,
        'num_orders': len(orders),
        'num_sells': len([o for o in orders if o.order_type == OrderType.SELL]),
        'num_buys': len([o for o in orders if o.order_type == OrderType.BUY]),
        'total_sell_value': sum(o.order_value for o in orders if o.order_type == OrderType.SELL),
        'total_buy_value': sum(o.order_value for o in orders if o.order_type == OrderType.BUY),
    }

    # Calculate portfolio turnover
    if current_portfolio.total_market_value > 0:
        sell_value = summary['total_sell_value']
        portfolio_value = current_portfolio.total_market_value
        summary['turnover_pct'] = (sell_value / portfolio_value) * 100
    else:
        summary['turnover_pct'] = 100.0 if summary['num_buys'] > 0 else 0.0

    # Expected cash after rebalancing
    summary['expected_cash'] = (
        current_portfolio.cash +
        summary['total_sell_value'] -
        summary['total_buy_value']
    )

    # Positions being added/removed
    current_tickers = set(current_portfolio.positions.keys())
    target_tickers = set(target_portfolio['ticker'].values) if not target_portfolio.empty else set()

    summary['positions_to_add'] = list(target_tickers - current_tickers)
    summary['positions_to_remove'] = list(current_tickers - target_tickers)
    summary['positions_to_keep'] = list(current_tickers & target_tickers)

    return summary


def simulate_rebalancing_execution(
    portfolio: Portfolio,
    orders: list[RebalancingOrder],
    stock_data: pd.DataFrame
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
    new_portfolio = Portfolio(
        cash=portfolio.cash,
        last_rebalance_date=datetime.now(UTC)
    )

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
                    atr=0.0  # Would be calculated from stock data in production
                )
                new_portfolio.add_position(position)

            new_portfolio.cash -= order.order_value

    logger.info(f"Simulated rebalancing: {new_portfolio.num_positions} positions, ${new_portfolio.cash:,.2f} cash")

    return new_portfolio
