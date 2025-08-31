"""
Tests for portfolio rebalancing module.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from clenow_momentum.strategy.rebalancing import (
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    RebalancingOrder,
    calculate_target_weights,
    create_rebalancing_summary,
    generate_rebalancing_orders,
    load_portfolio_state,
    save_portfolio_state,
    simulate_rebalancing_execution,
)


class TestPosition:
    """Test Position class."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )

        assert pos.ticker == "AAPL"
        assert pos.shares == 100
        assert pos.entry_price == 150.0
        assert pos.current_price == 160.0
        assert pos.atr == 5.0

    def test_position_calculations(self):
        """Test position value calculations."""
        pos = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )

        assert pos.market_value == 16000.0  # 100 * 160
        assert pos.unrealized_pnl == 1000.0  # (160 - 150) * 100
        assert pos.unrealized_pnl_pct == pytest.approx(0.0667, rel=1e-3)  # 10/150


class TestPortfolio:
    """Test Portfolio class."""

    def test_empty_portfolio(self):
        """Test empty portfolio."""
        portfolio = Portfolio()

        assert portfolio.num_positions == 0
        assert portfolio.total_market_value == 0.0
        assert portfolio.total_value == 0.0
        assert portfolio.cash == 0.0

    def test_portfolio_with_positions(self):
        """Test portfolio with positions."""
        portfolio = Portfolio(cash=10000.0)

        pos1 = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )

        pos2 = Position(
            ticker="MSFT",
            shares=50,
            entry_price=300.0,
            current_price=310.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=8.0
        )

        portfolio.add_position(pos1)
        portfolio.add_position(pos2)

        assert portfolio.num_positions == 2
        assert portfolio.total_market_value == 31500.0  # 16000 + 15500
        assert portfolio.total_value == 41500.0  # 31500 + 10000 cash
        assert portfolio.cash == 10000.0

    def test_portfolio_to_dataframe(self):
        """Test converting portfolio to DataFrame."""
        portfolio = Portfolio()

        pos = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )

        portfolio.add_position(pos)
        df = portfolio.to_dataframe()

        assert len(df) == 1
        assert df.iloc[0]['ticker'] == "AAPL"
        assert df.iloc[0]['shares'] == 100
        assert df.iloc[0]['market_value'] == 16000.0


class TestRebalancingOrder:
    """Test RebalancingOrder class."""

    def test_order_creation(self):
        """Test creating a rebalancing order."""
        order = RebalancingOrder(
            ticker="AAPL",
            order_type=OrderType.BUY,
            shares=100,
            current_price=150.0,
            order_value=15000.0,
            reason="New position",
            priority=1
        )

        assert order.ticker == "AAPL"
        assert order.order_type == OrderType.BUY
        assert order.shares == 100
        assert order.order_value == 15000.0
        assert order.status == OrderStatus.PENDING

    def test_order_to_dict(self):
        """Test converting order to dictionary."""
        order = RebalancingOrder(
            ticker="AAPL",
            order_type=OrderType.SELL,
            shares=50,
            current_price=160.0,
            order_value=8000.0,
            reason="Reduce position"
        )

        order_dict = order.to_dict()
        assert order_dict['ticker'] == "AAPL"
        assert order_dict['order_type'] == "SELL"
        assert order_dict['shares'] == 50
        assert order_dict['status'] == "PENDING"


class TestPortfolioPersistence:
    """Test portfolio state persistence."""

    def test_save_and_load_portfolio(self, tmp_path):
        """Test saving and loading portfolio state."""
        # Create a portfolio with positions
        portfolio = Portfolio(cash=25000.0)
        portfolio.last_rebalance_date = datetime(2025, 1, 1, tzinfo=UTC)

        pos1 = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0,
            stop_loss=145.0
        )

        pos2 = Position(
            ticker="MSFT",
            shares=50,
            entry_price=300.0,
            current_price=310.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=8.0
        )

        portfolio.add_position(pos1)
        portfolio.add_position(pos2)

        # Save portfolio
        filepath = tmp_path / "test_portfolio.json"
        saved_path = save_portfolio_state(portfolio, filepath)
        assert saved_path == filepath
        assert filepath.exists()

        # Load portfolio
        loaded = load_portfolio_state(filepath)

        assert loaded.cash == 25000.0
        assert loaded.num_positions == 2
        assert loaded.last_rebalance_date.date() == datetime(2025, 1, 1, tzinfo=UTC).date()

        # Check positions
        assert "AAPL" in loaded.positions
        assert loaded.positions["AAPL"].shares == 100
        assert loaded.positions["AAPL"].entry_price == 150.0
        assert loaded.positions["AAPL"].stop_loss == 145.0

        assert "MSFT" in loaded.positions
        assert loaded.positions["MSFT"].shares == 50

    def test_load_nonexistent_portfolio(self, tmp_path):
        """Test loading non-existent portfolio file."""
        filepath = tmp_path / "nonexistent.json"
        portfolio = load_portfolio_state(filepath)

        assert portfolio.num_positions == 0
        assert portfolio.cash == 0.0
        assert portfolio.last_rebalance_date is None


class TestTargetWeights:
    """Test target weight calculation."""

    def test_calculate_target_weights(self):
        """Test calculating target portfolio weights."""
        # Create sample momentum stocks
        momentum_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'momentum_score': [2.5, 2.3, 2.1, 1.9, 1.7]
        })

        # Calculate weights for top 3
        target = calculate_target_weights(momentum_df, max_positions=3)

        assert len(target) == 3
        assert target['ticker'].tolist() == ['AAPL', 'MSFT', 'GOOGL']

        # Check equal weighting
        expected_weight = 1.0 / 3
        for _, row in target.iterrows():
            assert row['target_weight'] == pytest.approx(expected_weight)
            assert row['target_value_pct'] == pytest.approx(expected_weight * 100)

    def test_calculate_target_weights_empty(self):
        """Test with empty momentum stocks."""
        empty_df = pd.DataFrame()
        target = calculate_target_weights(empty_df)
        assert target.empty


class TestRebalancingOrders:
    """Test rebalancing order generation."""

    def test_generate_sell_orders_for_exit(self):
        """Test generating sell orders for positions to exit."""
        # Current portfolio with 2 positions
        portfolio = Portfolio(cash=10000.0)

        pos1 = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )

        pos2 = Position(
            ticker="MSFT",
            shares=50,
            entry_price=300.0,
            current_price=310.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=8.0
        )

        portfolio.add_position(pos1)
        portfolio.add_position(pos2)

        # Target portfolio with only GOOGL (exit AAPL and MSFT)
        target_df = pd.DataFrame({
            'ticker': ['GOOGL'],
            'current_price': [2800.0],
            'shares': [10],
            'investment': [28000.0]
        })

        stock_data = pd.DataFrame()  # Not used in this test

        orders = generate_rebalancing_orders(
            portfolio, target_df, stock_data, account_value=100000
        )

        # Should have 2 sell orders
        sell_orders = [o for o in orders if o.order_type == OrderType.SELL]
        assert len(sell_orders) == 2

        # Check sell orders
        tickers_to_sell = {o.ticker for o in sell_orders}
        assert tickers_to_sell == {'AAPL', 'MSFT'}

        # AAPL sell order
        aapl_order = next(o for o in sell_orders if o.ticker == 'AAPL')
        assert aapl_order.shares == 100
        assert aapl_order.order_value == 16000.0

        # MSFT sell order
        msft_order = next(o for o in sell_orders if o.ticker == 'MSFT')
        assert msft_order.shares == 50
        assert msft_order.order_value == 15500.0

    def test_generate_buy_orders_for_new_positions(self):
        """Test generating buy orders for new positions."""
        # Empty current portfolio
        portfolio = Portfolio(cash=100000.0)

        # Target portfolio with 2 new positions
        target_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'current_price': [160.0, 310.0],
            'shares': [100, 50],
            'investment': [16000.0, 15500.0]
        })

        stock_data = pd.DataFrame()

        orders = generate_rebalancing_orders(
            portfolio, target_df, stock_data, account_value=100000
        )

        # Should have 2 buy orders
        buy_orders = [o for o in orders if o.order_type == OrderType.BUY]
        assert len(buy_orders) == 2

        # Check buy orders
        tickers_to_buy = {o.ticker for o in buy_orders}
        assert tickers_to_buy == {'AAPL', 'MSFT'}

        # Check order values
        aapl_order = next(o for o in buy_orders if o.ticker == 'AAPL')
        assert aapl_order.shares == 100
        assert aapl_order.order_value == 16000.0

        msft_order = next(o for o in buy_orders if o.ticker == 'MSFT')
        assert msft_order.shares == 50
        assert msft_order.order_value == 15500.0

    def test_generate_mixed_rebalancing_orders(self):
        """Test generating mixed buy/sell orders."""
        # Current portfolio
        portfolio = Portfolio(cash=20000.0)

        pos1 = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )

        portfolio.add_position(pos1)

        # Target: Keep AAPL (reduced), add MSFT, remove others
        target_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'current_price': [160.0, 310.0],
            'shares': [50, 50],  # Reduce AAPL from 100 to 50
            'investment': [8000.0, 15500.0]
        })

        stock_data = pd.DataFrame()

        orders = generate_rebalancing_orders(
            portfolio, target_df, stock_data, account_value=100000
        )

        # Should have 1 sell (reduce AAPL) and 1 buy (new MSFT)
        sell_orders = [o for o in orders if o.order_type == OrderType.SELL]
        buy_orders = [o for o in orders if o.order_type == OrderType.BUY]

        assert len(sell_orders) == 1
        assert len(buy_orders) == 1

        # Check AAPL reduction
        aapl_sell = sell_orders[0]
        assert aapl_sell.ticker == 'AAPL'
        assert aapl_sell.shares == 50  # Sell 50 to go from 100 to 50

        # Check MSFT addition
        msft_buy = buy_orders[0]
        assert msft_buy.ticker == 'MSFT'
        assert msft_buy.shares == 50


class TestRebalancingSummary:
    """Test rebalancing summary creation."""

    def test_create_rebalancing_summary(self):
        """Test creating rebalancing summary."""
        # Setup portfolio and orders
        portfolio = Portfolio(cash=20000.0)

        pos = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )
        portfolio.add_position(pos)

        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.SELL,
                shares=100,
                current_price=160.0,
                order_value=16000.0,
                reason="Exit"
            ),
            RebalancingOrder(
                ticker="MSFT",
                order_type=OrderType.BUY,
                shares=50,
                current_price=310.0,
                order_value=15500.0,
                reason="New"
            )
        ]

        target_df = pd.DataFrame({
            'ticker': ['MSFT'],
            'shares': [50]
        })

        summary = create_rebalancing_summary(portfolio, orders, target_df)

        assert summary['current_positions'] == 1
        assert summary['target_positions'] == 1
        assert summary['num_orders'] == 2
        assert summary['num_sells'] == 1
        assert summary['num_buys'] == 1
        assert summary['total_sell_value'] == 16000.0
        assert summary['total_buy_value'] == 15500.0
        assert summary['expected_cash'] == 20500.0  # 20000 + 16000 - 15500
        assert summary['positions_to_remove'] == ['AAPL']
        assert summary['positions_to_add'] == ['MSFT']


class TestRebalancingSimulation:
    """Test rebalancing execution simulation."""

    def test_simulate_rebalancing(self):
        """Test simulating rebalancing execution."""
        # Initial portfolio
        portfolio = Portfolio(cash=20000.0)

        pos = Position(
            ticker="AAPL",
            shares=100,
            entry_price=150.0,
            current_price=160.0,
            entry_date=datetime(2025, 1, 1, tzinfo=UTC),
            atr=5.0
        )
        portfolio.add_position(pos)

        # Orders to execute
        orders = [
            RebalancingOrder(
                ticker="AAPL",
                order_type=OrderType.SELL,
                shares=50,  # Partial sell
                current_price=160.0,
                order_value=8000.0,
                reason="Reduce"
            ),
            RebalancingOrder(
                ticker="MSFT",
                order_type=OrderType.BUY,
                shares=30,
                current_price=310.0,
                order_value=9300.0,
                reason="New"
            )
        ]

        stock_data = pd.DataFrame()  # Not used

        # Simulate execution
        new_portfolio = simulate_rebalancing_execution(portfolio, orders, stock_data)

        # Check results
        assert new_portfolio.num_positions == 2
        assert new_portfolio.cash == pytest.approx(18700.0)  # 20000 + 8000 - 9300

        # Check AAPL position (reduced)
        assert new_portfolio.positions['AAPL'].shares == 50

        # Check MSFT position (new)
        assert 'MSFT' in new_portfolio.positions
        assert new_portfolio.positions['MSFT'].shares == 30
        assert new_portfolio.positions['MSFT'].entry_price == 310.0
