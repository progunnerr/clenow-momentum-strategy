#!/usr/bin/env python3
"""
Run momentum analysis on S&P 500 stocks.

This script fetches S&P 500 data and displays the top momentum stocks
according to Clenow's momentum strategy.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Initialize logger configuration
from clenow_momentum.data.fetcher import get_sp500_tickers, get_stock_data
from clenow_momentum.indicators.filters import apply_all_filters
from clenow_momentum.indicators.momentum import (
    calculate_momentum_for_universe,
    get_top_momentum_stocks,
)
from clenow_momentum.indicators.risk import apply_risk_limits, build_portfolio
from clenow_momentum.strategy.market_regime import check_market_regime, should_trade_momentum
from clenow_momentum.strategy.rebalancing import (
    Portfolio,
    create_rebalancing_summary,
    generate_rebalancing_orders,
    load_portfolio_state,
)
from clenow_momentum.strategy.trading_schedule import (
    get_trading_calendar_summary,
    should_execute_trades,
)
from clenow_momentum.utils.config import get_position_sizing_guide, load_config, validate_config


def print_trading_schedule_status(config):
    """Print trading schedule status."""
    print("TRADING SCHEDULE STATUS")
    print("-" * 30)

    bypass_wednesday = config.get('bypass_wednesday_check', False)
    calendar_summary = get_trading_calendar_summary(bypass_wednesday)
    can_trade, trade_reason = should_execute_trades(bypass_wednesday=bypass_wednesday)

    print(f"📅 Today: {calendar_summary['current_date']} ({calendar_summary['current_weekday']})")
    if bypass_wednesday:
        print("⚠️  TESTING MODE: Wednesday-only restriction bypassed")
    print(f"🎯 Trading Day: {'✅ Yes' if calendar_summary['is_trading_day'] else '❌ No'}")
    print(f"🔄 Rebalancing Day: {'✅ Yes' if calendar_summary['is_rebalancing_day'] else '❌ No'}")
    print(f"📊 Trading Status: {trade_reason}")

    if not can_trade:
        print(f"\n⏭️  Next Trading Day: {calendar_summary['next_trading_day']} (in {calendar_summary['days_until_next_trading']} days)")

    print(f"🔄 Next Rebalancing: {calendar_summary['next_rebalancing_date']} (in {calendar_summary['days_until_next_rebalancing']} days)")
    print()

    return calendar_summary, can_trade, trade_reason


def print_account_sizing(config):
    """Print account and position sizing information."""
    sizing_guide = get_position_sizing_guide(config['account_value'])
    print("ACCOUNT & POSITION SIZING")
    print("-" * 30)
    print(f"💰 Account Value: ${config['account_value']:,.0f}")
    print(f"🎯 Risk per Trade: {config['risk_per_trade']:.3%} (${sizing_guide['risk_per_trade_dollars']:,.0f})")
    print(f"📊 Target Positions: {sizing_guide['recommended_positions']} ({sizing_guide['risk_level']})")
    print(f"⚖️  Min Position Size: ${config['min_position_value']:,.0f}")

    # Show any configuration warnings
    warnings = validate_config(config)
    if warnings:
        print("\nCONFIGURATION NOTES:")
        for warning in warnings:
            print(f"  {warning}")
    print()


def fetch_and_process_data(tickers, config):
    """Fetch stock data and calculate momentum scores."""
    # Get stock data
    print("Step 2: Fetching stock data (6 months)...")
    print("⏳ This may take a moment...")
    print(f"📈 Processing all {len(tickers)} S&P 500 stocks")

    stock_data = get_stock_data(tickers, period="6mo")

    if stock_data is None:
        print("❌ Could not retrieve stock data. Exiting.")
        return None, None

    print("✅ Stock data retrieved successfully")
    print(f"Data shape: {stock_data.shape}")
    print(
        f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}"
    )
    print()

    # Calculate momentum scores
    print(f"Step 3: Calculating momentum scores ({config['momentum_period']}-day period)...")
    momentum_df = calculate_momentum_for_universe(stock_data, period=config['momentum_period'])

    valid_scores = momentum_df.dropna(subset=["momentum_score"])
    print(f"✅ Calculated momentum for {len(valid_scores)} stocks")
    print()

    return stock_data, momentum_df


def check_and_filter_stocks(momentum_df, stock_data, config, trading_allowed):
    """Apply filters and select stocks for portfolio."""
    stocks_for_portfolio = pd.DataFrame()

    if trading_allowed:
        # Filter the entire universe of stocks with valid momentum scores
        stocks_to_filter = momentum_df.dropna(subset=['momentum_score'])
        print(f"🔍 Applying filters to all {len(stocks_to_filter)} stocks with valid momentum scores...")
        filtered_stocks = apply_all_filters(
            stocks_to_filter,
            stock_data,
            ma_period=config['ma_filter_period'],
            gap_threshold=config['gap_threshold']
        )

        print(f"✅ {len(filtered_stocks)} stocks passed all filters.")

        # For efficiency, select top stocks for portfolio construction before sizing
        stocks_for_portfolio = filtered_stocks.head(config['max_positions'])
        print(f"📈 Selecting top {len(stocks_for_portfolio)} momentum stocks for portfolio (max_positions: {config['max_positions']}).")

        final_stocks = filtered_stocks  # Keep all filtered stocks for display
    else:
        print("⛔ Market regime not favorable - skipping individual stock filters")
        print(f"📊 Showing top {config['top_momentum_pct']:.0%} momentum rankings for informational purposes only...")
        # For informational purposes, show the top 20% unfiltered
        final_stocks = get_top_momentum_stocks(momentum_df, top_pct=config['top_momentum_pct'])

    return stocks_for_portfolio, final_stocks


def display_portfolio_table(portfolio_df, config):
    """Display portfolio table with position sizing."""
    print("🎯 PORTFOLIO WITH POSITION SIZING")
    print("=" * 60)

    # Prepare portfolio table data
    table_data = []
    for _, row in portfolio_df.iterrows():
        table_data.append([
            int(row["portfolio_rank"]),
            row["ticker"],
            f"{row['momentum_score']:.3f}",
            f"${row['current_price']:.2f}",
            f"${row['atr']:.2f}",
            int(row['shares']),
            f"${row['investment']:,.0f}",
            f"{row['position_pct']:.1%}",
            f"${row['actual_risk']:,.0f}"
        ])

    headers = ["Rank", "Ticker", "Momentum", "Price", "ATR", "Shares", "Investment", "% Port", "Risk $"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Portfolio summary
    total_investment = portfolio_df['investment'].sum()
    total_positions = len(portfolio_df)
    avg_position = total_investment / total_positions
    total_risk = portfolio_df['actual_risk'].sum()

    print("PORTFOLIO SUMMARY")
    print("-" * 50)
    print(f"💰 Account Value: ${config['account_value']:,.0f}")
    print(f"📊 Total Positions: {total_positions}")
    print(f"💼 Total Investment: ${total_investment:,.0f}")
    print(f"💵 Cash Remaining: ${config['account_value'] - total_investment:,.0f}")
    print(f"📈 Capital Utilization: {total_investment/config['account_value']:.1%}")
    print(f"⚖️  Average Position: ${avg_position:,.0f}")
    print(f"⚠️  Total Portfolio Risk: ${total_risk:,.0f}")
    print(f"🎯 Risk per Trade: {config['risk_per_trade']:.3%} (${config['account_value'] * config['risk_per_trade']:,.0f})")


def display_stocks_table(final_stocks, config, trading_allowed, valid_scores, market_regime):
    """Display filtered stocks table without portfolio."""
    table_data = []
    for i, (_, row) in enumerate(final_stocks.iterrows()):
        # Check if we have filter data (MA info)
        if 'latest_price' in row and not pd.isna(row['latest_price']):
            price_val = row['latest_price']
            ma_info = f" (vs MA: {row.get('price_vs_ma', 0):+.1%})" if 'price_vs_ma' in row else ""
        else:
            price_val = row.get('current_price', 0)
            ma_info = ""

        table_data.append([
            i + 1,  # Re-rank after filtering
            row["ticker"],
            f"{row['momentum_score']:.3f}",
            f"{row['annualized_slope']:.3f}",
            f"{row['r_squared']:.3f}",
            f"${price_val:.2f}{ma_info}" if not pd.isna(price_val) else "N/A",
            f"{row['period_return_pct']:+.1f}%" if not pd.isna(row["period_return_pct"]) else "N/A",
        ])

    headers = ["Rank", "Ticker", "Momentum", "Ann. Slope", "R²", "Price", "90d Return"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Enhanced summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 50)
    print(f"📊 Stocks with valid scores: {len(valid_scores)}")
    if trading_allowed:
        print(f"✅ Stocks passing all filters: {len(final_stocks)}")
        if len(valid_scores) > 0:
            filter_rate = len(final_stocks) / len(valid_scores) * 100
            print(f"🔍 Filter pass rate: {filter_rate:.1f}%")
    else:
        # In the "no trade" case, final_stocks is the top 20%
        print(f"🎯 Top {config['top_momentum_pct']:.0%} stocks shown: {len(final_stocks)}")

    print(f"📈 Average Momentum Score: {final_stocks['momentum_score'].mean():.3f}")
    print(f"📐 Average R-squared: {final_stocks['r_squared'].mean():.3f}")
    print(f"📊 Average 90-day Return: {final_stocks['period_return_pct'].mean():.1f}%")

    if trading_allowed and len(final_stocks) > 0:
        print(f"🎯 Market Regime: {market_regime['regime'].upper()}")
        print("💼 Ready for position sizing and portfolio construction")
    elif not trading_allowed:
        print(f"⛔ Trading suspended due to {market_regime['regime']} market regime")
    print()


def main():
    """Main analysis function."""
    print("=" * 60)
    print("CLENOW MOMENTUM STRATEGY ANALYSIS")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    # Load configuration
    config = load_config()

    # Step 0: Check Trading Schedule
    calendar_summary, can_trade, trade_reason = print_trading_schedule_status(config)

    # Show position sizing guidance
    print_account_sizing(config)

    # Step 1: Get S&P 500 tickers
    print("Step 1: Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()

    if not tickers:
        print("❌ Could not retrieve tickers. Exiting.")
        return 1

    print(f"✅ Successfully fetched {len(tickers)} tickers")
    print("Sample tickers:", tickers[:10])
    print()

    # Step 2 & 3: Get stock data and calculate momentum
    stock_data, momentum_df = fetch_and_process_data(tickers, config)
    if stock_data is None:
        return 1

    valid_scores = momentum_df.dropna(subset=["momentum_score"])

    # Step 4: Check market regime
    print(f"Step 4: Checking market regime (SPX vs {config['market_regime_period']}-day MA)...")
    market_regime = check_market_regime(period=config['market_regime_period'])
    trading_allowed, regime_reason = should_trade_momentum(market_regime)

    print(f"📊 Market Regime: {market_regime.get('regime', 'unknown').upper()}")
    print(f"📈 SPX: ${market_regime.get('current_price', 'N/A')} vs {config['market_regime_period']}MA: ${market_regime.get('ma_value', 'N/A')}")
    print(f"🎯 Trading Status: {'✅ ALLOWED' if trading_allowed else '⛔ SUSPENDED'}")
    print(f"📝 Reason: {regime_reason}")
    print()

    # Step 5: Apply trading filters to find eligible stocks
    print("Step 5: Applying trading filters...")
    stocks_for_portfolio, final_stocks = check_and_filter_stocks(momentum_df, stock_data, config, trading_allowed)
    print()

    # Step 6: Portfolio Construction (if trading allowed)
    portfolio_df = pd.DataFrame()
    if trading_allowed and not stocks_for_portfolio.empty:
        print("Step 6: Portfolio Construction & Position Sizing...")

        # Build portfolio with position sizing (Clenow's risk parity method)
        portfolio_df = build_portfolio(
            filtered_stocks=stocks_for_portfolio,
            stock_data=stock_data,
            account_value=config['account_value'],
            risk_per_trade=config['risk_per_trade'],
            atr_period=config['atr_period'],
            allocation_method="equal_risk",
            stop_loss_multiplier=config['stop_loss_multiplier'],
            max_position_pct=config['max_position_pct']
        )

        if not portfolio_df.empty:
            # Apply minimum position value limit. max_positions is already handled.
            portfolio_df = apply_risk_limits(
                portfolio_df=portfolio_df,
                max_positions=None, # Already sliced to max_positions
                min_position_value=config['min_position_value']
            )

            print(f"✅ Portfolio constructed with {len(portfolio_df)} positions")
        else:
            print("❌ Failed to construct portfolio")

        print()

    # Step 7: Display results
    print("=" * 60)
    if trading_allowed:
        print("TOP MOMENTUM STOCKS (FILTERED)")
        print("🎯 These stocks passed all Clenow strategy filters")
    else:
        print("TOP MOMENTUM STOCKS (UNFILTERED)")
        print("⚠️  For informational purposes only - market regime not favorable")
    print("=" * 60)

    if final_stocks.empty:
        print("❌ No stocks passed all filters")
        print()
    else:
        # Display portfolio if constructed, otherwise show stock list
        if not portfolio_df.empty:
            display_portfolio_table(portfolio_df, config)
        else:
            display_stocks_table(final_stocks, config, trading_allowed, valid_scores, market_regime)

    # Step 8: Rebalancing Analysis (if it's a rebalancing day and we have a portfolio)
    if calendar_summary['is_rebalancing_day'] and not portfolio_df.empty and config.get('simulate_rebalancing', True):
        print()
        print("=" * 60)
        print("🔄 REBALANCING ANALYSIS")
        print("=" * 60)

        # Load current portfolio
        from pathlib import Path
        portfolio_file = Path(config.get('portfolio_state_file', 'data/portfolio_state.json'))
        current_portfolio = load_portfolio_state(portfolio_file)

        if current_portfolio.num_positions > 0:
            print(f"📊 Current Portfolio: {current_portfolio.num_positions} positions, ${current_portfolio.cash:,.0f} cash")

            # Generate rebalancing orders
            rebalancing_orders = generate_rebalancing_orders(
                current_portfolio=current_portfolio,
                target_portfolio=portfolio_df,
                stock_data=stock_data,
                account_value=config['account_value'],
                cash_buffer=config.get('cash_buffer', 0.02)
            )

            if rebalancing_orders:
                print(f"📝 Generated {len(rebalancing_orders)} rebalancing orders")
                print()

                # Show rebalancing summary
                summary = create_rebalancing_summary(current_portfolio, rebalancing_orders, portfolio_df)

                print("REBALANCING SUMMARY")
                print("-" * 50)
                print(f"Current Positions: {summary['current_positions']}")
                print(f"Target Positions: {summary['target_positions']}")
                print(f"Portfolio Turnover: {summary['turnover_pct']:.1f}%")
                print()
                print("Orders to Execute:")
                print(f"  - Sells: {summary['num_sells']} (${summary['total_sell_value']:,.0f})")
                print(f"  - Buys: {summary['num_buys']} (${summary['total_buy_value']:,.0f})")
                print()
                print(f"Positions to Add: {', '.join(summary['positions_to_add']) if summary['positions_to_add'] else 'None'}")
                print(f"Positions to Remove: {', '.join(summary['positions_to_remove']) if summary['positions_to_remove'] else 'None'}")
                print(f"Positions to Keep: {len(summary['positions_to_keep'])}")
                print()
                print(f"Expected Cash After: ${summary['expected_cash']:,.0f}")

                # Show detailed orders
                print()
                print("REBALANCING ORDERS")
                print("-" * 50)

                order_data = []
                # Show all SELL orders first
                sell_orders = [o for o in rebalancing_orders if o.order_type.value == "SELL"]
                buy_orders = [o for o in rebalancing_orders if o.order_type.value == "BUY"]

                print(f"\n📉 SELL ORDERS ({len(sell_orders)} total):")
                for order in sell_orders:
                    order_data.append([
                        order.order_type.value,
                        order.ticker,
                        order.shares,
                        f"${order.current_price:.2f}",
                        f"${order.order_value:,.0f}",
                        order.reason[:40] + "..." if len(order.reason) > 40 else order.reason
                    ])

                if sell_orders:
                    headers = ["Type", "Ticker", "Shares", "Price", "Value", "Reason"]
                    print(tabulate(order_data, headers=headers, tablefmt="grid"))

                print(f"\n📈 BUY ORDERS ({len(buy_orders)} total):")
                order_data = []
                for order in buy_orders:
                    order_data.append([
                        order.order_type.value,
                        order.ticker,
                        order.shares,
                        f"${order.current_price:.2f}",
                        f"${order.order_value:,.0f}",
                        order.reason[:40] + "..." if len(order.reason) > 40 else order.reason
                    ])

                if buy_orders:
                    print(tabulate(order_data, headers=headers, tablefmt="grid"))

                # Add execution note
                print("\n💡 EXECUTION NOTE:")
                print("Execute SELL orders first to free up capital, then execute BUY orders.")
                print("Some BUY orders may not be shown due to insufficient cash simulation.")
            else:
                print("✅ No rebalancing needed - portfolio is already aligned with targets")
        else:
            print("📊 No existing portfolio found - this would be initial portfolio construction")
            print(f"💼 Would invest in {len(portfolio_df)} positions")

            # Optionally save the new portfolio state
            if config.get('simulate_rebalancing', True):
                print()
                print("💾 Simulating initial portfolio creation...")
                # Create initial portfolio from target
                new_portfolio = Portfolio(cash=config['account_value'])
                new_portfolio.last_rebalance_date = datetime.now(UTC)
                # Note: In production, would create Position objects from portfolio_df
                # save_portfolio_state(new_portfolio, portfolio_file)
                print("✅ Portfolio state ready for tracking")

    elif calendar_summary['is_rebalancing_day'] and portfolio_df.empty:
        print()
        print("⚠️  Today is a rebalancing day but no valid portfolio could be constructed")
    elif calendar_summary['is_rebalancing_day']:
        print()
        print("ℹ️  Rebalancing simulation disabled in configuration")

    print("✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
