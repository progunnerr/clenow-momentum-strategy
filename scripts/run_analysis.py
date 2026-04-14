#!/usr/bin/env python3
"""
Run momentum analysis on S&P 500 stocks.

This script fetches S&P 500 data and displays the top momentum stocks
according to Clenow's momentum strategy.
"""

import sys
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Initialize logger configuration

from clenow_momentum.data import get_universe_tickers, get_stock_data
from clenow_momentum.data.universes import get_universe_spec
from clenow_momentum.data.sources.yfinance_adapter import (
    WikipediaTickerAdapter,
    YFinanceMarketDataAdapter,
)
from clenow_momentum.indicators.filters import apply_all_filters
from clenow_momentum.indicators.momentum import (
    calculate_momentum_for_universe,
    get_top_momentum_stocks,
)
from clenow_momentum.indicators.risk import apply_risk_limits, build_portfolio
from clenow_momentum.market_analysis import MarketAnalysisFacade
from clenow_momentum.strategy.rebalancing import (
    create_rebalancing_summary,
    generate_rebalancing_orders,
    load_portfolio_state,
)
from clenow_momentum.strategy.trading_schedule import (
    get_trading_calendar_summary,
    should_execute_trades,
)
from clenow_momentum.trading import get_trading_mode
from clenow_momentum.trading.portfolio_sync import PortfolioSynchronizer
from clenow_momentum.utils.config import get_position_sizing_guide, load_config, validate_config, validate_ibkr_config


def has_error(result: dict) -> bool:
    """Return True only when a result dict contains a real error message."""
    return bool(result.get("error"))


MA_ALIGNMENT_TARGET = "10 EMA > 20 SMA > 50 SMA > 200 SMA"


def format_ma_stack(short_mas: dict, long_term_ma: float | None) -> str:
    """Return moving averages ordered from highest value to lowest value."""
    ma_values = [
        ("10 EMA", short_mas.get("10_ema")),
        ("20 SMA", short_mas.get("20_sma")),
        ("50 SMA", short_mas.get("50_sma")),
        ("200 SMA", long_term_ma),
    ]

    available_mas = []
    for label, value in ma_values:
        if value is None:
            continue

        ma_value = float(value)
        if isfinite(ma_value):
            available_mas.append((label, ma_value))

    if not available_mas:
        return ""

    ordered_mas = sorted(available_mas, key=lambda item: item[1], reverse=True)
    return " > ".join(f"{label} (${value:,.2f})" for label, value in ordered_mas)


def print_trading_schedule_status(config):
    """Print trading schedule status."""
    print("TRADING SCHEDULE STATUS")
    print("-" * 30)

    bypass_wednesday = config.get("bypass_wednesday_check", False)
    calendar_summary = get_trading_calendar_summary(bypass_wednesday)
    can_trade, trade_reason = should_execute_trades(bypass_wednesday=bypass_wednesday)

    print(
        f"📅 Today: {calendar_summary['current_date']} ({calendar_summary['current_weekday']})"
    )
    if bypass_wednesday:
        print("⚠️  TESTING MODE: Wednesday-only restriction bypassed")
    print(
        f"🎯 Trading Day: {'✅ Yes' if calendar_summary['is_trading_day'] else '❌ No'}"
    )
    print(
        f"🔄 Rebalancing Day: {'✅ Yes' if calendar_summary['is_rebalancing_day'] else '❌ No'}"
    )
    print(f"📊 Trading Status: {trade_reason}")

    if not can_trade:
        print(
            f"\n⏭️  Next Trading Day: {calendar_summary['next_trading_day']} (in {calendar_summary['days_until_next_trading']} days)"
        )

    print(
        f"🔄 Next Rebalancing: {calendar_summary['next_rebalancing_date']} (in {calendar_summary['days_until_next_rebalancing']} days)"
    )
    print()

    return calendar_summary, can_trade, trade_reason


def print_account_sizing(config):
    """Print strategy allocation and position sizing information."""
    sizing_guide = get_position_sizing_guide(config["strategy_allocation"])
    print("STRATEGY ALLOCATION & POSITION SIZING")
    print("-" * 30)
    print(f"💰 Strategy Allocation: ${config['strategy_allocation']:,.0f}")
    print(
        f"🎯 Risk per Trade: {config['risk_per_trade']:.3%} (${sizing_guide['risk_per_trade_dollars']:,.0f})"
    )
    print(
        f"📊 Target Positions: {sizing_guide['recommended_positions']} ({sizing_guide['risk_level']})"
    )
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
    # Get stock data - need enough for 200-day MA plus some buffer
    # 200 trading days ≈ 10 months, so fetch 1 year to be safe
    print("Step 2: Fetching stock data (1 year for 200-day MA)...")
    print("⏳ This may take a moment...")
    print(f"📈 Processing all {len(tickers)} {config.get('universe', 'SP500')} stocks")

    stock_data = get_stock_data(tickers, period="1y")

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
    print(
        f"Step 3: Calculating momentum scores ({config['momentum_period']}-day period)..."
    )
    momentum_df = calculate_momentum_for_universe(
        stock_data, period=config["momentum_period"]
    )

    valid_scores = momentum_df.dropna(subset=["momentum_score"])
    print(f"✅ Calculated momentum for {len(valid_scores)} stocks")
    print()

    return stock_data, momentum_df


def check_and_filter_stocks(momentum_df, stock_data, config, trading_allowed):
    """Apply filters and select stocks for portfolio."""
    stocks_for_portfolio = pd.DataFrame()

    if trading_allowed:
        # Filter the entire universe of stocks with valid momentum scores
        stocks_to_filter = momentum_df.dropna(subset=["momentum_score"])
        print(
            f"🔍 Applying filters to all {len(stocks_to_filter)} stocks with valid momentum scores..."
        )
        filtered_stocks = apply_all_filters(
            stocks_to_filter,
            stock_data,
            ma_period=config["ma_filter_period"],
            gap_threshold=config["gap_threshold"],
        )

        print(f"✅ {len(filtered_stocks)} stocks passed all filters.")

        # For efficiency, select top stocks for portfolio construction before sizing
        stocks_for_portfolio = filtered_stocks.head(config["max_positions"])
        print(
            f"📈 Selecting top {len(stocks_for_portfolio)} momentum stocks for portfolio (max_positions: {config['max_positions']})."
        )

        final_stocks = filtered_stocks  # Keep all filtered stocks for display
    else:
        print("⛔ Market regime not favorable - skipping individual stock filters")
        print(
            f"📊 Showing top {config['top_momentum_pct']:.0%} momentum rankings for informational purposes only..."
        )
        # For informational purposes, show the top 20% unfiltered
        final_stocks = get_top_momentum_stocks(
            momentum_df, top_pct=config["top_momentum_pct"]
        )

    return stocks_for_portfolio, final_stocks


def display_portfolio_table(portfolio_df, config):
    """Display portfolio table with position sizing."""
    print("🎯 PORTFOLIO WITH RISK PARITY POSITION SIZING")
    print("=" * 60)

    # Prepare portfolio table data
    table_data = []
    for _, row in portfolio_df.iterrows():
        table_data.append(
            [
                int(row["portfolio_rank"]),
                row["ticker"],
                f"{row['momentum_score']:.3f}",
                f"${row['current_price']:.2f}",
                f"${row['atr']:.2f}",
                int(row["shares"]),
                f"${row['investment']:,.0f}",
                f"{row['position_pct']:.1%}",
                f"${row['actual_risk']:,.0f}",
            ]
        )

    headers = [
        "Rank",
        "Ticker",
        "Momentum",
        "Price",
        "ATR",
        "Shares",
        "Investment",
        "Weight %",
        "Risk $",
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Portfolio summary
    total_investment = portfolio_df["investment"].sum()
    total_positions = len(portfolio_df)
    avg_position = total_investment / total_positions
    total_risk = portfolio_df["actual_risk"].sum()

    print("PORTFOLIO SUMMARY")
    print("-" * 50)
    print(f"💰 Strategy Allocation: ${config['strategy_allocation']:,.0f}")
    print(f"📊 Total Positions: {total_positions}")
    print(f"💼 Total Investment: ${total_investment:,.0f}")
    print(
        f"💵 Cash Remaining: ${config['strategy_allocation'] - total_investment:,.0f}"
    )
    print(
        f"📈 Capital Utilization: {total_investment / config['strategy_allocation']:.1%}"
    )
    print(f"⚖️  Average Position: ${avg_position:,.0f}")
    print(f"⚠️  Total Portfolio Risk: ${total_risk:,.0f}")
    print(
        f"🎯 Risk per Trade: {config['risk_per_trade']:.3%} (${config['strategy_allocation'] * config['risk_per_trade']:,.0f})"
    )
    print("📊 Weighting Method: Risk Parity (Equal Risk Contribution)")


def display_stocks_table(
    final_stocks, config, trading_allowed, valid_scores, market_regime
):
    """Display filtered stocks table without portfolio."""
    table_data = []
    for i, (_, row) in enumerate(final_stocks.iterrows()):
        # Check if we have filter data (MA info)
        if "latest_price" in row and not pd.isna(row["latest_price"]):
            price_val = row["latest_price"]
            ma_info = (
                f" (vs MA: {row.get('price_vs_ma', 0):+.1%})"
                if "price_vs_ma" in row
                else ""
            )
        else:
            price_val = row.get("current_price", 0)
            ma_info = ""

        table_data.append(
            [
                i + 1,  # Re-rank after filtering
                row["ticker"],
                f"{row['momentum_score']:.3f}",
                f"{row['annualized_slope']:.3f}",
                f"{row['r_squared']:.3f}",
                f"${price_val:.2f}{ma_info}" if not pd.isna(price_val) else "N/A",
                (
                    f"{row['period_return_pct']:+.1f}%"
                    if not pd.isna(row["period_return_pct"])
                    else "N/A"
                ),
            ]
        )

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
        print(
            f"🎯 Top {config['top_momentum_pct']:.0%} stocks shown: {len(final_stocks)}"
        )

    print(f"📈 Average Momentum Score: {final_stocks['momentum_score'].mean():.3f}")
    print(f"📐 Average R-squared: {final_stocks['r_squared'].mean():.3f}")
    print(f"📊 Average 90-day Return: {final_stocks['period_return_pct'].mean():.1f}%")

    if trading_allowed and len(final_stocks) > 0:
        print(f"🎯 Market Regime: {market_regime['regime'].upper()}")
        print("💼 Ready for position sizing and portfolio construction")
    elif not trading_allowed:
        print(f"⛔ Trading suspended due to {market_regime['regime']} market regime")
    print()


def main(force_execution: bool = False):
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

    # Step 1: Get universe tickers (driven by MARKET_UNIVERSE env var / config)
    universe = config.get("universe", "SP500")
    print(f"Step 1: Fetching {universe} tickers...")
    tickers = get_universe_tickers(universe)

    if not tickers:
        print("❌ Could not retrieve tickers. Exiting.")
        return 1

    print(f"✅ Successfully fetched {len(tickers)} {universe} tickers")
    print("Sample tickers:", tickers[:10])
    print()

    # Step 2 & 3: Get stock data and calculate momentum
    stock_data, momentum_df = fetch_and_process_data(tickers, config)
    if stock_data is None:
        return 1

    valid_scores = momentum_df.dropna(subset=["momentum_score"])

    # Initialize new market analysis architecture
    universe_spec = get_universe_spec(universe)

    # Step 4: Check market regime with enhanced analysis
    print(
        f"Step 4: Checking market regime ({universe_spec.benchmark_etf} vs {config['market_regime_period']}-day MA)..."
    )
    market_data_source = YFinanceMarketDataAdapter()
    ticker_source = WikipediaTickerAdapter()
    market_analysis = MarketAnalysisFacade(
        market_data_source, ticker_source, benchmark_ticker=universe_spec.benchmark_etf
    )
    
    # Check regime using new architecture
    regime = market_analysis.check_regime(period=config["market_regime_period"])
    trading_allowed, regime_reason = market_analysis.should_trade_momentum(
        regime=regime, period=config["market_regime_period"]
    )
    
    # Get detailed market status
    detailed_status = market_analysis.get_detailed_status(period=config["market_regime_period"])
    
    # Calculate additional market metrics using new architecture
    breadth = market_analysis.calculate_breadth(
        period=config["market_regime_period"],
        stock_data=stock_data,  # Reuse the data we already fetched
        tickers=tickers,
    )
    momentum_metrics = market_analysis.calculate_absolute_momentum(period_months=12)
    
    # Calculate daily performance and short-term MAs
    daily_perf = market_analysis.calculate_daily_performance()
    short_mas = market_analysis.calculate_short_term_mas()
    ma_position = market_analysis.analyze_ma_position(
        long_term_ma=regime.ma_value, short_term_mas=short_mas
    )
    
    # Convert new dataclasses to old dict format for backward compatibility with display code
    market_regime = {
        "regime": regime.regime,
        "current_price": regime.current_price,
        "ma_value": regime.ma_value,
        "price_vs_ma": regime.price_vs_ma_pct / 100 if regime.price_vs_ma_pct else None,  # Convert pct to decimal
        "trading_allowed": regime.trading_allowed,
    }
    breadth_data = {
        "breadth_pct": breadth.breadth_pct,
        "above_ma": breadth.above_ma,
        "below_ma": breadth.below_ma,
        "total_checked": breadth.total_checked,
        "breadth_strength": breadth.breadth_strength,
        "error": breadth.error,
    }
    momentum_data = {
        "period_return": momentum_metrics.period_return,
        "period_months": momentum_metrics.period_months,
        "current_price": momentum_metrics.current_price,
        "past_price": momentum_metrics.past_price,
        "momentum_strength": momentum_metrics.momentum_strength,
        "all_returns": momentum_metrics.all_returns,
        "bullish": momentum_metrics.bullish,
        "error": momentum_metrics.error,
    }
    daily_perf_dict = {
        "current_price": daily_perf.current_price,
        "previous_close": daily_perf.previous_close,
        "daily_change": daily_perf.daily_change,
        "daily_change_pct": daily_perf.daily_change_pct,
        "daily_trend": daily_perf.daily_trend,
        "five_day_return": daily_perf.five_day_return,
        "error": daily_perf.error,
    }
    short_mas_dict = {
        "current_price": short_mas.current_price,
        "10_ema": short_mas.ema_10,
        "20_sma": short_mas.sma_20,
        "50_sma": short_mas.sma_50,
        "error": short_mas.error,
    }
    ma_analysis = {
        "current_price": ma_position.current_price,
        "mas_above": ma_position.mas_above,
        "mas_below": ma_position.mas_below,
        "total_mas": ma_position.total_mas,
        "market_structure": ma_position.market_structure,
        "ma_positions": ma_position.ma_positions,
        "mas_aligned": ma_position.mas_aligned,
        "200_ma": ma_position.ma_200,
        "error": ma_position.error,
    }
    daily_perf = daily_perf_dict
    short_mas = short_mas_dict

    # Display Daily Market Update FIRST
    print("\n" + "=" * 60)
    print("DAILY MARKET UPDATE")
    print("=" * 60)

    if not has_error(daily_perf):
        # Format daily change with color indicators
        change_pct = daily_perf.get("daily_change_pct", 0)
        change_emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "➖"
        trend_emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"

        print(
            f"{change_emoji} {universe_spec.display_name} Today: {change_pct:+.2f}% (${daily_perf.get('daily_change', 0):+.2f})"
        )
        print(f"   • Current: ${daily_perf.get('current_price', 0):,.2f}")
        print(f"   • Previous Close: ${daily_perf.get('previous_close', 0):,.2f}")
        print(f"   • Trend: {daily_perf.get('daily_trend', 'Unknown')}")

        if daily_perf.get("five_day_return") is not None:
            five_day = daily_perf.get("five_day_return", 0)
            five_emoji = "📈" if five_day > 0 else "📉"
            print(f"{five_emoji} 5-Day Performance: {five_day:+.2f}%")
    else:
        print(
            f"⚠️ Could not fetch daily performance: {daily_perf.get('error', 'Unknown error')}"
        )

    # Display Short-term Technical Analysis
    if not has_error(ma_analysis):
        print("\n" + "-" * 40)
        print("SHORT-TERM TECHNICALS")
        print("-" * 40)

        # Show position relative to MAs
        print(
            f"📊 Market Position: {ma_analysis.get('mas_above', 0)}/{ma_analysis.get('total_mas', 0)} MAs"
        )

        # Display each MA with distance
        for ma_name, position, distance in ma_analysis.get("ma_positions", []):

            # Get MA value
            ma_value = None
            if "10-day" in ma_name and "10_ema" in short_mas:
                ma_value = short_mas["10_ema"]
            elif "20-day" in ma_name and "20_sma" in short_mas:
                ma_value = short_mas["20_sma"]
            elif "50-day" in ma_name and "50_sma" in short_mas:
                ma_value = short_mas["50_sma"]
            elif "200-day" in ma_name:
                ma_value = ma_analysis.get("200_ma")

            if ma_value:
                print(f"   • {ma_name}: ${ma_value:,.2f} ({distance:+.1f}% {position})")

        # Market structure assessment
        structure = ma_analysis.get("market_structure", "Unknown")
        struct_emoji = (
            "💪"
            if "Strong Bullish" in structure
            else (
                "📈"
                if "Bullish" in structure
                else "⚖️" if "Mixed" in structure else "📉"
            )
        )
        print(f"\n{struct_emoji} Market Structure: {structure}")

        alignment_stack = format_ma_stack(short_mas, ma_analysis.get("200_ma"))
        if ma_analysis.get("mas_aligned"):
            print(f"   ✅ MA stack bullishly aligned ({MA_ALIGNMENT_TARGET})")
        else:
            print(
                f"   ⚠️ MA stack not bullishly aligned (target: {MA_ALIGNMENT_TARGET})"
            )
            if alignment_stack:
                print(f"   • Current stack: {alignment_stack}")

    print("\n" + "=" * 60)
    print("MARKET REGIME ANALYSIS")
    print("=" * 60)

    print(f"📊 Market Regime: {market_regime.get('regime', 'unknown').upper()}")
    print(
        f"📈 {universe_spec.benchmark_etf} ({universe_spec.display_name}): ${market_regime.get('current_price', 'N/A'):,.2f} vs {config['market_regime_period']}MA: ${market_regime.get('ma_value', 'N/A'):,.2f}"
    )

    if market_regime.get("price_vs_ma") is not None:
        price_vs_ma_pct = market_regime.get("price_vs_ma") * 100
        print(f"📍 Distance from MA: {price_vs_ma_pct:+.2f}%")

    print(f"🎯 Trading Status: {'✅ ALLOWED' if trading_allowed else '⛔ SUSPENDED'}")
    print(f"📝 Reason: {regime_reason}")

    # Display enhanced market metrics if available
    if not has_error(detailed_status):
        print("\n" + "-" * 40)
        print("REGIME HISTORY & METRICS")
        print("-" * 40)

        # Current regime information
        print(f"🏛️ Current Regime: {detailed_status.get('regime_type', 'Unknown')}")
        print(
            f"📅 {universe_spec.display_name} {detailed_status.get('crossover_type', '')} {config['market_regime_period']}MA on: {detailed_status.get('crossover_date', 'N/A')}"
        )
        print(
            f"⏱️ Days in {detailed_status.get('regime_type', 'regime')}: {detailed_status.get('days_since_crossover', 0)} days"
        )
        print(
            f"📊 Consecutive days {('above' if detailed_status.get('above_ma') else 'below')} MA: {detailed_status.get('consecutive_days_current_regime', 0)} days"
        )

        # Distance metrics
        print(
            f"\n📈 Distance from MA Statistics (since {detailed_status.get('crossover_date', 'N/A')}):"
        )
        print(f"   • Current: {detailed_status.get('price_vs_ma_pct', 0):+.2f}%")
        print(
            f"   • Maximum: {detailed_status.get('max_distance_from_ma_pct', 0):+.2f}%"
        )
        print(
            f"   • Minimum: {detailed_status.get('min_distance_from_ma_pct', 0):+.2f}%"
        )
        print(
            f"   • Average: {detailed_status.get('avg_distance_from_ma_pct', 0):+.2f}%"
        )

        # MA trend
        ma_trend = detailed_status.get("ma_trend", "unknown")
        ma_slope = detailed_status.get("ma_slope_10d", 0)
        trend_emoji = (
            "📈" if ma_trend == "rising" else "📉" if ma_trend == "falling" else "➡️"
        )
        print(
            f"\n{trend_emoji} {config['market_regime_period']}MA Trend: {ma_trend.capitalize()} (10-day slope: ${ma_slope:.2f}/day)"
        )

        # Recent behavior
        print("\n📊 Last 30 Days:")
        print(
            f"   • Days above {config['market_regime_period']}MA: {detailed_status.get('recent_30d_above_ma', 0)}"
        )
        print(
            f"   • Days below {config['market_regime_period']}MA: {detailed_status.get('recent_30d_below_ma', 0)}"
        )

    # Display Market Breadth
    print("\n" + "-" * 40)
    print("MARKET BREADTH ANALYSIS")
    print("-" * 40)

    if not has_error(breadth_data):
        breadth_pct = breadth_data.get("breadth_pct", 0)
        breadth_emoji = (
            "🟢" if breadth_pct >= 60 else "🟡" if breadth_pct >= 50 else "🔴"
        )

        ma_period_used = breadth_data.get("ma_period", config["market_regime_period"])

        print(
            f"{breadth_emoji} Market Breadth: {breadth_pct:.1f}% of {universe_spec.display_name} stocks above {ma_period_used}-day MA"
        )
        print(
            f"   • Stocks above MA: {breadth_data.get('above_ma', 0)}/{breadth_data.get('total_checked', 0)}"
        )
        print(
            f"   • Breadth Strength: {breadth_data.get('breadth_strength', 'Unknown')}"
        )
        print(
            f"   • Coverage: {breadth_data.get('total_checked', 0)}/{breadth_data.get('sample_size', 0)} stocks with valid data"
        )

        # Breadth interpretation
        if breadth_pct >= 60:
            print("   ✅ Broad participation - Healthy bull market")
        elif breadth_pct >= 50:
            print("   ⚠️ Neutral breadth - Use caution")
        else:
            print("   ❌ Weak breadth - Bear market conditions")
    else:
        print(
            f"   ⚠️ Could not calculate breadth: {breadth_data.get('error', 'Unknown error')}"
        )

    # Display Absolute Momentum
    print("\n" + "-" * 40)
    print("ABSOLUTE MOMENTUM ANALYSIS")
    print("-" * 40)

    if not has_error(momentum_data):
        period_return = momentum_data.get("period_return", 0)
        momentum_emoji = (
            "🚀" if period_return > 10 else "📈" if period_return > 0 else "📉"
        )

        print(f"{momentum_emoji} {universe_spec.display_name} 12-Month Return: {period_return:+.2f}%")
        print(
            f"   • Momentum Strength: {momentum_data.get('momentum_strength', 'Unknown')}"
        )

        # Show multiple timeframe returns if available
        all_returns = momentum_data.get("all_returns", {})
        if all_returns:
            print("\n   Returns by Period:")
            for period, ret in sorted(all_returns.items()):
                print(f"   • {period:>3}: {ret:+6.2f}%")

        # Momentum interpretation
        if period_return > 0:
            print("\n   ✅ Positive absolute momentum - Bull bias confirmed")
        else:
            print("\n   ❌ Negative absolute momentum - Bear market conditions")
    else:
        print(
            f"   ⚠️ Could not calculate momentum: {momentum_data.get('error', 'Unknown error')}"
        )

    # Combined Market Assessment
    print("\n" + "-" * 40)
    print("COMBINED MARKET ASSESSMENT")
    print("-" * 40)

    # Count bullish signals
    bullish_signals = 0
    total_signals = 0

    # 1. Price vs MA
    if market_regime.get("regime") == "bullish":
        bullish_signals += 1
    total_signals += 1

    # 2. Breadth
    if not has_error(breadth_data) and breadth_data.get("breadth_pct", 0) >= 50:
        bullish_signals += 1
    if not has_error(breadth_data):
        total_signals += 1

    # 3. Absolute Momentum
    if not has_error(momentum_data) and momentum_data.get("bullish", False):
        bullish_signals += 1
    if not has_error(momentum_data):
        total_signals += 1

    # Overall assessment
    signal_strength = (
        (bullish_signals / total_signals * 100) if total_signals > 0 else 0
    )

    print(
        f"📊 Bullish Signals: {bullish_signals}/{total_signals} ({signal_strength:.0f}%)"
    )
    print(
        f"   • {universe_spec.benchmark_etf} vs 200MA: {'✅ Above' if market_regime.get('regime') == 'bullish' else '❌ Below'}"
    )
    if not has_error(breadth_data):
        print(
            f"   • Market Breadth: {'✅ Positive' if breadth_data.get('breadth_pct', 0) >= 50 else '❌ Negative'} ({breadth_data.get('breadth_pct', 0):.1f}%)"
        )
    if not has_error(momentum_data):
        print(
            f"   • Absolute Momentum: {'✅ Positive' if momentum_data.get('bullish', False) else '❌ Negative'} ({momentum_data.get('period_return', 0):+.1f}%)"
        )

    if signal_strength >= 66:
        print("\n🟢 STRONG BULL MARKET - All systems go for momentum trading")
    elif signal_strength >= 33:
        print(
            "\n🟡 MIXED SIGNALS - Trade with caution, consider reduced position sizes"
        )
    else:
        print("\n🔴 BEAR MARKET CONDITIONS - Consider defensive positioning")

    print("=" * 60)
    print()

    # Step 5: Apply trading filters to find eligible stocks
    print("Step 5: Applying trading filters...")
    stocks_for_portfolio, final_stocks = check_and_filter_stocks(
        momentum_df, stock_data, config, trading_allowed
    )
    print()

    # Step 6: Portfolio Construction (if trading allowed)
    portfolio_df = pd.DataFrame()
    if trading_allowed and not stocks_for_portfolio.empty:
        print("Step 6: Portfolio Construction & Position Sizing...")

        # Build portfolio with position sizing (Clenow's risk parity method)
        portfolio_df = build_portfolio(
            filtered_stocks=stocks_for_portfolio,
            stock_data=stock_data,
            account_value=config["strategy_allocation"],
            risk_per_trade=config["risk_per_trade"],
            atr_period=config["atr_period"],
            allocation_method="equal_risk",
            stop_loss_multiplier=config["stop_loss_multiplier"],
            max_position_pct=config["max_position_pct"],
        )

        if not portfolio_df.empty:
            # Apply minimum position value limit. max_positions is already handled.
            portfolio_df = apply_risk_limits(
                portfolio_df=portfolio_df,
                max_positions=None,  # Already sliced to max_positions
                min_position_value=config["min_position_value"],
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
            display_stocks_table(
                final_stocks, config, trading_allowed, valid_scores, market_regime
            )

    # Step 8: Rebalancing Analysis (if it's a rebalancing day and we have a portfolio)
    rebalancing_orders = []  # Initialize for IBKR integration
    current_portfolio = None  # Will be loaded from IBKR or JSON

    if calendar_summary["is_rebalancing_day"] and not portfolio_df.empty:
        print()
        print("=" * 60)
        print("🔄 REBALANCING ANALYSIS")
        print("=" * 60)

        from pathlib import Path

        portfolio_file = Path(
            config.get("portfolio_state_file", "data/portfolio_state.json")
        )

        # Determine portfolio source based on IBKR configuration
        if config.get("enable_ibkr_trading", False):
            # IBKR is enabled - sync real portfolio from broker
            print("🔗 IBKR trading enabled - syncing real portfolio from broker...")

            try:
                # Import here to avoid issues when IBKR not configured
                from clenow_momentum.trading.trading_manager import TradingManager

                # Validate configuration first
                ibkr_config_issues = validate_ibkr_config(config)
                critical_issues = [
                    issue for issue in ibkr_config_issues if "❌" in issue
                ]

                if critical_issues:
                    print("❌ Cannot sync with IBKR due to configuration issues:")
                    for issue in critical_issues:
                        print(f"  {issue}")
                    print("⚠️  Falling back to local portfolio file...")
                    current_portfolio = load_portfolio_state(portfolio_file)
                else:
                    # Connect to IBKR and sync portfolio
                    try:
                        print("Syncing portfolio with IBKR...")
                        synchronizer = PortfolioSynchronizer(config)
                        current_portfolio = synchronizer.sync_with_ibkr()

                        print(f"✅ Portfolio synced: {current_portfolio.num_positions} positions, ${current_portfolio.cash:,.0f} cash")

                    except Exception as e:
                        print(f"❌ Failed to sync with IBKR: {e}")
                        current_portfolio = None

                    if current_portfolio is None:
                        print("⚠️  IBKR sync failed - cannot proceed with rebalancing")
                        print("Please check your IBKR connection and try again.")
                        # Don't use JSON fallback when IBKR is enabled but fails
                        portfolio_df = (
                            pd.DataFrame()
                        )  # Clear portfolio to prevent trading

            except ImportError as e:
                print(f"❌ IBKR modules not available: {e}")
                print("⚠️  Falling back to local portfolio file...")
                current_portfolio = load_portfolio_state(portfolio_file)
            except Exception as e:
                print(f"❌ Unexpected error syncing with IBKR: {e}")
                print("Cannot proceed with rebalancing when IBKR is enabled.")
                portfolio_df = pd.DataFrame()  # Clear portfolio to prevent trading

        else:
            # IBKR not enabled - use local JSON file for simulation
            print("📁 Using local portfolio file (simulation mode)...")
            current_portfolio = load_portfolio_state(portfolio_file)

        # Check if we have a valid portfolio to work with
        if current_portfolio is None:
            print("❌ No portfolio data available - cannot generate rebalancing orders")
        elif current_portfolio.num_positions > 0:
            print(
                f"📊 Current Portfolio: {current_portfolio.num_positions} positions, ${current_portfolio.cash:,.0f} cash"
            )

            # Generate rebalancing orders
            rebalancing_orders = generate_rebalancing_orders(
                current_portfolio=current_portfolio,
                target_portfolio=portfolio_df,
                stock_data=stock_data,
                account_value=config["strategy_allocation"],
                cash_buffer=config.get("cash_buffer", 0.02),
            )

            if rebalancing_orders:
                print(f"📝 Generated {len(rebalancing_orders)} rebalancing orders")
                print()

                # Show rebalancing summary
                summary = create_rebalancing_summary(
                    current_portfolio, rebalancing_orders, portfolio_df
                )

                print("REBALANCING SUMMARY")
                print("-" * 50)
                print(f"Current Positions: {summary['current_positions']}")
                print(f"Target Positions: {summary['target_positions']}")
                print(f"Portfolio Turnover: {summary['turnover_pct']:.1f}%")
                print()
                print("Orders to Execute:")
                print(
                    f"  - Sells: {summary['num_sells']} (${summary['total_sell_value']:,.0f})"
                )
                print(
                    f"  - Buys: {summary['num_buys']} (${summary['total_buy_value']:,.0f})"
                )
                print()
                print(
                    f"Positions to Add: {', '.join(summary['positions_to_add']) if summary['positions_to_add'] else 'None'}"
                )
                print(
                    f"Positions to Remove: {', '.join(summary['positions_to_remove']) if summary['positions_to_remove'] else 'None'}"
                )
                print(f"Positions to Keep: {len(summary['positions_to_keep'])}")
                print()
                print(f"Expected Cash After: ${summary['expected_cash']:,.0f}")

                # Show detailed orders
                print()
                print("REBALANCING ORDERS")
                print("-" * 50)

                order_data = []
                # Show all SELL orders first
                sell_orders = [
                    o for o in rebalancing_orders if o.order_type.value == "SELL"
                ]
                buy_orders = [
                    o for o in rebalancing_orders if o.order_type.value == "BUY"
                ]

                print(f"\n📉 SELL ORDERS ({len(sell_orders)} total):")
                for order in sell_orders:
                    order_data.append(
                        [
                            order.order_type.value,
                            order.ticker,
                            order.shares,
                            f"${order.current_price:.2f}",
                            f"${order.order_value:,.0f}",
                            (
                                order.reason[:40] + "..."
                                if len(order.reason) > 40
                                else order.reason
                            ),
                        ]
                    )

                if sell_orders:
                    headers = ["Type", "Ticker", "Shares", "Price", "Value", "Reason"]
                    print(tabulate(order_data, headers=headers, tablefmt="grid"))

                print(f"\n📈 BUY ORDERS ({len(buy_orders)} total):")
                # BUY orders are already in momentum rank order from generate_rebalancing_orders
                order_data = []
                for order in buy_orders:
                    order_data.append(
                        [
                            order.order_type.value,
                            order.ticker,
                            order.shares,
                            f"${order.current_price:.2f}",
                            f"${order.order_value:,.0f}",
                            (
                                order.reason[:40] + "..."
                                if len(order.reason) > 40
                                else order.reason
                            ),
                        ]
                    )

                if buy_orders:
                    headers = ["Type", "Ticker", "Shares", "Price", "Value", "Reason"]
                    print(tabulate(order_data, headers=headers, tablefmt="grid"))

                # Add execution note
                print("\n💡 EXECUTION NOTE:")
                print(
                    "Execute SELL orders first to free up capital, then execute BUY orders."
                )
                print(
                    "Some BUY orders may not be shown due to insufficient cash simulation."
                )
            else:
                print(
                    "✅ No rebalancing needed - portfolio is already aligned with targets"
                )
        else:
            # Empty portfolio - initial construction
            print("📊 Empty portfolio detected - initial portfolio construction")
            print(f"💰 Cash available: ${current_portfolio.cash:,.0f}")
            print(f"💼 Target: {len(portfolio_df)} positions")

            # Generate initial BUY orders for all target positions
            print()
            print("📝 Generating initial portfolio orders...")

            # Create an empty portfolio for initial construction
            if current_portfolio.cash < 1000:
                # If no cash, use strategy allocation as starting cash
                current_portfolio.cash = config["strategy_allocation"]
                print(
                    f"💵 Using strategy allocation: ${config['strategy_allocation']:,.0f}"
                )

            # Generate orders (will be all BUY orders since no current positions)
            rebalancing_orders = generate_rebalancing_orders(
                current_portfolio=current_portfolio,
                target_portfolio=portfolio_df,
                stock_data=stock_data,
                account_value=config["strategy_allocation"],
                cash_buffer=config.get("cash_buffer", 0.02),
            )

            if rebalancing_orders:
                print(f"✅ Generated {len(rebalancing_orders)} initial BUY orders")

                # Show summary for initial portfolio
                total_investment = sum(o.order_value for o in rebalancing_orders)
                print()
                print("INITIAL PORTFOLIO CONSTRUCTION")
                print("-" * 50)
                print(f"Number of positions: {len(rebalancing_orders)}")
                print(f"Total investment: ${total_investment:,.0f}")
                print(
                    f"Cash reserve: ${current_portfolio.cash - total_investment:,.0f}"
                )

                # Show the orders
                print()
                print("INITIAL BUY ORDERS")
                print("-" * 50)

                order_data = []
                for order in rebalancing_orders:
                    order_data.append(
                        [
                            order.ticker,
                            order.shares,
                            f"${order.current_price:.2f}",
                            f"${order.order_value:,.0f}",
                        ]
                    )

                headers = ["Ticker", "Shares", "Price", "Value"]
                print(tabulate(order_data, headers=headers, tablefmt="grid"))
            else:
                print("⚠️  Could not generate initial orders - check cash availability")

    elif calendar_summary["is_rebalancing_day"] and portfolio_df.empty:
        print()
        print(
            "⚠️  Today is a rebalancing day but no valid portfolio could be constructed"
        )

    # Step 9: IBKR Live Trading Execution (if enabled and conditions met)
    if (
        calendar_summary["is_rebalancing_day"]
        and not portfolio_df.empty
        and config.get("enable_ibkr_trading", False)
        and rebalancing_orders  # From Step 8
    ):
        print()
        print("=" * 60)
        print("🔗 IBKR LIVE TRADING EXECUTION")
        print("=" * 60)

        try:
            # Check IBKR configuration
            ibkr_config_issues = validate_ibkr_config(config)
            critical_issues = [issue for issue in ibkr_config_issues if "❌" in issue]

            if critical_issues:
                print("❌ IBKR configuration issues prevent trading:")
                for issue in critical_issues:
                    print(f"  {issue}")
                print("Please fix configuration issues before enabling live trading.")
            else:
                if ibkr_config_issues:
                    print("⚠️ IBKR configuration warnings:")
                    for issue in ibkr_config_issues:
                        print(f"  {issue}")
                    print()

                print("🚀 Executing rebalancing orders via IBKR...")
                trading_mode = get_trading_mode(config)
                if trading_mode == "live":
                    print("Trading mode: 🚨 LIVE TRADING")
                elif trading_mode == "paper":
                    print("Trading mode: PAPER TRADING")
                else:
                    print(f"Trading mode: {trading_mode.upper()}")

                # Execute trading
                try:
                    with TradingManager(config) as trading_manager:
                        print(f"Status: {trading_manager.get_status_summary()}")
                        sys.stdout.flush()  # Ensure status is shown before confirmations

                        # Execute rebalancing
                        execution_results = trading_manager.execute_rebalancing(
                            rebalancing_orders=rebalancing_orders,
                            dry_run=False,  # Use actual trading mode from config
                            force_execution=force_execution,
                        )

                except Exception as e:
                    print(f"❌ IBKR trading execution failed: {e}")
                    execution_results = None

                if execution_results:
                    print("\n🎯 TRADING EXECUTION RESULTS")
                    print("-" * 40)
                    print(f"Status: {execution_results.get('status', 'Unknown')}")
                    print(
                        f"Orders submitted: {execution_results.get('orders_submitted', 0)}"
                    )
                    print(
                        f"Orders executed: {execution_results.get('orders_executed', 0)}"
                    )
                    print(f"Orders failed: {execution_results.get('orders_failed', 0)}")
                    print(
                        f"Total value traded: ${execution_results.get('total_value_traded', 0):,.2f}"
                    )
                    print(
                        f"Total commission: ${execution_results.get('total_commission', 0):,.2f}"
                    )
                    print(
                        f"Execution time: {execution_results.get('execution_duration_minutes', 0):.1f} minutes"
                    )

                    if execution_results.get("discrepancies", 0) > 0:
                        print(
                            f"⚠️ Portfolio discrepancies detected: {execution_results['discrepancies']}"
                        )

                    if execution_results.get("post_trade_alerts", 0) > 0:
                        print(
                            f"🚨 Post-trade alerts: {execution_results['post_trade_alerts']}"
                        )

                    if execution_results.get("status") == "completed_successfully":
                        print("✅ Live trading execution completed successfully!")
                    else:
                        print("⚠️ Trading execution completed with issues")
                else:
                    print("❌ Trading execution failed")

        except Exception as e:
            print(f"❌ IBKR trading module error: {e}")
            print("Analysis will continue without live trading")

    elif config.get("enable_ibkr_trading", False):
        print()
        print("ℹ️  IBKR trading is enabled but conditions not met for execution:")
        if not calendar_summary["is_rebalancing_day"]:
            print("   - Not a rebalancing day")
        if portfolio_df.empty:
            print("   - No valid portfolio constructed")
        if not rebalancing_orders:
            print("   - No rebalancing orders generated")

    print("\n✅ Analysis complete!")

    # Clean up IBKR connection if it was used
    if config.get("enable_ibkr_trading", False):
        try:
            # Connection cleanup is handled by context manager
            pass
        except Exception:
            pass  # Ignore cleanup errors

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Clenow momentum strategy analysis"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without confirmation prompts",
    )
    args = parser.parse_args()

    exit_code = main(force_execution=args.force)
    sys.exit(exit_code)
