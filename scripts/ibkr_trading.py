#!/usr/bin/env python3
"""
IBKR Trading Script for Clenow Momentum Strategy.

This script provides command-line access to IBKR trading functionality.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from clenow_momentum.trading import get_trading_mode, validate_ibkr_config
from clenow_momentum.trading.trading_manager import TradingManager
from clenow_momentum.utils.config import load_config


async def test_connection():
    """Test IBKR connection."""
    print("🔌 Testing IBKR connection...")

    try:
        config = load_config()

        # Validate configuration first
        issues = validate_ibkr_config(config)
        if issues:
            print("\n⚠️ Configuration Issues:")
            for issue in issues:
                print(f"  {issue}")
            print()

        async with TradingManager(config) as trading_manager:
            status = await trading_manager.get_account_status()

            print("✅ Connection successful!")
            trading_mode = get_trading_mode(config)
            print(f"Trading Mode: {trading_mode.upper()}")
            print(f"Account ID: {status.get('account_id', 'N/A')}")
            print(f"Total Cash: ${status.get('total_cash', 0):,.2f}")
            print(f"Net Liquidation: ${status.get('net_liquidation', 0):,.2f}")
            print(f"Positions: {len(status.get('positions', {}))}")

            return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def sync_portfolio():
    """Sync portfolio with IBKR."""
    print("🔄 Syncing portfolio with IBKR...")

    try:
        config = load_config()

        async with TradingManager(config) as trading_manager:
            portfolio = await trading_manager.sync_portfolio_only()

            print("✅ Portfolio sync completed!")
            print(f"Positions: {portfolio.num_positions}")
            print(f"Cash: ${portfolio.cash:,.2f}")
            print(f"Total Value: ${portfolio.total_value:,.2f}")

            if portfolio.positions:
                print("\nPositions:")
                for ticker, position in list(portfolio.positions.items())[:10]:  # Show first 10
                    pnl_pct = position.unrealized_pnl_pct * 100
                    print(f"  {ticker}: {position.shares} shares @ ${position.current_price:.2f} ({pnl_pct:+.1f}%)")

                if len(portfolio.positions) > 10:
                    print(f"  ... and {len(portfolio.positions) - 10} more positions")

            return True

    except Exception as e:
        print(f"❌ Portfolio sync failed: {e}")
        return False


async def check_status():
    """Check comprehensive status."""
    print("📊 Checking trading status...")

    try:
        config = load_config()

        async with TradingManager(config) as trading_manager:
            status = await trading_manager.get_account_status()

            print("✅ Status retrieved successfully!")
            print(f"\nTrading Manager: {trading_manager.get_status_summary()}")

            print("\nAccount Information:")
            print(f"  Account ID: {status.get('account_id', 'N/A')}")
            print(f"  Total Cash: ${status.get('total_cash', 0):,.2f}")
            print(f"  Net Liquidation: ${status.get('net_liquidation', 0):,.2f}")
            print(f"  Buying Power: ${status.get('buying_power', 0):,.2f}")

            positions = status.get('positions', {})
            print(f"\nPositions ({len(positions)}):")
            for ticker, pos_info in list(positions.items())[:5]:  # Show first 5
                print(f"  {ticker}: {pos_info['shares']} shares, "
                      f"Value: ${pos_info['market_value']:,.2f}, "
                      f"P&L: ${pos_info['unrealized_pnl']:,.2f}")

            if len(positions) > 5:
                print(f"  ... and {len(positions) - 5} more positions")

            # Risk status
            risk_status = status.get('risk_controls', {})
            cb_status = risk_status.get('circuit_breaker_status', {})
            if cb_status.get('is_tripped'):
                print(f"\n🚨 CIRCUIT BREAKER TRIPPED: {cb_status.get('trip_reason')}")
            else:
                print("\n✅ Risk controls: Normal")

            daily_trades = risk_status.get('daily_trades', {})
            print(f"Daily trades: {daily_trades.get('count', 0)}/{daily_trades.get('limit', 0)}")

            return True

    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


async def async_main():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="IBKR Trading Interface for Clenow Momentum Strategy"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test connection
    subparsers.add_parser("test", help="Test IBKR connection")

    # Sync portfolio
    subparsers.add_parser("sync", help="Sync portfolio with IBKR")

    # Check status
    subparsers.add_parser("status", help="Check comprehensive trading status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Configure logger for script usage
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    print("🚀 Clenow Momentum Strategy - IBKR Trading Interface")
    print("=" * 60)

    try:
        if args.command == "test":
            success = await test_connection()
        elif args.command == "sync":
            success = await sync_portfolio()
        elif args.command == "status":
            success = await check_status()
        else:
            print(f"Unknown command: {args.command}")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error occurred")
        return 1


def main():
    """Main function."""
    try:
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an event loop, create a task
            import nest_asyncio
            nest_asyncio.apply()
            exit_code = asyncio.run(async_main())
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            exit_code = asyncio.run(async_main())
    except ImportError:
        # nest_asyncio not available, try the traditional approach
        try:
            exit_code = asyncio.run(async_main())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                print("\n❌ Event loop conflict. Install nest_asyncio: uv add nest-asyncio")
                exit_code = 1
            else:
                print(f"\n❌ Runtime error: {e}")
                exit_code = 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
