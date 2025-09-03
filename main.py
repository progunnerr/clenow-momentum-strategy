#!/usr/bin/env python3
"""
Main entry point for Clenow Momentum Strategy.

This script provides a simple way to run the momentum analysis.
For detailed analysis, use scripts/run_analysis.py instead.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.run_analysis import main as run_analysis


def main():
    """
    Main function - runs the momentum analysis.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clenow Momentum Strategy Analysis")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip order confirmation prompts and execute all orders automatically"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Clenow Momentum Strategy Analysis")
    print("=" * 50)
    
    if args.force:
        print("⚠️  Force mode enabled - will skip all confirmation prompts")
        print()

    # Run the full analysis with force flag
    exit_code = run_analysis(force_execution=args.force)

    if exit_code == 0:
        print("\n🎉 Analysis completed successfully!")
        print("\n✅ Full Clenow Momentum Strategy Implementation:")
        print("• Phase 1: Momentum calculation ✓")
        print("• Phase 2: Trading filters (MA, gap, market regime) ✓")
        print("• Phase 3: ATR-based position sizing ✓")
        print("• Phase 4: Wednesday trading & bi-monthly rebalancing ✓")
        print("• Phase 5: IBKR integration for live trading ✓")
        print("\n📊 Strategy Status: PRODUCTION READY")
        print("\n💡 What you can do now:")
        print("1. Review the complete analysis output above")
        print("2. Check if today is a trading/rebalancing day")
        print("3. Test IBKR connection: uv run python scripts/ibkr_trading.py test")
        print("4. Enable live trading by setting ENABLE_IBKR_TRADING=true in .env")
        print("5. Run with --force flag to skip order confirmations")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
