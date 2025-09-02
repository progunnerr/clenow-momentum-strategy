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
        print("\nNext steps:")
        print("1. Review the momentum rankings above")
        print("2. Consider adding trading filters (Phase 2)")
        print("3. Test with paper trading when ready")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
