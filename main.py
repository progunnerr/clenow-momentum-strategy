#!/usr/bin/env python3
"""
Main entry point for Clenow Momentum Strategy.

This script provides a simple way to run the momentum analysis.
For detailed analysis, use scripts/run_analysis.py instead.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.run_analysis import main as run_analysis


def main():
    """
    Main function - runs the momentum analysis.
    """
    print("🚀 Starting Clenow Momentum Strategy Analysis")
    print("=" * 50)

    # Run the full analysis
    exit_code = run_analysis()

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
