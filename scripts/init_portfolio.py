#!/usr/bin/env python3
"""
Initialize an empty portfolio state file.

This script creates a clean portfolio_state.json with no positions,
ready for either IBKR sync or initial portfolio construction.
"""

import json
from datetime import UTC, datetime
from pathlib import Path


def create_empty_portfolio() -> dict:
    """
    Create an empty portfolio state.
    
    The cash amount will be determined by:
    - IBKR sync if enabled (real cash balance)
    - STRATEGY_ALLOCATION from .env if IBKR disabled
        
    Returns:
        Dictionary representing empty portfolio state
    """
    return {
        "cash": 0.0,  # Will be set from IBKR or STRATEGY_ALLOCATION
        "last_rebalance_date": datetime.now(UTC).isoformat(),
        "positions": {}
    }


def main():
    """Initialize portfolio state file."""
    # Get the data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    portfolio_file = data_dir / "portfolio_state.json"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Check if file already exists
    if portfolio_file.exists():
        print(f"⚠️  Portfolio file already exists: {portfolio_file}")
        response = input("Do you want to reset it to empty? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Cancelled - portfolio file unchanged")
            return 1
    
    # Create empty portfolio (cash will be set from IBKR or config)
    portfolio_state = create_empty_portfolio()
    
    # Save to file
    with open(portfolio_file, 'w') as f:
        json.dump(portfolio_state, f, indent=2)
    
    print(f"\n✅ Created empty portfolio file: {portfolio_file}")
    print(f"📊 Positions: 0")
    print(f"💵 Cash: Will be set from:")
    print(f"   • IBKR account balance (if IBKR enabled)")
    print(f"   • STRATEGY_ALLOCATION in .env (if IBKR disabled)")
    print(f"📅 Date: {portfolio_state['last_rebalance_date']}")
    
    print("\n📝 Next steps:")
    print("1. Set STRATEGY_ALLOCATION in .env file")
    print("2. Configure IBKR settings if using live/paper trading")
    print("3. Run 'uv run python main.py' to start trading")
    
    return 0


if __name__ == "__main__":
    exit(main())