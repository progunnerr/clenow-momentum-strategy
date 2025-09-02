"""
Configuration management for Clenow Momentum Strategy.

Loads settings from .env file with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def load_config() -> dict:
    """
    Load configuration from .env file.

    Returns:
        Dictionary with all configuration settings
    """
    # Find .env file in project root
    project_root = Path(__file__).parent.parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded configuration from {env_file}")
    else:
        logger.warning(f".env file not found at {env_file}, using defaults")

    config = {
        # Strategy Settings
        "strategy_allocation": float(os.getenv("STRATEGY_ALLOCATION", "100000")),
        "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.001")),
        # Portfolio Construction
        "max_positions": int(os.getenv("MAX_POSITIONS", "20")),
        "min_position_value": float(os.getenv("MIN_POSITION_VALUE", "5000")),
        "max_position_pct": float(os.getenv("MAX_POSITION_PCT", "0.05")),
        # Strategy Parameters
        "top_momentum_pct": float(os.getenv("TOP_MOMENTUM_PCT", "0.20")),
        "momentum_period": int(os.getenv("MOMENTUM_PERIOD", "90")),
        "ma_filter_period": int(os.getenv("MA_FILTER_PERIOD", "100")),
        "market_regime_period": int(os.getenv("MARKET_REGIME_PERIOD", "200")),
        "gap_threshold": float(os.getenv("GAP_THRESHOLD", "0.15")),
        "atr_period": int(os.getenv("ATR_PERIOD", "14")),
        "stop_loss_multiplier": float(
            os.getenv("STOP_LOSS_MULTIPLIER", "3.0")
        ),  # Clenow uses 3x ATR
        # Rebalancing Parameters
        "rebalancing_frequency": os.getenv(
            "REBALANCING_FREQUENCY", "bi-monthly"
        ),  # bi-monthly or monthly
        "cash_buffer": float(os.getenv("CASH_BUFFER", "0.02")),  # 2% cash buffer
        "portfolio_state_file": os.getenv("PORTFOLIO_STATE_FILE", "data/portfolio_state.json"),
        "bypass_wednesday_check": os.getenv("BYPASS_WEDNESDAY_CHECK", "false").lower()
        == "true",  # For testing - also triggers rebalancing
        # IBKR Integration Settings
        "ibkr_host": os.getenv("IBKR_HOST", "127.0.0.1"),
        "ibkr_port": int(os.getenv("IBKR_PORT", "7497")),  # Default to TWS Paper trading
        "ibkr_client_id": int(os.getenv("IBKR_CLIENT_ID", "1")),
        "ibkr_account_id": os.getenv("IBKR_ACCOUNT_ID", ""),
        "ibkr_timeout": int(os.getenv("IBKR_TIMEOUT", "60")),
        "ibkr_auto_reconnect": os.getenv("IBKR_AUTO_RECONNECT", "true").lower() == "true",
    }

    # Log key settings
    logger.info(f"Strategy Allocation: ${config['strategy_allocation']:,.0f}")
    logger.info(f"Risk per Trade: {config['risk_per_trade']:.3%}")
    logger.info(f"Max Positions: {config['max_positions']}")
    logger.info(f"Top Momentum %: {config['top_momentum_pct']:.0%}")

    return config


def get_position_sizing_guide(strategy_allocation: float) -> dict:
    """
    Provide position sizing guidance based on strategy allocation.

    Args:
        strategy_allocation: Strategy allocation in dollars

    Returns:
        Dictionary with position sizing recommendations
    """
    if strategy_allocation < 25000:
        recommended_positions = 5
        risk_level = "Conservative (fewer positions due to minimums)"
    elif strategy_allocation < 100000:
        recommended_positions = 10
        risk_level = "Moderate"
    elif strategy_allocation < 500000:
        recommended_positions = 15
        risk_level = "Balanced"
    else:
        recommended_positions = 20
        risk_level = "Fully Diversified"

    risk_per_trade_dollars = strategy_allocation * 0.001  # 0.1%

    return {
        "strategy_allocation": strategy_allocation,
        "recommended_positions": recommended_positions,
        "risk_level": risk_level,
        "risk_per_trade_dollars": risk_per_trade_dollars,
        "min_position_for_diversification": strategy_allocation / recommended_positions,
        "guidance": {
            "risk_per_trade": f"${risk_per_trade_dollars:.0f} (0.1% of strategy allocation)",
            "position_sizing": f"Target {recommended_positions} positions for optimal diversification",
            "min_stock_price": f"Can trade stocks up to ${risk_per_trade_dollars * 10:.0f} effectively",
        },
    }


def validate_config(config: dict) -> list:
    """
    Validate configuration settings and return warnings.

    Args:
        config: Configuration dictionary

    Returns:
        List of warning messages
    """
    warnings = []

    # Strategy allocation checks
    if config["strategy_allocation"] < 10000:
        warnings.append("⚠️  Strategy allocation under $10,000 may limit diversification")

    # Risk per trade checks
    if config["risk_per_trade"] > 0.005:
        warnings.append("⚠️  Risk per trade > 0.5% is quite aggressive")
    elif config["risk_per_trade"] < 0.0005:
        warnings.append("⚠️  Risk per trade < 0.05% is very conservative")

    # Position count checks
    if config["max_positions"] > config["strategy_allocation"] / config["min_position_value"]:
        warnings.append("⚠️  Too many positions for strategy allocation given minimum position value")

    # Diversification checks
    if config["max_positions"] < 10:
        warnings.append("ℹ️  Less than 10 positions reduces diversification")
    elif config["max_positions"] > 25:
        warnings.append("ℹ️  More than 25 positions may be over-diversified")

    return warnings
