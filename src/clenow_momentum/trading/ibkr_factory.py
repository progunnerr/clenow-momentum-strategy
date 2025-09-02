"""
IBKR connector factory for creating configured connector instances.

This module provides factory methods to create IBKR connectors with proper
configuration based on the environment settings.
"""

from loguru import logger

from ..utils.config import load_config
from .ibkr_connector import IBKRConnector


def is_paper_trading_port(port: int) -> bool:
    """
    Determine if port corresponds to paper trading.

    Args:
        port: IBKR connection port

    Returns:
        True if paper trading port, False if live trading port

    Raises:
        ValueError: If port is not a standard IBKR port
    """
    if port == 7497:  # TWS Paper
        return True
    if port == 7496:  # TWS Live
        return False
    if port == 4002:  # Gateway Paper
        return True
    if port == 4001:  # Gateway Live
        return False
    # Non-standard port - warn but assume based on typical patterns
    logger.warning(f"Non-standard IBKR port {port} - cannot auto-detect trading mode")
    return True  # Default to paper trading for safety


def get_trading_mode_from_port(port: int) -> str:
    """
    Get trading mode string from port.

    Args:
        port: IBKR connection port

    Returns:
        'paper' or 'live' or 'unknown'
    """
    try:
        return "paper" if is_paper_trading_port(port) else "live"
    except ValueError:
        return "unknown"


def create_ibkr_connector(config: dict = None) -> IBKRConnector:
    """
    Create an IBKR connector with configuration.

    Args:
        config: Configuration dictionary (loads from env if None)

    Returns:
        Configured IBKRConnector instance
    """
    if config is None:
        config = load_config()

    # Use the port directly as specified by user
    port = config["ibkr_port"]

    connector = IBKRConnector(
        host=config["ibkr_host"],
        port=port,
        client_id=config["ibkr_client_id"],
        account_id=config["ibkr_account_id"],
        timeout=config["ibkr_timeout"],
        auto_reconnect=config["ibkr_auto_reconnect"],
    )

    # Log trading mode detected from port
    trading_mode = get_trading_mode_from_port(port).upper()
    logger.info(f"Created IBKR connector for {trading_mode} trading (auto-detected from port {port})")
    logger.info(f"Connection: {config['ibkr_host']}:{port}")

    return connector


def get_trading_mode(config: dict = None) -> str:
    """
    Get the current trading mode (paper or live) based on port.

    Args:
        config: Configuration dictionary (loads from env if None)

    Returns:
        'paper', 'live', or 'unknown'
    """
    if config is None:
        config = load_config()

    return get_trading_mode_from_port(config["ibkr_port"])


def validate_ibkr_config(config: dict = None) -> list[str]:
    """
    Validate IBKR configuration and return any issues.

    Args:
        config: Configuration dictionary (loads from env if None)

    Returns:
        List of validation warnings/errors
    """
    if config is None:
        config = load_config()

    issues = []

    # Check required settings
    if not config["ibkr_host"]:
        issues.append("❌ IBKR_HOST is required")

    if not config["ibkr_port"]:
        issues.append("❌ IBKR_PORT is required")

    # Check port validity
    valid_ports = [7496, 7497, 4001, 4002]  # Standard TWS/Gateway ports
    if config["ibkr_port"] not in valid_ports:
        issues.append(
            f"⚠️ IBKR_PORT {config['ibkr_port']} is not a standard port "
            f"(expected: {valid_ports})"
        )

    # Check client ID
    if config["ibkr_client_id"] < 1:
        issues.append("❌ IBKR_CLIENT_ID must be >= 1")

    # Check timeout
    if config["ibkr_timeout"] < 10:
        issues.append("⚠️ IBKR_TIMEOUT < 10 seconds may cause connection issues")

    # Trading mode detection and warnings
    port = config["ibkr_port"]
    trading_mode = get_trading_mode_from_port(port)

    if trading_mode == "live":
        issues.append("🚨 LIVE TRADING MODE DETECTED - Real money will be used!")
        if port == 7496:
            issues.append("📊 Detected: TWS Live Trading (port 7496)")
        elif port == 4001:
            issues.append("📊 Detected: IB Gateway Live Trading (port 4001)")
    elif trading_mode == "paper":
        if port == 7497:
            issues.append("✅ Detected: TWS Paper Trading (port 7497)")
        elif port == 4002:
            issues.append("✅ Detected: IB Gateway Paper Trading (port 4002)")
    else:
        issues.append(f"⚠️ Unknown trading mode for port {port}")

    return issues
