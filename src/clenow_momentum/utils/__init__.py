"""
Utility modules for Clenow Momentum Strategy.

This module configures logging for the entire application.
"""

import sys
from pathlib import Path

from loguru import logger

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure loguru logger
logger.remove()  # Remove default handler

# Add console handler with nice formatting
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler with rotation
logger.add(
    logs_dir / "momentum_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    compression="zip"
)

# Export the configured logger
__all__ = ["logger"]
