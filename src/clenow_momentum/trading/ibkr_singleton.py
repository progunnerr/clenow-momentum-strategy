"""
IBKR Connection Singleton Manager.

This module ensures that only one IBKR connection exists at a time,
preventing multiple connections with the same client ID.
"""

import asyncio
from typing import Any

from loguru import logger

from .ibkr_connector import IBKRConnector
from .ibkr_factory import create_ibkr_connector


class IBKRConnectionManager:
    """Manages a singleton IBKR connection across the application."""

    _instance: "IBKRConnectionManager | None" = None
    _connector: IBKRConnector | None = None
    _lock: asyncio.Lock = asyncio.Lock()
    _reference_count: int = 0

    def __new__(cls) -> "IBKRConnectionManager":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_connector(self, config: dict[str, Any]) -> IBKRConnector:
        """
        Get or create the IBKR connector.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            The singleton IBKR connector instance
        """
        async with self._lock:
            if self._connector is None:
                logger.info("Creating new IBKR connector instance")
                self._connector = create_ibkr_connector(config)
                
                # Connect if not already connected
                if not self._connector.is_connected():
                    logger.info("Connecting to IBKR...")
                    success = await self._connector.connect()
                    if not success:
                        self._connector = None
                        raise ConnectionError("Failed to connect to IBKR")
            else:
                logger.debug("Reusing existing IBKR connector")
                
                # Verify connection is still alive
                if not self._connector.is_connected():
                    logger.warning("IBKR connection lost, reconnecting...")
                    success = await self._connector.connect()
                    if not success:
                        self._connector = None
                        raise ConnectionError("Failed to reconnect to IBKR")
            
            self._reference_count += 1
            logger.debug(f"IBKR connector reference count: {self._reference_count}")
            return self._connector

    async def release_connector(self) -> None:
        """
        Release a reference to the connector.
        Keep the connection alive between uses to avoid reconnection issues.
        """
        async with self._lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                logger.debug(f"Released IBKR connector reference, count: {self._reference_count}")
                
                # Keep the connection alive even when reference count reaches 0
                # This avoids issues with recreating the IB() object
                if self._reference_count == 0:
                    logger.info("No active references, keeping IBKR connection alive for reuse")

    async def force_disconnect(self) -> None:
        """Force disconnect the IBKR connection regardless of reference count."""
        async with self._lock:
            if self._connector is not None:
                logger.warning("Force disconnecting IBKR")
                await self._connector.disconnect()
                self._connector = None
                self._reference_count = 0

    def is_connected(self) -> bool:
        """Check if the connector is currently connected."""
        return self._connector is not None and self._connector.is_connected()


# Global singleton instance
_connection_manager = IBKRConnectionManager()


async def get_ibkr_connector(config: dict[str, Any]) -> IBKRConnector:
    """
    Get the singleton IBKR connector.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        The IBKR connector instance
    """
    return await _connection_manager.get_connector(config)


async def release_ibkr_connector() -> None:
    """Release a reference to the IBKR connector."""
    await _connection_manager.release_connector()


async def force_disconnect_ibkr() -> None:
    """Force disconnect the IBKR connection."""
    await _connection_manager.force_disconnect()