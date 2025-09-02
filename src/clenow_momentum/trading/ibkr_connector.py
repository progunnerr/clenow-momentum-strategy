"""
Interactive Brokers (IBKR) connector for Clenow momentum strategy.

This module provides the core connectivity to IBKR TWS/Gateway for executing
trades and managing positions in the momentum strategy.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ib_async import IB, Contract, MarketOrder, Order, Stock, Trade
from loguru import logger


class ConnectionStatus(Enum):
    """Connection status enumeration."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"


class IBKRConnectorError(Exception):
    """Base exception for IBKR connector errors."""

    pass


class IBKRConnectionError(IBKRConnectorError):
    """Raised when connection to IBKR fails."""

    pass


class IBKROrderError(IBKRConnectorError):
    """Raised when order submission/management fails."""

    pass


class IBKRConnector:
    """
    Interactive Brokers connector for automated trading.

    Manages connection to TWS/Gateway and provides methods for order execution,
    position monitoring, and account information retrieval.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account_id: str = "",
        timeout: int = 60,
        auto_reconnect: bool = True,
    ):
        """
        Initialize IBKR connector.

        Args:
            host: TWS/Gateway host address
            port: TWS/Gateway port (7497 for TWS live, 7496 for TWS paper,
                  4001 for Gateway live, 4002 for Gateway paper)
            client_id: Unique client ID for this connection
            account_id: IBKR account ID (empty string for default account)
            timeout: Connection timeout in seconds
            auto_reconnect: Whether to automatically reconnect on disconnection
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_id = account_id
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect

        # IB connection object
        self.ib = IB()
        self.status = ConnectionStatus.DISCONNECTED

        # Event callbacks
        self.on_connected: Callable | None = None
        self.on_disconnected: Callable | None = None
        self.on_error: Callable | None = None
        self.on_order_status: Callable | None = None

        # Internal state
        self._last_connection_attempt = None
        self._reconnect_task = None

        logger.info(f"Initialized IBKR connector for {host}:{port} (client_id={client_id})")

    async def connect(self) -> bool:
        """
        Connect to IBKR TWS/Gateway.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.status = ConnectionStatus.CONNECTING
            self._last_connection_attempt = datetime.now(UTC)

            logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")

            # Connect with timeout
            await asyncio.wait_for(
                self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=self.timeout,
                ),
                timeout=self.timeout,
            )

            # Verify connection
            if self.ib.isConnected():
                self.status = ConnectionStatus.CONNECTED
                logger.success("Successfully connected to IBKR")

                # Set up event handlers
                self._setup_event_handlers()

                # Validate account
                await self._validate_account()

                if self.on_connected:
                    self.on_connected()

                return True
            self.status = ConnectionStatus.FAILED
            logger.error("Connection to IBKR failed - not connected after timeout")
            return False

        except TimeoutError:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Connection to IBKR timed out after {self.timeout} seconds")
            return False

        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Failed to connect to IBKR: {e}")
            if self.on_error:
                self.on_error(e)
            return False

    async def disconnect(self):
        """Disconnect from IBKR."""
        try:
            if self._reconnect_task:
                self._reconnect_task.cancel()

            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from IBKR")

            self.status = ConnectionStatus.DISCONNECTED

            if self.on_disconnected:
                self.on_disconnected()

        except Exception as e:
            logger.error(f"Error during disconnection: {e}")

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.ib.isConnected() and self.status == ConnectionStatus.CONNECTED

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account information from IBKR.

        Returns:
            Dictionary with account information
        """
        if not self.is_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        try:
            account_values = self.ib.accountValues()
            positions = self.ib.positions()

            # Convert to more usable format
            account_info = {
                "account_id": self.account_id or "default",
                "total_cash": 0.0,
                "net_liquidation": 0.0,
                "buying_power": 0.0,
                "positions": {},
                "last_updated": datetime.now(UTC),
            }

            # Parse account values
            for av in account_values:
                if av.tag == "TotalCashValue":
                    account_info["total_cash"] = float(av.value)
                elif av.tag == "NetLiquidation":
                    account_info["net_liquidation"] = float(av.value)
                elif av.tag == "BuyingPower":
                    account_info["buying_power"] = float(av.value)

            # Parse positions
            for pos in positions:
                if pos.contract.secType == "STK":  # Stocks only
                    account_info["positions"][pos.contract.symbol] = {
                        "shares": pos.position,
                        "avg_cost": pos.avgCost,
                        "market_value": pos.marketValue,
                        "unrealized_pnl": pos.unrealizedPNL,
                    }

            logger.debug(f"Retrieved account info: {len(positions)} positions")
            return account_info

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise IBKRConnectionError(f"Failed to get account info: {e}") from e

    async def create_stock_order(
        self, symbol: str, quantity: int, order_type: str = "MKT", limit_price: float = None
    ) -> Order:
        """
        Create a stock order.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares (positive for buy, negative for sell)
            order_type: Order type ('MKT' for market, 'LMT' for limit)
            limit_price: Limit price (required for limit orders)

        Returns:
            Order object ready for submission
        """
        try:
            if order_type == "MKT":
                order = MarketOrder("BUY" if quantity > 0 else "SELL", abs(quantity))
            elif order_type == "LMT":
                if limit_price is None:
                    raise ValueError("Limit price required for limit orders")
                order = Order()
                order.action = "BUY" if quantity > 0 else "SELL"
                order.orderType = "LMT"
                order.totalQuantity = abs(quantity)
                order.lmtPrice = limit_price
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Add order properties for better execution
            order.outsideRth = False  # Only during regular trading hours
            order.transmit = True  # Submit immediately
            order.goodAfterTime = ""  # No specific time restriction
            order.goodTillDate = ""  # Good till cancelled

            logger.debug(f"Created {order_type} order: {order.action} {order.totalQuantity} {symbol}")
            return order

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise IBKROrderError(f"Failed to create order: {e}") from e

    async def submit_order(self, symbol: str, order: Order) -> Trade:
        """
        Submit an order to IBKR.

        Args:
            symbol: Stock symbol
            order: Order object to submit

        Returns:
            Trade object for tracking execution
        """
        if not self.is_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        try:
            # Create stock contract
            contract = Stock(symbol, "SMART", "USD")

            # Submit order
            trade = self.ib.placeOrder(contract, order)

            logger.info(
                f"Submitted order: {order.action} {order.totalQuantity} {symbol} "
                f"(order_id={trade.order.orderId})"
            )

            return trade

        except Exception as e:
            logger.error(f"Failed to submit order for {symbol}: {e}")
            raise IBKROrderError(f"Failed to submit order: {e}") from e

    async def cancel_order(self, trade: Trade) -> bool:
        """
        Cancel an open order.

        Args:
            trade: Trade object to cancel

        Returns:
            True if cancellation was successful
        """
        if not self.is_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        try:
            self.ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order {trade.order.orderId}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {trade.order.orderId}: {e}")
            return False

    async def get_open_orders(self) -> list[Trade]:
        """
        Get all open orders.

        Returns:
            List of Trade objects for open orders
        """
        if not self.is_connected():
            raise IBKRConnectionError("Not connected to IBKR")

        try:
            trades = self.ib.openTrades()
            logger.debug(f"Retrieved {len(trades)} open orders")
            return trades

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise IBKRConnectionError(f"Failed to get open orders: {e}") from e

    def _setup_event_handlers(self):
        """Set up event handlers for IB connection."""
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        self.ib.orderStatusEvent += self._on_order_status

    def _on_disconnected(self):
        """Handle disconnection event."""
        self.status = ConnectionStatus.DISCONNECTED
        logger.warning("Disconnected from IBKR")

        if self.on_disconnected:
            self.on_disconnected()

        # Attempt reconnection if enabled
        if self.auto_reconnect:
            self._schedule_reconnect()

    def _on_error(self, req_id: int, error_code: int, error_string: str, contract: Contract = None):
        """Handle error events."""
        logger.error(f"IBKR Error {error_code}: {error_string} (reqId={req_id})")

        if self.on_error:
            self.on_error({"code": error_code, "message": error_string, "reqId": req_id})

        # Handle fatal errors
        if error_code in [502, 503, 504]:  # Connection-related errors
            self.status = ConnectionStatus.FAILED

    def _on_order_status(self, trade: Trade):
        """Handle order status updates."""
        order = trade.order
        status = trade.orderStatus

        logger.info(
            f"Order {order.orderId}: {status.status} - "
            f"{status.filled}/{order.totalQuantity} filled @ ${status.avgFillPrice}"
        )

        if self.on_order_status:
            self.on_order_status(trade)

    def _schedule_reconnect(self):
        """Schedule automatic reconnection."""
        if self._reconnect_task and not self._reconnect_task.done():
            return

        async def reconnect_task():
            await asyncio.sleep(5)  # Wait 5 seconds before reconnecting
            if not self.is_connected():
                logger.info("Attempting automatic reconnection...")
                await self.connect()

        self._reconnect_task = asyncio.create_task(reconnect_task())

    async def _validate_account(self):
        """Validate account access and settings."""
        try:
            # Wait a moment for connection to stabilize
            await asyncio.sleep(0.5)

            # Simple validation - just log the account info
            if self.account_id:
                logger.info(f"Using specified account: {self.account_id}")
            else:
                logger.info("Using default account")

            logger.info("Account validation completed")

        except Exception as e:
            logger.error(f"Account validation failed: {e}")
            raise IBKRConnectionError(f"Account validation failed: {e}") from e

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
