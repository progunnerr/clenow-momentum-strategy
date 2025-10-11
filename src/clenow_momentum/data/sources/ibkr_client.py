"""
Simple synchronous IBKR client that works with ib_async's patterns.

This version doesn't fight with ib_async's event loop management.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ib_async import IB, Contract, MarketOrder, Stock, Trade
from loguru import logger


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


def get_trading_mode(config: dict = None) -> str:
    """
    Get the current trading mode (paper or live) based on port.

    Args:
        config: Configuration dictionary with ibkr_port

    Returns:
        'paper', 'live', or 'unknown'
    """
    if config is None:
        from ..utils.config import load_config

        config = load_config()

    # Handle both flat config and nested ibkr config
    if "ibkr_port" in config:
        port = config["ibkr_port"]
    elif "ibkr" in config and "port" in config["ibkr"]:
        port = config["ibkr"]["port"]
    else:
        logger.warning("No IBKR port found in config, defaulting to paper trading")
        return "paper"

    return get_trading_mode_from_port(port)


class IBKRError(Exception):
    """Base exception for IBKR client errors."""

    pass


@dataclass
class IBKRContract:
    """
    Represents an IBKR contract with full details.

    Matches the structure of ib_async's Contract objects.
    """

    conId: int  # IB's internal contract ID
    symbol: str  # Ticker symbol (e.g., 'AAPL')
    secType: str = "STK"  # Security type (STK, OPT, FUT, etc.)
    exchange: str = "SMART"  # Exchange (NYSE, NASDAQ, SMART, etc.)
    primaryExchange: str | None = None  # Primary listing exchange
    currency: str = "USD"  # Currency denomination
    localSymbol: str | None = None  # Local exchange symbol
    tradingClass: str | None = None  # Trading class

    @classmethod
    def from_ib_contract(cls, contract: Contract) -> "IBKRContract":
        """Create from ib_async Contract object."""
        return cls(
            conId=getattr(contract, "conId", 0),
            symbol=getattr(contract, "symbol", ""),
            secType=getattr(contract, "secType", "STK"),
            exchange=getattr(contract, "exchange", "SMART"),
            primaryExchange=getattr(contract, "primaryExchange", None),
            currency=getattr(contract, "currency", "USD"),
            localSymbol=getattr(contract, "localSymbol", None),
            tradingClass=getattr(contract, "tradingClass", None),
        )


@dataclass
class IBKRPosition:
    """
    Represents an IBKR position with full details.

    Matches the structure returned by ib_async's positions() method.
    """

    account: str
    contract: Contract  # Raw Contract from ib_async
    position: float  # Number of shares (positive for long, negative for short)
    avg_cost: float  # Average cost per share

    @property
    def symbol(self) -> str:
        """Get the ticker symbol."""
        return getattr(self.contract, "symbol", "")

    @property
    def quantity(self) -> float:
        """Get the position quantity (alias for position)."""
        return self.position


@dataclass
class AccountSummary:
    """Account summary information."""

    net_liquidation: float
    buying_power: float
    total_cash: float
    excess_liquidity: float


@dataclass
class IBKRPortfolio:
    """
    Complete portfolio state from IBKR.

    This is the source of truth for account state.
    """

    account: str  # Account ID
    positions: list[IBKRPosition]  # All positions
    cash: float  # Total cash (from AccountSummary)
    net_liquidation: float  # Total account value
    buying_power: float  # Available buying power
    timestamp: datetime  # When this snapshot was taken

    @property
    def num_positions(self) -> int:
        """Number of positions (excluding cash)."""
        return len([p for p in self.positions if p.position != 0])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "account": self.account,
            "cash": self.cash,
            "net_liquidation": self.net_liquidation,
            "buying_power": self.buying_power,
            "timestamp": self.timestamp.isoformat(),
            "positions": [
                {
                    "symbol": p.symbol,
                    "position": p.position,
                    "avg_cost": p.avg_cost,
                    "account": p.account,
                    "contract": {
                        "conId": getattr(p.contract, "conId", 0),
                        "symbol": getattr(p.contract, "symbol", ""),
                        "exchange": getattr(p.contract, "exchange", ""),
                        "currency": getattr(p.contract, "currency", "USD"),
                    },
                }
                for p in self.positions
                if p.position != 0
            ],
        }


class IBKRClient:
    """
    Simple synchronous IBKR client.

    Works with ib_async's event loop instead of fighting it.

    Usage:
        # With context manager - using individual params
        with IBKRClient(host="127.0.0.1", port=7497) as client:
            summary = client.get_account_summary()
            positions = client.get_positions()

        # With context manager - using config dict
        with IBKRClient.from_config(config["ibkr"]) as client:
            summary = client.get_account_summary()

        # Manual
        client = IBKRClient()
        client.connect()
        # ... do stuff
        client.disconnect()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: str = "",
    ):
        """Initialize client."""
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self._ib: IB | None = None

    @classmethod
    def from_config(cls, ibkr_config: dict[str, Any]) -> "IBKRClient":
        """
        Create IBKRClient from config dictionary.

        Args:
            ibkr_config: IBKR configuration dict with keys like 'host', 'port', 'client_id', 'account_id'

        Returns:
            IBKRClient instance

        Example:
            ibkr_config = config.get("ibkr", {})
            with IBKRClient.from_config(ibkr_config) as client:
                positions = client.get_positions()
        """
        return cls(
            host=ibkr_config.get("host", "127.0.0.1"),
            port=ibkr_config.get("port", 7497),
            client_id=ibkr_config.get("client_id", 1),
            account=ibkr_config.get("account_id", ""),
        )

    def connect(self) -> None:
        """Connect to IBKR."""
        # Always create fresh IB instance
        self._ib = IB()

        try:
            # Don't use async, just connect synchronously
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            logger.success(f"Connected to IBKR at {self.host}:{self.port}")

            # Give IB a moment to populate account data
            import time

            time.sleep(0.2)  # Small delay for data to be ready

        except Exception as e:
            self._ib = None
            raise IBKRError(f"Connection failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")
        self._ib = None

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._ib is not None and self._ib.isConnected()

    def _require_connection(self) -> None:
        """Ensure connected."""
        if not self.is_connected():
            raise IBKRError("Not connected to IBKR")

    def buy(self, symbol: str, quantity: int) -> Trade:
        """Place market buy order."""
        self._require_connection()

        contract = Stock(symbol.upper(), "SMART", "USD")
        order = MarketOrder("BUY", quantity)

        trade = self._ib.placeOrder(contract, order)
        logger.info(
            f"Buy order placed: {quantity} {symbol} - Order ID: {trade.order.orderId}"
        )

        # Allow IB to process the order
        self._ib.sleep(0.1)

        return trade

    def sell(self, symbol: str, quantity: int) -> Trade:
        """Place market sell order."""
        self._require_connection()

        contract = Stock(symbol.upper(), "SMART", "USD")
        order = MarketOrder("SELL", quantity)

        trade = self._ib.placeOrder(contract, order)
        logger.info(
            f"Sell order placed: {quantity} {symbol} - Order ID: {trade.order.orderId}"
        )

        # Allow IB to process the order
        self._ib.sleep(0.1)

        return trade

    def get_positions(self) -> list[IBKRPosition]:
        """
        Get current positions with full IBKR details.

        Returns:
            List of IBKRPosition objects with complete position information
        """
        self._require_connection()

        raw_positions = self._ib.positions()
        positions = []

        for pos in raw_positions:
            positions.append(
                IBKRPosition(
                    account=pos.account,
                    contract=pos.contract,
                    position=float(pos.position),
                    avg_cost=float(pos.avgCost),
                )
            )

        logger.debug(f"Retrieved {len(positions)} positions from IBKR")
        return positions

    def get_portfolio(self) -> IBKRPortfolio:
        """
        Get complete portfolio snapshot from IBKR.

        This is the main method to get the full account state.

        Returns:
            IBKRPortfolio with all account and position data
        """
        self._require_connection()

        # Get account summary
        account_summary = self.get_account_summary()

        # Get all positions
        positions = self.get_positions()

        # Get account ID (from first position or default)
        account_id = positions[0].account if positions else self.account

        # Create portfolio snapshot
        portfolio = IBKRPortfolio(
            account=account_id,
            positions=positions,
            cash=account_summary.total_cash,
            net_liquidation=account_summary.net_liquidation,
            buying_power=account_summary.buying_power,
            timestamp=datetime.now(UTC),
        )

        logger.info(
            f"Portfolio snapshot: {portfolio.num_positions} positions, "
            f"${portfolio.cash:,.0f} cash, "
            f"${portfolio.net_liquidation:,.0f} total value"
        )

        return portfolio

    def get_account_summary(self) -> AccountSummary:
        """Get account summary."""
        self._require_connection()

        account_values = self._ib.accountValues()
        summary_dict: dict[str, float] = {}

        for av in account_values:
            # Check all currencies, not just BASE/USD
            if av.tag == "NetLiquidation":
                summary_dict["net_liquidation"] = float(av.value)
            elif av.tag == "BuyingPower":
                summary_dict["buying_power"] = float(av.value)
            elif av.tag == "TotalCashValue":
                summary_dict["total_cash"] = float(av.value)
            elif av.tag == "ExcessLiquidity":
                summary_dict["excess_liquidity"] = float(av.value)

        return AccountSummary(
            net_liquidation=summary_dict.get("net_liquidation", 0.0),
            buying_power=summary_dict.get("buying_power", 0.0),
            total_cash=summary_dict.get("total_cash", 0.0),
            excess_liquidity=summary_dict.get("excess_liquidity", 0.0),
        )

    def get_open_orders(self) -> list[Trade]:
        """Get open orders."""
        self._require_connection()
        return self._ib.openTrades()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self.is_connected() else "disconnected"
        return f"IBKRClient(host='{self.host}', port={self.port}, status='{status}')"


# Example usage
if __name__ == "__main__":
    # This works cleanly with ib_async
    with IBKRClient(port=7497, client_id=127) as client:
        print(f"Connected: {client.is_connected()}")

        summary = client.get_account_summary()
        print(f"Account Value: ${summary.net_liquidation:,.2f}")

        positions = client.get_positions()
        print(f"Positions: {len(positions)}")

    print("Disconnected")
