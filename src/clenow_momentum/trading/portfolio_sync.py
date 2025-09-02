"""
Portfolio synchronization with Interactive Brokers.

This module handles synchronization between the internal portfolio state
and the actual positions in the IBKR account.
"""

import asyncio
from datetime import UTC, datetime

import yfinance as yf
from loguru import logger

from ..strategy.rebalancing import Portfolio, Position
from .ibkr_connector import IBKRConnector


class PortfolioSyncError(Exception):
    """Base exception for portfolio synchronization errors."""

    pass


class PortfolioSynchronizer:
    """
    Synchronizes portfolio state with IBKR account.

    This class handles:
    - Fetching current positions from IBKR
    - Updating internal portfolio state
    - Detecting discrepancies between internal and broker state
    - Reconciling differences
    """

    def __init__(self, ibkr_connector: IBKRConnector):
        """
        Initialize portfolio synchronizer.

        Args:
            ibkr_connector: Connected IBKR connector instance
        """
        self.ibkr = ibkr_connector
        self.last_sync: datetime | None = None

    async def sync_portfolio_from_ibkr(self, portfolio: Portfolio) -> Portfolio:
        """
        Synchronize portfolio with IBKR account positions.

        Args:
            portfolio: Current portfolio to update

        Returns:
            Updated portfolio with IBKR data

        Raises:
            PortfolioSyncError: If synchronization fails
        """
        if not self.ibkr.is_connected():
            raise PortfolioSyncError("IBKR not connected")

        try:
            logger.info("Synchronizing portfolio with IBKR account...")

            # Get account info from IBKR
            account_info = await self.ibkr.get_account_info()

            # Update cash position
            portfolio.cash = account_info.get("total_cash", 0.0)

            # Update positions
            ibkr_positions = account_info.get("positions", {})
            await self._sync_positions(portfolio, ibkr_positions)

            # Update last sync time
            self.last_sync = datetime.now(UTC)
            logger.success(f"Portfolio synchronized successfully with {len(ibkr_positions)} positions")

            return portfolio

        except Exception as e:
            logger.error(f"Portfolio synchronization failed: {e}")
            raise PortfolioSyncError(f"Portfolio synchronization failed: {e}") from e

    async def _sync_positions(self, portfolio: Portfolio, ibkr_positions: dict) -> None:
        """
        Synchronize individual positions.

        Args:
            portfolio: Portfolio to update
            ibkr_positions: Positions from IBKR
        """
        # Get current market prices for all tickers
        all_tickers = set(portfolio.positions.keys()) | set(ibkr_positions.keys())
        current_prices = await self._get_current_prices(list(all_tickers))

        # Update existing positions and add new ones
        for ticker, ibkr_pos in ibkr_positions.items():
            shares = int(ibkr_pos["shares"])
            if shares == 0:
                # Position was closed
                if ticker in portfolio.positions:
                    logger.info(f"Position {ticker} was closed")
                    portfolio.remove_position(ticker)
                continue

            current_price = current_prices.get(ticker, ibkr_pos.get("market_value", 0) / shares)

            if ticker in portfolio.positions:
                # Update existing position
                position = portfolio.positions[ticker]
                position.shares = shares
                position.current_price = current_price
                position.avg_cost_basis = ibkr_pos.get("avg_cost", position.entry_price)
                position.realized_pnl = ibkr_pos.get("unrealized_pnl", 0.0)
                position.last_updated = datetime.now(UTC)

                # Update entry price if we have better cost basis data
                if position.avg_cost_basis and position.avg_cost_basis > 0:
                    position.entry_price = position.avg_cost_basis

                logger.debug(f"Updated position {ticker}: {shares} shares @ ${current_price:.2f}")

            else:
                # New position found in IBKR that we don't have
                logger.info(f"New position found in IBKR: {ticker}")
                new_position = Position(
                    ticker=ticker,
                    shares=shares,
                    entry_price=ibkr_pos.get("avg_cost", current_price),
                    current_price=current_price,
                    entry_date=datetime.now(UTC),  # We don't know the actual entry date
                    atr=0.0,  # Will be calculated later
                    avg_cost_basis=ibkr_pos.get("avg_cost"),
                    realized_pnl=ibkr_pos.get("unrealized_pnl", 0.0),
                    last_updated=datetime.now(UTC),
                )
                portfolio.add_position(new_position)

        # Check for positions in our portfolio that are no longer in IBKR
        portfolio_tickers = set(portfolio.positions.keys())
        ibkr_tickers = set(ibkr_positions.keys())
        missing_tickers = portfolio_tickers - ibkr_tickers

        for ticker in missing_tickers:
            logger.warning(f"Position {ticker} in portfolio but not found in IBKR - may have been closed")
            # Optionally remove these positions or mark them for investigation
            # For now, we'll keep them but log the discrepancy

    async def _get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """
        Get current prices for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to current price
        """
        if not tickers:
            return {}

        try:
            # Use yfinance to get current prices
            prices = {}

            # Process in batches to avoid overwhelming yfinance
            batch_size = 20
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                ticker_data = yf.download(
                    " ".join(batch), period="1d", interval="1m", progress=False, threads=True
                )

                if len(batch) == 1:
                    # Single ticker
                    ticker = batch[0]
                    if not ticker_data.empty and "Close" in ticker_data.columns:
                        prices[ticker] = float(ticker_data["Close"].iloc[-1])
                else:
                    # Multiple tickers
                    if "Close" in ticker_data.columns:
                        for ticker in batch:
                            if (("Close", ticker) in ticker_data.columns
                                and not ticker_data[("Close", ticker)].empty):
                                prices[ticker] = float(ticker_data[("Close", ticker)].iloc[-1])

                # Small delay between batches
                await asyncio.sleep(0.1)

            logger.debug(f"Fetched prices for {len(prices)}/{len(tickers)} tickers")
            return prices

        except Exception as e:
            logger.warning(f"Failed to fetch current prices: {e}")
            return {}

    async def detect_portfolio_discrepancies(self, portfolio: Portfolio) -> list[dict]:
        """
        Detect discrepancies between internal portfolio and IBKR account.

        Args:
            portfolio: Internal portfolio state

        Returns:
            List of discrepancy reports
        """
        if not self.ibkr.is_connected():
            raise PortfolioSyncError("IBKR not connected")

        discrepancies = []

        try:
            account_info = await self.ibkr.get_account_info()
            ibkr_positions = account_info.get("positions", {})

            # Check cash discrepancy
            ibkr_cash = account_info.get("total_cash", 0.0)
            if abs(portfolio.cash - ibkr_cash) > 1.0:  # More than $1 difference
                discrepancies.append({
                    "type": "cash",
                    "ticker": "CASH",
                    "portfolio_value": portfolio.cash,
                    "ibkr_value": ibkr_cash,
                    "difference": portfolio.cash - ibkr_cash,
                    "severity": "high" if abs(portfolio.cash - ibkr_cash) > 100 else "low",
                })

            # Check position discrepancies
            all_tickers = set(portfolio.positions.keys()) | set(ibkr_positions.keys())

            for ticker in all_tickers:
                portfolio_pos = portfolio.positions.get(ticker)
                ibkr_pos = ibkr_positions.get(ticker)

                if portfolio_pos and not ibkr_pos:
                    # Position in portfolio but not in IBKR
                    discrepancies.append({
                        "type": "missing_in_ibkr",
                        "ticker": ticker,
                        "portfolio_shares": portfolio_pos.shares,
                        "ibkr_shares": 0,
                        "severity": "high",
                    })

                elif ibkr_pos and not portfolio_pos:
                    # Position in IBKR but not in portfolio
                    discrepancies.append({
                        "type": "missing_in_portfolio",
                        "ticker": ticker,
                        "portfolio_shares": 0,
                        "ibkr_shares": int(ibkr_pos["shares"]),
                        "severity": "medium",
                    })

                elif portfolio_pos and ibkr_pos:
                    # Both have the position, check quantities
                    portfolio_shares = portfolio_pos.shares
                    ibkr_shares = int(ibkr_pos["shares"])

                    if portfolio_shares != ibkr_shares:
                        discrepancies.append({
                            "type": "quantity_mismatch",
                            "ticker": ticker,
                            "portfolio_shares": portfolio_shares,
                            "ibkr_shares": ibkr_shares,
                            "difference": portfolio_shares - ibkr_shares,
                            "severity": "medium",
                        })

            if discrepancies:
                logger.warning(f"Found {len(discrepancies)} portfolio discrepancies")
            else:
                logger.success("No portfolio discrepancies found")

            return discrepancies

        except Exception as e:
            logger.error(f"Failed to detect portfolio discrepancies: {e}")
            raise PortfolioSyncError(f"Failed to detect discrepancies: {e}") from e

    async def reconcile_discrepancies(
        self, portfolio: Portfolio, discrepancies: list[dict], auto_fix: bool = False
    ) -> list[str]:
        """
        Reconcile portfolio discrepancies.

        Args:
            portfolio: Portfolio to reconcile
            discrepancies: List of discrepancies to fix
            auto_fix: Whether to automatically fix discrepancies

        Returns:
            List of actions taken
        """
        actions = []

        for discrepancy in discrepancies:
            if discrepancy["type"] == "cash":
                if auto_fix or discrepancy["severity"] == "low":
                    old_cash = portfolio.cash
                    portfolio.cash = discrepancy["ibkr_value"]
                    actions.append(f"Updated cash: ${old_cash:.2f} -> ${portfolio.cash:.2f}")

            elif discrepancy["type"] == "missing_in_portfolio":
                if auto_fix:
                    ticker = discrepancy["ticker"]
                    shares = discrepancy["ibkr_shares"]

                    # Create new position based on IBKR data
                    account_info = await self.ibkr.get_account_info()
                    ibkr_pos = account_info["positions"].get(ticker, {})

                    current_price = ibkr_pos.get("market_value", 0) / shares if shares > 0 else 0
                    new_position = Position(
                        ticker=ticker,
                        shares=shares,
                        entry_price=ibkr_pos.get("avg_cost", current_price),
                        current_price=current_price,
                        entry_date=datetime.now(UTC),
                        atr=0.0,
                        avg_cost_basis=ibkr_pos.get("avg_cost"),
                        realized_pnl=ibkr_pos.get("unrealized_pnl", 0.0),
                        last_updated=datetime.now(UTC),
                    )
                    portfolio.add_position(new_position)
                    actions.append(f"Added missing position: {ticker} ({shares} shares)")

            elif discrepancy["type"] == "quantity_mismatch" and auto_fix:
                ticker = discrepancy["ticker"]
                portfolio.positions[ticker].shares = discrepancy["ibkr_shares"]
                actions.append(
                    f"Updated {ticker} quantity: {discrepancy['portfolio_shares']} -> {discrepancy['ibkr_shares']}"
                )

        logger.info(f"Reconciliation completed: {len(actions)} actions taken")
        return actions

    def get_sync_status(self) -> dict:
        """
        Get synchronization status information.

        Returns:
            Dictionary with sync status details
        """
        return {
            "last_sync": self.last_sync,
            "is_connected": self.ibkr.is_connected(),
            "minutes_since_sync": (
                int((datetime.now(UTC) - self.last_sync).total_seconds() / 60)
                if self.last_sync
                else None
            ),
            "sync_required": (
                self.last_sync is None
                or (datetime.now(UTC) - self.last_sync).total_seconds() > 300  # 5 minutes
            ),
        }
