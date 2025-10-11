"""
Portfolio synchronization with Interactive Brokers.

This module handles synchronization between the internal portfolio state
and the actual positions in the IBKR account.
"""

import json
from pathlib import Path

from loguru import logger

from clenow_momentum.data.sources.ibkr_client import IBKRClient, IBKRPortfolio
from clenow_momentum.utils.config import load_config


class PortfolioSyncError(Exception):
    """Base exception for portfolio synchronization errors."""

    pass


class PortfolioSynchronizer:
    """
    Synchronizes portfolio state between IBKR account and local storage.

    This class handles:
    - Loading portfolio state from local file
    - Fetching current positions from IBKR
    - Updating and saving portfolio state
    - Detecting discrepancies between internal and broker state
    - Reconciling differences
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize portfolio synchronizer.

        Args:
            config: Configuration dictionary. If None, loads from config file.
        """
        self.config = config or load_config()
        self.portfolio_file = Path(self.config.get("portfolio_state_file", "data/portfolio_state.json"))

    def sync_with_ibkr(self, save: bool = True) -> IBKRPortfolio:
        """
        Get complete portfolio snapshot from IBKR.
        
        This method:
        1. Connects to IBKR and fetches complete portfolio state
        2. Returns IBKRPortfolio (IBKR is source of truth)
        3. Saves portfolio to file (if save=True)

        Args:
            save: Whether to save the synchronized portfolio to file

        Returns:
            IBKRPortfolio with current IBKR data

        Raises:
            PortfolioSyncError: If synchronization fails
        """
        try:
            logger.info("Fetching portfolio from IBKR (source of truth)...")
            
            # Connect to IBKR and get complete portfolio
            with IBKRClient.from_config(self.config.get("ibkr", {})) as ibkr_client:
                logger.info("Connected to IBKR, fetching portfolio snapshot...")
                
                # Get complete portfolio snapshot
                ibkr_portfolio = ibkr_client.get_portfolio()
                
                logger.success(
                    f"Portfolio fetched from IBKR: {ibkr_portfolio.num_positions} positions, "
                    f"${ibkr_portfolio.cash:,.0f} cash, "
                    f"${ibkr_portfolio.net_liquidation:,.0f} total value"
                )
            
            # Save portfolio to file if requested
            if save:
                self.save_portfolio(ibkr_portfolio)
            
            return ibkr_portfolio

        except Exception as e:
            logger.error(f"Portfolio synchronization failed: {e}")
            raise PortfolioSyncError(f"Portfolio synchronization failed: {e}") from e

    def save_portfolio(self, portfolio: IBKRPortfolio) -> Path:
        """
        Save IBKRPortfolio to JSON file.

        Args:
            portfolio: IBKRPortfolio to save

        Returns:
            Path to saved file
        """
        # Ensure directory exists
        self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        with open(self.portfolio_file, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2, default=str)

        logger.info(f"Portfolio saved to {self.portfolio_file}")
        return self.portfolio_file
