"""
Adapter utilities to convert broker-specific portfolio snapshots to the
internal domain Portfolio used by the strategy and simulations.
"""

from __future__ import annotations

from typing import Mapping

from loguru import logger

from ..data.sources.ibkr_client import IBKRPortfolio
from ..strategy.rebalancing import Portfolio, Position


def to_domain_portfolio(
    ibkr: IBKRPortfolio,
    price_map: Mapping[str, float] | None = None,
) -> Portfolio:
    """
    Convert an IBKRPortfolio (broker DTO) into the domain Portfolio.

    Args:
        ibkr: IBKR portfolio snapshot from broker
        price_map: Optional mapping from ticker -> latest price to set current prices

    Returns:
        Portfolio: domain model with positions keyed by ticker
    """
    price_map = price_map or {}

    portfolio = Portfolio(cash=ibkr.cash)

    added = 0
    for p in ibkr.positions:
        # Skip flat positions
        if getattr(p, "position", 0) == 0:
            continue

        ticker = getattr(p, "symbol", "")
        if not ticker:
            # Try contract symbol as fallback
            ticker = getattr(getattr(p, "contract", object()), "symbol", "")
        if not ticker:
            continue

        shares = int(round(getattr(p, "position", 0)))
        entry = float(getattr(p, "avg_cost", 0.0))
        current = float(price_map.get(ticker, entry)) if entry is not None else float(price_map.get(ticker, 0.0))

        pos = Position(
            ticker=ticker,
            shares=shares,
            entry_price=entry or 0.0,
            current_price=current or 0.0,
        )
        portfolio.add_position(pos)
        added += 1

    logger.info(f"Converted IBKR portfolio to domain model: {added} positions, ${portfolio.cash:,.2f} cash")
    return portfolio

