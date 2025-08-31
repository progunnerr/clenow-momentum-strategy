"""
Trading schedule management for Clenow momentum strategy.

This module implements the Wednesday-only trading schedule and bi-monthly
rebalancing logic according to Clenow's systematic trading rules.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
from loguru import logger


def is_wednesday(date: datetime) -> bool:
    """
    Check if a given date is a Wednesday.

    Args:
        date: Date to check

    Returns:
        True if Wednesday (weekday == 2), False otherwise
    """
    return date.weekday() == 2


def is_market_open(date: datetime) -> bool:
    """
    Check if US stock market is open on a given date.

    Simplified implementation - checks for weekdays only.
    A production system would check for market holidays.

    Args:
        date: Date to check

    Returns:
        True if market is likely open, False otherwise
    """
    # Basic check: market is open on weekdays
    # TODO: Add holiday calendar check (New Year's, Christmas, etc.)
    # For now, we'll assume all weekdays are trading days
    # In production, use pandas_market_calendars or similar
    return date.weekday() < 5  # Monday = 0, Friday = 4


def is_trading_day(date: datetime | None = None, bypass_wednesday: bool = False) -> bool:
    """
    Check if a given date is a valid trading day (Wednesday with market open).

    According to Clenow's strategy, trades are only executed on Wednesdays.

    Args:
        date: Date to check (defaults to today UTC)
        bypass_wednesday: If True, bypass Wednesday requirement (for testing)

    Returns:
        True if valid trading day, False otherwise
    """
    if date is None:
        date = datetime.now(UTC)

    # Check if market is open
    if not is_market_open(date):
        return False

    # Check Wednesday requirement (unless bypassed)
    if bypass_wednesday:
        logger.debug(f"{date.strftime('%Y-%m-%d')} - Wednesday check bypassed for testing")
        return True

    # Must be Wednesday
    is_valid = is_wednesday(date)

    if is_valid:
        logger.debug(f"{date.strftime('%Y-%m-%d')} is a valid trading day (Wednesday)")

    return is_valid


def get_next_trading_day(from_date: datetime | None = None) -> datetime:
    """
    Get the next valid trading day (Wednesday with market open).

    Args:
        from_date: Start searching from this date (defaults to today)

    Returns:
        Next Wednesday when market is open
    """
    if from_date is None:
        from_date = datetime.now(UTC)

    # Start from tomorrow to avoid returning today
    current = from_date + timedelta(days=1)

    # Search for next Wednesday (max 7 days ahead)
    for _ in range(7):
        if is_trading_day(current):
            logger.debug(f"Next trading day: {current.strftime('%Y-%m-%d')}")
            return current
        current += timedelta(days=1)

    # Should never reach here, but just in case
    raise ValueError("Could not find next trading day within 7 days")


def get_rebalancing_months() -> list[int]:
    """
    Get the months when rebalancing should occur.

    Following bi-monthly schedule on odd months:
    January (1), March (3), May (5), July (7), September (9), November (11)

    Returns:
        List of month numbers for rebalancing
    """
    return [1, 3, 5, 7, 9, 11]


def is_rebalancing_month(date: datetime) -> bool:
    """
    Check if the given date falls in a rebalancing month.

    Args:
        date: Date to check

    Returns:
        True if in a rebalancing month, False otherwise
    """
    return date.month in get_rebalancing_months()


def get_first_wednesday_of_month(year: int, month: int) -> datetime:
    """
    Get the first Wednesday of a given month.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        First Wednesday of the month as datetime
    """
    # Start from the first day of the month
    first_day = datetime(year, month, 1, tzinfo=UTC)

    # Find the first Wednesday
    days_until_wednesday = (2 - first_day.weekday()) % 7
    if days_until_wednesday == 0 and first_day.weekday() != 2:
        days_until_wednesday = 7

    return first_day + timedelta(days=days_until_wednesday)


def is_rebalancing_day(date: datetime | None = None, bypass_wednesday: bool = False) -> bool:
    """
    Check if a given date is a rebalancing day.

    Rebalancing occurs on the first Wednesday of odd months
    (January, March, May, July, September, November).

    Args:
        date: Date to check (defaults to today)
        bypass_wednesday: If True, bypass Wednesday requirement and treat as rebalancing day (for testing)

    Returns:
        True if rebalancing day, False otherwise
    """
    if date is None:
        date = datetime.now(UTC)

    # If bypassing for testing, always treat as rebalancing day
    if bypass_wednesday:
        logger.info(f"🔄 {date.strftime('%Y-%m-%d')} treated as REBALANCING DAY (bypass active)")
        return True

    # Must be a trading day first
    if not is_trading_day(date, bypass_wednesday):
        return False

    # Must be in a rebalancing month
    if not is_rebalancing_month(date):
        return False

    # Must be the first Wednesday of the month
    first_wednesday = get_first_wednesday_of_month(date.year, date.month)
    is_rebalancing = date.date() == first_wednesday.date()

    if is_rebalancing:
        logger.info(f"🔄 {date.strftime('%Y-%m-%d')} is a REBALANCING DAY")

    return is_rebalancing


def get_next_rebalancing_date(from_date: datetime | None = None) -> datetime:
    """
    Get the next rebalancing date.

    Args:
        from_date: Start searching from this date (defaults to today)

    Returns:
        Next rebalancing date (first Wednesday of next odd month)
    """
    if from_date is None:
        from_date = datetime.now(UTC)

    current_date = from_date

    # Search through the next 6 months maximum
    for _ in range(6):
        # If we're before the first Wednesday of a rebalancing month, return it
        if is_rebalancing_month(current_date):
            first_wednesday = get_first_wednesday_of_month(
                current_date.year, current_date.month
            )
            if first_wednesday.date() > from_date.date():
                logger.debug(
                    f"Next rebalancing date: {first_wednesday.strftime('%Y-%m-%d')}"
                )
                return first_wednesday

        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1, day=1)

    raise ValueError("Could not find next rebalancing date within 6 months")


def get_rebalancing_schedule(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    num_periods: int = 6
) -> pd.DataFrame:
    """
    Generate a rebalancing schedule for the specified period.

    Args:
        start_date: Schedule start date (defaults to today)
        end_date: Schedule end date (if None, use num_periods)
        num_periods: Number of rebalancing periods to generate (if end_date is None)

    Returns:
        DataFrame with rebalancing dates and details
    """
    if start_date is None:
        start_date = datetime.now(UTC)

    rebalancing_dates = []
    current_date = start_date

    # Generate rebalancing dates
    while len(rebalancing_dates) < num_periods:
        next_date = get_next_rebalancing_date(current_date)

        if end_date and next_date > end_date:
            break

        rebalancing_dates.append({
            'rebalancing_date': next_date,
            'year': next_date.year,
            'month': next_date.month,
            'month_name': next_date.strftime('%B'),
            'weekday': next_date.strftime('%A'),
        })

        current_date = next_date

    if not rebalancing_dates:
        logger.warning("No rebalancing dates found in specified period")
        return pd.DataFrame()

    schedule_df = pd.DataFrame(rebalancing_dates)
    schedule_df['rebalancing_number'] = range(1, len(schedule_df) + 1)

    # Calculate days until each rebalancing
    today = datetime.now(UTC)
    schedule_df['days_until'] = schedule_df['rebalancing_date'].apply(
        lambda x: (x.date() - today.date()).days
    )

    logger.info(f"Generated rebalancing schedule with {len(schedule_df)} dates")

    return schedule_df


def get_trading_calendar_summary(bypass_wednesday: bool = False) -> dict:
    """
    Get a summary of the current trading calendar status.

    Args:
        bypass_wednesday: If True, bypass Wednesday requirement (for testing)

    Returns:
        Dictionary with trading calendar information
    """
    now = datetime.now(UTC)

    summary = {
        'current_date': now.strftime('%Y-%m-%d'),
        'current_weekday': now.strftime('%A'),
        'is_trading_day': is_trading_day(now, bypass_wednesday),
        'is_rebalancing_day': is_rebalancing_day(now, bypass_wednesday),
        'next_trading_day': get_next_trading_day(now).strftime('%Y-%m-%d') if not bypass_wednesday else now.strftime('%Y-%m-%d'),
        'next_rebalancing_date': get_next_rebalancing_date(now).strftime('%Y-%m-%d'),
        'rebalancing_months': get_rebalancing_months(),
        'bypass_active': bypass_wednesday
    }

    # Calculate days until events
    if not bypass_wednesday:
        next_trading = get_next_trading_day(now)
        next_rebalancing = get_next_rebalancing_date(now)
        summary['days_until_next_trading'] = (next_trading.date() - now.date()).days
        summary['days_until_next_rebalancing'] = (next_rebalancing.date() - now.date()).days
    else:
        summary['days_until_next_trading'] = 0
        next_rebalancing = get_next_rebalancing_date(now)
        summary['days_until_next_rebalancing'] = (next_rebalancing.date() - now.date()).days

    return summary


def should_execute_trades(date: datetime | None = None, bypass_wednesday: bool = False) -> tuple[bool, str]:
    """
    Determine if trades should be executed on the given date.

    Args:
        date: Date to check (defaults to today)
        bypass_wednesday: If True, bypass Wednesday requirement (for testing)

    Returns:
        Tuple of (should_execute, reason_message)
    """
    if date is None:
        date = datetime.now(UTC)

    # Check if it's a trading day
    if not is_trading_day(date, bypass_wednesday):
        if bypass_wednesday:
            return False, "Market is closed"
        if not is_wednesday(date):
            return False, f"Not a Wednesday (today is {date.strftime('%A')})"
        return False, "Wednesday but market is closed"

    # Check if it's a rebalancing day
    if is_rebalancing_day(date, bypass_wednesday):
        return True, "✅ REBALANCING DAY - Execute all rebalancing trades"

    # Regular trading day
    if bypass_wednesday:
        return True, "✅ Trading day (Wednesday check bypassed for testing)"
    return True, "✅ Regular Wednesday trading day"
