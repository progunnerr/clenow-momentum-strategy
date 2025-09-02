"""
Tests for trading schedule module.
"""

from datetime import UTC, datetime

from clenow_momentum.strategy.trading_schedule import (
    get_first_wednesday_of_month,
    get_next_rebalancing_date,
    get_next_trading_day,
    get_rebalancing_months,
    get_rebalancing_schedule,
    get_trading_calendar_summary,
    is_market_open,
    is_rebalancing_day,
    is_rebalancing_month,
    is_trading_day,
    is_wednesday,
    should_execute_trades,
)


class TestBasicFunctions:
    """Test basic helper functions."""

    def test_is_wednesday(self):
        """Test Wednesday detection."""
        # Wednesday
        wed = datetime(2025, 1, 1, tzinfo=UTC)  # Jan 1, 2025 is Wednesday
        assert is_wednesday(wed) is True

        # Not Wednesday
        thu = datetime(2025, 1, 2, tzinfo=UTC)  # Thursday
        assert is_wednesday(thu) is False

        # Another Wednesday
        wed2 = datetime(2025, 1, 8, tzinfo=UTC)  # Jan 8, 2025 is Wednesday
        assert is_wednesday(wed2) is True

    def test_is_market_open(self):
        """Test market open detection."""
        # Weekday - market open
        monday = datetime(2025, 1, 6, tzinfo=UTC)
        assert is_market_open(monday) is True

        # Weekend - market closed
        saturday = datetime(2025, 1, 4, tzinfo=UTC)
        assert is_market_open(saturday) is False

        sunday = datetime(2025, 1, 5, tzinfo=UTC)
        assert is_market_open(sunday) is False

    def test_get_rebalancing_months(self):
        """Test rebalancing months."""
        months = get_rebalancing_months()
        assert months == [1, 3, 5, 7, 9, 11]
        assert all(m % 2 == 1 for m in months)  # All odd months

    def test_is_rebalancing_month(self):
        """Test rebalancing month detection."""
        # Odd months - rebalancing
        jan = datetime(2025, 1, 15, tzinfo=UTC)
        assert is_rebalancing_month(jan) is True

        mar = datetime(2025, 3, 15, tzinfo=UTC)
        assert is_rebalancing_month(mar) is True

        # Even months - not rebalancing
        feb = datetime(2025, 2, 15, tzinfo=UTC)
        assert is_rebalancing_month(feb) is False

        apr = datetime(2025, 4, 15, tzinfo=UTC)
        assert is_rebalancing_month(apr) is False


class TestTradingDays:
    """Test trading day logic."""

    def test_is_trading_day(self):
        """Test trading day detection."""
        # Wednesday - trading day
        wed = datetime(2025, 1, 1, tzinfo=UTC)
        assert is_trading_day(wed) is True

        # Thursday - not trading day
        thu = datetime(2025, 1, 2, tzinfo=UTC)
        assert is_trading_day(thu) is False

        # Weekend Wednesday (doesn't exist in reality but test the logic)
        # We can't create a Wednesday that's also weekend, so skip this

    def test_get_next_trading_day(self):
        """Test getting next trading day."""
        # From Tuesday, next is Wednesday (tomorrow)
        tuesday = datetime(2024, 12, 31, tzinfo=UTC)
        next_day = get_next_trading_day(tuesday)
        assert next_day.weekday() == 2  # Wednesday
        assert next_day.date() == datetime(2025, 1, 1, tzinfo=UTC).date()

        # From Wednesday, next is next Wednesday
        wednesday = datetime(2025, 1, 1, tzinfo=UTC)
        next_day = get_next_trading_day(wednesday)
        assert next_day.weekday() == 2
        assert next_day.date() == datetime(2025, 1, 8, tzinfo=UTC).date()

        # From Thursday, next Wednesday is in 6 days
        thursday = datetime(2025, 1, 2, tzinfo=UTC)
        next_day = get_next_trading_day(thursday)
        assert next_day.weekday() == 2
        assert next_day.date() == datetime(2025, 1, 8, tzinfo=UTC).date()


class TestRebalancingSchedule:
    """Test rebalancing schedule logic."""

    def test_get_first_wednesday_of_month(self):
        """Test getting first Wednesday of month."""
        # January 2025 - first Wednesday is Jan 1
        first_wed = get_first_wednesday_of_month(2025, 1)
        assert first_wed.date() == datetime(2025, 1, 1, tzinfo=UTC).date()
        assert first_wed.weekday() == 2

        # February 2025 - first Wednesday is Feb 5
        first_wed = get_first_wednesday_of_month(2025, 2)
        assert first_wed.date() == datetime(2025, 2, 5, tzinfo=UTC).date()
        assert first_wed.weekday() == 2

        # March 2025 - first Wednesday is Mar 5
        first_wed = get_first_wednesday_of_month(2025, 3)
        assert first_wed.date() == datetime(2025, 3, 5, tzinfo=UTC).date()
        assert first_wed.weekday() == 2

    def test_is_rebalancing_day(self):
        """Test rebalancing day detection."""
        # First Wednesday of January 2025 - rebalancing day
        jan_1 = datetime(2025, 1, 1, tzinfo=UTC)
        assert is_rebalancing_day(jan_1) is True

        # Second Wednesday of January - not rebalancing day
        jan_8 = datetime(2025, 1, 8, tzinfo=UTC)
        assert is_rebalancing_day(jan_8) is False

        # First Wednesday of February (even month) - not rebalancing
        feb_5 = datetime(2025, 2, 5, tzinfo=UTC)
        assert is_rebalancing_day(feb_5) is False

        # First Wednesday of March - rebalancing day
        mar_5 = datetime(2025, 3, 5, tzinfo=UTC)
        assert is_rebalancing_day(mar_5) is True

        # Not a Wednesday - not rebalancing
        jan_2 = datetime(2025, 1, 2, tzinfo=UTC)  # Thursday
        assert is_rebalancing_day(jan_2) is False

    def test_get_next_rebalancing_date(self):
        """Test getting next rebalancing date."""
        # From December 2024, next is Jan 1, 2025
        dec_15 = datetime(2024, 12, 15, tzinfo=UTC)
        next_rebal = get_next_rebalancing_date(dec_15)
        assert next_rebal.date() == datetime(2025, 1, 1, tzinfo=UTC).date()

        # From Jan 2, 2025, next is Mar 5, 2025
        jan_2 = datetime(2025, 1, 2, tzinfo=UTC)
        next_rebal = get_next_rebalancing_date(jan_2)
        assert next_rebal.date() == datetime(2025, 3, 5, tzinfo=UTC).date()

        # From Feb 2025, next is Mar 5, 2025
        feb_15 = datetime(2025, 2, 15, tzinfo=UTC)
        next_rebal = get_next_rebalancing_date(feb_15)
        assert next_rebal.date() == datetime(2025, 3, 5, tzinfo=UTC).date()

    def test_get_rebalancing_schedule(self):
        """Test generating rebalancing schedule."""
        start = datetime(2025, 1, 1, tzinfo=UTC)
        schedule = get_rebalancing_schedule(start, num_periods=6)

        assert len(schedule) == 6

        # Check dates are first Wednesdays of odd months
        expected_dates = [
            datetime(2025, 3, 5, tzinfo=UTC),  # Mar (Jan 1 is today, so skipped)
            datetime(2025, 5, 7, tzinfo=UTC),  # May
            datetime(2025, 7, 2, tzinfo=UTC),  # Jul
            datetime(2025, 9, 3, tzinfo=UTC),  # Sep
            datetime(2025, 11, 5, tzinfo=UTC),  # Nov
            datetime(2026, 1, 7, tzinfo=UTC),  # Jan next year
        ]

        for i, expected in enumerate(expected_dates):
            if i < len(schedule):
                actual = schedule.iloc[i]["rebalancing_date"]
                assert actual.date() == expected.date()
                assert actual.weekday() == 2  # Wednesday

        # Check all are in odd months
        for _, row in schedule.iterrows():
            assert row["month"] in [1, 3, 5, 7, 9, 11]
            assert row["weekday"] == "Wednesday"


class TestTradingDecisions:
    """Test trading decision logic."""

    def test_should_execute_trades(self):
        """Test trade execution decision."""
        # Rebalancing day - should execute
        jan_1 = datetime(2025, 1, 1, tzinfo=UTC)
        should_trade, reason = should_execute_trades(jan_1)
        assert should_trade is True
        assert "REBALANCING DAY" in reason

        # Regular Wednesday - should execute
        jan_8 = datetime(2025, 1, 8, tzinfo=UTC)
        should_trade, reason = should_execute_trades(jan_8)
        assert should_trade is True
        assert "Regular Wednesday" in reason

        # Not Wednesday - should not execute
        jan_2 = datetime(2025, 1, 2, tzinfo=UTC)
        should_trade, reason = should_execute_trades(jan_2)
        assert should_trade is False
        assert "Not a Wednesday" in reason

        # Weekend - should not execute
        jan_4 = datetime(2025, 1, 4, tzinfo=UTC)  # Saturday
        should_trade, reason = should_execute_trades(jan_4)
        assert should_trade is False

    def test_get_trading_calendar_summary(self):
        """Test trading calendar summary."""
        # Mock a specific date for testing
        # This test will be date-dependent, so we just check structure
        summary = get_trading_calendar_summary()

        assert "current_date" in summary
        assert "current_weekday" in summary
        assert "is_trading_day" in summary
        assert "is_rebalancing_day" in summary
        assert "next_trading_day" in summary
        assert "next_rebalancing_date" in summary
        assert "rebalancing_months" in summary
        assert "days_until_next_trading" in summary
        assert "days_until_next_rebalancing" in summary

        # Check rebalancing months
        assert summary["rebalancing_months"] == [1, 3, 5, 7, 9, 11]
