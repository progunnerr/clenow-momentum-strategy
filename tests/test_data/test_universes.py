"""Tests for the universe registry (data/universes.py)."""

import pytest

from src.clenow_momentum.data.universes import UNIVERSES, UniverseSpec, get_universe_spec


class TestUniverseRegistry:
    """UNIVERSES dict contains the expected entries."""

    def test_sp500_registered(self):
        assert "SP500" in UNIVERSES

    def test_russell1000_registered(self):
        assert "RUSSELL1000" in UNIVERSES

    def test_sp500_spec_shape(self):
        spec = UNIVERSES["SP500"]
        assert isinstance(spec, UniverseSpec)
        assert spec.symbol == "SP500"
        assert spec.display_name == "S&P 500"
        assert spec.benchmark_etf == "SPY"
        assert spec.benchmark_index == "^GSPC"
        assert spec.wiki_table_id == "constituents"
        assert "Symbol" in spec.symbol_column_candidates
        assert 450 <= spec.expected_row_range[0] < spec.expected_row_range[1]

    def test_russell1000_spec_shape(self):
        spec = UNIVERSES["RUSSELL1000"]
        assert isinstance(spec, UniverseSpec)
        assert spec.symbol == "RUSSELL1000"
        assert spec.display_name == "Russell 1000"
        assert spec.benchmark_etf == "IWB"
        assert spec.benchmark_index == "^RUI"
        assert spec.wiki_table_id is None  # uses filter-based selection
        assert any(c in spec.symbol_column_candidates for c in ("Symbol", "Ticker"))
        assert 900 <= spec.expected_row_range[0] < spec.expected_row_range[1]

    def test_all_specs_are_frozen(self):
        for spec in UNIVERSES.values():
            with pytest.raises((AttributeError, TypeError)):
                spec.symbol = "MUTATED"  # type: ignore[misc]


class TestGetUniverseSpec:
    """get_universe_spec() lookup behaviour."""

    def test_exact_match(self):
        spec = get_universe_spec("SP500")
        assert spec.symbol == "SP500"

    def test_case_insensitive(self):
        assert get_universe_spec("sp500") == get_universe_spec("SP500")
        assert get_universe_spec("russell1000") == get_universe_spec("RUSSELL1000")

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe_spec("SPX")

    def test_error_message_lists_registered(self):
        with pytest.raises(ValueError) as exc_info:
            get_universe_spec("INVALID")
        msg = str(exc_info.value)
        assert "SP500" in msg
        assert "RUSSELL1000" in msg

    def test_russell2000_not_registered(self):
        """RUSSELL2000 is in IndexSymbol vocabulary but not yet in UNIVERSES."""
        with pytest.raises(ValueError):
            get_universe_spec("RUSSELL2000")
