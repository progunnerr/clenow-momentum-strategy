"""Market universe registry.

Defines UniverseSpec — a frozen dataclass describing each supported trading
universe — and the UNIVERSES dict that maps IndexSymbol keys to their spec.

A symbol in IndexSymbol is a *vocabulary* entry; a symbol in UNIVERSES is
*runtime-available*. Prefer get_universe_spec() over direct dict access so
callers get a clear error for unregistered symbols.
"""

from dataclasses import dataclass

from .interfaces import IndexSymbol


@dataclass(frozen=True)
class UniverseSpec:
    """Describes a market universe and its data sources.

    Attributes:
        symbol:                   IndexSymbol key ("SP500", "RUSSELL1000", ...)
        display_name:             Human-readable name ("S&P 500")
        wiki_url:                 Wikipedia page URL for constituent list
        wiki_table_id:            HTML table id to select (None → auto-select by filter)
        symbol_column_candidates: Column name(s) to try in order for ticker symbols
        expected_row_range:       Plausible (min, max) row count for the constituents table
        benchmark_etf:            Tradable ETF used for regime detection ("SPY", "IWB")
        benchmark_index:          Index quote symbol for analytics ("^GSPC", "^RUI")
    """

    symbol: IndexSymbol
    display_name: str
    wiki_url: str
    wiki_table_id: str | None
    symbol_column_candidates: tuple[str, ...]
    expected_row_range: tuple[int, int]
    benchmark_etf: str
    benchmark_index: str


UNIVERSES: dict[str, UniverseSpec] = {
    "SP500": UniverseSpec(
        symbol="SP500",
        display_name="S&P 500",
        wiki_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        wiki_table_id="constituents",
        symbol_column_candidates=("Symbol",),
        expected_row_range=(450, 560),
        benchmark_etf="SPY",
        benchmark_index="^GSPC",
    ),
    "RUSSELL1000": UniverseSpec(
        symbol="RUSSELL1000",
        display_name="Russell 1000",
        wiki_url="https://en.wikipedia.org/wiki/Russell_1000_Index",
        wiki_table_id=None,
        symbol_column_candidates=("Symbol", "Ticker"),
        expected_row_range=(900, 1100),
        benchmark_etf="IWB",
        benchmark_index="^RUI",
    ),
}


def get_universe_spec(symbol: str) -> UniverseSpec:
    """Return the UniverseSpec for a registered universe symbol.

    Args:
        symbol: Universe key, case-insensitive (e.g. "SP500", "russell1000").

    Returns:
        Matching UniverseSpec.

    Raises:
        ValueError: If the symbol is not registered in UNIVERSES.
    """
    key = symbol.upper()
    if key not in UNIVERSES:
        registered = sorted(UNIVERSES.keys())
        raise ValueError(
            f"Unknown universe {symbol!r}. Registered: {registered}"
        )
    return UNIVERSES[key]
