"""Wikipedia constituent fetching for market universes.

Provides a generic fetcher that works for any universe described by a
UniverseSpec, plus back-compat wrappers for the S&P 500 names that
existed before the universe registry was introduced.
"""

from io import StringIO
from typing import TYPE_CHECKING

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

if TYPE_CHECKING:
    from ..universes import UniverseSpec

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def _fetch_wikipedia_html(
    spec: "UniverseSpec",
    timeout: int = 10,
    headers: dict | None = None,
) -> str:
    """Fetch the Wikipedia page HTML for a universe spec."""
    if headers is None:
        headers = _DEFAULT_HEADERS

    logger.debug(f"Fetching {spec.display_name} constituents from Wikipedia: {spec.wiki_url}")

    response = requests.get(spec.wiki_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def _select_constituents_table(spec: "UniverseSpec", html: str) -> pd.DataFrame:
    """Select the strict constituent table for a universe spec.

    Table selection is strict:
    - If spec.wiki_table_id is set, find that table by HTML id (required).
    - Otherwise, scan all tables for one that (a) contains one of
      spec.symbol_column_candidates and (b) has a row count within
      spec.expected_row_range. Exactly one match is required; zero or multiple
      matches raises ValueError with a diagnostic summary.
    """
    if spec.wiki_table_id is not None:
        # --- id-based lookup (fast, unambiguous) ---
        soup = BeautifulSoup(html, "html.parser")
        table_tag = soup.find("table", {"id": spec.wiki_table_id})
        if table_tag is None:
            raise ValueError(
                f"[{spec.display_name}] Could not find <table id={spec.wiki_table_id!r}> "
                f"on {spec.wiki_url}"
            )
        df_list = pd.read_html(StringIO(str(table_tag)))
        if not df_list:
            raise ValueError(
                f"[{spec.display_name}] pandas could not parse table id={spec.wiki_table_id!r}"
            )
        df = df_list[0]
    else:
        # --- filter-based lookup (strict) ---
        all_tables = pd.read_html(StringIO(html))
        lo, hi = spec.expected_row_range
        candidates = []
        for i, df in enumerate(all_tables):
            has_col = any(c in df.columns for c in spec.symbol_column_candidates)
            in_range = lo <= len(df) <= hi
            if has_col and in_range:
                candidates.append((i, df))

        if len(candidates) == 0:
            summary = "; ".join(
                f"table[{i}] cols={list(t.columns)[:6]} rows={len(t)}"
                for i, t in enumerate(all_tables)
            )
            raise ValueError(
                f"[{spec.display_name}] No table matched filter "
                f"(symbol_columns={spec.symbol_column_candidates}, "
                f"row_range={spec.expected_row_range}). "
                f"Tables found: {summary}"
            )
        if len(candidates) > 1:
            summary = "; ".join(
                f"table[{i}] cols={list(df.columns)[:6]} rows={len(df)}"
                for i, df in candidates
            )
            raise ValueError(
                f"[{spec.display_name}] Ambiguous — {len(candidates)} tables matched filter. "
                f"Matches: {summary}"
            )

        _, df = candidates[0]

    return df


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name from candidate names."""
    return next((c for c in candidates if c in df.columns), None)


def fetch_index_constituents_from_wikipedia(
    spec: "UniverseSpec",
    timeout: int = 10,
    headers: dict | None = None,
) -> pd.DataFrame:
    """Fetch raw constituent metadata for any universe described by a UniverseSpec.

    Returns source-table symbols and optional metadata. Tickers are not
    yfinance-normalized here; normalization happens in the provider layer.

    Args:
        spec: UniverseSpec from data/universes.py
        timeout: HTTP timeout in seconds
        headers: Optional HTTP headers (defaults to a browser-like UA)

    Returns:
        DataFrame with source_symbol, company_name, and sector columns.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If table selection is ambiguous, missing, or the symbol
                    column cannot be found.
    """
    html = _fetch_wikipedia_html(spec, timeout=timeout, headers=headers)
    df = _select_constituents_table(spec, html)

    # --- resolve symbol column ---
    col = _resolve_column(df, spec.symbol_column_candidates)
    if col is None:
        raise ValueError(
            f"[{spec.display_name}] None of {spec.symbol_column_candidates} "
            f"found in table columns: {df.columns.tolist()}"
        )

    company_col = _resolve_column(df, spec.company_column_candidates)
    sector_col = _resolve_column(df, spec.sector_column_candidates)

    if company_col is None:
        logger.warning(
            f"[{spec.display_name}] Company metadata unavailable. "
            f"Tried columns: {spec.company_column_candidates}"
        )
    if sector_col is None:
        logger.warning(
            f"[{spec.display_name}] Sector metadata unavailable. "
            f"Tried columns: {spec.sector_column_candidates}"
        )

    constituents = pd.DataFrame(
        {
            "source_symbol": df[col],
            "company_name": df[company_col] if company_col else pd.NA,
            "sector": df[sector_col] if sector_col else pd.NA,
        }
    )
    constituents = constituents.dropna(subset=["source_symbol"])
    constituents["source_symbol"] = constituents["source_symbol"].astype(str).str.strip()
    constituents = constituents[constituents["source_symbol"] != ""]
    constituents = constituents.reset_index(drop=True)

    logger.info(
        f"Fetched {len(constituents)} {spec.display_name} constituents from Wikipedia"
    )
    return constituents


def fetch_index_tickers_from_wikipedia(
    spec: "UniverseSpec",
    timeout: int = 10,
    headers: dict | None = None,
) -> list[str]:
    """Fetch raw constituent tickers for any universe described by a UniverseSpec.

    This is the ticker-only compatibility wrapper around
    fetch_index_constituents_from_wikipedia().
    """
    constituents = fetch_index_constituents_from_wikipedia(
        spec, timeout=timeout, headers=headers
    )
    tickers = constituents["source_symbol"].dropna().astype(str).tolist()
    logger.info(f"Fetched {len(tickers)} {spec.display_name} tickers from Wikipedia")
    return tickers


# ---------------------------------------------------------------------------
# Back-compat aliases (pre-registry names)
# ---------------------------------------------------------------------------

def fetch_sp500_tickers_from_wikipedia(
    timeout: int = 10,
    headers: dict | None = None,
) -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia (back-compat wrapper).

    Delegates to fetch_index_tickers_from_wikipedia with the SP500 spec.
    """
    from ..universes import UNIVERSES
    return fetch_index_tickers_from_wikipedia(UNIVERSES["SP500"], timeout=timeout, headers=headers)


def get_sp500_tickers_wikipedia() -> list[str]:
    """Alias used by WikipediaTickerAdapter (back-compat)."""
    return fetch_sp500_tickers_from_wikipedia()
