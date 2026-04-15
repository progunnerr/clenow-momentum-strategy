"""Tests for yfinance source helpers."""

from src.clenow_momentum.data.sources.yfinance_source import convert_ticker_for_yfinance


def test_convert_ticker_for_yfinance_converts_share_class_dots():
    assert convert_ticker_for_yfinance("BRK.B") == "BRK-B"
    assert convert_ticker_for_yfinance("BF.B") == "BF-B"


def test_convert_ticker_for_yfinance_preserves_tsx_suffix():
    assert convert_ticker_for_yfinance("SHOP.TO") == "SHOP.TO"
    assert convert_ticker_for_yfinance("BIP.UN.TO") == "BIP-UN.TO"
