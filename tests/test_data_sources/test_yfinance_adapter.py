from unittest.mock import patch

import pandas as pd

from clenow_momentum.data.sources.yfinance_adapter import YFinanceMarketDataAdapter


def make_price_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Open": [100.0, 100.5],
            "Volume": [1_000_000, 1_100_000],
        },
        index=dates,
    )


def test_index_data_uses_raw_close_for_broker_compatible_technicals():
    with patch(
        "clenow_momentum.data.sources.yfinance_adapter.yf.download",
        return_value=make_price_data(),
    ) as mock_download:
        adapter = YFinanceMarketDataAdapter()

        adapter.get_index_data("IWB", period="1y")

    assert mock_download.call_args.kwargs["auto_adjust"] is False


def test_stock_data_keeps_adjusted_prices_for_momentum_inputs():
    with patch(
        "clenow_momentum.data.sources.yfinance_adapter.yf.download",
        return_value=make_price_data(),
    ) as mock_download:
        adapter = YFinanceMarketDataAdapter()

        adapter.get_stock_data(["AAPL"], period="1y")

    assert mock_download.call_args.kwargs["auto_adjust"] is True
