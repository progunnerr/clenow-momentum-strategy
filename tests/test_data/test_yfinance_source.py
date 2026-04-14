import pandas as pd

from clenow_momentum.data.sources.yfinance_source import (
    download_index_data,
    download_stock_data,
)


class FakeYFinance:
    def __init__(self):
        self.call_kwargs = None

    def download(self, *_args, **kwargs):
        self.call_kwargs = kwargs
        return pd.DataFrame({"Close": [100.0]})


def test_download_index_data_uses_raw_close_for_broker_compatible_technicals():
    fake_yf = FakeYFinance()

    download_index_data("IWB", yf=fake_yf)

    assert fake_yf.call_kwargs["auto_adjust"] is False


def test_download_stock_data_keeps_adjusted_prices_for_momentum_inputs():
    fake_yf = FakeYFinance()

    download_stock_data(["AAPL"], yf=fake_yf)

    assert fake_yf.call_kwargs["auto_adjust"] is True
