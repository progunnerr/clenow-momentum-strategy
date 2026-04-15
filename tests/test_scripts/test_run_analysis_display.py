import importlib.util
from pathlib import Path

import pandas as pd


def load_run_analysis_module():
    module_path = Path(__file__).parents[2] / "scripts" / "run_analysis.py"
    spec = importlib.util.spec_from_file_location("run_analysis", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_analysis = load_run_analysis_module()


def test_format_ma_stack_orders_mas_by_value():
    short_mas = {
        "10_ema": 673.59,
        "20_sma": 660.36,
        "50_sma": 672.95,
    }

    result = run_analysis.format_ma_stack(short_mas, long_term_ma=363.18)

    assert result == (
        "10 EMA ($673.59) > 50 SMA ($672.95) > "
        "20 SMA ($660.36) > 200 SMA ($363.18)"
    )


def test_format_ma_stack_returns_empty_string_when_no_mas_are_available():
    assert run_analysis.format_ma_stack({}, long_term_ma=None) == ""


def test_display_portfolio_table_includes_metadata_and_diagnostic_headers(capsys):
    """Portfolio table should show metadata and diagnostic context columns."""
    portfolio_df = pd.DataFrame(
        {
            "portfolio_rank": [1],
            "ticker": ["BRK-B"],
            "company_name": ["Berkshire Hathaway Inc."],
            "sector": ["Financials"],
            "momentum_score": [1.234],
            "r_squared": [0.876],
            "period_return_pct": [14.5],
            "price_vs_ma": [0.123],
            "current_price": [400.0],
            "atr": [8.5],
            "shares": [6],
            "investment": [2400.0],
            "position_pct": [0.048],
            "actual_atr_impact": [51.0],
            "stop_loss_risk": [153.0],
        }
    )
    config = {
        "strategy_allocation": 50000,
        "risk_per_trade": 0.001,
        "stop_loss_multiplier": 3.0,
        "ma_filter_period": 100,
    }

    run_analysis.display_portfolio_table(portfolio_df, config)

    output = capsys.readouterr().out
    assert "Company" in output
    assert "Sector" in output
    assert "R²" in output
    assert "90d Return %" in output
    assert "Price vs 100d MA %" in output
