import importlib.util
from pathlib import Path


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
