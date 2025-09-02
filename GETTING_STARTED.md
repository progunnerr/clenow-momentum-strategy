# Getting Started - Clenow Momentum Strategy

## Quick Start (5 minutes)

### 1. Run the Momentum Analysis
```bash
# Main analysis script
uv run python scripts/run_analysis.py

# Or use the main entry point
uv run python main.py

# Run with automatic execution (skip order confirmations)
uv run python main.py --force
```

### 2. Run Tests to Verify Everything Works
```bash
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=src/clenow_momentum

# Run specific test file
uv run pytest tests/test_indicators/test_momentum.py -v
```

### 3. Check Code Quality
```bash
# Lint and format code
uvx ruff check .
uvx ruff format .
```

## What to Expect

### First Run Output
The analysis will:
1. ✅ Fetch ~500 S&P 500 tickers from Wikipedia
2. ✅ Download 6 months of stock data (uses first 50 stocks for testing)
3. ✅ Calculate momentum scores for each stock
4. ✅ Display top 20% ranked by momentum

### Sample Output
```
============================================================
CLENOW MOMENTUM STRATEGY ANALYSIS  
============================================================
Analysis Date: 2024-08-31 10:30:15 UTC

2024-08-31 10:30:15 | SUCCESS  | fetcher:get_sp500_tickers:35 - Successfully fetched 503 S&P 500 tickers
Step 1: Fetching S&P 500 tickers...
✅ Successfully fetched 503 tickers
Sample tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'NVDA', 'UNH', 'JNJ']

2024-08-31 10:30:16 | INFO     | fetcher:get_stock_data:51 - Fetching stock data for 50 tickers (period: 6mo)

Step 2: Fetching stock data (6 months)...
⏳ This may take a moment...
🧪 Using first 50 stocks for testing
✅ Stock data retrieved successfully
Data shape: (130, 250)
Date range: 2024-02-29 to 2024-08-30

Step 3: Calculating momentum scores...
✅ Calculated momentum for 47 stocks

Step 4: Selecting top 20% momentum stocks...
✅ Selected 9 top momentum stocks

============================================================
TOP MOMENTUM STOCKS
============================================================
╒════════╤══════════╤═══════════╤═════════════╤═══════╤═════════╤═════════════╕
│   Rank │ Ticker   │ Momentum  │ Ann. Slope  │    R² │ Price   │ 90d Return  │
╞════════╪══════════╪═══════════╪═════════════╪═══════╪═════════╪═════════════╡
│      1 │ NVDA     │     0.823 │       0.892 │ 0.923 │ $125.34 │      +45.2% │
│      2 │ AMD      │     0.671 │       0.768 │ 0.874 │ $156.78 │      +38.1% │
│      3 │ TSLA     │     0.634 │       0.756 │ 0.838 │ $245.67 │      +35.4% │
╘════════╧══════════╧═══════════╧═════════════╧═══════╧═════════╧═════════════╛

SUMMARY STATISTICS
------------------------------
Average Momentum Score: 0.542
Average R-squared: 0.812
Average 90-day Return: 28.4%

✅ Analysis complete!
```

## Trading with Order Confirmations (NEW)

When IBKR trading is enabled and it's a rebalancing day, you'll see interactive order confirmations:

### Order Confirmation Display
```
============================================================
📋 ORDER CONFIRMATION REQUIRED
============================================================

📈 BUY ORDER
Symbol: NVDA
Shares: 150
Current Price: $485.50
Total Value: $72,825.00

📊 REASON:
  • Momentum score: 2.145
  • Rank #1
  • New position entry based on strong momentum signal

📈 TRADE METRICS:
  • Momentum Rank: #1
  • Momentum Score: 2.145
  • R-squared: 0.923
  • Position Size: 7.3% of portfolio

⚠️  This will buy 150 shares of NVDA
    Impact: $72,825.00

Options:
  [y] Execute this order
  [n] Skip this order
  [a] Approve all remaining orders
  [q] Cancel all orders and quit

Your choice: y
✅ Order confirmed
```

### Trading Options
- **Interactive Mode** (default): Review each order individually
- **Force Mode** (`--force`): Skip all confirmations for automated execution
- **Approve All**: Press 'a' during confirmation to approve remaining orders
- **Safety Exit**: Press 'q' to cancel all orders and exit safely

## Understanding the Results

### Key Metrics Explained

| Metric | What It Means | Good Values |
|--------|---------------|-------------|
| **Momentum** | Slope × R² (trend strength × consistency) | > 0.5 |
| **Ann. Slope** | Annualized price growth rate | > 0.3 |
| **R²** | How consistent the trend is (0-1) | > 0.7 |
| **90d Return** | Actual return over 90 days | Positive |

### Reading the Rankings
- **Rank 1** = Strongest momentum (best trend + consistency)
- **High R²** = Very consistent trend (not choppy)
- **Positive Slope** = Stock is trending upward
- **High Momentum** = Strong upward trend that's consistent

## Logging Features

### Log Files
The application now creates detailed log files in the `logs/` directory:
- **Console logs**: Colored output with timestamps and levels
- **File logs**: Detailed logs saved to `logs/momentum_YYYY-MM-DD.log`
- **Automatic rotation**: New log file each day, keeps 7 days, compresses old logs

### Log Levels
- **SUCCESS**: Green - Operations completed successfully
- **INFO**: Blue - General information and progress
- **WARNING**: Yellow - Issues that don't stop execution
- **ERROR**: Red - Failures that prevent operation

### Viewing Logs
```bash
# View today's log file
cat logs/momentum_$(date +%Y-%m-%d).log

# Follow live logs (if supported)
tail -f logs/momentum_$(date +%Y-%m-%d).log
```

## Troubleshooting

### Common Issues

#### 1. Wikipedia 403 Forbidden Error
If you see a 403 error from Wikipedia, don't worry! The application automatically uses yfinance as a fallback:

```
2025-08-31 09:50:58 | WARNING  | Failed to fetch S&P 500 tickers from Wikipedia: 403 Client Error: Forbidden
2025-08-31 09:50:58 | INFO     | Trying yfinance as fallback method...
2025-08-31 09:50:58 | INFO     | Attempting to fetch S&P 500 tickers via yfinance...
2025-08-31 09:50:59 | SUCCESS  | Verified 45 major S&P 500 tickers
```

**This is completely normal!** The fallback system:
1. **First tries**: SPY ETF holdings via yfinance
2. **Then tries**: Verified list of major S&P 500 companies
3. **Ensures quality**: Only includes tradeable, liquid stocks

#### 2. "Could not retrieve stock data"
```bash
# yfinance API might be rate-limited, try again in a few minutes
# Or check if you can access Yahoo Finance in your browser
```

#### 3. Import Errors
```bash
# Make sure dependencies are installed
uv sync --dev

# Check if you're in the right directory
pwd  # Should show: .../clenow_momentum_project
```

#### 4. Test Failures
```bash
# Run tests in verbose mode to see details
uv run pytest -v

# Run a specific failing test
uv run pytest tests/test_indicators/test_momentum.py::TestMomentumScore::test_strong_momentum -v
```

### Performance Notes
- **First run**: Takes 1-2 minutes (downloading data)
- **Subsequent runs**: Faster (but still downloads fresh data)
- **50 stocks**: Used for testing (full universe would be slower)

## Next Steps After Testing

### Phase 1 Success Checklist
- [ ] Analysis runs without errors
- [ ] Shows momentum rankings table
- [ ] All tests pass (`uv run pytest`)
- [ ] Code quality checks pass (`uvx ruff check .`)

### Project Complete - All Phases Implemented!
The Clenow Momentum Strategy now includes:
1. ✅ **Phase 1**: Momentum calculation and ranking
2. ✅ **Phase 2**: Trading filters (MA, gap detection, market regime)
3. ✅ **Phase 3**: ATR-based position sizing
4. ✅ **Phase 4**: Trading schedule and rebalancing
5. ✅ **Phase 5**: Full IBKR integration with live trading
6. ✅ **NEW**: Interactive order confirmation system

## Quick Commands Reference

```bash
# Development
uv run python main.py                    # Run analysis (with order confirmations)
uv run python main.py --force            # Run analysis (skip confirmations)
uv run pytest                           # Run tests  
uvx ruff check .                         # Lint code
uvx ruff format .                        # Format code

# IBKR Trading
uv run python scripts/ibkr_trading.py test   # Test IBKR connection
uv run python scripts/ibkr_trading.py sync   # Sync portfolio
uv run python scripts/ibkr_trading.py status # Check status

# Testing specific components
uv run pytest tests/test_data/ -v       # Test data fetching
uv run pytest tests/test_indicators/ -v # Test momentum calculations
uv run pytest tests/test_trading/ -v    # Test IBKR integration
uv run pytest --cov=src/clenow_momentum # Coverage report

# Project info
uv run python -c "import src.clenow_momentum; print(src.clenow_momentum.__version__)"
```

---

## 🚀 Ready to Start?

Run your first analysis:
```bash
uv run python main.py
```

Let me know how it goes! 🎯