# Lumibot Signal Executor

*Since you are here, please leave a ⭐*

A production-ready Lumibot backtesting engine for executing pre-computed trading signals with proper risk management, multiple exit conditions, and comprehensive analytics.

## Why This Exists

Lumibot's documentation shows you how to "buy AAPL on day 1 and hold." But what about real strategies with:
- Pre-computed signals from technical indicators
- Dynamic stop losses and take profits
- Multiple exit conditions (signal-based, SL, TP)
- Proper position tracking and trade analytics
- Walk-forward optimization compatibility

This repository provides a **real-world implementation** that professionals can actually use.

## Features

✅ **Signal-Based Execution**: Execute pre-computed entry/exit signals  
✅ **Dynamic Risk Management**: Use calculated SL/TP values per trade  
✅ **Multiple Exit Conditions**: Signal exits, stop loss, take profit  
✅ **Position Tracking**: Track entry prices, position types, trade metadata  
✅ **Comprehensive Stats**: 15+ performance metrics via QuantStats  
✅ **Timezone Handling**: Proper UTC → America/New_York conversion  
✅ **Trade Storage**: Full trade history with entry/exit reasons  

## Installation
```bash
pip install lumibot quantstats pandas numpy
```

## Quick Start
```python
from lumibot_backtest import run_backtest
import pandas as pd

# Your data with OHLCV + signal columns
data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Required columns:
# - Open, High, Low, Close, Volume (OHLCV)
# - long_entries, long_exits (boolean signals)
# - short_entries, short_exits (boolean signals)
# - long_sl, long_tp (float, as decimals e.g. 0.05 = 5%)
# - short_sl, short_tp (float)

# Run backtest
trades, stats = run_backtest(
    signals_df=data,
    symbol="EURUSD",
    start_date=data.index[0],
    end_date=data.index[-1],
    init_cash=10000,
    leverage=10,
    fees=1,
    freq="1m"
)

# View results
print(f"Total Trades: {stats['Total Trades']}")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"Total Return: {stats['Total Return [%]']:.2f}%")
print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
```

## Data Format

Your `signals_df` must include:

### Required OHLCV Columns
- `Open`, `High`, `Low`, `Close`, `Volume`

### Required Signal Columns
- `long_entries` (bool): True when long entry conditions met
- `long_exits` (bool): True when long exit conditions met
- `short_entries` (bool): True when short entry conditions met
- `short_exits` (bool): True when short exit conditions met

### Required Risk Columns
- `long_sl` (float): Stop loss as decimal (e.g., 0.05 = 5%)
- `long_tp` (float): Take profit as decimal (e.g., 0.0215 = 2.15%)
- `short_sl` (float): Stop loss for shorts
- `short_tp` (float): Take profit for shorts

### Index
- Must be `DatetimeIndex` (timezone-aware or naive, will be converted)

## How It Works

### Execution Flow

1. **Initialization**: Strategy initializes with your parameters
2. **Time Loop**: Lumibot iterates through every minute (or your `sleeptime`)
3. **Signal Check**: At each timestamp, checks your pre-computed signals
4. **Entry**: If `long_entries=True` and no position, enters long
5. **Exit**: Checks (in order):
   - Signal-based exit (`long_exits=True`)
   - Stop loss hit (`price <= stop_loss_price`)
   - Take profit hit (`price >= take_profit_price`)
6. **Trade Storage**: Saves complete trade metadata
7. **Statistics**: Calculates 15+ performance metrics

### Key Insights

⚠️ **Orders fill at the CLOSE of the current bar** (not next bar's open)  
⚠️ **Timezone must be America/New_York** (handled automatically)  
⚠️ **Signals must be reindexed** if generated on different timeframe  
⚠️ **Exit priority**: Signal exits → Stop loss → Take profit  

## Statistics Provided

The backtest returns comprehensive statistics:

**Returns & Risk:**
- Total Return %
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio

**Trade Analysis:**
- Total Trades
- Win Rate %
- Best/Worst Trade %
- Avg Winning/Losing Trade %
- Profit Factor
- Expectancy

**Drawdown:**
- Max Drawdown %
- Max Drawdown Duration

**Position Metrics:**
- Position Coverage %
- Max Gross Exposure %
- Avg Winning/Losing Trade Duration

## Example: Multi-Timeframe Strategy
```python
# Generate signals on 5-minute data
signal_data_5m = resample_to_5min(data_1m)
check_data_5h = resample_to_5hour(data_1m)

# Calculate indicators
signal_data_5m['EMA_8'] = signal_data_5m['Close'].ewm(span=8).mean()
signal_data_5m['VWMA_20'] = calculate_vwma(signal_data_5m, 20)

# Generate entry signals (with 5H trend filter)
signal_data_5m['long_entries'] = (
    (signal_data_5m['EMA_8'].shift(1) < signal_data_5m['VWMA_20'].shift(1)) &
    (signal_data_5m['EMA_8'] > signal_data_5m['VWMA_20']) &
    (check_data_5h['Close'].reindex(signal_data_5m.index, method='ffill') > 
     check_data_5h['EMA_20'].reindex(signal_data_5m.index, method='ffill'))
)

# Reindex signals to 1-minute data
data_1m['long_entries'] = signal_data_5m['long_entries'].reindex(
    data_1m.index, 
    fill_value=False
)

# Run backtest on 1-minute data
trades, stats = run_backtest(signals_df=data_1m, ...)
```

## Common Pitfalls

### 1. Index Misalignment
**Problem:** Signals on 5-min data, execution on 1-min data  
**Solution:** Reindex signals to match execution timeframe

### 2. Timezone Issues
**Problem:** Data in UTC, Lumibot expects America/New_York  
**Solution:** Code handles this automatically

### 3. All False Signals
**Problem:** No trades executed, all signals are False  
**Solution:** Check if 5H trend filter was reindexed properly

### 4. Lookahead Bias
**Problem:** Using current bar data to generate entry signal  
**Solution:** Check setup conditions on `.shift(1)` (prior bar)

## File Structure
```
lumibot_backtest.py       # Main backtesting engine
README.md                 # This file
examples/
  ├── basic_strategy.py   # Simple example
  └── multi_timeframe.py  # Multi-timeframe example
```

## Performance Notes

- **Sleeptime**: Set to `"1M"` for minute-by-minute execution
- **Leverage**: Applies to position sizing calculation
- **Fees**: Applied on both entry and exit
- **Cash Usage**: Uses 95% of available cash (configurable via `CASH_USAGE`)

## Differences from Lumibot Docs

| Feature | Lumibot Docs | This Implementation |
|---------|--------------|---------------------|
| Signal Generation | Inside `on_trading_iteration()` | Pre-computed |
| Exit Conditions | Single condition | Multiple (signal, SL, TP) |
| Position Tracking | Minimal | Full metadata |
| Statistics | Basic | 15+ metrics |
| Timeframe Handling | Single | Multi-timeframe ready |
| Trade Storage | None | Complete history |

## Related Article

📖 Read the full breakdown: [A Real-World Trade Strategy Lumibot Backtest](link-to-your-medium-article)

This article walks through **exactly** what happens line-by-line when Lumibot executes a trade—no toy examples, just production code.

## Contributing

Found a bug? Have a suggestion? Open an issue or PR.

## License

MIT

## Disclaimer

This is for educational and research purposes. Past performance does not guarantee future results. Always test thoroughly before live trading.

---

**Built because the Lumibot docs gave us "buy AAPL and hold" examples.**  
**We needed something that actually works in production.**
