# Market Data Service

Complete documentation for the `MarketDataService` class - real-time market data fetching and management.

## Overview

The `MarketDataService` provides multi-threaded, real-time market data fetching from yfinance. It manages spot prices, option chains, and historical data with intelligent update scheduling and persistent storage.

### Key Features

- **Multi-threaded Updates**: Separate threads for spot prices, options, and historical data
- **Priority-based Scheduling**: Near-term options updated more frequently
- **Atomic File Writes**: Thread-safe data persistence
- **Stale Data Detection**: Automatic cleanup of outdated information
- **Error Handling**: Robust retry logic with exponential backoff

---

## Quick Start

```python
from python.option_pricer.market_data import MarketDataService

# Initialize service
symbols = ['SPY', 'AAPL', 'TSLA']
service = MarketDataService(symbols)

# Start data fetching
service.start()

# Get data
aapl_spot = service.get_spot_price('AAPL')
aapl_options = service.get_option_chain('AAPL')

print(f"AAPL: ${aapl_spot['price']:.2f}")

# Stop when done
service.stop()
```

---

## Class Reference

### Constructor

```python
MarketDataService(symbols, config=None)
```

**Parameters:**
- `symbols` (list): List of ticker symbols to track
- `config` (dict, optional): Configuration dictionary

**Config Options:**
```python
config = {
    'save_to_file': True,              # Enable file persistence
    'data_directory': 'market_data',   # Data storage directory
    'save_interval': 5,                # Save interval in seconds
    'file_format': 'json'              # File format (json only)
}
```

---

## Methods

### start()

Starts the market data service and begins fetching data.

```python
success = service.start()
```

**Returns:**
- `bool`: True if service started successfully

**What it does:**
1. Performs initial data load for all symbols
2. Starts spot price update thread
3. Starts option chain update threads
4. Starts historical data thread
5. Starts file saving thread (if enabled)

---

### stop()

Gracefully stops the service and saves all data.

```python
service.stop()
```

**What it does:**
1. Sets running flag to False
2. Waits for threads to finish
3. Performs final data save
4. Cleans up resources

---

### get_spot_price(symbol)

Get current spot price data for a symbol.

```python
data = service.get_spot_price('AAPL')
```

**Returns:**
```python
{
    'price': 175.50,          # Current price
    'bid': 175.48,
    'ask': 175.52,
    'mid': 175.50,
    'volume': 52431000,
    'bid_size': 100,
    'ask_size': 200,
    'avg_volume': 54230000,
    'open': 174.20,
    'high': 176.10,
    'low': 174.00,
    'prev_close': 174.50,
    'change': 1.00,
    'change_pct': 0.573,
    'timestamp': 1698345678.123,
    'market_time': 1698345600.0,
    'quote_type': 'EQUITY',
    'halted': False,
    'currency': 'USD',
    'exchange': 'NMS'
}
```

---

### get_option_chain(symbol, expiry=None)

Get option chain data for a symbol.

```python
# Get all expiries
all_options = service.get_option_chain('AAPL')

# Get specific expiry
options_2024_01 = service.get_option_chain('AAPL', '2024-01-19')
```

**Returns:**
```python
{
    '2024-01-19': {
        'calls': [
            {
                'contractSymbol': 'AAPL240119C00170000',
                'strike': 170.0,
                'lastPrice': 8.50,
                'bid': 8.45,
                'ask': 8.55,
                'mid': 8.50,
                'volume': 1250,
                'openInterest': 5430,
                'impliedVolatility': 0.235,
                'type': 'call',
                'last_updated': 1698345678.123
            },
            ...
        ],
        'puts': [...],
        'last_updated': 1698345678.123
    },
    ...
}
```

---

### get_all_spot_prices()

Get spot prices for all tracked symbols.

```python
all_spots = service.get_all_spot_prices()
```

**Returns:**
```python
{
    'AAPL': {...},  # Same format as get_spot_price()
    'SPY': {...},
    'TSLA': {...}
}
```

---

### get_data_summary()

Get a summary of all available data.

```python
summary = service.get_data_summary()
```

**Returns:**
```python
{
    'spot_symbols': ['AAPL', 'SPY', 'TSLA'],
    'option_symbols': ['AAPL', 'SPY'],
    'historical_symbols': ['AAPL', 'SPY', 'TSLA'],
    'total_option_chains': 45,
    'last_updates': {
        'spot': {
            'AAPL': '2024-10-30T14:25:30',
            ...
        },
        'options': 38,
        'history': 3
    },
    'errors': 2,
    'running': True,
    'market_open': True
}
```

---

### is_market_open()

Check if the market is currently open.

```python
is_open = service.is_market_open()
```

**Returns:**
- `bool`: True if market is open

---

### load_saved_data()

Load previously saved data from files.

```python
data = service.load_saved_data()
```

**Returns:**
```python
{
    'spot': {...},      # Spot price data
    'options': {...},   # Option chain data
    'history': {...}    # Historical data
}
```

---

### get_data_files()

Get paths to all data files.

```python
files = service.get_data_files()
```

**Returns:**
```python
{
    'spot': './market_data/spot_data.json',
    'options': './market_data/option_chains.json',
    'history': './market_data/historical_data.json',
    'metadata': './market_data/metadata.json',
    'directory': './market_data'
}
```

---

## Update Frequencies

The service uses different update frequencies for different data types:

| Data Type | Default Interval | Notes |
|-----------|-----------------|-------|
| Spot Prices | 5 seconds | When market is open |
| ATM Options | 30 seconds | Near-term expiries |
| OTM Options | 5 minutes | Far-term expiries |
| Historical Data | 1 hour | Daily bars |

---

## Priority Symbols

Default priority symbols (updated more frequently):
- `SPY` - S&P 500 ETF
- `QQQ` - Nasdaq-100 ETF
- `AAPL` - Apple Inc.

---

## Complete Example

```python
from python.option_pricer.market_data import MarketDataService
import time

# Configure service
symbols = [
    'SPY', 'QQQ', 'IWM',  # ETFs
    'AAPL', 'MSFT', 'GOOGL'  # Stocks
]

config = {
    'save_to_file': True,
    'data_directory': 'my_market_data',
    'save_interval': 10
}

# Start service
service = MarketDataService(symbols, config)
print("Starting market data service...")

if service.start():
    print("Service started successfully!")

    try:
        # Let it collect initial data
        time.sleep(10)

        # Get summary
        summary = service.get_data_summary()
        print(f"\nTracking {len(summary['spot_symbols'])} symbols")
        print(f"Market is {'open' if summary['market_open'] else 'closed'}")

        # Get specific data
        spy_spot = service.get_spot_price('SPY')
        if spy_spot:
            print(f"\nSPY: ${spy_spot['price']:.2f}")
            print(f"  Volume: {spy_spot.get('volume', 0):,}")
            print(f"  Change: {spy_spot.get('change', 0):+.2f} ({spy_spot.get('change_pct', 0):+.2f}%)")

        # Get options
        spy_options = service.get_option_chain('SPY')
        if spy_options:
            print(f"\nSPY has {len(spy_options)} expiries available")

            # Show first expiry
            first_expiry = list(spy_options.keys())[0]
            first_chain = spy_options[first_expiry]
            print(f"  {first_expiry}: {len(first_chain['calls'])} calls, {len(first_chain['puts'])} puts")

        # Run for a while
        print("\nPress Ctrl+C to stop...\n")
        while True:
            time.sleep(30)

            # Show periodic updates
            summary = service.get_data_summary()
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Tracked: {len(summary['spot_symbols'])} symbols, "
                  f"Errors: {summary['errors']}")

    except KeyboardInterrupt:
        print("\n\nStopping service...")
        service.stop()
        print("Service stopped successfully")
else:
    print("Failed to start service!")
```

---

## Threading Architecture

```
MarketDataService
├── Main Thread (control)
├── SpotPriceUpdater Thread
│   └── Updates every 5 seconds (market hours)
├── OptionChainUpdater Threads (10 workers)
│   ├── Worker 1
│   ├── Worker 2
│   └── ...
├── HistoricalDataUpdater Thread
│   └── Updates every hour
└── FileSaver Thread
    └── Saves every 5 seconds
```

---

## File Format

Data is saved in JSON format with the following structure:

**spot_data.json:**
```json
{
  "AAPL": {
    "price": 175.50,
    "bid": 175.48,
    ...
  }
}
```

**option_chains.json:**
```json
{
  "AAPL": {
    "2024-01-19": {
      "calls": [...],
      "puts": [...],
      "last_updated": 1698345678.123
    }
  }
}
```

**metadata.json:**
```json
{
  "last_save_time": "2024-10-30T14:25:30",
  "symbols_tracked": ["AAPL", "SPY"],
  "service_running": true,
  "data_counts": {...},
  "market_open": true
}
```

---

## Error Handling

The service implements robust error handling:

1. **Retryable Errors**: ConnectionError, TimeoutError, HTTP 5xx
   - Exponential backoff (1s, 2s, 4s)
   - Max 3 retries per operation

2. **Non-retryable Errors**: Invalid symbol, API errors
   - Logged and skipped
   - Service continues with other symbols

3. **Thread Safety**: All data access protected by locks
   - `data_lock`: Protects in-memory data
   - `file_lock`: Coordinates file writes

---

## Performance Tips

1. **Limit Symbols**: Track only symbols you need (< 30 recommended)
2. **Adjust Save Interval**: Increase to 10-30 seconds if I/O is slow
3. **Priority Symbols**: Set priority for symbols needing frequent updates
4. **Market Hours**: Service automatically sleeps when market is closed

---

## See Also

- [API Reference](API_REFERENCE.md)
- [Calibration Service](CALIBRATION.md)
- [Usage Guide](USAGE_GUIDE.md)
