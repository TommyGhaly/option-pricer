# Calibration Service

Complete documentation for the `CalibrationService` class - multi-model options calibration.

## Overview

The `CalibrationService` automatically calibrates multiple volatility models (SABR, Heston, Merton, Local Vol) to real-time market data. It calculates implied volatilities, calibrates model parameters, and exports results for analysis and visualization.

### Supported Models

1. **SABR**: Stochastic Alpha Beta Rho model with Hagan approximation
2. **Heston**: Stochastic volatility model with mean reversion
3. **Merton**: Jump diffusion model
4. **LocalVol**: Dupire local volatility surface

---

## Quick Start

```python
from python.option_pricer.calibration import CalibrationService

# Initialize service
symbols = ['SPY', 'AAPL', 'TSLA']
config = {
    'calibration_interval': 30,
    'max_calibrations': 50,
    'max_expiries_per_symbol': 8
}

service = CalibrationService(symbols, config)

# Start calibration (runs in background)
service.start()

# Results are automatically saved to ./calibration_data_realtime/calibrations.json
```

---

## Class Reference

### Constructor

```python
CalibrationService(symbols=None, config=None)
```

**Parameters:**
- `symbols` (list, optional): List of symbols to calibrate. Defaults to all predefined tickers.
- `config` (dict, optional): Configuration dictionary

**Config Options:**
```python
config = {
    'calibration_interval': 30,        # Calibration cycle interval (seconds)
    'max_calibrations': 50,            # Max calibrations per cycle
    'max_expiries_per_symbol': 8       # Max expiries to process per symbol
}
```

**Default Symbols:**
```python
# ETFs
'SPY', 'QQQ', 'IWM', 'DIA'

# Tech Stocks
'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMD'

# Financials
'JPM', 'BAC', 'GS'

# Volatility Products
'UVXY', 'SVXY'

# Sector ETFs
'XLF', 'XLE', 'XLK', 'GLD'
```

---

## Methods

### start()

Starts the calibration service.

```python
service.start()
```

**What it does:**
1. Loads market data from MarketDataService files
2. Calculates initial implied volatilities
3. Runs initial calibration cycle
4. Starts background calibration thread

---

### stop()

Stops the calibration service and saves final results.

```python
service.stop()
```

---

## Output Format

Calibrations are saved to `./calibration_data_realtime/calibrations.json`:

```json
{
  "updated": "2024-10-30T14:30:00",
  "data": {
    "AAPL": {
      "2024-01-19": {
        "spot": 175.50,
        "timestamp": "2024-10-30T14:30:00",
        "models": {
          "SABR": {
            "params": {
              "alpha": 0.295,
              "beta": 0.5,
              "rho": -0.312,
              "nu": 0.428
            },
            "prices": {
              "170.0": {
                "call": {
                  "model": 8.52,
                  "market": 8.50,
                  "iv": 0.235
                },
                "put": {
                  "model": 2.98,
                  "market": 3.00,
                  "iv": 0.237
                }
              },
              ...
            }
          },
          "Heston": {...},
          "Merton": {...},
          "LocalVol": {...}
        }
      }
    }
  }
}
```

---

## Model Details

### SABR Model

**Parameters Calibrated:**
- `alpha`: Initial volatility level
- `rho`: Correlation between asset and volatility
- `nu`: Volatility of volatility
- `beta`: Fixed at 0.5 for equities

**Calibration Method:**
- Least-squares optimization
- Minimizes error between model IV and market IV
- Uses Hagan approximation formula

**Output:**
```python
{
    'params': {
        'alpha': 0.295,
        'beta': 0.5,
        'rho': -0.312,
        'nu': 0.428
    },
    'rmse': 0.0045,  # Root mean squared error
    'projected_prices': {...}
}
```

---

### Heston Model

**Parameters:**
- `v0`: Initial variance
- `theta`: Long-term variance
- `kappa`: Mean reversion speed (fixed at 2.0)
- `sigma`: Vol of vol (fixed at 0.3)
- `rho`: Correlation (fixed at -0.5)

**Calibration Method:**
- Simplified approach using ATM volatility
- Parameters estimated from market IV levels

---

### Merton Jump Diffusion

**Parameters:**
- `sigma`: Continuous volatility component
- `lambda`: Jump intensity (jumps per year)
- `mu_jump`: Mean jump size
- `sigma_jump`: Jump size volatility

**Calibration Method:**
- ATM volatility determines base sigma
- Jump parameters set to typical values
- Prices calculated via Poisson series expansion

---

### Local Volatility

**Parameters:**
- Local volatility surface (strike-dependent)
- Built using Dupire's formula

**Calibration Method:**
- Constructs local vol from market implied vols
- Uses linear interpolation for surface
- Strike-dependent volatility adjustments

---

## Complete Example

```python
from python.option_pricer.calibration import CalibrationService
import time
import json

# Configure service
symbols = ['SPY', 'AAPL', 'MSFT']

config = {
    'calibration_interval': 60,  # Calibrate every minute
    'max_calibrations': 30,
    'max_expiries_per_symbol': 5
}

# Start service
print("Starting calibration service...")
service = CalibrationService(symbols, config)
service.start()

try:
    # Let it run for a while
    print("Calibrating... Press Ctrl+C to stop\n")

    while True:
        time.sleep(60)

        # Load latest calibrations
        with open('./calibration_data_realtime/calibrations.json', 'r') as f:
            data = json.load(f)

        # Show summary
        print(f"[{time.strftime('%H:%M:%S')}] Calibration Summary:")
        for symbol in symbols:
            if symbol in data['data']:
                expiries = len(data['data'][symbol])
                print(f"  {symbol}: {expiries} expiries calibrated")

                # Show SABR params for first expiry
                first_expiry = list(data['data'][symbol].keys())[0]
                sabr = data['data'][symbol][first_expiry]['models'].get('SABR', {})
                if 'params' in sabr:
                    params = sabr['params']
                    print(f"    SABR: α={params['alpha']:.3f}, "
                          f"ρ={params['rho']:.3f}, ν={params['nu']:.2f}")

        print()

except KeyboardInterrupt:
    print("\nStopping calibration service...")
    service.stop()
    print("Calibration service stopped")
```

---

## Implied Volatility Calculation

The service calculates implied volatilities from market prices using Newton-Raphson method:

```python
# Pseudo-code
def calculate_implied_vol(market_price, S, K, r, T, is_call):
    sigma = 0.3  # Initial guess
    for iteration in range(50):
        model_price = black_scholes(S, K, r, T, sigma, is_call)
        vega = calculate_vega(S, K, r, T, sigma, is_call)

        diff = model_price - market_price
        if abs(diff) < 0.00001:
            return sigma  # Converged

        sigma = sigma - diff / vega  # Newton step
        sigma = max(0.01, min(sigma, 5.0))  # Bounds

    return None  # Failed to converge
```

**Features:**
- Max 50 iterations
- Convergence tolerance: $0.00001
- Bounds: 1% to 500% annualized volatility
- Returns None for deep ITM/OTM or invalid data

---

## Calibration Workflow

```
1. Load Market Data
   ├── Read spot_data.json
   ├── Read option_chains.json
   └── Validate data freshness

2. Calculate Implied Volatilities
   ├── For each symbol/expiry
   ├── Extract market prices (mid of bid/ask)
   ├── Calculate time to maturity
   └── Compute IV for each strike

3. Calibrate Models (per symbol/expiry)
   ├── SABR Calibration
   │   ├── Extract strikes and IVs
   │   ├── Optimize (alpha, rho, nu)
   │   └── Calculate RMSE
   ├── Heston Calibration
   │   ├── Estimate initial variance
   │   └── Compute model IVs
   ├── Merton Calibration
   │   └── Simulate jump component
   └── Local Vol Calibration
       └── Build local vol surface

4. Consolidate Results
   ├── Merge model prices
   ├── Calculate differences
   └── Compute error metrics

5. Save Results
   └── Write to calibrations.json
```

---

## Priority and Scheduling

**Expiry Priority:**
- Near-term expiries (< 7 days) processed first
- Up to `max_expiries_per_symbol` per symbol
- Expired expiries automatically skipped

**Symbol Processing:**
- All symbols processed each cycle
- No artificial limits on total calibrations
- Controlled by `max_expiries_per_symbol`

---

## Data Requirements

For successful calibration, each symbol/expiry needs:

**Minimum Requirements:**
- Valid spot price > 0
- At least 3 liquid strikes with market prices
- Time to maturity > 1 day
- Valid bid/ask spreads

**Optimal Data:**
- 10+ strikes spanning 80% to 120% moneyness
- Tight bid/ask spreads (< 5% of mid)
- Open interest > 10 contracts
- Volume > 0

---

## Performance Metrics

**Per Calibration Cycle:**
- Processing time: 5-30 seconds (depends on #symbols and expiries)
- Memory usage: ~100-500 MB
- CPU usage: ~50-80% of single core

**Typical Throughput:**
- 50 calibrations per minute
- 100+ symbols per hour
- 500+ expiry/model combinations per cycle

---

## Error Handling

**Automatic Skip Conditions:**
- Spot price unavailable or invalid
- Insufficient option data (< 3 strikes)
- Time to maturity <= 0 (expired)
- All market prices = 0

**Calibration Failures:**
- Logged but don't stop service
- Failed models excluded from output
- Service continues with other models/symbols

---

## File Output

**calibrations.json Structure:**

```
Size: ~50-500 KB per symbol (depends on #expiries and strikes)
Update: Every calibration cycle (default 30s)
Format: Pretty-printed JSON (2-space indent)
```

**Atomic Writes:**
- Uses temp file + rename for safety
- No partial writes or corruption
- Safe for concurrent reads

---

## Integration with Visualization

The calibrations.json file is automatically read by the visualization dashboard:

```bash
# Start calibration service (terminal 1)
python -c "from python.option_pricer.calibration import CalibrationService; \
           service = CalibrationService(['SPY', 'AAPL']); \
           service.start()"

# Start visualization (terminal 2)
python python/option_pricer/visualization.py
```

Dashboard will display:
- Real-time SABR parameters
- Model vs market price comparisons
- Implied volatility smiles
- Greeks from calibrated models

---

## Advanced Usage

### Custom Model Selection

Modify the calibration to use only specific models:

```python
# In calibration.py, modify _calibrate_all_models()
results = {}

# Only calibrate SABR and Heston
sabr = self._calibrate_sabr(S, T, r, q, ivs, market_prices)
if sabr:
    results['SABR'] = sabr

heston = self._calibrate_heston(S, T, r, q, ivs, market_prices)
if heston:
    results['Heston'] = heston

return results
```

### Custom Risk-Free Rates

```python
# In calibration.py constructor
self.risk_free_rate = {
    '1m': 0.045,
    '3m': 0.042,
    '1y': 0.040
}
```

### Export Calibrations to CSV

```python
import json
import pandas as pd

# Load calibrations
with open('./calibration_data_realtime/calibrations.json', 'r') as f:
    data = json.load(f)

# Extract SABR parameters
records = []
for symbol, expiries in data['data'].items():
    for expiry, cal_data in expiries.items():
        sabr = cal_data['models'].get('SABR', {}).get('params', {})
        if sabr:
            records.append({
                'symbol': symbol,
                'expiry': expiry,
                'spot': cal_data['spot'],
                'alpha': sabr.get('alpha'),
                'rho': sabr.get('rho'),
                'nu': sabr.get('nu'),
                'timestamp': cal_data['timestamp']
            })

df = pd.DataFrame(records)
df.to_csv('sabr_calibrations.csv', index=False)
print(f"Exported {len(df)} calibrations")
```

---

## See Also

- [Market Data Service](MARKET_DATA.md)
- [Visualization](VISUALIZATION.md)
- [API Reference](API_REFERENCE.md)
