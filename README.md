# Option Pricer

A comprehensive, high-performance option pricing and risk management library combining C++ computational engine with Python interface. Features real-time market data integration, multiple volatility models, and interactive visualization dashboards.

## Features

### Pricing Models
- **Black-Scholes**: Analytical pricing for European options
- **Binomial Trees**: Both European and American option pricing
- **Monte Carlo**: American and Asian options with parallel execution
- **Heston Model**: Stochastic volatility pricing
- **SABR Model**: Stochastic Alpha Beta Rho with calibration
- **Local Volatility**: Dupire local volatility with FDM
- **Jump Diffusion**: Merton jump diffusion model

### Greeks Calculation
- **First-order Greeks**: Delta, Gamma, Vega, Theta, Rho
- **Second-order Greeks**: Vanna, Charm, Vomma, Veta
- Available for all major pricing models
- Model-specific Greeks for SABR (including Volga)

### Real-Time Market Data
- **MarketDataService**: Multi-threaded data fetching from yfinance
- Tracks spot prices, option chains, and historical data
- Priority-based update scheduling
- Thread-safe file persistence with atomic writes
- Supports 20+ symbols including SPY, QQQ, AAPL, TSLA, etc.

### Model Calibration
- **CalibrationService**: Real-time multi-model calibration
- Calibrates SABR, Heston, Merton, and Local Vol models
- Processes multiple expiries per symbol (up to 8)
- Calculates implied volatilities from market prices
- Exports calibrated parameters and model comparisons

### Interactive Visualization
- **Dash/Plotly Dashboard**: Real-time options analytics
- Volatility smile and surface visualization
- 3D implied volatility surfaces
- Model vs market price comparisons
- Greeks heatmaps
- Term structure analysis

### Performance
- C++ core (C++11) for computational efficiency
- Vectorized NumPy operations
- Parallel Monte Carlo simulations
- Multi-threaded market data updates
- Optimized calibration algorithms

## Installation

### Prerequisites
- Python 3.9+
- C++11 compatible compiler (GCC, Clang, or MSVC)
- CMake 3.12+ (optional, for manual builds)

### Quick Install

```bash
# Clone repository
git clone https://github.com/tommyghaly/option-pricer.git
cd option-pricer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build C++ extensions
python setup.py build_ext --inplace
```

For detailed installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md)

## Quick Start

### Basic Option Pricing

```python
import option_pricer as op

# Price a European call option using Black-Scholes
S = 100.0    # Spot price
K = 105.0    # Strike price
r = 0.05     # Risk-free rate
q = 0.02     # Dividend yield
sigma = 0.2  # Volatility
T = 0.25     # Time to maturity (years)

call_price = op.black_scholes(S, K, r, q, sigma, T, is_call=True)
put_price = op.black_scholes(S, K, r, q, sigma, T, is_call=False)

print(f"Call Price: ${call_price:.2f}")
print(f"Put Price: ${put_price:.2f}")
```

### Calculate Greeks

```python
# First-order Greeks
delta = op.delta(S, K, r, q, sigma, T, is_call=True)
gamma = op.gamma(S, K, r, q, sigma, T, is_call=True)
vega = op.vega(S, K, r, q, sigma, T, is_call=True)
theta = op.theta(S, K, r, q, sigma, T, is_call=True)
rho = op.rho(S, K, r, q, sigma, T, is_call=True)

# Second-order Greeks
vanna = op.vanna(S, K, r, q, sigma, T, is_call=True)
charm = op.charm(S, K, r, q, sigma, T, is_call=True)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega: {vega:.4f}")
```

### Advanced Models

```python
# Price with SABR model
F = S * np.exp((r - q) * T)  # Forward price
alpha = 0.3  # Volatility
beta = 0.5   # Beta parameter
rho = -0.3   # Correlation
nu = 0.4     # Vol of vol

sabr_iv = op.sabr_implied_vol(S, K, r, T, alpha, beta, rho, nu)
sabr_price = op.sabr_option(S, K, r, T, F, alpha, beta, rho, nu, is_call=True)

# Price with Heston model
v0 = 0.04      # Initial variance
kappa = 2.0    # Mean reversion speed
theta = 0.04   # Long-term variance
sigma_v = 0.3  # Vol of vol
rho_h = -0.7   # Correlation

heston_price = op.heston_model(S, K, r, q, T, kappa, theta, sigma_v, rho_h, v0, is_call=True)
```

### Real-Time Market Data

```python
from python.option_pricer.market_data import MarketDataService

# Initialize service
symbols = ['SPY', 'AAPL', 'TSLA']
config = {
    'save_to_file': True,
    'data_directory': 'market_data_realtime',
    'save_interval': 5
}

service = MarketDataService(symbols, config)
service.start()

# Get data
aapl_spot = service.get_spot_price('AAPL')
aapl_options = service.get_option_chain('AAPL')

print(f"AAPL Price: ${aapl_spot['price']:.2f}")
```

### Model Calibration

```python
from python.option_pricer.calibration import CalibrationService

# Initialize calibration service
symbols = ['SPY', 'QQQ', 'AAPL']
calibration_config = {
    'calibration_interval': 30,
    'max_calibrations': 50,
    'max_expiries_per_symbol': 8
}

calib_service = CalibrationService(symbols, calibration_config)
calib_service.start()

# Calibrated parameters are saved to ./calibration_data_realtime/calibrations.json
```

### Interactive Dashboard

```bash
# Start the visualization dashboard
python python/option_pricer/visualization.py

# Open browser to http://localhost:8050
```

The dashboard provides:
- Real-time volatility smile visualization
- 3D implied volatility surfaces
- Model vs market price comparisons
- Greeks heatmaps
- Term structure charts

## Project Structure

```
option_pricer/
├── cpp/                      # C++ source code
│   ├── core/                 # Core pricing algorithms
│   ├── models/               # Advanced models (Heston, SABR, etc.)
│   ├── include/              # Header files
│   └── bindings/             # Python bindings
├── python/
│   ├── option_pricer/
│   │   ├── __init__.py       # Python API
│   │   ├── market_data.py    # Market data service
│   │   ├── calibration.py    # Model calibration
│   │   └── visualization.py  # Dash dashboard
│   └── tests/                # Test suite
├── docs/                     # Documentation
├── benchmarks/               # Performance benchmarks
└── setup.py                  # Build configuration
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Usage Guide](docs/USAGE_GUIDE.md) - Examples and tutorials
- [Market Data](docs/MARKET_DATA.md) - MarketDataService documentation
- [Calibration](docs/CALIBRATION.md) - CalibrationService guide
- [Visualization](docs/VISUALIZATION.md) - Dashboard usage

## Testing

```bash
# Run all tests
pytest python/tests/ -v

# Run specific test file
pytest python/tests/test_black_scholes.py -v

# Run with coverage
pytest python/tests/ --cov=option_pricer --cov-report=html
```

## Performance

The library is optimized for performance:
- C++ core provides 10-100x speedup over pure Python
- Parallel Monte Carlo simulations scale with CPU cores
- Efficient Greeks calculation using analytical formulas
- Real-time market data with multi-threaded updates

Benchmark results available in [benchmarks/](benchmarks/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Author

Tommy Ghaly

## Acknowledgments

- Built with pybind11 for seamless C++/Python integration
- Market data provided by yfinance
- Visualization powered by Plotly and Dash
