# Option Pricer

A high-performance option pricing library combining C++ computational engine with Python interface.

## Features

- **Pricing Models**
  - Black-Scholes analytical pricing
  - Monte Carlo simulation
  - Binomial trees (European & American options)
  - Heston stochastic volatility
  - SABR model
  - Jump diffusion models

- **Greeks Calculation**
  - First-order: Delta, Gamma, Theta, Vega, Rho
  - Second-order: Vanna, Charm, Vomma, Veta, Vera
  - Portfolio-level aggregation

- **Performance**
  - C++ core for speed
  - Vectorized operations
  - Parallel Monte Carlo
  - GPU acceleration support

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/option-pricer.git
cd option-pricer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Build C++ extensions
python setup.py build_ext --inplace
```

## Quick Start

```python
from option_pricer import OptionPricer

# Initialize pricer
pricer = OptionPricer()

# Price a European call option
S = 100    # Current price
K = 105    # Strike price
r = 0.05   # Risk-free rate
T = 0.25   # Time to maturity
sigma = 0.2  # Volatility

price = pricer.calculate_price(S, K, r, T, sigma, is_call=True)
greeks = pricer.calculate_greeks(S, K, r, T, sigma, is_call=True)

print(f"Option Price: ${price:.2f}")
print(f"Delta: {greeks.delta:.4f}")
```

## Documentation

Full documentation available at: [docs/](docs/)

## Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_black_scholes.py -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.