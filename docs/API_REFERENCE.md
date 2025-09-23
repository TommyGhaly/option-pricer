# API Reference

## Python API

### Overview
The Python API provides a high-level interface for option pricing and risk management. It leverages the C++ core for computational efficiency while offering Python's ease of use.

### Modules
- **`option_pricer`**: Core module for pricing and Greeks calculation.
- **`visualization`**: Tools for plotting and analyzing results.
- **`calibration`**: Functions for model calibration.

### Key Functions
#### `calculate_price`
- **Description**: Computes the price of an option.
- **Parameters**:
  - `S` (float): Current price of the underlying asset.
  - `K` (float): Strike price.
  - `r` (float): Risk-free rate.
  - `T` (float): Time to maturity.
  - `sigma` (float): Volatility.
  - `is_call` (bool): `True` for call option, `False` for put option.
- **Returns**: Option price (float).

#### `calculate_greeks`
- **Description**: Computes the Greeks for an option.
- **Parameters**: Same as `calculate_price`.
- **Returns**: A dictionary containing Delta, Gamma, Vega, Theta, and Rho.

### Example Usage
```python
from option_pricer import calculate_price, calculate_greeks

# Price a European call option
price = calculate_price(S=100, K=105, r=0.05, T=0.25, sigma=0.2, is_call=True)
print(f"Option Price: {price}")

greeks = calculate_greeks(S=100, K=105, r=0.05, T=0.25, sigma=0.2, is_call=True)
print(f"Greeks: {greeks}")
```

## C++ Bindings

### Overview
The C++ core is exposed to Python using `pybind11`. It provides high-performance implementations of various option pricing models and risk metrics.

### Key Functions
#### `black_scholes`
- **Description**: Prices a European option using the Black-Scholes formula.
- **Parameters**:
  - `S`, `K`, `r`, `q`, `T`, `sigma`, `is_call` (same as Python API).
- **Returns**: Option price (double).

#### `heston_model`
- **Description**: Prices an option using the Heston stochastic volatility model.
- **Parameters**:
  - `S`, `K`, `r`, `q`, `T` (same as above).
  - `v0`, `kappa`, `theta`, `sigma`, `rho`, `lambda` (model-specific parameters).
  - `is_call` (bool).
- **Returns**: Option price (double).

#### `monte_carlo_american_option`
- **Description**: Prices an American option using Monte Carlo simulation.
- **Parameters**:
  - `S`, `K`, `r`, `q`, `sigma`, `T` (same as above).
  - `steps` (int): Number of time steps.
  - `num_simulations` (int): Number of simulations.
  - `is_call` (bool).
- **Returns**: Option price (double).

### Example Usage
```cpp
#include <iostream>
#include "option_pricer.h"

int main() {
    double price = black_scholes(100, 105, 0.05, 0.02, 0.25, 0.2, true);
    std::cout << "Option Price: " << price << std::endl;
    return 0;
}
```
