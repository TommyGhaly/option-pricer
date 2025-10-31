# API Reference

Complete API documentation for the Option Pricer library.

## Table of Contents

- [Core Pricing Functions](#core-pricing-functions)
- [Greeks Functions](#greeks-functions)
- [Advanced Models](#advanced-models)
- [SABR Model](#sabr-model)
- [Python Services](#python-services)

---

## Core Pricing Functions

### black_scholes

Calculate option price using the Black-Scholes model (European options).

```python
option_pricer.black_scholes(S, K, r, q, sigma, T, is_call)
```

**Parameters:**
- `S` (float): Spot price of the underlying asset
- `K` (float): Strike price
- `r` (float): Risk-free interest rate (annualized)
- `q` (float): Dividend yield (annualized)
- `sigma` (float): Volatility (annualized)
- `T` (float): Time to maturity (in years)
- `is_call` (bool): True for call option, False for put option

**Returns:**
- `float`: Option price

**Example:**
```python
import option_pricer as op

price = op.black_scholes(
    S=100.0,      # Spot price
    K=105.0,      # Strike
    r=0.05,       # 5% risk-free rate
    q=0.02,       # 2% dividend yield
    sigma=0.2,    # 20% volatility
    T=0.25,       # 3 months
    is_call=True  # Call option
)
print(f"Call Price: ${price:.2f}")
```

---

### binomial_tree

Calculate option price using the binomial tree model.

```python
option_pricer.binomial_tree(S, K, r, q, sigma, T, steps, is_call, american=True)
```

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `q` (float): Dividend yield
- `sigma` (float): Volatility
- `T` (float): Time to maturity
- `steps` (int): Number of time steps in the tree
- `is_call` (bool): True for call, False for put
- `american` (bool, optional): True for American option, False for European (default: True)

**Returns:**
- `float`: Option price

**Example:**
```python
# Price an American put option
price = op.binomial_tree(
    S=100.0,
    K=105.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    T=0.25,
    steps=100,
    is_call=False,
    american=True
)
```

---

### monte_carlo

Calculate option price using Monte Carlo simulation.

```python
option_pricer.monte_carlo(S, K, r, q, sigma, T, simulations, is_call, american=True)
```

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `q` (float): Dividend yield
- `sigma` (float): Volatility
- `T` (float): Time to maturity
- `simulations` (int): Number of simulation paths
- `is_call` (bool): True for call, False for put
- `american` (bool, optional): True for American, False for Asian (default: True)

**Returns:**
- `float`: Option price

**Example:**
```python
# Price an American call with 10,000 simulations
price = op.monte_carlo(
    S=100.0,
    K=105.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    T=0.25,
    simulations=10000,
    is_call=True,
    american=True
)
```

---

## Greeks Functions

### delta

Calculate option delta (∂V/∂S).

```python
option_pricer.delta(S, K, r, q, sigma, T, is_call)
```

**Parameters:** Same as `black_scholes`

**Returns:**
- `float`: Delta value (typically between 0 and 1 for calls, -1 and 0 for puts)

**Example:**
```python
d = op.delta(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
print(f"Delta: {d:.4f}")  # e.g., 0.4523
```

---

### gamma

Calculate option gamma (∂²V/∂S²).

```python
option_pricer.gamma(S, K, r, q, sigma, T, is_call)
```

**Parameters:** Same as `black_scholes`

**Returns:**
- `float`: Gamma value (same for calls and puts)

**Example:**
```python
g = op.gamma(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
print(f"Gamma: {g:.4f}")
```

---

### vega

Calculate option vega (∂V/∂σ).

```python
option_pricer.vega(S, K, r, q, sigma, T, is_call)
```

**Parameters:** Same as `black_scholes`

**Returns:**
- `float`: Vega value (sensitivity to 1% change in volatility)

**Example:**
```python
v = op.vega(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
print(f"Vega: {v:.2f}")  # e.g., 18.45
```

---

### theta

Calculate option theta (∂V/∂t).

```python
option_pricer.theta(S, K, r, q, sigma, T, is_call)
```

**Parameters:** Same as `black_scholes`

**Returns:**
- `float`: Theta value (time decay per day)

**Example:**
```python
t = op.theta(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
print(f"Theta: {t:.4f}")  # Typically negative
```

---

### rho

Calculate option rho (∂V/∂r).

```python
option_pricer.rho(S, K, r, q, sigma, T, is_call)
```

**Parameters:** Same as `black_scholes`

**Returns:**
- `float`: Rho value (sensitivity to 1% change in interest rate)

**Example:**
```python
r_greek = op.rho(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
print(f"Rho: {r_greek:.4f}")
```

---

### Second-Order Greeks

#### vanna

Calculate option vanna (∂²V/∂S∂σ).

```python
option_pricer.vanna(S, K, r, q, sigma, T, is_call)
```

---

#### charm

Calculate option charm (∂²V/∂S∂t).

```python
option_pricer.charm(S, K, r, q, sigma, T, is_call)
```

---

#### vomma

Calculate option vomma (∂²V/∂σ²).

```python
option_pricer.vomma(S, K, r, q, sigma, T, is_call)
```

---

#### veta

Calculate option veta (∂²V/∂σ∂t).

```python
option_pricer.veta(S, K, r, q, sigma, T, is_call)
```

**Example:**
```python
# Calculate all second-order Greeks
vanna_val = op.vanna(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
charm_val = op.charm(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
vomma_val = op.vomma(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)
veta_val = op.veta(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, is_call=True)

print(f"Vanna: {vanna_val:.4f}")
print(f"Charm: {charm_val:.4f}")
print(f"Vomma: {vomma_val:.4f}")
print(f"Veta: {veta_val:.4f}")
```

---

## Advanced Models

### heston_model

Calculate option price using the Heston stochastic volatility model.

```python
option_pricer.heston_model(S, K, r, q, T, kappa, theta, sigma, rho, v0, is_call)
```

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `q` (float): Dividend yield
- `T` (float): Time to maturity
- `kappa` (float): Mean reversion speed
- `theta` (float): Long-term variance
- `sigma` (float): Volatility of volatility
- `rho` (float): Correlation between spot and variance (-1 to 1)
- `v0` (float): Initial variance
- `is_call` (bool): True for call, False for put

**Returns:**
- `float`: Option price

**Example:**
```python
price = op.heston_model(
    S=100.0,
    K=105.0,
    r=0.05,
    q=0.02,
    T=0.25,
    kappa=2.0,      # Mean reversion speed
    theta=0.04,     # Long-term variance
    sigma=0.3,      # Vol of vol
    rho=-0.7,       # Correlation
    v0=0.04,        # Initial variance
    is_call=True
)
```

---

### jump_diffusion

Calculate option price using the Merton jump diffusion model.

```python
option_pricer.jump_diffusion(S, K, r, q, sigma, T, lambda_, mu_j, sigma_j, simulations, is_call)
```

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `q` (float): Dividend yield
- `sigma` (float): Volatility (continuous component)
- `T` (float): Time to maturity
- `lambda_` (float): Jump intensity (jumps per year)
- `mu_j` (float): Mean jump size
- `sigma_j` (float): Jump size volatility
- `simulations` (int): Number of Monte Carlo simulations
- `is_call` (bool): True for call, False for put

**Returns:**
- `float`: Option price

**Example:**
```python
price = op.jump_diffusion(
    S=100.0,
    K=105.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    T=0.25,
    lambda_=0.1,    # 0.1 jumps per year
    mu_j=-0.10,     # -10% average jump
    sigma_j=0.15,   # 15% jump volatility
    simulations=10000,
    is_call=True
)
```

---

### local_volatility

Calculate option price using local volatility model with finite difference method (FDM).

```python
option_pricer.local_volatility(S, K, r, q, T, is_call, iv_surface, american=True)
```

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `q` (float): Dividend yield
- `T` (float): Time to maturity
- `is_call` (bool): True for call, False for put
- `iv_surface` (list): List of dicts with 'K', 'T', 'iv' keys defining the implied volatility surface
- `american` (bool, optional): True for American, False for European (default: True)

**Returns:**
- `float`: Option price

**Example:**
```python
# Define implied volatility surface
iv_surface = [
    {'K': 95.0, 'T': 0.25, 'iv': 0.22},
    {'K': 100.0, 'T': 0.25, 'iv': 0.20},
    {'K': 105.0, 'T': 0.25, 'iv': 0.21},
    {'K': 110.0, 'T': 0.25, 'iv': 0.23}
]

price = op.local_volatility(
    S=100.0,
    K=105.0,
    r=0.05,
    q=0.02,
    T=0.25,
    is_call=True,
    iv_surface=iv_surface,
    american=True
)
```

---

## SABR Model

### sabr_implied_vol

Calculate implied volatility using the SABR model.

```python
option_pricer.sabr_implied_vol(S, K, r, T, alpha, beta, rho, nu)
```

**Parameters:**
- `S` (float): Spot price (or forward price)
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `T` (float): Time to maturity
- `alpha` (float): Volatility parameter
- `beta` (float): Beta parameter (CEV exponent, typically 0.5 for equity)
- `rho` (float): Correlation between asset and volatility
- `nu` (float): Volatility of volatility

**Returns:**
- `float`: Implied volatility

**Example:**
```python
iv = op.sabr_implied_vol(
    S=100.0,
    K=105.0,
    r=0.05,
    T=0.25,
    alpha=0.3,
    beta=0.5,
    rho=-0.3,
    nu=0.4
)
print(f"SABR IV: {iv:.4f}")
```

---

### sabr_option

Calculate option price using SABR model.

```python
option_pricer.sabr_option(S, K, r, T, F, alpha, beta, rho, nu, is_call)
```

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `T` (float): Time to maturity
- `F` (float): Forward price
- `alpha` (float): Volatility parameter
- `beta` (float): Beta parameter
- `rho` (float): Correlation
- `nu` (float): Vol of vol
- `is_call` (bool): True for call, False for put

**Returns:**
- `float`: Option price

---

### sabr_calibrate

Calibrate SABR parameters to market data.

```python
option_pricer.sabr_calibrate(market_data, F, T, beta)
```

**Parameters:**
- `market_data` (dict): Market implied volatilities {strike: iv}
- `F` (float): Forward price
- `T` (float): Time to maturity
- `beta` (float): Fixed beta parameter

**Returns:**
- `tuple`: Calibrated (alpha, rho, nu) parameters

**Example:**
```python
# Market IV data
market_ivs = {
    95.0: 0.22,
    100.0: 0.20,
    105.0: 0.21,
    110.0: 0.23
}

F = 101.0  # Forward price
T = 0.25
beta = 0.5

alpha, rho, nu = op.sabr_calibrate(market_ivs, F, T, beta)
print(f"Calibrated SABR: α={alpha:.4f}, ρ={rho:.4f}, ν={nu:.4f}")
```

---

### SABR Greeks

The SABR model provides its own Greeks:

- `sabr_delta(S, K, r, T, F, alpha, beta, rho, nu, is_call)` - Delta
- `sabr_gamma(S, K, r, T, F, alpha, beta, rho, nu, is_call)` - Gamma
- `sabr_vega(S, K, r, T, F, alpha, beta, rho, nu, is_call)` - Vega
- `sabr_volga(S, K, r, T, F, alpha, beta, rho, nu, is_call)` - Volga (vomma)
- `sabr_vanna(S, K, r, T, F, alpha, beta, rho, nu, is_call)` - Vanna

**Example:**
```python
# Calculate SABR Greeks
F = 101.0
alpha, beta, rho, nu = 0.3, 0.5, -0.3, 0.4

delta = op.sabr_delta(100.0, 105.0, 0.05, 0.25, F, alpha, beta, rho, nu, True)
gamma = op.sabr_gamma(100.0, 105.0, 0.05, 0.25, F, alpha, beta, rho, nu, True)
vega = op.sabr_vega(100.0, 105.0, 0.05, 0.25, F, alpha, beta, rho, nu, True)
volga = op.sabr_volga(100.0, 105.0, 0.05, 0.25, F, alpha, beta, rho, nu, True)

print(f"SABR Delta: {delta:.4f}")
print(f"SABR Gamma: {gamma:.4f}")
print(f"SABR Vega: {vega:.2f}")
print(f"SABR Volga: {volga:.4f}")
```

---

## Python Services

### MarketDataService

See [MARKET_DATA.md](MARKET_DATA.md) for complete documentation of the `MarketDataService` class.

**Quick Example:**
```python
from python.option_pricer.market_data import MarketDataService

service = MarketDataService(['SPY', 'AAPL', 'TSLA'])
service.start()

# Get data
spot = service.get_spot_price('AAPL')
options = service.get_option_chain('AAPL')
```

---

### CalibrationService

See [CALIBRATION.md](CALIBRATION.md) for complete documentation of the `CalibrationService` class.

**Quick Example:**
```python
from python.option_pricer.calibration import CalibrationService

service = CalibrationService(['SPY', 'AAPL'])
service.start()

# Calibrations run automatically and are saved to JSON
```

---

### Visualization

See [VISUALIZATION.md](VISUALIZATION.md) for complete documentation of the interactive dashboard.

**Quick Example:**
```bash
python python/option_pricer/visualization.py
```

---

## Complete Example

Here's a complete example that uses multiple features:

```python
import option_pricer as op
import numpy as np

# Market parameters
S = 100.0
K_strikes = np.arange(90, 111, 5)  # Strikes from 90 to 110
r = 0.05
q = 0.02
T = 0.25
sigma = 0.2

print("Options Pricing Analysis")
print("=" * 60)
print(f"Spot: ${S:.2f}, Time: {T:.2f} years, Vol: {sigma:.1%}\n")

for K in K_strikes:
    # Price with different models
    bs_call = op.black_scholes(S, K, r, q, sigma, T, is_call=True)
    binom_call = op.binomial_tree(S, K, r, q, sigma, T, 100, is_call=True, american=True)

    # Calculate Greeks
    delta = op.delta(S, K, r, q, sigma, T, is_call=True)
    gamma = op.gamma(S, K, r, q, sigma, T, is_call=True)
    vega = op.vega(S, K, r, q, sigma, T, is_call=True)

    # Display
    moneyness = "ITM" if K < S else ("ATM" if K == S else "OTM")
    print(f"Strike ${K:.0f} ({moneyness}):")
    print(f"  BS: ${bs_call:.2f}, Binomial: ${binom_call:.2f}")
    print(f"  Δ={delta:.3f}, Γ={gamma:.4f}, ν={vega:.2f}")
    print()
```

---

## Error Handling

All pricing functions return valid numerical results. For invalid inputs (negative prices, negative time, etc.), functions may return 0 or raise exceptions. Always validate inputs before calling pricing functions.

**Best Practices:**
```python
# Validate inputs
assert S > 0, "Spot price must be positive"
assert K > 0, "Strike must be positive"
assert T > 0, "Time to maturity must be positive"
assert sigma > 0, "Volatility must be positive"
assert 0 <= r <= 1, "Interest rate must be reasonable"
```

---

## Performance Tips

1. **Use C++ functions directly**: All core pricing and Greeks functions are implemented in C++ for maximum performance
2. **Batch calculations**: When pricing multiple options, reuse common calculations
3. **Monte Carlo**: Use more simulations (100,000+) for accurate American option pricing
4. **Binomial trees**: 100-500 steps typically provide good accuracy
5. **Greeks**: Use analytical Greeks when available (Black-Scholes) rather than numerical approximations

---

## See Also

- [Usage Guide](USAGE_GUIDE.md) - Detailed examples and tutorials
- [Market Data](MARKET_DATA.md) - Real-time data integration
- [Calibration](CALIBRATION.md) - Model calibration
- [Installation](INSTALLATION.md) - Setup instructions
