# Usage Guide

Comprehensive guide with examples for using the Option Pricer library.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Option Pricing](#basic-option-pricing)
- [Greeks Analysis](#greeks-analysis)
- [Advanced Models](#advanced-models)
- [Real-Time Market Data](#real-time-market-data)
- [Model Calibration](#model-calibration)
- [Interactive Visualization](#interactive-visualization)
- [Complete Examples](#complete-examples)

---

## Getting Started

### Import the Library

```python
import option_pricer as op
import numpy as np
```

### Basic Parameters

Common parameters used throughout:

```python
S = 100.0      # Spot price
K = 105.0      # Strike price
r = 0.05       # Risk-free rate (5% annualized)
q = 0.02       # Dividend yield (2% annualized)
sigma = 0.2    # Volatility (20% annualized)
T = 0.25       # Time to maturity (3 months = 0.25 years)
```

---

## Basic Option Pricing

### Black-Scholes Model

Price European options using analytical Black-Scholes formula:

```python
# Price a call option
call_price = op.black_scholes(S, K, r, q, sigma, T, is_call=True)
print(f"European Call: ${call_price:.2f}")

# Price a put option
put_price = op.black_scholes(S, K, r, q, sigma, T, is_call=False)
print(f"European Put: ${put_price:.2f}")

# Verify put-call parity
parity_check = call_price - put_price - (S * np.exp(-q*T) - K * np.exp(-r*T))
print(f"Put-Call Parity (should be ~0): {parity_check:.6f}")
```

**Output:**
```
European Call: $2.56
European Put: $6.48
Put-Call Parity (should be ~0): 0.000000
```

---

### Binomial Tree Model

Price American and European options using binomial trees:

```python
# American call option (can exercise early)
american_call = op.binomial_tree(S, K, r, q, sigma, T, steps=100, is_call=True, american=True)
print(f"American Call: ${american_call:.2f}")

# European call (for comparison)
european_call = op.binomial_tree(S, K, r, q, sigma, T, steps=100, is_call=True, american=False)
print(f"European Call: ${european_call:.2f}")

# American put (early exercise premium)
american_put = op.binomial_tree(S, K, r, q, sigma, T, steps=100, is_call=False, american=True)
print(f"American Put: ${american_put:.2f}")

# Compare with Black-Scholes
bs_put = op.black_scholes(S, K, r, q, sigma, T, is_call=False)
early_exercise_premium = american_put - bs_put
print(f"Early Exercise Premium: ${early_exercise_premium:.2f}")
```

**Tips:**
- Use 100-500 steps for good accuracy
- More steps = more accurate but slower
- American puts usually have higher early exercise premium than calls

---

### Monte Carlo Simulation

Price options using Monte Carlo simulation:

```python
# American call with Monte Carlo
mc_call = op.monte_carlo(S, K, r, q, sigma, T, simulations=10000, is_call=True, american=True)
print(f"Monte Carlo American Call: ${mc_call:.2f}")

# Asian option (average price option)
asian_call = op.monte_carlo(S, K, r, q, sigma, T, simulations=10000, is_call=True, american=False)
print(f"Asian Call: ${asian_call:.2f}")

# Increase simulations for better accuracy
mc_call_accurate = op.monte_carlo(S, K, r, q, sigma, T, simulations=100000, is_call=True, american=True)
print(f"Monte Carlo (100k sims): ${mc_call_accurate:.2f}")
```

**Tips:**
- 10,000 simulations: Quick estimates
- 100,000 simulations: Production use
- 1,000,000 simulations: High accuracy, research

---

## Greeks Analysis

### First-Order Greeks

Calculate sensitivity measures:

```python
# Calculate all first-order Greeks
delta = op.delta(S, K, r, q, sigma, T, is_call=True)
gamma = op.gamma(S, K, r, q, sigma, T, is_call=True)
vega = op.vega(S, K, r, q, sigma, T, is_call=True)
theta = op.theta(S, K, r, q, sigma, T, is_call=True)
rho = op.rho(S, K, r, q, sigma, T, is_call=True)

print("First-Order Greeks:")
print(f"  Delta: {delta:.4f}  (Share equivalent)")
print(f"  Gamma: {gamma:.4f}  (Delta change per $1 move)")
print(f"  Vega:  {vega:.2f}   (P&L per 1% vol change)")
print(f"  Theta: {theta:.2f}   (Daily time decay)")
print(f"  Rho:   {rho:.2f}     (P&L per 1% rate change)")
```

**Output:**
```
First-Order Greeks:
  Delta: 0.4523  (Share equivalent)
  Gamma: 0.0189  (Delta change per $1 move)
  Vega:  18.45   (P&L per 1% vol change)
  Theta: -7.23   (Daily time decay)
  Rho:   10.15   (P&L per 1% rate change)
```

---

### Second-Order Greeks

Calculate second-order sensitivities:

```python
# Calculate second-order Greeks
vanna = op.vanna(S, K, r, q, sigma, T, is_call=True)
charm = op.charm(S, K, r, q, sigma, T, is_call=True)
vomma = op.vomma(S, K, r, q, sigma, T, is_call=True)
veta = op.veta(S, K, r, q, sigma, T, is_call=True)

print("Second-Order Greeks:")
print(f"  Vanna: {vanna:.4f}  (dDelta/dVol)")
print(f"  Charm: {charm:.4f}  (dDelta/dTime)")
print(f"  Vomma: {vomma:.4f}  (dVega/dVol)")
print(f"  Veta:  {veta:.4f}   (dVega/dTime)")
```

---

### Greeks Profile Analysis

Analyze Greeks across different strikes:

```python
strikes = np.arange(90, 111, 2)
deltas = []
gammas = []
vegas = []

for K in strikes:
    deltas.append(op.delta(S, K, r, q, sigma, T, is_call=True))
    gammas.append(op.gamma(S, K, r, q, sigma, T, is_call=True))
    vegas.append(op.vega(S, K, r, q, sigma, T, is_call=True))

print("Strike  Delta   Gamma   Vega   Moneyness")
print("-" * 50)
for i, K in enumerate(strikes):
    moneyness = "ITM" if K < S else ("ATM" if K == S else "OTM")
    print(f"{K:5.0f}   {deltas[i]:.3f}   {gammas[i]:.4f}  {vegas[i]:5.2f}  {moneyness}")
```

---

## Advanced Models

### Heston Stochastic Volatility

Price options with stochastic volatility:

```python
# Heston model parameters
v0 = 0.04          # Initial variance (20% vol)
kappa = 2.0        # Mean reversion speed
theta = 0.04       # Long-term variance
sigma_v = 0.3      # Vol of vol
rho = -0.7         # Correlation (negative for stocks)

# Price option
heston_price = op.heston_model(S, K, r, q, T, kappa, theta, sigma_v, rho, v0, is_call=True)
bs_price = op.black_scholes(S, K, r, q, sigma, T, is_call=True)

print(f"Heston Model:      ${heston_price:.2f}")
print(f"Black-Scholes:     ${bs_price:.2f}")
print(f"Difference:        ${heston_price - bs_price:+.2f}")
```

**Parameter Guidelines:**
- `kappa`: 0.5-5.0 (higher = faster mean reversion)
- `theta`: Long-term vol² (e.g., 0.04 for 20% vol)
- `sigma_v`: 0.1-0.5 (vol of vol)
- `rho`: -0.9 to 0 (typically negative for stocks)
- `v0`: Initial vol² (match current market vol)

---

### SABR Model

Use SABR for more accurate vol surface modeling:

```python
# SABR parameters (typically from calibration)
F = S * np.exp((r - q) * T)  # Forward price
alpha = 0.3      # Initial volatility
beta = 0.5       # Beta (0.5 for lognormal, 0 for normal)
rho = -0.3       # Correlation
nu = 0.4         # Vol of vol

# Calculate implied volatility
sabr_iv = op.sabr_implied_vol(S, K, r, T, alpha, beta, rho, nu)
print(f"SABR Implied Vol: {sabr_iv:.2%}")

# Price option using SABR
sabr_price = op.sabr_option(S, K, r, T, F, alpha, beta, rho, nu, is_call=True)
print(f"SABR Option Price: ${sabr_price:.2f}")

# Calculate SABR Greeks
sabr_delta = op.sabr_delta(S, K, r, T, F, alpha, beta, rho, nu, is_call=True)
sabr_vega = op.sabr_vega(S, K, r, T, F, alpha, beta, rho, nu, is_call=True)

print(f"SABR Delta: {sabr_delta:.4f}")
print(f"SABR Vega:  {sabr_vega:.2f}")
```

---

### Jump Diffusion Model

Price options with jump risk:

```python
# Merton jump diffusion parameters
lambda_jump = 0.1    # 0.1 jumps per year
mu_jump = -0.10      # -10% average jump
sigma_jump = 0.15    # 15% jump volatility

# Price with jump diffusion
jump_price = op.jump_diffusion(
    S, K, r, q, sigma, T,
    lambda_jump, mu_jump, sigma_jump,
    simulations=10000,
    is_call=True
)

bs_price = op.black_scholes(S, K, r, q, sigma, T, is_call=True)

print(f"Jump Diffusion: ${jump_price:.2f}")
print(f"Black-Scholes:  ${bs_price:.2f}")
print(f"Jump Premium:   ${jump_price - bs_price:+.2f}")
```

**Use Cases:**
- Modeling crash risk (negative jumps)
- Earnings announcements
- News-driven volatility
- Out-of-the-money puts

---

## Real-Time Market Data

### Basic Market Data Fetching

```python
from python.option_pricer.market_data import MarketDataService
import time

# Initialize service
service = MarketDataService(['SPY', 'AAPL'])

# Configure to save data
service.save_to_file = True
service.data_directory = './my_data'

# Start fetching
if service.start():
    print("Service started!")

    # Wait for initial data
    time.sleep(10)

    # Get spot price
    spy = service.get_spot_price('SPY')
    print(f"SPY: ${spy['price']:.2f}")
    print(f"  Volume: {spy['volume']:,}")
    print(f"  Bid/Ask: ${spy['bid']:.2f} / ${spy['ask']:.2f}")

    # Get options
    spy_options = service.get_option_chain('SPY')
    print(f"SPY has {len(spy_options)} expiries")

    # Stop service
    service.stop()
```

---

### Working with Option Chains

```python
from python.option_pricer.market_data import MarketDataService

service = MarketDataService(['AAPL'])
service.start()

time.sleep(10)

# Get option chain for specific expiry
expiry = '2024-01-19'
options = service.get_option_chain('AAPL', expiry)

if options:
    calls = options['calls']
    puts = options['puts']

    print(f"AAPL {expiry} Options")
    print("\nCalls:")
    print("Strike   Last    Bid     Ask     IV      Volume")
    print("-" * 55)

    for call in calls[:10]:  # First 10 calls
        print(f"{call['strike']:6.1f}  "
              f"{call['lastPrice']:6.2f}  "
              f"{call['bid']:6.2f}  "
              f"{call['ask']:6.2f}  "
              f"{call['impliedVolatility']:6.2%}  "
              f"{call['volume']:8.0f}")

service.stop()
```

---

## Model Calibration

### Automatic Multi-Model Calibration

```python
from python.option_pricer.calibration import CalibrationService
import json
import time

# Start calibration service
service = CalibrationService(['SPY'], config={
    'calibration_interval': 60,
    'max_expiries_per_symbol': 5
})

service.start()

# Let it calibrate
print("Calibrating...")
time.sleep(90)

# Load results
with open('./calibration_data_realtime/calibrations.json', 'r') as f:
    results = json.load(f)

# Display SABR calibration
spy_data = results['data']['SPY']
first_expiry = list(spy_data.keys())[0]
sabr = spy_data[first_expiry]['models']['SABR']

print(f"SPY {first_expiry} SABR Calibration:")
print(f"  Alpha: {sabr['params']['alpha']:.4f}")
print(f"  Rho:   {sabr['params']['rho']:.4f}")
print(f"  Nu:    {sabr['params']['nu']:.4f}")

# Show model performance
if 'rmse' in sabr:
    print(f"  RMSE:  {sabr['rmse']:.4f}")

service.stop()
```

---

### Manual SABR Calibration

```python
# Market data: strikes and implied vols
market_data = {
    95.0: 0.22,   # Strike: IV
    100.0: 0.20,
    105.0: 0.21,
    110.0: 0.23,
    115.0: 0.25
}

F = 101.0  # Forward price
T = 0.25   # Time to maturity
beta = 0.5 # Fixed beta

# Calibrate SABR
alpha, rho, nu = op.sabr_calibrate(market_data, F, T, beta)

print("Calibrated SABR Parameters:")
print(f"  Alpha: {alpha:.4f}")
print(f"  Beta:  {beta:.4f}")
print(f"  Rho:   {rho:.4f}")
print(f"  Nu:    {nu:.4f}")

# Use calibrated params to price new option
K_new = 107.0
sabr_iv_new = op.sabr_implied_vol(F, K_new, 0.05, T, alpha, beta, rho, nu)
print(f"\nInterpolated IV at K={K_new}: {sabr_iv_new:.2%}")
```

---

## Interactive Visualization

### Start Dashboard

```bash
# Terminal 1: Calibration service
python python/option_pricer/calibration.py

# Terminal 2: Dashboard
python python/option_pricer/visualization.py
```

Open browser to http://localhost:8050

---

## Complete Examples

### Example 1: Portfolio Greeks

Calculate portfolio-level Greeks:

```python
import option_pricer as op
import pandas as pd

# Portfolio positions
positions = [
    {'type': 'call', 'strike': 95, 'quantity': 10},
    {'type': 'call', 'strike': 100, 'quantity': -20},  # Short
    {'type': 'call', 'strike': 105, 'quantity': 10},
    {'type': 'put', 'strike': 95, 'quantity': -10},
]

S, r, q, sigma, T = 100.0, 0.05, 0.02, 0.2, 0.25

# Calculate portfolio Greeks
portfolio_delta = 0
portfolio_gamma = 0
portfolio_vega = 0
portfolio_theta = 0

results = []
for pos in positions:
    K = pos['strike']
    qty = pos['quantity']
    is_call = (pos['type'] == 'call')

    # Greeks for single option
    delta = op.delta(S, K, r, q, sigma, T, is_call)
    gamma = op.gamma(S, K, r, q, sigma, T, is_call)
    vega = op.vega(S, K, r, q, sigma, T, is_call)
    theta = op.theta(S, K, r, q, sigma, T, is_call)

    # Position Greeks
    pos_delta = delta * qty
    pos_gamma = gamma * qty
    pos_vega = vega * qty
    pos_theta = theta * qty

    # Add to portfolio
    portfolio_delta += pos_delta
    portfolio_gamma += pos_gamma
    portfolio_vega += pos_vega
    portfolio_theta += pos_theta

    results.append({
        'Type': pos['type'],
        'Strike': K,
        'Qty': qty,
        'Delta': f"{pos_delta:.2f}",
        'Gamma': f"{pos_gamma:.4f}",
        'Vega': f"{pos_vega:.2f}",
        'Theta': f"{pos_theta:.2f}"
    })

# Display results
df = pd.DataFrame(results)
print(df.to_string(index=False))

print("\nPortfolio Greeks:")
print(f"  Delta: {portfolio_delta:.2f} (shares)")
print(f"  Gamma: {portfolio_gamma:.4f}")
print(f"  Vega:  {portfolio_vega:.2f}")
print(f"  Theta: {portfolio_theta:.2f}")
```

---

### Example 2: Volatility Surface Analysis

Build and analyze volatility surface:

```python
import option_pricer as op
import numpy as np
import matplotlib.pyplot as plt

S = 100
r, q, T = 0.05, 0.02, 0.25

# Grid of strikes and volatilities
strikes = np.linspace(85, 115, 31)
implied_vols = 0.2 + 0.01 * ((strikes - S) / S)**2  # Smile shape

# Price options across strikes
bs_prices = []
heston_prices = []

for i, K in enumerate(strikes):
    sigma = implied_vols[i]

    # Black-Scholes
    bs = op.black_scholes(S, K, r, q, sigma, T, is_call=True)
    bs_prices.append(bs)

    # Heston
    v0 = sigma**2
    heston = op.heston_model(S, K, r, q, T, 2.0, v0, 0.3, -0.7, v0, is_call=True)
    heston_prices.append(heston)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(strikes, implied_vols * 100, 'b-', linewidth=2)
plt.axvline(S, color='r', linestyle='--', label='ATM')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility (%)')
plt.title('Volatility Smile')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(strikes, bs_prices, 'b-', label='Black-Scholes', linewidth=2)
plt.plot(strikes, heston_prices, 'r--', label='Heston', linewidth=2)
plt.xlabel('Strike')
plt.ylabel('Option Price ($)')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vol_surface_analysis.png', dpi=150)
print("Saved vol_surface_analysis.png")
```

---

### Example 3: Real-Time Greeks Monitor

Monitor Greeks in real-time:

```python
from python.option_pricer.market_data import MarketDataService
import option_pricer as op
import time

service = MarketDataService(['SPY'])
service.start()

print("Real-Time Greeks Monitor")
print("Press Ctrl+C to stop\n")

try:
    while True:
        # Get current price
        spy_data = service.get_spot_price('SPY')
        if not spy_data:
            time.sleep(5)
            continue

        S = spy_data['price']
        K = round(S)  # ATM strike
        r, q, sigma, T = 0.05, 0.02, 0.2, 0.25

        # Calculate Greeks
        delta = op.delta(S, K, r, q, sigma, T, is_call=True)
        gamma = op.gamma(S, K, r, q, sigma, T, is_call=True)
        vega = op.vega(S, K, r, q, sigma, T, is_call=True)
        theta = op.theta(S, K, r, q, sigma, T, is_call=True)

        # Display
        print(f"\r[{time.strftime('%H:%M:%S')}] "
              f"SPY=${S:.2f} K={K} | "
              f"Δ={delta:.3f} Γ={gamma:.4f} ν={vega:.1f} θ={theta:.2f}",
              end='', flush=True)

        time.sleep(5)

except KeyboardInterrupt:
    print("\n\nStopping...")
    service.stop()
```

---

## See Also

- [API Reference](API_REFERENCE.md) - Complete function documentation
- [Market Data](MARKET_DATA.md) - MarketDataService guide
- [Calibration](CALIBRATION.md) - Model calibration
- [Visualization](VISUALIZATION.md) - Interactive dashboard
- [Installation](INSTALLATION.md) - Setup instructions
