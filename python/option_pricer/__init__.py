"""
Option Pricer Package
A comprehensive options pricing library with C++ implementations
"""

import importlib.util
import os
import sys

# Get the directory where this __init__.py file is located
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to find the .so file with different naming conventions
_so_extensions = [
    f"option_pricer.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so",
    "option_pricer.so",
    f"option_pricer.cpython-{sys.version_info.major}{sys.version_info.minor}-linux-x86_64.so",
    f"option_pricer.cpython-{sys.version_info.major}{sys.version_info.minor}-win_amd64.pyd"
]

_option_pricer = None
for ext in _so_extensions:
    _so_path = os.path.join(_current_dir, ext)
    if os.path.exists(_so_path):
        spec = importlib.util.spec_from_file_location("option_pricer", _so_path)
        _option_pricer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_option_pricer)
        break

if _option_pricer is None:
    raise ImportError("Could not find option_pricer compiled module. Please ensure it's built and in the package directory.")

# Core Options Pricing Functions
def black_scholes(S, K, r, q, sigma, T, is_call):
    """
    Calculate option price using Black-Scholes model

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    sigma: Volatility
    T: Time to maturity
    is_call: True for call, False for put
    """
    return _option_pricer.black_scholes(S, K, r, q, sigma, T, is_call)

def binomial_tree(S, K, r, q,  sigma, T, steps, is_call, american=True):
    """
    Calculate option price using binomial tree model

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    sigma: Volatility
    T: Time to maturity
    steps: Number of time steps
    is_call: True for call, False for put
    american: True for American, False for European
    """
    if american:
        return _option_pricer.binomial_tree_american(S, K, r, q, sigma, T, steps, is_call)
    else:
        return _option_pricer.binomial_tree_european(S, K, r, q, sigma, T, steps, is_call)

def monte_carlo(S, K, r, q, sigma, T, simulations, is_call, american=True):
    """
    Calculate option price using Monte Carlo simulation

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    sigma: Volatility
    T: Time to maturity
    simulations: Number of simulation paths
    is_call: True for call, False for put
    american: True for American, False for Asian
    """
    if american:
        return _option_pricer.monte_carlo_american_option(S, K, r, q, sigma, T, simulations, is_call)
    else:
        return _option_pricer.monte_carlo_asian_option(S, K, r, q, sigma, T, simulations, is_call)

# Greek Calculations
def delta(S, K, r, q, sigma, T, is_call):
    """Calculate option delta"""
    return _option_pricer.delta(S, K, r, q, sigma, T, is_call)

def gamma(S, K, r, q, sigma, T, is_call):
    """Calculate option gamma"""
    return _option_pricer.gamma(S, K, r, q, sigma, T, is_call)

def vega(S, K, r, q, sigma, T, is_call):
    """Calculate option vega"""
    return _option_pricer.vega(S, K, r, q, sigma, T, is_call)

def theta(S, K, r, q, sigma, T, is_call):
    """Calculate option theta"""
    return _option_pricer.theta(S, K, r, q, sigma, T, is_call)

def rho(S, K, r, q, sigma, T, is_call):
    """Calculate option rho"""
    return _option_pricer.rho(S, K, r, q, sigma, T, is_call)

def vanna(S, K, r, q, sigma, T, is_call):
    """Calculate option vanna (dVega/dSpot)"""
    return _option_pricer.vanna(S, K, r, q, sigma, T, is_call)

def charm(S, K, r, q, sigma, T, is_call):
    """Calculate option charm (dDelta/dTime)"""
    return _option_pricer.charm(S, K, r, q, sigma, T, is_call)

def vomma(S, K, r, q, sigma, T, is_call):
    """Calculate option vomma (dVega/dVol)"""
    return _option_pricer.vomma(S, K, r, q, sigma, T, is_call)

def veta(S, K, r, q, sigma, T, is_call):
    """Calculate option veta (dVega/dTime)"""
    return _option_pricer.veta(S, K, r, q, sigma, T, is_call)

# Advanced Models
def heston_model(S, K, r, q, T, kappa, theta, sigma, rho, v0, is_call):
    """
    Calculate option price using Heston stochastic volatility model

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    T: Time to maturity
    kappa: Mean reversion speed
    theta: Long-term variance
    sigma: Volatility of volatility
    rho: Correlation between spot and variance
    v0: Initial variance
    is_call: True for call, False for put
    """
    return _option_pricer.heston_model(S, K, r, q, T, kappa, theta, sigma, rho, v0, is_call)

def jump_diffusion(S, K, r, q, sigma, T, lambda_, mu_j, sigma_j, simulations, is_call):
    """
    Calculate option price using Merton jump diffusion model

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    sigma: Volatility
    T: Time to maturity
    lambda_: Jump intensity
    mu_j: Mean jump size
    sigma_j: Jump size volatility
    simulations: Number of simulation paths
    is_call: True for call, False for put
    """
    return _option_pricer.jump_diffusion(S, K, r, q, sigma, T, lambda_, mu_j, sigma_j, simulations, is_call)

def local_volatility(S, K, r, q, T, is_call, american=True):
    """
    Calculate option price using local volatility model with FDM

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    T: Time to maturity
    is_call: True for call, False for put
    american: True for American, False for European
    """
    if american:
        return _option_pricer.american_local_vol_fdm(S, K, r, q, T, is_call)
    else:
        return _option_pricer.european_local_vol_fdm(S, K, r, q, T, is_call)

# SABR Model Functions
def sabr_implied_vol(S, K, r, T, alpha, beta, rho, nu):
    """
    Calculate implied volatility using SABR model

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    T: Time to maturity
    alpha: Volatility parameter
    beta: Beta parameter (CEV exponent)
    rho: Correlation between spot and volatility
    nu: Volatility of volatility
    """
    return _option_pricer.SABRImpliedVol(S, K, r, T, alpha, beta, rho, nu)

def sabr_option(S, K, r, T, F, alpha, beta, rho, nu, is_call):
    """
    Calculate option price using SABR model

    Parameters:
    S: Spot price
    K: Strike price
    r: Risk-free rate
    T: Time to maturity
    F: Forward price
    alpha: Volatility parameter
    beta: Beta parameter
    rho: Correlation
    nu: Vol of vol
    is_call: True for call, False for put
    """
    return _option_pricer.SABROptionPrice(S, K, r, T, F, alpha, beta, rho, nu, is_call)

def sabr_calibrate(market_data, F, T, beta):
    """
    Calibrate SABR parameters to market data

    Parameters:
    market_data: Market implied volatilities
    F: Forward price
    T: Time to maturity
    beta: Fixed beta parameter

    Returns:
    Calibrated (alpha, rho, nu) parameters
    """
    return _option_pricer.SABRCalibrate(market_data, F, T, beta)

# SABR Greeks
def sabr_delta(S, K, r, T, F, alpha, beta, rho, nu, is_call):
    """Calculate SABR model delta"""
    return _option_pricer.SABRDelta(S, K, r, T, F, alpha, beta, rho, nu, is_call)

def sabr_gamma(S, K, r, T, F, alpha, beta, rho, nu, is_call):
    """Calculate SABR model gamma"""
    return _option_pricer.SABRGamma(S, K, r, T, F, alpha, beta, rho, nu, is_call)

def sabr_vega(S, K, r, T, F, alpha, beta, rho, nu, is_call):
    """Calculate SABR model vega"""
    return _option_pricer.SABRVega(S, K, r, T, F, alpha, beta, rho, nu, is_call)

def sabr_volga(S, K, r, T, F, alpha, beta, rho, nu, is_call):
    """Calculate SABR model volga (dVega/dVol)"""
    return _option_pricer.SABRVolga(S, K, r, T, F, alpha, beta, rho, nu, is_call)

def sabr_vanna(S, K, r, T, F, alpha, beta, rho, nu, is_call):
    """Calculate SABR model vanna (dVega/dSpot)"""
    return _option_pricer.SABRVanna(S, K, r, T, F, alpha, beta, rho, nu, is_call)

# Aliases for consistent naming
Heston_Model = heston_model
SABR_implied_vol = sabr_implied_vol
SABR_option = sabr_option
SABR_calibrate = sabr_calibrate
SABR_delta = sabr_delta
SABR_gamma = sabr_gamma
SABR_vega = sabr_vega
SABR_volga = sabr_volga
SABR_vanna = sabr_vanna

# Export all functions
__all__ = [
    # Core pricing
    'black_scholes',
    'binomial_tree',
    'monte_carlo',

    # Greeks
    'delta',
    'gamma',
    'vega',
    'theta',
    'rho',
    'vanna',
    'charm',
    'vomma',
    'veta',

    # Advanced models
    'heston_model',
    'Heston_Model',
    'jump_diffusion',
    'local_volatility',

    # SABR
    'sabr_implied_vol',
    'SABR_implied_vol',
    'sabr_option',
    'SABR_option',
    'sabr_calibrate',
    'SABR_calibrate',
    'sabr_delta',
    'SABR_delta',
    'sabr_gamma',
    'SABR_gamma',
    'sabr_vega',
    'SABR_vega',
    'sabr_volga',
    'SABR_volga',
    'sabr_vanna',
    'SABR_vanna',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Tommy Ghaly'
__description__ = 'High-performance options pricing library with C++ backend'
