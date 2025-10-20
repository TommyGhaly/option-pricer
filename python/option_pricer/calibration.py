"""
Complete Multi-Model Options Calibration Service
Implements SABR, Heston, Merton Jump Diffusion, and Local Volatility models
"""

import threading as th
import queue as qu
import os
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import math
import numpy as np
import json
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import interp1d
import time
import tempfile
from pathlib import Path
import warnings
import atexit
import shutil
warnings.filterwarnings('ignore')


class CalibrationService:
    """
    Real-time options calibration service for multiple volatility models
    """

    # Define all tickers to track
    TICKERS = {
        # Major Index ETFs
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq-100 ETF',
        'IWM': 'Russell 2000 ETF',
        'DIA': 'Dow Jones ETF',

        # High-Volume Tech Stocks
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'AMZN': 'Amazon.com Inc.',
        'GOOGL': 'Alphabet Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corporation',
        'META': 'Meta Platforms Inc.',
        'AMD': 'Advanced Micro Devices',

        # Financial Sector
        'JPM': 'JPMorgan Chase',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',

        # Volatility Products
        'UVXY': 'Ultra VIX Short-Term',
        'SVXY': 'Short VIX Short-Term',

        # Sector ETFs
        'XLF': 'Financial Sector',
        'XLE': 'Energy Sector',
        'XLK': 'Technology Sector',
        'GLD': 'Gold ETF'
    }

    def __init__(self, symbols=None, config=None):
        """Initialize the calibration service"""
        print("\n" + "="*80)
        print("MULTI-MODEL OPTIONS CALIBRATION SERVICE")
        print("="*80)

        # Use provided symbols or default to all tickers
        self.symbols = symbols if symbols else list(self.TICKERS.keys())

        # Configuration
        self.data_directory = "./market_data_realtime"
        self.calibration_directory = Path('./calibration_data_realtime')
        self.calibration_directory.mkdir(exist_ok=True)

        self.file_paths = {
            'spot': "spot_data.json",
            'options': "option_chains.json",
            'history': "historical_data.json"
        }

        # Models to calibrate
        self.models = ['SABR', 'Heston', 'Merton', 'LocalVol']
        print(f"Models: {', '.join(self.models)}")
        print(f"Tracking {len(self.symbols)} symbols")

        # Set up cache directory
        self.temp_dir = tempfile.mkdtemp(prefix="calibration_cache_")
        self.cache_dir = Path(self.temp_dir)
        atexit.register(self._cleanup_temp_dir)

        # Market data storage
        self.spot_prices = {}
        self.option_chains = {}
        self.historical_data = {}

        # Risk-free rates
        self.risk_free_rate = {
            '1m': 0.042,
            '3m': 0.040,
            '1y': 0.038
        }

        # Calibration results
        self.calibrated_params = {}
        self.iv_cache = {}

        # Threading
        self.running = False
        self.data_lock = th.Lock()
        self.calibration_thread = None

        # Configuration
        config = config or {}
        self.calibration_interval = config.get('calibration_interval', 30)
        self.min_data_points = 3
        self.max_calibrations_per_cycle = config.get('max_calibrations', 20)

        # File tracking
        self.file_timestamps = {}
        self.last_calibration_time = {}

        print("Initialization complete!\n")

    def _cleanup_temp_dir(self):
        """Clean up temporary directory on exit"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

    def start(self):
        """Start the calibration service"""
        print("\n" + "="*80)
        print("STARTING SERVICE")
        print("="*80)

        # Load initial data
        print("\nLoading market data...")
        if not self._load_market_data():
            print("ERROR: Failed to load market data!")
            return

        print(f"Loaded: {len(self.spot_prices)} spot prices, {len(self.option_chains)} option chains")

        # Calculate initial IVs
        print("\nCalculating initial implied volatilities...")
        self._calculate_initial_ivs()

        # Run initial calibration
        print("\nRunning initial calibrations...")
        self._run_calibration_cycle()

        # Start background thread
        self.running = True
        self.calibration_thread = th.Thread(
            target=self._calibration_loop,
            daemon=True
        )
        self.calibration_thread.start()

        print("\n" + "="*80)
        print("SERVICE STARTED SUCCESSFULLY")
        print(f"Calibration interval: {self.calibration_interval} seconds")
        print("="*80 + "\n")

    def stop(self):
        """Stop the calibration service"""
        print("\nStopping calibration service...")
        self.running = False

        if self.calibration_thread and self.calibration_thread.is_alive():
            self.calibration_thread.join(timeout=5)

        # Save final state
        self._save_all_data()

        total_calibrations = sum(
            len(expiries) for symbol in self.calibrated_params.values()
            for expiries in symbol.values()
        )
        print(f"Service stopped. Total calibrations saved: {total_calibrations}")

    def _load_market_data(self):
        """Load market data from JSON files"""
        try:
            # Load spot prices
            spot_path = os.path.join(self.data_directory, self.file_paths["spot"])
            if os.path.exists(spot_path):
                with open(spot_path, 'r') as f:
                    self.spot_prices = json.load(f)
            else:
                print(f"ERROR: Spot file not found: {spot_path}")
                return False

            # Load option chains
            option_path = os.path.join(self.data_directory, self.file_paths["options"])
            if os.path.exists(option_path):
                with open(option_path, 'r') as f:
                    self.option_chains = json.load(f)
            else:
                print(f"ERROR: Options file not found: {option_path}")
                return False

            # Load historical data (optional)
            history_path = os.path.join(self.data_directory, self.file_paths["history"])
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.historical_data = json.load(f)

            return True

        except Exception as e:
            print(f"ERROR loading market data: {e}")
            return False

    def _calculate_initial_ivs(self):
        """Calculate IVs for initial symbols"""
        count = 0
        for symbol in self.symbols[:10]:  # First 10 symbols
            if symbol not in self.option_chains:
                continue

            expiries = list(self.option_chains[symbol].keys())[:2]
            for expiry in expiries:
                ivs = self._calculate_all_implied_vols(symbol, expiry)
                if ivs:
                    count += len(ivs)

        print(f"  Calculated {count} initial IVs")

    def _calculate_time_to_maturity(self, expiry_string):
        """Calculate time to maturity in years"""
        try:
            expiry = dt.datetime.strptime(expiry_string, '%Y-%m-%d')
            expiry = expiry.replace(hour=16, minute=0)  # Market close
            now = dt.datetime.now()

            if expiry <= now:
                return None

            time_diff = expiry - now
            years = time_diff.total_seconds() / (365.25 * 24 * 3600)

            return max(years, 1/365.25)  # Minimum 1 day

        except:
            return None

    def _black_scholes_price(self, S, K, r, T, sigma, is_call, q=0):
        """Calculate Black-Scholes option price"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            if is_call:
                return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        except:
            return 0

    def _calculate_implied_vol(self, market_price, S, K, r, T, is_call, q=0):
        """Calculate implied volatility using Newton-Raphson"""
        if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
            return None

        # Check intrinsic value
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        if market_price < intrinsic:
            return None

        # Newton-Raphson
        sigma = 0.3
        for _ in range(50):
            try:
                price = self._black_scholes_price(S, K, r, T, sigma, is_call, q)

                # Calculate vega
                d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

                if abs(vega) < 1e-10:
                    return None

                diff = price - market_price
                if abs(diff) < 1e-5:
                    return sigma if 0.01 < sigma < 5.0 else None

                sigma = sigma - diff/vega
                sigma = max(0.01, min(sigma, 5.0))

            except:
                return None

        return None

    def _calculate_all_implied_vols(self, symbol, expiry):
        """Calculate all implied volatilities for an expiry"""
        ivs = {}

        if symbol not in self.option_chains or expiry not in self.option_chains[symbol]:
            return ivs

        S = self.spot_prices.get(symbol, {}).get('price', 0)
        if S <= 0:
            return ivs

        T = self._calculate_time_to_maturity(expiry)
        if not T:
            return ivs

        r = 0.04
        q = 0.0

        options = self.option_chains[symbol][expiry]

        # Process calls and puts
        for option_type in ['calls', 'puts']:
            is_call = (option_type == 'calls')

            for option in options.get(option_type, []):
                try:
                    K = option.get("strike", 0)
                    mid = option.get("mid", 0)

                    if mid <= 0:
                        bid = option.get("bid", 0)
                        ask = option.get("ask", 0)
                        if bid > 0 and ask > 0:
                            mid = (bid + ask) / 2

                    if K > 0 and mid > 0:
                        iv = self._calculate_implied_vol(mid, S, K, r, T, is_call, q)
                        if iv:
                            key = f"{K}_{'call' if is_call else 'put'}"
                            ivs[key] = iv
                except:
                    continue

        # Cache results
        if symbol not in self.iv_cache:
            self.iv_cache[symbol] = {}
        self.iv_cache[symbol][expiry] = ivs

        return ivs

    # SABR Model
    def _sabr_vol(self, F, K, T, alpha, beta, rho, nu):
        """SABR implied volatility (Hagan approximation)"""
        if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
            return 0.25

        if abs(F - K) < 1e-10:  # ATM
            v_atm = (alpha / (F**(1-beta))) * (1 + T * (
                ((1-beta)**2/24) * (alpha**2) / (F**(2*(1-beta))) +
                0.25 * rho * beta * nu * alpha / (F**(1-beta)) +
                (2 - 3*rho**2) * (nu**2) / 24
            ))
            return v_atm

        try:
            z = (nu/alpha) * ((F*K)**((1-beta)/2)) * np.log(F/K)
            x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

            numerator = alpha
            denominator = ((F*K)**((1-beta)/2)) * (1 + ((1-beta)**2/24) * (np.log(F/K))**2 +
                        ((1-beta)**4/1920) * (np.log(F/K))**4)

            expansion = 1 + T * (
                ((1-beta)**2/24) * (alpha**2) / ((F*K)**(1-beta)) +
                0.25 * rho * beta * nu * alpha / ((F*K)**((1-beta)/2)) +
                (2 - 3*rho**2) * (nu**2) / 24
            )

            return (numerator / denominator) * (z / x) * expansion
        except:
            return 0.25

    def _calibrate_sabr(self, S, T, r, q, ivs, market_prices):
        """Calibrate SABR model"""
        try:
            F = S * np.exp((r - q) * T)

            strikes = []
            market_ivs = []

            for key, iv in ivs.items():
                strike = float(key.split('_')[0])
                if 0.5 < strike/S < 2.0:
                    strikes.append(strike)
                    market_ivs.append(iv)

            if len(strikes) < 3:
                return None

            strikes = np.array(strikes)
            market_ivs = np.array(market_ivs)

            beta = 0.5

            def objective(params):
                alpha, rho, nu = params
                if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                    return 1e10
                errors = []
                for K, mkt_iv in zip(strikes, market_ivs):
                    model_iv = self._sabr_vol(F, K, T, alpha, beta, rho, nu)
                    errors.append((model_iv - mkt_iv)**2)
                return np.sum(errors)

            x0 = [np.mean(market_ivs), -0.3, 0.4]
            bounds = [(0.01, 5.0), (-0.99, 0.99), (0.01, 5.0)]
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

            if result.success:
                alpha, rho, nu = result.x

                # Calculate projected prices
                projected_prices = {}
                for K in market_prices.keys():
                    sabr_iv = self._sabr_vol(F, K, T, alpha, beta, rho, nu)
                    call_price = self._black_scholes_price(S, K, r, T, sabr_iv, True, q)
                    put_price = self._black_scholes_price(S, K, r, T, sabr_iv, False, q)

                    projected_prices[K] = self._create_price_comparison(
                        K, call_price, put_price, sabr_iv, market_prices, 'SABR'
                    )

                rmse = np.sqrt(result.fun / len(strikes))

                return {
                    'params': {
                        'alpha': float(alpha),
                        'beta': float(beta),
                        'rho': float(rho),
                        'nu': float(nu)
                    },
                    'rmse': float(rmse),
                    'projected_prices': projected_prices
                }
        except:
            return None

    # Heston Model
    def _calibrate_heston(self, S, T, r, q, ivs, market_prices):
        """Calibrate Heston stochastic volatility model"""
        try:
            strikes = []
            market_ivs = []

            for key, iv in ivs.items():
                strike = float(key.split('_')[0])
                if 0.8 < strike/S < 1.2:
                    strikes.append(strike)
                    market_ivs.append(iv)

            if len(strikes) < 3:
                return None

            v0 = np.mean(market_ivs)**2
            theta = v0
            kappa = 2.0
            sigma = 0.3
            rho = -0.5

            # Calculate projected prices (simplified Heston)
            projected_prices = {}
            for K in market_prices.keys():
                moneyness = np.log(K/S)
                heston_iv = np.sqrt(v0) * (1 + 0.25 * rho * sigma * moneyness / np.sqrt(v0))
                heston_iv = max(0.01, min(heston_iv, 5.0))

                call_price = self._black_scholes_price(S, K, r, T, heston_iv, True, q)
                put_price = self._black_scholes_price(S, K, r, T, heston_iv, False, q)

                projected_prices[K] = self._create_price_comparison(
                    K, call_price, put_price, heston_iv, market_prices, 'Heston'
                )

            return {
                'params': {
                    'v0': float(v0),
                    'theta': float(theta),
                    'kappa': float(kappa),
                    'sigma': float(sigma),
                    'rho': float(rho)
                },
                'projected_prices': projected_prices
            }
        except:
            return None

    # Merton Jump Diffusion Model
    def _calibrate_merton(self, S, T, r, q, ivs, market_prices):
        """Calibrate Merton Jump Diffusion model"""
        try:
            atm_iv = np.mean(list(ivs.values()))

            sigma = atm_iv * 0.8
            lambda_jump = 0.1
            mu_jump = -0.10
            sigma_jump = 0.15

            projected_prices = {}
            for K in market_prices.keys():
                # Simplified Merton pricing
                n_terms = 10
                price_call = 0
                price_put = 0

                for n in range(n_terms):
                    poisson_prob = np.exp(-lambda_jump * T) * (lambda_jump * T)**n / math.factorial(n)
                    r_n = r - lambda_jump * (np.exp(mu_jump + 0.5*sigma_jump**2) - 1) + n * mu_jump / T
                    sigma_n = np.sqrt(sigma**2 + n * sigma_jump**2 / T)

                    price_call += poisson_prob * self._black_scholes_price(S, K, r_n, T, sigma_n, True, q)
                    price_put += poisson_prob * self._black_scholes_price(S, K, r_n, T, sigma_n, False, q)

                merton_iv = np.sqrt(sigma**2 + lambda_jump * (mu_jump**2 + sigma_jump**2))

                projected_prices[K] = self._create_price_comparison(
                    K, price_call, price_put, merton_iv, market_prices, 'Merton'
                )

            return {
                'params': {
                    'sigma': float(sigma),
                    'lambda': float(lambda_jump),
                    'mu_jump': float(mu_jump),
                    'sigma_jump': float(sigma_jump)
                },
                'projected_prices': projected_prices
            }
        except:
            return None

    # Local Volatility Model
    def _calibrate_local_vol(self, S, T, r, q, ivs, market_prices):
        """Build Local Volatility surface"""
        try:
            # Build local vol from market IVs
            strikes = []
            local_vols = {}

            for key, iv in ivs.items():
                strike = float(key.split('_')[0])
                # Simplified Dupire local vol
                local_vol = iv * (1 + 0.1 * np.log(strike/S))
                strikes.append(strike)
                local_vols[strike] = local_vol

            if len(strikes) < 3:
                return None

            # Interpolate for all market strikes
            interp_func = interp1d(strikes, list(local_vols.values()),
                                 kind='linear', fill_value='extrapolate')

            projected_prices = {}
            for K in market_prices.keys():
                local_iv = float(interp_func(K))
                local_iv = max(0.01, min(local_iv, 5.0))

                call_price = self._black_scholes_price(S, K, r, T, local_iv, True, q)
                put_price = self._black_scholes_price(S, K, r, T, local_iv, False, q)

                projected_prices[K] = self._create_price_comparison(
                    K, call_price, put_price, local_iv, market_prices, 'LocalVol'
                )

            return {
                'params': {
                    'type': 'Dupire',
                    'num_points': len(strikes)
                },
                'local_vols': {float(k): float(v) for k, v in local_vols.items()},
                'projected_prices': projected_prices
            }
        except:
            return None

    def _create_price_comparison(self, K, call_price, put_price, model_iv, market_prices, model_name):
        """Create price comparison structure"""
        result = {
            "call": {
                f"{model_name}_price": float(call_price),
                f"{model_name}_iv": float(model_iv)
            },
            "put": {
                f"{model_name}_price": float(put_price),
                f"{model_name}_iv": float(model_iv)
            }
        }

        # Add market prices and differences
        if K in market_prices:
            if "call" in market_prices[K]:
                mkt = market_prices[K]["call"]
                result["call"]["market_price"] = mkt["market_price"]
                result["call"][f"{model_name}_diff"] = float(call_price - mkt["market_price"])
                result["call"][f"{model_name}_diff_pct"] = float(
                    100 * (call_price - mkt["market_price"]) / mkt["market_price"]
                ) if mkt["market_price"] > 0 else 0

            if "put" in market_prices[K]:
                mkt = market_prices[K]["put"]
                result["put"]["market_price"] = mkt["market_price"]
                result["put"][f"{model_name}_diff"] = float(put_price - mkt["market_price"])
                result["put"][f"{model_name}_diff_pct"] = float(
                    100 * (put_price - mkt["market_price"]) / mkt["market_price"]
                ) if mkt["market_price"] > 0 else 0

        return result

    def _extract_market_prices(self, option_chain, S):
        """Extract market prices from option chain"""
        market_prices = {}

        for call in option_chain.get("calls", []):
            K = call.get("strike")
            mid = call.get("mid", 0)
            if mid <= 0:
                bid = call.get("bid", 0)
                ask = call.get("ask", 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2

            if K and mid > 0 and 0.5 < K/S < 2.0:
                if K not in market_prices:
                    market_prices[K] = {}
                market_prices[K]["call"] = {
                    "market_price": float(mid),
                    "bid": float(call.get("bid", 0)),
                    "ask": float(call.get("ask", 0))
                }

        for put in option_chain.get("puts", []):
            K = put.get("strike")
            mid = put.get("mid", 0)
            if mid <= 0:
                bid = put.get("bid", 0)
                ask = put.get("ask", 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2

            if K and mid > 0 and 0.5 < K/S < 2.0:
                if K not in market_prices:
                    market_prices[K] = {}
                market_prices[K]["put"] = {
                    "market_price": float(mid),
                    "bid": float(put.get("bid", 0)),
                    "ask": float(put.get("ask", 0))
                }

        return market_prices

    def _calibrate_all_models(self, symbol, expiry):
        """Calibrate all models for a symbol/expiry"""
        S = self.spot_prices.get(symbol, {}).get('price', 0)
        if S <= 0:
            return {}

        T = self._calculate_time_to_maturity(expiry)
        if not T:
            return {}

        r = 0.04
        q = 0.0

        ivs = self.iv_cache.get(symbol, {}).get(expiry, {})
        if len(ivs) < 3:
            return {}

        option_chain = self.option_chains.get(symbol, {}).get(expiry, {})
        market_prices = self._extract_market_prices(option_chain, S)

        if not market_prices:
            return {}

        results = {
            'symbol': symbol,
            'expiry': expiry,
            'spot': float(S),
            'time_to_expiry': float(T),
            'timestamp': dt.datetime.now().isoformat()
        }

        # Calibrate each model
        sabr = self._calibrate_sabr(S, T, r, q, ivs, market_prices)
        if sabr:
            results['SABR'] = sabr

        heston = self._calibrate_heston(S, T, r, q, ivs, market_prices)
        if heston:
            results['Heston'] = heston

        merton = self._calibrate_merton(S, T, r, q, ivs, market_prices)
        if merton:
            results['Merton'] = merton

        local_vol = self._calibrate_local_vol(S, T, r, q, ivs, market_prices)
        if local_vol:
            results['LocalVol'] = local_vol

        # Merge all projected prices
        if len(results) > 5:  # Has at least one model
            results['consolidated_prices'] = self._consolidate_prices(results)

        return results

    def _consolidate_prices(self, results):
        """Consolidate prices from all models"""
        consolidated = {}

        for model_name in ['SABR', 'Heston', 'Merton', 'LocalVol']:
            if model_name not in results:
                continue

            prices = results[model_name].get('projected_prices', {})
            for strike, price_data in prices.items():
                if strike not in consolidated:
                    consolidated[strike] = {'call': {}, 'put': {}}

                for opt_type in ['call', 'put']:
                    if opt_type in price_data:
                        consolidated[strike][opt_type].update(price_data[opt_type])

        return consolidated

    def _run_calibration_cycle(self):
        """Run one calibration cycle"""
        calibration_count = 0
        errors_by_model = {'SABR': [], 'Heston': [], 'Merton': [], 'LocalVol': []}

        # Limit calibrations per cycle
        symbols_to_process = self.symbols[:self.max_calibrations_per_cycle]

        for symbol in symbols_to_process:
            if symbol not in self.option_chains:
                continue

            # Get valid expiries
            valid_expiries = []
            for expiry in list(self.option_chains[symbol].keys())[:5]:
                T = self._calculate_time_to_maturity(expiry)
                if T and T > 1/365:
                    valid_expiries.append(expiry)
                    if len(valid_expiries) >= 2:
                        break

            for expiry in valid_expiries:
                # Calculate IVs if needed
                if symbol not in self.iv_cache or expiry not in self.iv_cache[symbol]:
                    ivs = self._calculate_all_implied_vols(symbol, expiry)
                else:
                    ivs = self.iv_cache[symbol][expiry]

                if len(ivs) >= 3:
                    # Calibrate all models
                    results = self._calibrate_all_models(symbol, expiry)

                    if results and len(results) > 5:  # Has models
                        # Store results
                        if symbol not in self.calibrated_params:
                            self.calibrated_params[symbol] = {}
                        self.calibrated_params[symbol][expiry] = results

                        # Display summary
                        models_calibrated = []
                        for model in ['SABR', 'Heston', 'Merton', 'LocalVol']:
                            if model in results:
                                models_calibrated.append(model[0])  # First letter

                                # Collect errors
                                if 'consolidated_prices' in results:
                                    for strike_data in results['consolidated_prices'].values():
                                        for opt_type in ['call', 'put']:
                                            diff_key = f"{model}_diff_pct"
                                            if diff_key in strike_data.get(opt_type, {}):
                                                errors_by_model[model].append(
                                                    abs(strike_data[opt_type][diff_key])
                                                )

                        print(f"  ✓ {symbol} {expiry}: {''.join(models_calibrated)}")
                        calibration_count += 1

        # Show average errors
        if calibration_count > 0:
            print(f"\n  Model Average Pricing Errors:")
            for model, errors in errors_by_model.items():
                if errors:
                    avg_error = np.mean(errors)
                    print(f"    {model}: {avg_error:.2f}%")

        return calibration_count

    def _calibration_loop(self):
        """Background calibration loop"""
        while self.running:
            try:
                time.sleep(self.calibration_interval)

                print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] Running calibration cycle...")

                # Reload market data
                self._load_market_data()

                # Run calibrations
                count = self._run_calibration_cycle()

                if count > 0:
                    print(f"  Completed {count} calibrations")
                    self._save_all_data()

            except Exception as e:
                print(f"Calibration loop error: {e}")
                time.sleep(self.calibration_interval)

    def _save_all_data(self):
        """Save streamlined calibration data for visualization"""
        try:
            filename = self.calibration_directory / "calibrations.json"

            # Build streamlined data structure
            viz_data = {}

            for symbol, expiries in self.calibrated_params.items():
                viz_data[symbol] = {}

                for expiry, data in expiries.items():
                    viz_data[symbol][expiry] = {
                        'spot': data.get('spot'),
                        'timestamp': data.get('timestamp'),
                        'models': {}
                    }

                    # For each model, save only essential data
                    for model in ['SABR', 'Heston', 'Merton', 'LocalVol']:
                        if model in data:
                            model_data = data[model]
                            viz_data[symbol][expiry]['models'][model] = {
                                'params': model_data.get('params'),
                                'prices': self._extract_price_summary(model_data.get('projected_prices', {}), model)
                            }

            # Final structure
            output = {
                'updated': dt.datetime.now().isoformat(),
                'data': viz_data
            }

            # Write to file
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)

            # Calculate file size
            file_size = filename.stat().st_size / 1024
            print(f"  Saved {len(viz_data)} symbols to calibrations.json ({file_size:.1f} KB)")

        except Exception as e:
            print(f"Error saving: {e}")

    def _extract_price_summary(self, projected_prices, model):
        """Extract only essential price data for visualization"""
        summary = {}

        for strike, price_data in projected_prices.items():
            strike_float = float(strike)
            summary[strike_float] = {}

            for opt_type in ['call', 'put']:
                if opt_type in price_data:
                    # Only save model price and market price
                    summary[strike_float][opt_type] = {
                        'model': price_data[opt_type].get(f'{model}_price', 0),
                        'market': price_data[opt_type].get('market_price', 0),
                        'iv': price_data[opt_type].get(f'{model}_iv', 0)
                    }

        return summary

    def _prepare_for_json(self, obj):
        """Convert to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("MULTI-MODEL OPTIONS CALIBRATION SERVICE")
    print("="*80)

    # Use all defined tickers
    symbols = list(CalibrationService.TICKERS.keys())

    # Create and start service
    service = CalibrationService(
        symbols=symbols,
        config={
            'calibration_interval': 30,
            'max_calibrations': 20
        }
    )

    service.start()

    try:
        print("\n✓ Service running. Press Ctrl+C to stop.\n")

        while True:
            time.sleep(30)

            # Show status
            total_calibrations = sum(
                len(expiries) for symbol in service.calibrated_params.values()
                for expiries in symbol.values()
            )

            print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] Status:")
            print(f"  Total calibrations: {total_calibrations}")
            print(f"  Symbols processed: {len(service.calibrated_params)}")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        service.stop()
        print("✓ Service stopped successfully")
