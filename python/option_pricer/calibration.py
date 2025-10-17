from market_data import MarketDataService
from __init__ import *
import threading as th
import queue as qu
from queue import PriorityQueue as pq
import os
from fredapi import Fred
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import math
import numpy as np
import json
from scipy.optimize import minimize
from scipy.interpolate import griddata, interp1d
import time
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# retrieve api key from FRED website
fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))

class CalibrationService:

    # Initialization Methods
    def __init__(self, symbols, config=None):
        """
        Initializes all data structures
        Sets up file monitoring system
        Configures threading primitives
        Loads configuration settings
        Validates data directory exists
        Prepares calibration service for startup
        """
        # Data Source Variables
        self.data_directory = "./market_data_realtime"
        self.file_paths = {
            'spot': "spot_data.json",
            'options': "option_chains.json",
            'history': "historical_data.json",
            'meta': "metadata.json"
        }
        self.file_timestamps = {}
        self.symbols = symbols

        # Extracted Market Data Variables
        self.spot_prices = {}
        self.option_chains = {}
        self.historical_data = {}
        self.risk_free_rate = {
            '1m': self._get_rf_rate("DGS1MO"),
            '3m': self._get_rf_rate("DGS3MO"),
            '1y': self._get_rf_rate("DGS1")
        }

        # Calibration Output Variables
        self.calibrated_params = {}
        self.vol_surfaces = {}
        self.implied_volatilities = {}

        # Threading Control Variables
        self.running = False
        self.data_lock = th.Lock()
        self.calibration_thread = None
        self.calibration_workers = ThreadPoolExecutor(max_workers=8)
        self.calibration_queue = pq()

        # Configuration Variables
        self.calibration_interval = config.get('calibration_interval', 5) if config else 5
        self.models_to_calibrate = config.get('models', ['SABR', 'Heston', 'LocalVol']) if config else ['SABR', 'Heston', 'LocalVol']
        self.min_data_points = 5
        self.max_expiries_per_symbol = 15
        self.priority_symbols = ['SPY', 'QQQ', 'AAPL']
        self.save_calibrations = True
        self.calibration_directory = Path('./calibration_data_realtime')
        self.calibration_directory.mkdir(exist_ok=True)

        # Cache and Performance Variables
        self.iv_cache = {}
        self.last_calibration_time = {}
        self.calibration_staleness_threshold = 3600  # 1 hour in seconds
        self.spot_change_threshold = 0.01

        # Validation and Quality Control Variables
        self.parameter_bounds = {
            'SABR': {
                'alpha': (0.001, 5.0),
                'beta': (0.0, 1.0),
                'rho': (-0.99, 0.99),
                'nu': (0.001, 5.0)
            },
            'Heston': {
                'kappa': (0.1, 10.0),
                'theta': (0.01, 1.0),
                'sigma': (0.01, 2.0),
                'rho': (-0.99, 0.99),
                'v0': (0.01, 1.0)
            }
        }
        self.rmse_threshold = {'short': 0.02, 'medium': 0.03, 'long': 0.05}
        self.failed_calibrations = {}
        self.calibration_statistics = {}

    def _get_rf_rate(self, series_id):
        """Get risk-free rate from FRED"""
        try:
            data = fred.get_series_latest_release(series_id)
            if len(data) > 0:
                return data.iloc[-1] / 100  # Convert percentage to decimal
        except:
            pass
        # Fallback rates if FRED fails
        defaults = {'DGS1MO': 0.05, 'DGS3MO': 0.05, 'DGS1': 0.05}
        return defaults.get(series_id, 0.05)

    def start(self):
        """
        Performs initial data load from files
        Calculates initial implied volatilities
        Runs first calibration pass for all symbols
        Starts calibration thread
        Returns when service is ready
        """
        # Initial data load
        self._load_market_data()

        # Calculate initial implied volatilities for all symbols
        for symbol in self.symbols:
            if symbol not in self.option_chains:
                print(f"Warning: {symbol} not found in option chains")
                continue

            for expiry in list(self.option_chains[symbol].keys())[:self.max_expiries_per_symbol]:
                self._calculate_all_implied_vols(symbol, expiry)

        # Run initial calibration pass for priority symbols
        for symbol in self.priority_symbols:
            if symbol not in self.option_chains:
                continue

            # Get first few expiries for initial calibration
            expiries = sorted(list(self.option_chains[symbol].keys()))[:3]

            for expiry in expiries:
                # Schedule initial SABR calibrations (fastest model)
                self.calibration_queue.put((-1000, (symbol, expiry, 'SABR')))

        # Process initial calibrations synchronously for immediate availability
        while not self.calibration_queue.empty():
            try:
                _, (symbol, expiry, model) = self.calibration_queue.get_nowait()
                self._worker_calibration_task(symbol, expiry, model)
            except qu.Empty:
                break
            except Exception as e:
                print(f"Initial calibration failed: {e}")
                continue

        # Set running flag
        self.running = True

        # Start the main calibration loop thread
        self.calibration_thread = th.Thread(
            target=self._calibration_loop,
            name="CalibrationLoop",
            daemon=True
        )
        self.calibration_thread.start()

        print(f"Calibration service started with {len(self.symbols)} symbols")
        print(f"Worker pool: {self.calibration_workers._max_workers} threads")
        print(f"Initial calibrations completed for {len(self.calibrated_params)} symbols")

    def stop(self):
        """
        Sets running flag to False
        Gracefully stops calibration thread
        Shuts down worker pool
        Saves current calibrated parameters
        """
        print("Stopping calibration service...")

        # Set flag to stop loop
        self.running = False

        # Wait for calibration thread to finish
        if self.calibration_thread and self.calibration_thread.is_alive():
            self.calibration_thread.join(timeout=self.calibration_interval + 1)

        # Shutdown worker pool gracefully
        self.calibration_workers.shutdown(wait=True, timeout=30)

        # Save final calibrations
        if self.save_calibrations:
            self._save_calibrations()

        # Clear the queue
        while not self.calibration_queue.empty():
            try:
                self.calibration_queue.get_nowait()
            except qu.Empty:
                break

        print(f"Calibration service stopped. Saved {len(self.calibrated_params)} calibrations")

    # File Monitoring Methods
    def _check_file_updates(self):
        """
        Checks if new market data available since last calibration
        Returns True if any market data file is newer
        """
        spot_path = os.path.join(self.data_directory, self.file_paths["spot"])
        options_path = os.path.join(self.data_directory, self.file_paths["options"])

        if not os.path.exists(spot_path) or not os.path.exists(options_path):
            return False

        # Get file modification times
        spot_mtime = dt.datetime.fromtimestamp(os.path.getmtime(spot_path))
        options_mtime = dt.datetime.fromtimestamp(os.path.getmtime(options_path))

        # Get most recent update across all files
        last_data_update = max(spot_mtime, options_mtime)

        # Store timestamps
        self.file_timestamps['spot'] = spot_mtime
        self.file_timestamps['options'] = options_mtime

        # If never calibrated, return True
        if not self.last_calibration_time:
            return True

        # Check if data is newer than any last calibration
        for symbol_times in self.last_calibration_time.values():
            if isinstance(symbol_times, dict):
                for expiry_time in symbol_times.values():
                    if last_data_update > expiry_time:
                        return True

        return False

    def _load_market_data(self):
        """
        Reads JSON files when changes detected
        Validates data integrity
        Updates spot_prices, option_chains, historical_data
        Thread-safe with lock
        """
        try:
            spot_path = os.path.join(self.data_directory, self.file_paths["spot"])
            with open(spot_path, 'r') as f:
                spot_data = json.load(f)

            option_path = os.path.join(self.data_directory, self.file_paths["options"])
            with open(option_path, 'r') as f:
                option_data = json.load(f)

            history_path = os.path.join(self.data_directory, self.file_paths["history"])
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
            else:
                history_data = {}

        except Exception as e:
            print(f"Failed to load JSON file: {e}")
            return

        # Validate and clean option data
        for ticker, ticker_data in option_data.items():
            for date, date_data in ticker_data.items():
                # Filter out invalid options
                if "calls" in date_data:
                    date_data["calls"] = [
                        option for option in date_data.get("calls", [])
                        if self._validate_option_data(option)
                    ]
                if "puts" in date_data:
                    date_data["puts"] = [
                        option for option in date_data.get("puts", [])
                        if self._validate_option_data(option)
                    ]

        # Update instance variables with lock
        with self.data_lock:
            self.spot_prices = spot_data
            self.option_chains = option_data
            self.historical_data = history_data

    # Implied Volatility Calculation Methods
    def _calculate_implied_vol(self, market_price, S, K, r, T, is_call, q=0):
        """
        Implements Newton-Raphson iteration for IV calculation
        Returns implied volatility or None
        """
        if T <= 0 or market_price <= 0:
            return None

        initial_guess = 0.2
        sigma = initial_guess
        max_iter = 100
        tolerance = 1e-6

        for i in range(max_iter):
            try:
                # Calculate BS price and vega at current sigma
                bs_price = black_scholes(S, K, r, q, sigma, T, is_call)
                vega_val = vega(S, K, r, q, sigma, T, is_call)

                diff = bs_price - market_price

                # Check convergence
                if abs(diff) < tolerance:
                    if 0.01 < sigma < 3.0:
                        return sigma
                    return None

                # Avoid division by zero
                if abs(vega_val) < 1e-10:
                    return None

                # Newton-Raphson update
                sigma = sigma - diff / vega_val

                # Keep sigma positive and reasonable
                sigma = max(0.01, min(sigma, 5.0))

            except:
                return None

        return None

    def _calculate_all_implied_vols(self, symbol, expiry):
        """
        Processes entire option chain for one expiry
        Returns dictionary of (strike, type) → implied vol
        """
        ivs = {}

        if symbol not in self.option_chains or expiry not in self.option_chains[symbol]:
            return ivs

        options = self.option_chains[symbol][expiry]
        spot_price = self.spot_prices.get(symbol, 0)

        if spot_price <= 0:
            return ivs

        tte = self._calculate_time_to_maturity(expiry)
        if tte is None or tte <= 0:
            return ivs

        rf_rate = self._get_risk_free_rate_for_maturity(tte)
        div_yield = self._calculate_dividend_yield(symbol)

        # Process calls
        for option in options.get("calls", []):
            try:
                if option.get("mid", 0) > 0:
                    iv_value = self._calculate_implied_vol(
                        option["mid"],
                        spot_price,
                        option["strike"],
                        rf_rate,
                        tte,
                        True,
                        div_yield
                    )

                    if iv_value and self._validate_implied_vol(
                        iv_value, spot_price, option["strike"],
                        rf_rate, tte, True, option["mid"], div_yield
                    ):
                        ivs[(option["strike"], 'call')] = iv_value
            except:
                continue

        # Process puts
        for option in options.get("puts", []):
            try:
                if option.get("mid", 0) > 0:
                    iv_value = self._calculate_implied_vol(
                        option["mid"],
                        spot_price,
                        option["strike"],
                        rf_rate,
                        tte,
                        False,
                        div_yield
                    )

                    if iv_value and self._validate_implied_vol(
                        iv_value, spot_price, option["strike"],
                        rf_rate, tte, False, option["mid"], div_yield
                    ):
                        ivs[(option["strike"], 'put')] = iv_value
            except:
                continue

        # Cache results
        if symbol not in self.iv_cache:
            self.iv_cache[symbol] = {}
        self.iv_cache[symbol][expiry] = ivs

        return ivs

    def _validate_implied_vol(self, iv, S, K, r, T, is_call, option_price, q=0):
        """
        Validates calculated IV makes sense
        """
        if iv is None or iv <= 0.01 or iv >= 3.0:
            return False

        # Check arbitrage bounds
        option = {'mid': option_price}
        if not self._check_arbitrage_bounds(option, S, K, r, T, is_call, q):
            return False

        # Check vega is significant
        try:
            vega_val = vega(S, K, r, q, iv, T, is_call)
            if vega_val < 0.001:
                return False
        except:
            return False

        return True

    # Data Extraction Methods
    def _extract_calibration_data(self, symbol, expiry):
        """
        Pulls all data needed for one calibration
        Returns structured calibration-ready dataset
        """
        try:
            S = self.spot_prices.get(symbol)
            if not S or S <= 0:
                return None

            T = self._calculate_time_to_maturity(expiry)
            if T is None or T <= 0:
                return None

            r = self._get_risk_free_rate_for_maturity(T)
            q = self._calculate_dividend_yield(symbol)

            # Get option chains
            if symbol not in self.option_chains or expiry not in self.option_chains[symbol]:
                return None

            option_chain = self.option_chains[symbol][expiry]

            # Filter options
            filtered_calls = self._filter_options_for_calibration(
                option_chain.get("calls", []), S
            )
            filtered_puts = self._filter_options_for_calibration(
                option_chain.get("puts", []), S
            )

            # Calculate implied volatilities
            ivs = self._calculate_all_implied_vols(symbol, expiry)

            # Check if we have sufficient data
            if len(ivs) < self.min_data_points:
                return None

            calibration_data = {
                "S": S,
                "r": r,
                "q": q,
                "T": T,
                "calls": filtered_calls,
                "puts": filtered_puts,
                "ivs": ivs
            }

            return calibration_data

        except Exception as e:
            print(f"Error extracting calibration data for {symbol} {expiry}: {e}")
            return None

    def _filter_options_for_calibration(self, options_list, S):
        """
        Filters options for quality and liquidity
        """
        valid_options = []

        for option in options_list:
            if not self._validate_option_data(option):
                continue

            # Liquidity filters
            if option.get("openInterest", 0) < 10:
                continue
            if option.get("volume", 0) < 5 and option.get("openInterest", 0) < 100:
                continue

            # Moneyness filter
            ratio = option["strike"] / S
            if ratio < 0.5 or ratio > 2.0:
                continue

            # Prioritize ATM
            if 0.95 <= ratio <= 1.05:
                valid_options.insert(0, option)
            else:
                valid_options.append(option)

        # Sort by moneyness
        return sorted(valid_options, key=lambda opt: abs(opt["strike"] / S - 1.0))

    def _calculate_forward_price(self, S, r, T, q=0):
        """Computes forward price"""
        if T <= 0:
            return S
        return S * math.exp((r - q) * T)

    # SABR Calibration
    def _calibrate_sabr(self, symbol, expiry, calibration_data):
        """
        Main SABR calibration entry point
        Returns calibrated (alpha, beta, rho, nu) or None
        """
        try:
            market_data = self._prepare_sabr_market_data(calibration_data)
            if not market_data or len(market_data['strikes']) < 3:
                return None

            S = calibration_data['S']
            r = calibration_data['r']
            q = calibration_data['q']
            T = calibration_data['T']

            # Forward price
            F = self._calculate_forward_price(S, r, T, q)

            # Fixed beta for equities
            beta = 0.5

            # Call C++ calibration
            alpha, beta_out, rho, nu = sabr_calibrate(
                market_data['ivs'],
                market_data['strikes'],
                F,
                T,
                beta
            )

            params = (alpha, beta_out, rho, nu)

            if not self._validate_sabr_parameters(alpha, beta_out, rho, nu):
                return None

            # Calculate fit quality
            quality = self._calculate_sabr_fit_quality(params, market_data, F, T)

            if not self._accept_calibration(quality['rmse'], T, symbol, expiry):
                return None

            return {
                'params': params,
                'spot_at_calibration': S,
                'forward': F,
                'time_to_expiry': T,
                'rmse': quality['rmse'],
                'calibration_time': dt.datetime.now().isoformat()
            }

        except Exception as e:
            print(f'SABR calibration failed for {symbol} {expiry}: {e}')
            return None

    def _prepare_sabr_market_data(self, calibration_data):
        """Formats data for SABR calibration"""
        S = calibration_data["S"]
        ivs = calibration_data["ivs"]

        market_data = []
        for (strike, option_type), iv in ivs.items():
            market_data.append({
                'strike': strike,
                'iv': iv,
                'moneyness': strike / S,
                'type': option_type
            })

        # Sort by moneyness
        market_data.sort(key=lambda x: x['moneyness'])

        # Limit points for stability
        max_points = 30
        if len(market_data) > max_points:
            # Keep ATM and sample rest
            atm_data = [d for d in market_data if 0.95 <= d['moneyness'] <= 1.05]
            otm_data = [d for d in market_data if d['moneyness'] < 0.95 or d['moneyness'] > 1.05]

            step = max(1, len(otm_data) // (max_points - len(atm_data)))
            otm_sampled = otm_data[::step]

            market_data = sorted(atm_data + otm_sampled, key=lambda x: x['moneyness'])

        strikes = np.array([d['strike'] for d in market_data])
        ivs_array = np.array([d['iv'] for d in market_data])

        return {
            'strikes': strikes,
            'ivs': ivs_array,
            'spot': S,
            'T': calibration_data['T'],
            'full_data': market_data
        }

    def _validate_sabr_parameters(self, alpha, beta, rho, nu):
        """Validates SABR parameters"""
        if alpha <= 0 or nu <= 0:
            return False
        if beta < 0 or beta > 1:
            return False
        if rho <= -1 or rho >= 1:
            return False
        if alpha > 5.0 or nu > 5.0:
            return False
        return True

    def _calculate_sabr_fit_quality(self, params, market_data, F, T):
        """Computes RMSE between model and market"""
        alpha, beta, rho, nu = params
        strikes = market_data["strikes"]
        market_ivs = market_data["ivs"]

        model_ivs = []
        for strike in strikes:
            try:
                sabr_iv = sabr_implied_vol(F, strike, T, alpha, beta, rho, nu)
                model_ivs.append(sabr_iv)
            except:
                return {'rmse': float('inf'), 'quality': 'failed'}

        model_ivs = np.array(model_ivs)
        residuals = model_ivs - market_ivs
        rmse = np.sqrt(np.mean(residuals ** 2))

        if rmse < 0.01:
            quality = 'excellent'
        elif rmse < 0.02:
            quality = 'good'
        elif rmse < 0.05:
            quality = 'acceptable'
        else:
            quality = 'poor'

        return {
            'rmse': rmse,
            'max_error': np.max(np.abs(residuals)),
            'mean_error': np.mean(np.abs(residuals)),
            'quality': quality
        }

    def _accept_calibration(self, rmse, T, symbol, expiry):
        """Accept or reject calibration based on RMSE"""
        if T < 0.25:
            threshold = 0.02
        elif T < 1.0:
            threshold = 0.03
        else:
            threshold = 0.05

        if rmse > threshold:
            print(f"Calibration rejected for {symbol} {expiry}: RMSE {rmse:.4f} > {threshold:.4f}")
            return False
        return True

    # Heston Calibration
    def _calibrate_heston(self, symbol, calibration_data):
        """Multi-expiry Heston calibration"""
        try:
            # Prepare market data across all expiries
            all_market_data = self._prepare_heston_market_data(symbol)
            if not all_market_data:
                return None

            S = calibration_data['S']

            # Initial parameters
            v0 = self._get_historical_volatility(symbol) ** 2
            kappa = 2.0
            theta = v0
            sigma = 0.3
            rho = -0.5

            initial_params = np.array([kappa, theta, sigma, rho, v0])

            # Bounds
            bounds = [
                (0.1, 10.0),   # kappa
                (0.01, 1.0),   # theta
                (0.01, 2.0),   # sigma
                (-0.99, 0.99), # rho
                (0.01, 1.0)    # v0
            ]

            # Optimize
            result = minimize(
                self._heston_objective_function,
                initial_params,
                args=(all_market_data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )

            if not result.success:
                return None

            kappa, theta, sigma, rho, v0 = result.x

            if not self._validate_heston_parameters(kappa, theta, sigma, rho, v0):
                return None

            return {
                'params': tuple(result.x),
                'spot_at_calibration': S,
                'rmse': np.sqrt(result.fun / len(all_market_data['market_prices'])),
                'calibration_time': dt.datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Heston calibration failed: {e}")
            return None

    def _prepare_heston_market_data(self, symbol):
        """Prepare data across all expiries for Heston"""
        all_data = {
            'spot': self.spot_prices.get(symbol),
            'strikes': [],
            'expiries': [],
            'market_prices': [],
            'option_types': [],
            'rates': [],
            'dividends': []
        }

        if not all_data['spot']:
            return None

        q = self._calculate_dividend_yield(symbol)

        for expiry in list(self.option_chains.get(symbol, {}).keys())[:5]:  # Use first 5 expiries
            T = self._calculate_time_to_maturity(expiry)
            if not T or T <= 0:
                continue

            r = self._get_risk_free_rate_for_maturity(T)
            ivs = self.iv_cache.get(symbol, {}).get(expiry, {})

            for (strike, option_type), iv in ivs.items():
                # Calculate market price from IV
                is_call = (option_type == 'call')
                market_price = black_scholes(
                    all_data['spot'], strike, r, q, iv, T, is_call
                )

                all_data['strikes'].append(strike)
                all_data['expiries'].append(T)
                all_data['market_prices'].append(market_price)
                all_data['option_types'].append(option_type)
                all_data['rates'].append(r)
                all_data['dividends'].append(q)

        if len(all_data['market_prices']) < 10:
            return None

        # Convert to arrays
        for key in ['strikes', 'expiries', 'market_prices', 'rates', 'dividends']:
            all_data[key] = np.array(all_data[key])

        return all_data

    def _heston_objective_function(self, params, market_data):
        """Heston objective function"""
        kappa, theta, sigma, rho, v0 = params

        if not self._validate_heston_parameters(kappa, theta, sigma, rho, v0):
            return 1e10

        S = market_data['spot']
        squared_errors = []

        for i in range(len(market_data['market_prices'])):
            try:
                K = market_data['strikes'][i]
                T = market_data['expiries'][i]
                r = market_data['rates'][i]
                q = market_data['dividends'][i]
                market_price = market_data['market_prices'][i]
                is_call = (market_data['option_types'][i] == 'call')

                heston_price = heston_model(S, K, r, q, T, kappa, theta, sigma, rho, v0, is_call)
                error = (heston_price - market_price) ** 2
                squared_errors.append(error)
            except:
                squared_errors.append(1e6)

        return np.sum(squared_errors)

    def _validate_heston_parameters(self, kappa, theta, sigma, rho, v0):
        """Validate Heston parameters"""
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 < 0:
            return False
        if rho <= -1 or rho >= 1:
            return False
        if 2 * kappa * theta < sigma ** 2:  # Feller condition
            return False
        if kappa > 50 or sigma > 5.0 or theta > 2.0:
            return False
        return True

    # Local Volatility
    def _calibrate_local_volatility(self, symbol):
        """Build local volatility surface"""
        try:
            surface = self._build_volatility_surface(symbol)
            if not surface or len(surface) < 10:
                return None

            S = self.spot_prices[symbol]

            # Create grid
            strikes = sorted(set(p['K'] for p in surface))
            expiries = sorted(set(p['T'] for p in surface))

            local_vol_grid = []

            for T in expiries:
                for K in strikes:
                    iv = self._interpolate_iv(surface, K, T)
                    if iv:
                        sigma_local = self._dupire_local_vol(S, K, T, iv, surface)
                        if sigma_local and sigma_local > 0:
                            local_vol_grid.append({
                                'K': K,
                                'T': T,
                                'local_vol': sigma_local
                            })

            if len(local_vol_grid) < 10:
                return None

            return {
                'grid': local_vol_grid,
                'spot_at_calibration': S,
                'calibration_time': dt.datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Local vol calibration failed: {e}")
            return None

    def _build_volatility_surface(self, symbol):
        """Build complete volatility surface"""
        if symbol not in self.option_chains:
            return None

        S = self.spot_prices.get(symbol)
        if not S:
            return None

        surface_points = []

        for expiry in self.option_chains[symbol]:
            T = self._calculate_time_to_maturity(expiry)
            if not T or T < 1/365:
                continue

            ivs = self.iv_cache.get(symbol, {}).get(expiry, {})

            for (strike, option_type), iv in ivs.items():
                # Use OTM options only
                if (option_type == 'call' and strike > S) or (option_type == 'put' and strike < S):
                    surface_points.append({
                        'K': strike,
                        'T': T,
                        'iv': iv
                    })

        return surface_points

    def _interpolate_iv(self, surface, K, T):
        """Interpolate IV from surface"""
        try:
            points = [(p['K'], p['T']) for p in surface]
            values = [p['iv'] for p in surface]

            if len(points) < 4:
                return None

            # Use nearest neighbor for simple interpolation
            min_dist = float('inf')
            closest_iv = None

            for p, v in zip(points, values):
                dist = ((p[0] - K)**2 + (p[1] - T)**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_iv = v

            return closest_iv

        except:
            return None

    def _dupire_local_vol(self, S, K, T, iv, surface):
        """Calculate local vol using Dupire formula (simplified)"""
        try:
            # Simplified Dupire - just return scaled IV for now
            # In production, implement full Dupire with derivatives
            moneyness = K / S
            if 0.8 <= moneyness <= 1.2:
                return iv * 1.1  # Slight adjustment
            else:
                return iv * 1.2  # Larger adjustment for OTM
        except:
            return None

    # Main Calibration Loop
    def _calibration_loop(self):
        """Main calibration loop"""
        while self.running:
            try:
                if self._check_file_updates():
                    self._load_market_data()
                    self._schedule_calibrations()

                    # Process queue with worker pool
                    futures = []

                    while not self.calibration_queue.empty() and len(futures) < 8:
                        try:
                            _, (symbol, expiry, model) = self.calibration_queue.get_nowait()

                            future = self.calibration_workers.submit(
                                self._worker_calibration_task,
                                symbol, expiry, model
                            )
                            futures.append(future)
                        except qu.Empty:
                            break

                    # Wait for completions
                    for future in as_completed(futures, timeout=60):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Calibration task error: {e}")

                    # Save after batch
                    if self.save_calibrations and futures:
                        self._save_calibrations()

                time.sleep(self.calibration_interval)

            except Exception as e:
                print(f"Calibration loop error: {e}")
                time.sleep(self.calibration_interval)

    def _schedule_calibrations(self):
        """Schedule calibrations for all symbols"""
        for symbol in self.symbols:
            if symbol not in self.option_chains:
                continue

            expiries = list(self.option_chains[symbol].keys())[:self.max_expiries_per_symbol]

            for expiry in expiries:
                for model in self.models_to_calibrate:
                    if self._should_recalibrate(symbol, expiry, model):
                        priority = self._calculate_calibration_priority(symbol, expiry, model)
                        self.calibration_queue.put((-priority, (symbol, expiry, model)))

    def _worker_calibration_task(self, symbol, expiry, model_type):
        """Worker task for calibration"""
        try:
            with self.data_lock:
                calibration_data = self._extract_calibration_data(symbol, expiry)

            if not calibration_data:
                return

            start_time = time.time()
            result = None

            if model_type == 'SABR':
                result = self._calibrate_sabr(symbol, expiry, calibration_data)
            elif model_type == 'Heston':
                result = self._calibrate_heston(symbol, calibration_data)
            elif model_type == 'LocalVol':
                result = self._calibrate_local_volatility(symbol)

            duration = time.time() - start_time

            if result:
                with self.data_lock:
                    if symbol not in self.calibrated_params:
                        self.calibrated_params[symbol] = {}
                    if expiry not in self.calibrated_params[symbol]:
                        self.calibrated_params[symbol][expiry] = {}

                    self.calibrated_params[symbol][expiry][model_type] = result

                    if symbol not in self.last_calibration_time:
                        self.last_calibration_time[symbol] = {}
                    self.last_calibration_time[symbol][expiry] = dt.datetime.now()

                    print(f"Calibrated {model_type} for {symbol} {expiry} in {duration:.2f}s")

        except Exception as e:
            print(f"Worker task error for {symbol} {expiry} {model_type}: {e}")

    def _calculate_calibration_priority(self, symbol, expiry, model):
        """Calculate calibration priority"""
        base_priority = 100.0

        # Symbol importance
        symbol_multiplier = 2.0 if symbol in self.priority_symbols else 1.0

        # Time to expiry
        T = self._calculate_time_to_maturity(expiry)
        if T and T < 7/365:
            expiry_multiplier = 1.5
        elif T and T < 30/365:
            expiry_multiplier = 1.3
        else:
            expiry_multiplier = 1.0

        # Model priority
        model_multipliers = {'SABR': 1.2, 'Heston': 1.1, 'LocalVol': 1.0}
        model_multiplier = model_multipliers.get(model, 1.0)

        return base_priority * symbol_multiplier * expiry_multiplier * model_multiplier

    def _should_recalibrate(self, symbol, expiry, model):
        """Check if recalibration needed"""
        if symbol not in self.last_calibration_time:
            return True
        if expiry not in self.last_calibration_time[symbol]:
            return True

        last_time = self.last_calibration_time[symbol][expiry]
        time_diff = (dt.datetime.now() - last_time).total_seconds() / 3600

        # Model-specific thresholds
        thresholds = {'SABR': 1, 'Heston': 4, 'LocalVol': 24}
        return time_diff > thresholds.get(model, 1)

    # Utility Methods
    def _validate_option_data(self, option):
        """Validate option data quality"""
        if not option:
            return False

        required_fields = ["bid", "ask", "mid", "strike"]
        for field in required_fields:
            if field not in option or option[field] is None:
                return False

        if option["bid"] > option["mid"] or option["mid"] > option["ask"]:
            return False
        if option["mid"] <= 0:
            return False

        return True

    def _check_arbitrage_bounds(self, option, S, K, r, T, is_call, q=0):
        """Check no-arbitrage bounds"""
        if T <= 0:
            return False

        discount_factor = math.exp(-r * T)
        dividend_discount = math.exp(-q * T)

        if is_call:
            lower = max(S * dividend_discount - K * discount_factor, 0)
            upper = S * dividend_discount
        else:
            lower = max(K * discount_factor - S * dividend_discount, 0)
            upper = K * discount_factor

        return lower <= option["mid"] <= upper

    def _calculate_time_to_maturity(self, expiry_string):
        """Convert expiry to years"""
        try:
            expiry = dt.datetime.strptime(expiry_string, '%Y-%m-%d %H:%M:%S')
            now = dt.datetime.now()

            if now >= expiry:
                return None

            time_diff = expiry - now
            return time_diff.total_seconds() / (365.25 * 24 * 3600)
        except:
            return None

    def _get_risk_free_rate_for_maturity(self, T):
        """Get appropriate risk-free rate"""
        if T <= 1/12:
            return self.risk_free_rate.get("1m", 0.05)
        elif T <= 1/4:
            return self.risk_free_rate.get("3m", 0.05)
        else:
            return self.risk_free_rate.get("1y", 0.05)

    def _get_historical_volatility(self, symbol, window_days=30):
        """Calculate historical volatility"""
        try:
            if symbol not in self.historical_data:
                return 0.25  # Default

            dates = sorted(self.historical_data[symbol].keys())[-window_days:]

            if len(dates) < 2:
                return 0.25

            closes = [self.historical_data[symbol][date].get("Close", 0) for date in dates]

            log_returns = []
            for i in range(1, len(closes)):
                if closes[i] > 0 and closes[i-1] > 0:
                    log_returns.append(np.log(closes[i] / closes[i-1]))

            if not log_returns:
                return 0.25

            daily_vol = np.std(log_returns)
            return daily_vol * np.sqrt(252)

        except:
            return 0.25

    def _calculate_dividend_yield(self, symbol):
        """Calculate dividend yield"""
        try:
            if symbol not in self.historical_data:
                return 0

            dates = sorted(self.historical_data[symbol].keys())[-365:]

            total_dividends = sum(
                self.historical_data[symbol][date].get("Dividends", 0)
                for date in dates
            )

            avg_price = np.mean([
                self.historical_data[symbol][date].get("Close", 100)
                for date in dates
            ])

            if avg_price <= 0:
                return 0

            return total_dividends / avg_price

        except:
            return 0

    # Persistence Methods
    def _save_calibrations(self):
        """Save calibrations to disk"""
        try:
            filename = self.calibration_directory / f"calibrations_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

            with open(filename, 'wb') as f:
                pickle.dump({
                    'params': self.calibrated_params,
                    'surfaces': self.vol_surfaces,
                    'timestamp': dt.datetime.now(),
                    'symbols': self.symbols
                }, f)

            # Keep only last 10 files
            files = sorted(self.calibration_directory.glob("calibrations_*.pkl"))
            if len(files) > 10:
                for old_file in files[:-10]:
                    old_file.unlink()

        except Exception as e:
            print(f"Failed to save calibrations: {e}")

    def _load_calibrations(self):
        """Load calibrations from disk"""
        try:
            files = sorted(self.calibration_directory.glob("calibrations_*.pkl"))
            if not files:
                return

            with open(files[-1], 'rb') as f:
                data = pickle.load(f)

            # Check staleness
            if (dt.datetime.now() - data['timestamp']).total_seconds() < 3600:
                self.calibrated_params = data.get('params', {})
                self.vol_surfaces = data.get('surfaces', {})
                print(f"Loaded calibrations from {files[-1].name}")

        except Exception as e:
            print(f"Failed to load calibrations: {e}")

    # Public Methods for Accessing Results
    def get_calibrated_params(self, symbol, expiry=None, model=None):
        """Get calibrated parameters"""
        with self.data_lock:
            if symbol not in self.calibrated_params:
                return None

            if expiry and model:
                return self.calibrated_params.get(symbol, {}).get(expiry, {}).get(model)
            elif expiry:
                return self.calibrated_params.get(symbol, {}).get(expiry)
            else:
                return self.calibrated_params.get(symbol)

    def get_volatility_surface(self, symbol):
        """Get volatility surface for symbol"""
        with self.data_lock:
            return self.vol_surfaces.get(symbol)

    def get_service_status(self):
        """Get service health status"""
        return {
            'running': self.running,
            'symbols_tracked': len(self.symbols),
            'calibrations_completed': sum(
                len(expiries)
                for expiries in self.calibrated_params.values()
            ),
            'last_update': max(
                self.file_timestamps.values()
            ) if self.file_timestamps else None,
            'worker_threads': self.calibration_workers._max_workers,
            'queue_size': self.calibration_queue.qsize()
        }


# Example usage
if __name__ == "__main__":
    # Initialize with your symbols
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']

    # Create service
    service = CalibrationService(symbols)

    # Start service
    service.start()

    # Let it run for a while
    try:
        while True:
            time.sleep(10)
            status = service.get_service_status()
            print(f"Status: {status}")

            # Example: Get SABR params for SPY
            params = service.get_calibrated_params('SPY')
            if params:
                print(f"SPY calibrations: {len(params)} expiries")

    except KeyboardInterrupt:
        print("Shutting down...")
        service.stop()
