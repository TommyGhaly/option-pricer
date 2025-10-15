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

# retrieve api key from FRED website
fred = Fred(api_key=os.environ["FRED_API_KEY"])

class CalibrationService:

    # Initialization Methods
    def __init__(self, symbols, config = None):
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
            'spot' : "spot_data.json",
            'options' : "option_chains.json",
            'history' : "historical_data.json",
            'meta' : "metadata.json"
        }
        self.file_timestamps = {}
        self.symbols = symbols


        # Extracted Market Data Variables
        self.spot_prices = {}
        self.options_chains = {}
        self.historical_data = {}
        self.risk_free_rate = {
            '1m' : self._get_rf_rate("1m"),
            '3m' : self._get_rf_rate("3m"),
            '1y' : self._get_rf_rate("1y")
        }


        # Calibration Output Variables
        self.calibrated_params = {}
        self.vol_surfaces = {}
        self.implied_volatilities = {}


        # Threading Control Variables
        self.running = False
        self.data_lock = th.Lock()
        self.calibration_thread = th.Thread()
        self.calibration_workers = ThreadPoolExecutor(max_workers=8)
        self.calibration_queue = pq()


        # Configuration Variables
        self.calibration_interval = 5
        self.models_to_calibrate = ['SABR', 'Heston', 'LocalVa']
        self.min_data_points = 5
        self.max_expiries_per_symbol = 15
        self.priority_symbols = ['SPY', 'QQQ', 'AAPL']
        self.save_calibrations = True
        self.calibration_directory = './calibration_data_realtime'


        # Cache and Performance Variables
        self.iv_cache = {}
        self.last_calibration_time = {}
        self.calibration_staleness_threshold = {}
        self.spot_change_threshold = 0.01


        # Validation and Quality Control Variables
        self.parameter_bounds = {}
        self.rmse_threshold = {}
        self.failed_calibrations = {}
        self.calibration_statistics = {}



    def start(self):
        """
        Performs initial data load from files
        Calculates initial implied volatilities
        Runs first calibration pass for all symbols
        Starts calibration thread
        Starts worker thread pool
        Returns when service is ready
        Provides initial parameters immediately
        """
        pass



    def stop(self):
        """
        Sets running flag to False
        Gracefully stops calibration thread
        Shuts down worker pool
        Saves current calibrated parameters
        Cleans up resources
        Waits for in-progress calibrations
        """
        pass


    # File Monitoring Methods
    def _check_file_data(self, symbol):
        """
        Checks if new market data available since last calibration
        Compares metadata timestamps vs last calibration time
        Returns True if data is newer
        Avoids redundant calibrations
        """
        # Load metadata
        metadata_path = os.path.join(self.data_directory, self.file_paths["meta"])
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Get last data update time
        last_spot_update = metadata['last_fetched_times']['spot'].get(symbol)
        last_options_update = metadata['last_fetched_times']['options'].get(f"{symbol}_<expiry>")

        # Compare to last calibration time
        if self.last_calibration_time[symbol] is None:
            return True  # Never calibrated

        # Parse timestamps and compare
        last_data_time = max(
            dt.fromisoformat(last_spot_update),
            dt.fromisoformat(last_options_update)
        )

        return last_data_time > self.last_calibration_time[symbol]


    def _load_market_data(self):
        """
        Reads JSON files when changes detected
        Parses into Python data structures
        Validates data integrity
        Updates spot_prices, option_chains, historical_data
        Thread-safe with lock
        Handles file read errors gracefully
        """
        # Read and load the json files
        try:
            spot_path = os.path.join(self.data_directory, self.file_paths["spot"])
            with open(spot_path, 'r') as f:
                spot_data = json.load(f)

            option_path = os.path.join(self.data_directory, self.file_paths["option"])
            with open(option_path, 'r') as f:
                option_data = json.load(f)

            history_path = os.path.join(self.data_directory, self.file_paths["history"])
            with open(history_path, 'r') as f:
                history_data = json.load(f)

        except Exception as e:
            print(f"Failed to load JSON file. Error: {e}")
            return  # Don't proceed if files couldn't be loaded

        # Validate data integrity - iterate correctly over nested dict
        for ticker, ticker_value in option_data.items():  # .items() not .keys()
            for date, date_value in ticker_value.items():  # .items() not .keys()
                # Filter out invalid options instead of modifying during iteration
                date_value["calls"] = [
                    option for option in date_value.get("calls", [])
                    if self._validate_option_data(option)
                ]
                date_value["puts"] = [
                    option for option in date_value.get("puts", [])
                    if self._validate_option_data(option)
                ]

        # Update instance variables
        with self.lock:  # If you have threading
            self.spot_prices = spot_data
            self.option_chains = option_data
            self.historical_data = history_data



        """
        Reads JSON files when changes detected
        Parses into Python data structures
        Validates data integrity
        Updates spot_prices, option_chains, historical_data
        Thread-safe with lock
        Handles file read errors gracefully
        """
        # Read and laod the json files
        try:
            spot_path = os.path.join(self.data_directory, self.file_paths["spot"])
            with open(spot_path, 'r') as f:
                spot_data = json.load(f)


            option_path = os.path.join(self.data_directory, self.file_paths["option"])
            with open(option_path, 'r') as f:
                option_data = json.load(f)

            history_path = os.path.join(self.data_directory, self.file_paths["history"])
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        except Exception as e:
            print(f"failed to load json file \n Code ended on exception {e}")
        # Validate data integrity
        for _, ticker_value in option_data.keys():
            for _, date_value in ticker_value.keys():
                for option in date_value["calls"]:
                    if not self._validate_option_data(option):
                        date_value["calls"].pop(option)
                for option in date_value["puts"]:
                    if not self._validate_option_data(option):
                        date_value["puts"].pop(option)


        self.spot_prices = spot_data
        self.options_chains = option_data
        self.historical_data = history_data

    def _detect_significant_changes(self):
        """
        Compares new data to cached data
        Identifies price moves exceeding threshold
        Returns list of symbols needing recalibration
        Considers both spot and option price changes
        Implements smart triggering logic
        Prevents excessive recalibrations
        """
        pass


    # Implied Volatility Calculation Methods
    def _calculate_implied_vol(self, market_price, S, K, r, T, is_call, q=0):
        """
        Implements Newton-Raphson iteration
        Starts with reasonable initial guess (20%)
        Uses Black-Scholes and vega from your C++ library
        Iterates until convergence or max iterations
        Returns implied volatility or None
        Handles edge cases (deep ITM/OTM)
        Validates result is reasonable (0.01 < iv < 3.0)
        """
        initial_guess = 0.2
        sigma = initial_guess
        max_iter = 100  # 10000 is way too many

        for i in range(max_iter):
            # Calculate BS price and vega at current sigma
            bs_price = black_scholes(S, K, r, q, sigma, T, is_call)
            vega_val = vega(S, K, r, q, sigma, T, is_call)  # Don't shadow function name

            diff = bs_price - market_price

            # Check convergence (use abs() to check both positive and negative)
            if abs(diff) < 1e-6:
                # Validate result is reasonable
                if 0.01 < sigma < 3.0:
                    return sigma
                else:
                    return None  # IV outside reasonable bounds

            # Avoid division by zero
            if vega_val == 0 or abs(vega_val) < 1e-10:
                return None

            # Newton-Raphson update (correct formula)
            sigma = sigma - diff / vega_val  # Not (sigma - diff) / vega

            # Keep sigma positive and reasonable
            if sigma <= 0:
                sigma = 0.01
            elif sigma > 5.0:  # Prevent explosion
                sigma = 5.0

        return None  # Failed to converge



    def _calculate_all_implied_vols(self, symbol, expiry):
        """
        Processes entire option chain for one expiry
        Calls _calculate_implied_vol for each contract
        Filters out invalid/illiquid options
        Returns dictionary of (strike, type) → implied vol
        Caches results for performance
        Used by calibration methods
        """

        # Implied Volatility Dict
        ivs = {}

        options = self.option_chains[symbol][expiry]
        spot_price = self.spot_prices[symbol]

        tte = self._calculate_time_to_maturity(expiry)

        # Determine proper risk-free rate (tte is a number, not a string)
        rf_rate = self._get_risk_free_rate_for_maturity(tte)

        # Calculate dividend yield once
        div_yield = self._calculate_dividend_yield(symbol)

        # Iterating through the options
        for option in options["calls"]:
            iv_value = self._calculate_implied_vol(
                option["mid"],
                spot_price,
                option["strike"],
                rf_rate,
                tte,
                True,  # is_call
                div_yield
            )

            # Validating implied volatility
            validation = self._validate_implied_vol(
                iv_value, spot_price, option["strike"],
                rf_rate, tte, True,
                option["mid"], div_yield
            )

            if iv_value is not None and validation:
                ivs[(option["strike"], 'call')] = iv_value  # Key: (strike, 'call')

        for option in options["puts"]:
            iv_value = self._calculate_implied_vol(
                option["mid"],
                spot_price,
                option["strike"],
                rf_rate,
                tte,
                False,  # is_call
                div_yield
            )

            # validating implied volatility
            validation = self._validate_implied_vol(
                iv_value, spot_price, option["strike"],
                rf_rate, tte, False,
                option["mid"], div_yield
            )

            if iv_value is not None and validation:
                ivs[(option["strike"], 'put')] = iv_value  # Key: (strike, 'put')

        # Initialize cache if needed
        if symbol not in self.iv_cache:
            self.iv_cache[symbol] = {}

        # Cache results for performance
        self.iv_cache[symbol][expiry] = ivs

        # return dictionary
        return ivs



    def _validate_implied_vol(self, iv, S, K, r, T, is_call, option_price, q=0):
        """
        Checks if calculated IV makes sense
        Validates against put-call parity
        Ensures price is within no-arbitrage bounds
        Rejects if vega near zero
        Returns boolean validity flag
        Prevents bad data from corrupting calibration
        """
        # Check IV exists and is in reasonable range
        if iv is None or iv <= 0.01 or iv >= 3.0:
            return False

        # Create option dict for arbitrage bounds check
        option = {'mid': option_price}
        if not self._check_arbitrage_bounds(option, S, K, r, T, is_call, q):
            return False

        # Check vega is significant
        vega_val = vega(S, K, r, q, iv, T, is_call)
        if vega_val < 0.01:  # Changed from 0.09
            return False

        return True



    # Data Extraction Methods
    def _extract_calibration_data(self, symbol, expiry):
        """
        Pulls all data needed for one calibration
        Gets spot price, time to maturity, risk-free rate
        Extracts call and put chains
        Calculates implied volatilities for all strikes
        Filters by quality metrics (liquidity, spreads)
        Returns structured calibration-ready dataset
        Returns None if insufficient data
        """
        try:
            S = self.spot_prices[symbol]
            T = self._calculate_time_to_maturity(expiry)

            # Determine risk-free rate based on time to maturity
            r = self._get_risk_free_rate_for_maturity(T)
            q = self._calculate_dividend_yield(symbol)

            # Get option chains and filter
            option_chain = self.option_chains[symbol][expiry]

            # Filter calls and puts separately
            filtered_calls = self._filter_options_for_calibration(
                option_chain.get("calls", []), S
            )
            filtered_puts = self._filter_options_for_calibration(
                option_chain.get("puts", []), S
            )

            # Calculate implied volatilities
            ivs = self._calculate_all_implied_vols(symbol, expiry)

            # Check if we have sufficient data
            if len(ivs) < 5:  # Need at least 5 data points for calibration
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

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error extracting calibration data for {symbol} {expiry}: {e}")
            return None


    def _filter_options_for_calibration(self, options_list, S):
        """
        Removes illiquid options (low volume/OI)
        Filters out wide bid-ask spreads
        Excludes deep OTM options (poor quality)
        Prioritizes ATM options (most informative)
        Limits to reasonable strike range
        Returns filtered list sorted by moneyness
        """
        valid_options = []
        for option in options_list:
            if not self._validate_option_data(option):
                continue


            elif option.get("openInterest", 0) < 100:
                continue


            elif option.get("volume", 0) < 10 and option.get("openInterest", 0) < 500:
                continue


            ratio = option["strike"] / S
            if ratio > 3:
                continue

            if 0.98 <= ratio <= 1.02:
                valid_options.insert(0, option)
                continue

            valid_options.append(option)

        # Sort by moneyness (strike relative to spot) for better organization
        # ATM options already at front, but sort the rest
        atm_count = sum(1 for opt in valid_options if 0.98 <= opt["strike"]/S <= 1.02)
        rest = valid_options[atm_count:]
        rest.sort(key=lambda opt: abs(opt["strike"] / S - 1.0))  # Sort by distance from ATM

        return valid_options[:atm_count] + rest



    def _calculate_forward_price(self, S, r, T,  q=0,):
        """
        Computes forward price: F = S * exp(r*T)
        Used in SABR calibration
        Handles dividend yield if available
        Simple but essential calculation
        Cached for performance
        """
        if T <= 0:
            return S  # At expiry, forward = spot

        return S * math.exp((r - q) * T)


    # SABR Calibration
    def _calibrate_sabr(self, symbol, expiry, calibration_data):
        """
        Main SABR calibration entry point
        Extracts strikes and implied vols
        Calculates forward price
        Fixes beta parameter (typically 0.5 for equities)
        Calls C++ sabr_calibrate function
        Returns calibrated (alpha, beta, rho, nu)
        Calculates and stores RMSE
        Returns None if calibration fails
        """
        try:
            # Extraction
            market_data = self._prepare_sabr_market_data(calibration_data)
            S = market_data['strike']
            r = calibration_data['r']
            q = calibration_data['q']
            T = self._calculate_time_to_maturity(expiry)
            ivs = market_data['ivs']

            # Calculating forward price
            fp = self._calculate_forward_price(S, r, T, q)
            alpha, beta, rho, nu = sabr_calibrate(ivs, fp, T, 0.5)
            params = (alpha, beta, rho, nu)

            if not self._validate_sabr_parameters(alpha, beta, rho, nu):
                print(f"SABR calibration rejected for {symbol} {expiry}: Invalid parameters")
                return None

            quality = self._calculate_sabr_fit_quality(params, market_data)

            # Quality-based acceptance/rejection using RMSE
            if not self._accept_calibration(quality['rmse'], T, symbol, expiry):
                return None

            return params

        except Exception as e:
            print(f'Failed to calibrate SABR. Code exited with exception: {e}')
            return None



    def _accept_calibration(self, rmse, T, symbol, expiry):
        """
        Accept or reject calibration based on RMSE
        Different thresholds for different expiries
        """
        # Time-dependent thresholds
        if T < 0.25:  # Short-dated: expect tight fit
            threshold = 0.02  # 2% vol error
        elif T < 1.0:  # Medium-dated
            threshold = 0.03  # 3% vol error
        else:  # Long-dated: more tolerance
            threshold = 0.05  # 5% vol error

        if rmse > threshold:
            print(f"SABR calibration rejected for {symbol} {expiry}: "
                  f"RMSE {rmse:.4f} exceeds threshold {threshold:.4f} (T={T:.2f})")
            return False

        return True




    def _prepare_sabr_market_data(self, calibration_data):
        """
        Formats data for C++ calibration function
        Creates list of (strike, implied_vol) pairs
        Sorts by moneyness for stability
        Limits number of points for efficiency
        Handles both calls and puts
        Returns properly formatted input
        """
        S = calibration_data["S"]
        ivs = calibration_data["ivs"]

        # Extract (strike, iv) pairs from the ivs dict
        # ivs has keys like (strike, 'call') or (strike, 'put')
        market_data = []

        for (strike, option_type), iv in ivs.items():
            market_data.append({
                'strike': strike,
                'iv': iv,
                'moneyness': strike / S,
                'log_moneyness': np.log(strike / S),
                'type': option_type
            })

        # Sort by moneyness for stability
        market_data.sort(key=lambda x: x['moneyness'])

        # Limit to reasonable number of points (SABR typically uses 10-30)
        max_points = 30
        if len(market_data) > max_points:
            # Keep ATM options and evenly space the rest
            atm_data = [d for d in market_data if 0.95 <= d['moneyness'] <= 1.05]
            otm_data = [d for d in market_data if d['moneyness'] < 0.95 or d['moneyness'] > 1.05]

            # Sample OTM points evenly
            step = max(1, len(otm_data) // (max_points - len(atm_data)))
            otm_sampled = otm_data[::step]

            market_data = sorted(atm_data + otm_sampled, key=lambda x: x['moneyness'])

        # Format for C++ function (typically wants arrays)
        strikes = np.array([d['strike'] for d in market_data])
        ivs_array = np.array([d['iv'] for d in market_data])

        return {
            'strikes': strikes,
            'ivs': ivs_array,
            'spot': S,
            'full_data': market_data  # Keep full info for debugging
        }


    def _validate_sabr_parameters(self, alpha, beta, rho, nu):
        """
        Checks parameters are within valid ranges
        Ensures rho between -1 and 1
        Validates alpha, nu are positive
        Tests stability conditions
        Returns boolean validity
        Prevents storing nonsensical parameters
        """
        # Check positivity constraints
        if alpha <= 0 or nu <= 0:  # Should be strictly positive, not just non-negative
            return False

        # Check beta range
        if beta < 0 or beta > 1:
            return False

        # Check correlation bounds (strict inequality to avoid numerical issues)
        if rho <= -1 or rho >= 1:
            return False

        # Check reasonable upper bounds to prevent explosion
        if alpha > 5.0 or nu > 5.0:  # Unrealistically high values
            return False

        # Optional: Check Feller condition for CIR-like behavior
        # More relevant when beta is close to 0.5
        # if 2 * alpha * beta < nu ** 2:
        #     return False

        return True



    def _calculate_sabr_fit_quality(self, params, market_data):
        """
        Computes RMSE between model and market
        Uses sabr_implied_vol for each strike
        Calculates residuals
        Returns quality metrics
        Used to accept/reject calibration
        Stores for monitoring
        """
        alpha, beta, rho, nu  = params
        S = market_data["spot"]
        T = market_data["T"]
        strikes = market_data["strikes"]
        market_ivs = market_data["ivs"]

        model_ivs = []

        for strike in strikes:
            try:

                # Call your C++ SABR IV function
                sabr_iv = sabr_implied_vol(S, strike, T, alpha, beta, rho, nu)
                model_ivs.append(sabr_iv)


            except Exception as e:
                # If SABR calculation fails, return poor quality
                print(f"SABR IV calculation failed for strike {strike}: {e}")
                return {
                    'rmse': float('inf'),
                    'max_error': float('inf'),
                    'mean_error': float('inf'),
                    'quality': 'failed'
                }

        model_ivs = np.array(model_ivs)
        market_ivs = np.array(market_ivs)

        # Calculate residuals
        residuals = model_ivs - market_ivs

        # Calculate quality metrics
        rmse = np.sqrt(np.mean(residuals ** 2))
        max_error = np.max(np.abs(residuals))
        mean_error = np.mean(np.abs(residuals))

        # Determine quality grade
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
            'max_error': max_error,
            'mean_error': mean_error,
            'residuals': residuals,
            'model_ivs': model_ivs,
            'quality': quality,
            'num_points': len(strikes)
        }


    # Heston Calibration Methods
    def _calibrate_heston(self, symbol, calibration_data):
        """
        Multi-expiry calibration (uses entire surface)
        Extracts all available market data
        Sets up optimization problem
        Initial parameter guesses from historical volatility
        Calls optimization routine with heston_model pricing
        Returns (kappa, theta, sigma, rho, v0)
        More computationally expensive than SABR
        Typically only for priority symbols
        """
        S = calibration_data['S']
        K = calibration_data['K']  # Array of strikes across all expiries
        T = calibration_data['T']  # Array of maturities
        r = calibration_data['r']
        q = calibration_data['q']
        market_prices = calibration_data['prices']  # Market option prices
        option_types = calibration_data['types']  # 'call' or 'put' for each

        # Initial parameter guesses
        v0 = self._get_historical_volatility(symbol) ** 2  # Variance not vol
        kappa = 2.0  # Mean reversion speed
        theta = v0  # Long-term variance (start at current)
        sigma = 0.3  # Vol-of-vol
        rho = -0.5  # Stock-vol correlation (leverage effect)

        # Bundle initial params
        initial_params = np.array([kappa, theta, sigma, rho, v0])

        # Parameter bounds (ensures valid Heston parameters)
        bounds = [
            (0.1, 10.0),   # kappa: mean reversion speed
            (0.01, 1.0),   # theta: long-term variance
            (0.01, 2.0),   # sigma: vol-of-vol
            (-0.99, 0.99), # rho: correlation
            (0.01, 1.0)    # v0: initial variance
        ]

        try:
            # Optimize using scipy
            result = minimize(
                self._heston_objective_function,
                initial_params,
                args=(S, K, T, r, q, market_prices, option_types),
                method='L-BFGS-B',  # Handles bounds well
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9}
            )

            if not result.success:
                print(f"Heston optimization failed for {symbol}: {result.message}")
                return None

            optimized_params = result.x
            kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt = optimized_params

            # Validate parameters
            if not self._validate_heston_parameters(kappa_opt, theta_opt, sigma_opt, rho_opt, v0_opt):
                print(f"Heston calibration rejected for {symbol}: Invalid parameters")
                return None

            # Calculate fit quality
            quality = self._calculate_heston_fit_quality(
                optimized_params, S, K, T, r, q, market_prices, option_types
            )

            # Accept/reject based on quality
            if not self._accept_heston_calibration(quality['rmse'], symbol):
                return None

            return tuple(optimized_params)

        except Exception as e:
            print(f"Heston calibration failed for {symbol}: {e}")
            return None




    def _estimate_long_term_variance(self, market_ivs, T, K, S):
        """
        Estimate theta from long-dated ATM options
        """
        # Find longest maturity ATM options
        long_dated_mask = T > 0.5  # Options > 6 months

        if not np.any(long_dated_mask):
            # Fallback: use overall ATM average
            atm_mask = np.abs(K - S) / S < 0.05  # Within 5% of spot
            return np.mean(market_ivs[atm_mask]) ** 2 if np.any(atm_mask) else 0.04

        # Get ATM IVs for long-dated options
        long_dated_ivs = market_ivs[long_dated_mask]
        long_dated_K = K[long_dated_mask]

        atm_mask = np.abs(long_dated_K - S) / S < 0.05

        if np.any(atm_mask):
            atm_iv = np.mean(long_dated_ivs[atm_mask])
            return atm_iv ** 2  # Convert to variance

        return 0.04  # Default 20% vol



    def _heston_objective_function(self, params, market_data):
        """
        Calculates sum of squared pricing errors
        Used by optimizer
        Iterates over all strikes and expiries
        Calls heston_model for each option
        Returns scalar error metric
        Includes regularization penalties
        """
        kappa, theta, sigma, rho, v0 = params

        # Validate parameters first
        if not self._validate_heston_parameters(kappa, theta, sigma, rho, v0):
            return 1e10  # Return huge penalty if invalid parameters

        S = market_data["spot"]
        T = market_data["T"]
        r = market_data["r"]
        q = market_data["q"]
        strikes = market_data["strikes"]
        market_prices = market_data["market_prices"]  # Need actual market prices
        option_types = market_data["option_types"]  # 'call' or 'put' for each

        squared_errors = []

        for i, strike in enumerate(strikes):
            try:
                # Get market price
                market_price = market_prices[i]
                is_call = (option_types[i] == 'call')

                # Calculate Heston model price
                heston_price = heston_model(S, strike, r, q, T, kappa, theta, sigma, rho, v0, is_call)

                # Calculate squared error
                error = (heston_price - market_price) ** 2
                squared_errors.append(error)

            except Exception as e:
                # If pricing fails, add large penalty
                print(f"Heston pricing failed for strike {strike}: {e}")
                squared_errors.append(1e6)

        # Sum of squared errors (SSE)
        sse = np.sum(squared_errors)

        # Add regularization to prevent extreme parameters
        # Penalize parameters far from typical values
        regularization = 0.0
        regularization += 0.1 * (kappa - 2.0) ** 2  # Prefer kappa near 2
        regularization += 0.1 * (theta - 0.04) ** 2  # Prefer theta near 4%
        regularization += 0.1 * (sigma - 0.3) ** 2   # Prefer sigma near 0.3

        # Total objective (minimize this)
        return sse + regularization

    def _validate_heston_parameters(self, kappa, theta, sigma, rho, v0):
        """
        Checks Feller condition: 2*kappa*theta ≥ sigma^2
        Ensures rho between -1 and 1
        Validates all parameters positive (except rho)
        Tests for numerical stability
        Returns boolean validity
        Critical for meaningful parameters
        """
        # Check positivity (strictly positive for most parameters)
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 < 0:
            return False

        # Check correlation bounds (strict inequality to avoid numerical issues)
        if rho <= -1 or rho >= 1:
            return False

        # Check Feller condition: 2*kappa*theta ≥ sigma^2
        # Ensures variance process stays positive
        if 2 * kappa * theta < sigma ** 2:
            return False

        # Check reasonable upper bounds to prevent instability
        if kappa > 50 or sigma > 5.0 or theta > 2.0:
            return False

        # Check v0 is reasonable relative to theta
        if v0 > 10 * theta or v0 > 2.0:  # Initial var shouldn't be too extreme
            return False

        return True


    # Local Volatility Methods
    def _calibrate_local_volatility(self, symbol):
        """
        Constructs local volatility surface
        Requires complete implied volatility surface
        Uses Dupire's formula implementation
        Calls local_volatility from C++ library
        Returns 2D grid of local volatilities
        Computationally intensive
        Typically run less frequently
        """
        surface = self._build_volatility_surface(symbol)
        if surface is None:
            print(f"Failed to build vol surface for {symbol}")
            return None




    def _build_volatility_surface(self, symbol):
        """
        Aggregates implied vols across all expiries
        Creates regularly-spaced strike grid
        Interpolates missing points
        Handles irregular data
        Returns complete surface structure
        Required input for local vol calculation
        """
        option_data = self.option_chains[symbol]
        S = self.spot_prices[symbol]

        surface_points = []

        for expiry, value in option_data.items():
            T = self._calculate_time_to_maturity(expiry)

            # Skip very short-dated options (< 1 day)
            if T < 1/365:
                continue

            r = self._get_risk_free_rate_for_maturity(T)
            q = self._calculate_dividend_yield(symbol)

            # Process calls (use for K > S, OTM calls only)
            for call in value.get("calls", []):
                try:
                    K = call["strike"]

                    # Use OTM calls only
                    if K <= S:
                        continue

                    # Skip if no valid price
                    if call.get('mid') is None or call['mid'] <= 0:
                        continue

                    iv = self._calculate_implied_vol(call['mid'], S, K, r, T, True, q)

                    # Validate IV
                    if iv is None or iv < 0.01 or iv > 3.0:
                        continue

                    surface_points.append({
                        'K': K,
                        'T': T,
                        'iv': iv
                    })

                except Exception as e:
                    print(f"Error processing call option for {symbol} {expiry}: {e}")
                    continue

            # Process puts (use for K < S, OTM puts)
            for put in value.get("puts", []):
                try:
                    K = put["strike"]

                    # Use OTM puts only
                    if K >= S:
                        continue

                    # Skip if no valid price
                    if put.get('mid') is None or put['mid'] <= 0:
                        continue

                    iv = self._calculate_implied_vol(put['mid'], S, K, r, T, False, q)

                    # Validate IV
                    if iv is None or iv < 0.01 or iv > 3.0:
                        continue

                    surface_points.append({
                        'K': K,
                        'T': T,
                        'iv': iv
                    })

                except Exception as e:
                    print(f"Error processing put option for {symbol} {expiry}: {e}")
                    continue

        # Need at least some points to build surface
        if len(surface_points) < 30:
            print(f"Insufficient data for {symbol} surface: only {len(surface_points)} points")
            return None

        # Extract ranges from data
        T_vals = np.array([p['T'] for p in surface_points])
        K_vals = np.array([p['K'] for p in surface_points])
        iv_vals = np.array([p['iv'] for p in surface_points])

        min_T, max_T = T_vals.min(), T_vals.max()
        min_K, max_K = K_vals.min(), K_vals.max()

        # Grid resolution
        n_time_steps = min(50, int(np.sqrt(len(surface_points))))
        n_strike_steps = min(100, int(2 * np.sqrt(len(surface_points))))

        # Create regular grid
        T_grid = np.linspace(min_T, max_T, n_time_steps)
        K_grid = np.linspace(min_K, max_K, n_strike_steps)
        grid_T, grid_K = np.meshgrid(T_grid, K_grid)

        # Interpolate with nearest neighbor for extrapolation
        from scipy.interpolate import griddata

        grid_iv = griddata(
            (T_vals, K_vals),
            iv_vals,
            (grid_T, grid_K),
            method='cubic',
            fill_value=np.nan
        )

        # Fill NaNs using nearest neighbor
        nan_mask = np.isnan(grid_iv)
        if nan_mask.any():
            grid_iv_nearest = griddata(
                (T_vals, K_vals),
                iv_vals,
                (grid_T, grid_K),
                method='nearest'
            )
            grid_iv[nan_mask] = grid_iv_nearest[nan_mask]

        return surface_points


    def _get_risk_free_rate_for_maturity(self, T):
        """
        Select appropriate risk-free rate based on maturity
        """
        if T <= 1/12:
            return self.risk_free_rate["1m"]
        elif T <= 1/4:
            return self.risk_free_rate["3m"]
        else:
            return self.risk_free_rate["1y"]



    def _interpolate_volatility(self, surface, K, T):
            """
            Bilinear interpolation on vol surface
            Handles strikes between grid points
            Handles times between grid points
            Returns interpolated implied vol
            Used for pricing at arbitrary strikes/expiries
            """
            strikes = surface['strikes']
            expiries = surface['expiries']
            vols = surface['volatilities']  # 2D array: vols[expiry_idx][strike_idx]

            # Handle edge cases - extrapolation
            if T <= expiries[0]:
                T_idx = 0
                T_weight = 0.0
                use_time_interp = False
            elif T >= expiries[-1]:
                T_idx = len(expiries) - 2
                T_weight = 1.0
                use_time_interp = False
            else:
                # Find surrounding expiries
                T_idx = np.searchsorted(expiries, T) - 1
                T_lower, T_upper = expiries[T_idx], expiries[T_idx + 1]
                T_weight = (T - T_lower) / (T_upper - T_lower)
                use_time_interp = True

            # Interpolate at lower expiry
            vol_lower = self._interpolate_strike(strikes, vols[T_idx], K)

            if not use_time_interp:
                return vol_lower

            # Interpolate at upper expiry
            vol_upper = self._interpolate_strike(strikes, vols[T_idx + 1], K)

            # Linear interpolation across time
            return vol_lower * (1 - T_weight) + vol_upper * T_weight




    def _interpolate_strike(self, strikes, vols_at_expiry, K):
        """
        Linear interpolation across strikes for a single expiry
        """
        if K <= strikes[0]:
            return vols_at_expiry[0]  # Flat extrapolation
        elif K >= strikes[-1]:
            return vols_at_expiry[-1]  # Flat extrapolation

        # Find surrounding strikes
        K_idx = np.searchsorted(strikes, K) - 1
        K_lower, K_upper = strikes[K_idx], strikes[K_idx + 1]
        vol_lower, vol_upper = vols_at_expiry[K_idx], vols_at_expiry[K_idx + 1]

        # Linear interpolation
        K_weight = (K - K_lower) / (K_upper - K_lower)
        return vol_lower * (1 - K_weight) + vol_upper * K_weight




    # Main Calibration Loop Methods
    def _calibration_loop():
        """
        Main thread that runs continuously
        Checks for file updates every interval
        Loads new data when detected
        Prioritizes symbols for calibration
        Submits calibration tasks to worker pool
        Monitors worker completion
        Updates calibrated_params atomically
        Saves results periodically
        Handles errors without crashing
        """
        pass



    def _schedule_calibration(self, changed_symbols):
        """
        Determines which calibrations to run
        Prioritizes based on multiple factors
        Considers data freshness, symbol importance, model complexity
        Adds tasks to priority queue
        Balances compute load
        Avoids redundant calibrations
        """
        pass



    def _worker_calibration_task(self, symbol, expiry, model_type):
        """
        Worker thread entry point
        Pulls task from queue
        Extracts relevant data
        Calls appropriate calibration method
        Validates results
        Updates shared data structures with lock
        Logs performance metrics
        Handles exceptions locally
        """
        pass


    # Calibration Priority Methods
    def _calculate_calibration_priority(self, symbol, expiry, model, staleness):
        """
        Computes numeric priority score
        Higher score = higher priority
        Factors: symbol importance, time to expiry, staleness, model type
        Priority symbols get 2x multiplier
        Near-term expiries get 1.5x multiplier
        Stale calibrations get staleness_ratio multiplier
        Returns float priority value for queue
        """
        pass



    def _should_realibrate(self, symbol, expiry, model):
        """
        Decides if recalibration is needed
        Checks time since last calibration
        Checks if underlying price changed significantly
        Considers model-specific staleness thresholds
        Returns boolean decision
        Prevents unnecessary compute
        """
        pass


    # Utility and Validation Methods
    def _validate_option_data(self, option):
        """
        Checks option contract data quality
        Validates bid ≤ mid ≤ ask
        Checks for positive prices
        Verifies reasonable implied volatility
        Returns boolean validity
        Filters out bad data before calibration
        """
        # Check for None values first
        if option.get("bid") is None or option.get("ask") is None or option.get("mid") is None:
            return False

        if option.get("impliedVolatility") is None:
            return False

        # Validate bid-ask relationship
        if option["bid"] > option["mid"] or option["mid"] > option["ask"]:
            return False

        # Check for positive prices
        if option["mid"] <= 0 or option["bid"] < 0:
            return False

        # Verify reasonable IV
        if option["impliedVolatility"] <= 0.05 or option["impliedVolatility"] >= 2.0:
            return False

        return True



    def _check_arbitrage_bounds(self, option, S, K, r, T, is_call, q=0):
        """
        Verifies option price within no-arbitrage bounds
        Call: max(S*exp(-qT) - K*exp(-rT), 0) ≤ price ≤ S*exp(-qT)
        Put: max(K*exp(-rT) - S*exp(-qT), 0) ≤ price ≤ K*exp(-rT)
        Returns boolean validity
        Critical for data quality
        """
        discount_factor = math.exp(-r * T)
        forward_discount = math.exp(-q * T)
        discounted_spot = S * forward_discount
        discounted_strike = K * discount_factor

        if is_call:
            lower_bound = max(discounted_spot - discounted_strike, 0)
            upper_bound = discounted_spot
        else:
            lower_bound = max(discounted_strike - discounted_spot, 0)
            upper_bound = discounted_strike  # Fixed: was S, should be K*exp(-rT)

        return lower_bound <= option["mid"] <= upper_bound




    def _calculate_time_to_maturity(self, expiry_string, use_trading_days=False):
        """
        Converts expiry date string to years

        Args:
            expiry_string: Date string in format '%Y-%m-%d %H:%M:%S'
            use_trading_days: If True, uses 252 trading days/year. If False, uses 365.25 calendar days/year

        Returns:
            float: Fractional years to maturity
            None: If option has expired
        """
        expiry = dt.strptime(expiry_string, '%Y-%m-%d %H:%M:%S')
        now = dt.now()

        # Check if expired
        if now > expiry:
            return None

        time_diff = expiry - now

        # Choose day count convention
        days_per_year = 252 if use_trading_days else 365.25
        years = time_diff.total_seconds() / (days_per_year * 24 * 3600)

        return years


    def _get_historical_volatility(self, symbol, window_days=30):
        """
        Calculates realized volatility from historical data
        Used for parameter initialization
        Returns annualized volatility
        Fallback when calibration unavailable
        """
        if window_days > 365:
            window_days = 360

        symbol_data = self.historical_data[symbol]
        dates = sorted(symbol_data.keys())
        closes = [symbol_data[date]["Close"] for date in dates[-window_days:]]

        # Calculate log returns
        log_returns = []
        for i in range(1, len(closes)):
            log_return = np.log(closes[i] / closes[i-1])
            log_returns.append(log_return)

        # Calculate annualized volatility
        daily_vol = np.std(log_returns)
        annual_vol = daily_vol * np.sqrt(252)

        return annual_vol



    def _calculate_dividend_yield(self, symbol, lookback_days=365):
        """
        Calculate annualized dividend yield from historical data

        Parameters:
        - symbol: ticker symbol
        - lookback_days: period to calculate yield over (default 1 year)

        Returns:
        - Annualized dividend yield (q)
        """
        symbol_data = self.historical_data[symbol]
        dates = sorted(symbol_data.keys())[-lookback_days:]

        # Sum all dividends paid in the period
        total_dividends = sum(symbol_data[date]["Dividends"] for date in dates)

        # Get average price over the period
        avg_price = np.mean([symbol_data[date]["Close"] for date in dates])

        # Annualize the yield
        days_in_period = len(dates)
        annual_dividend = total_dividends * (365 / days_in_period)
        dividend_yield = annual_dividend / avg_price

        return dividend_yield


    # Persistent Methods
    def _save_calibrations(self):
        """
        Writes calibrated_params to disk
        JSON or pickle format
        Organized by date and symbol
        Includes metadata (timestamp, versions)
        Atomic writes to prevent corruption
        Creates backup before overwrite
        """
        pass



    def _load_calibrations(self):
        """
        Reads persisted calibrations on startup
        Validates loaded parameters
        Checks staleness
        Populates calibrated_params
        Enables quick restart
        Handles missing/corrupted files gracefully
        """
        pass



    def _save_volatility_surfaces(self):
        """
        Saves vol surfaces for analysis
        CSV or numpy format
        Enables offline analysis
        Creates visualization-ready data
        Periodic saves for monitoring
        """
        pass


    # Error Handling and Monitoring Methods
    def _handel_calibration_error(self, error, symbol, expiry, model):
        """
        Centralized error handler
        Logs errors with context
        Updates failed_calibrations counter
        Implements retry logic for transient failures
        Prevents error propagation
        Sends alerts for persistent failures
        """
        pass



    def _log_calibration_metrics(self, symbol, expiry, model, params, duration, rmse):
        """
        Records performance metrics
        Tracks calibration times
        Monitors fit quality trends
        Enables performance optimization
        Creates audit trail
        """
        pass



    def _check_service_health(self):
        """
        Monitors overall calibration health
        Checks thread status
        Verifies data freshness
        Counts recent failures
        Returns health status dict
        Used by monitoring systems
        """
        pass
