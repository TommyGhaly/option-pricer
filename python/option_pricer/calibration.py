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
        if tte <= 1/12:  # 1 month in years
            rf_rate = self.risk_free_rate["1m"]
        elif tte <= 3/12:  # 3 months in years
            rf_rate = self.risk_free_rate["3m"]
        else:
            rf_rate = self.risk_free_rate["1y"]

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
        pass



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
        pass



    def _prepare_sabr_market_data(self, calibration_data):
        """
        Formats data for C++ calibration function
        Creates list of (strike, implied_vol) pairs
        Sorts by moneyness for stability
        Limits number of points for efficiency
        Handles both calls and puts
        Returns properly formatted input
        """
        pass



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
        pass


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
        pass



    def _heston_objective_function(self, params, market_data):
        """
        Calculates sum of squared pricing errors
        Used by optimizer
        Iterates over all strikes and expiries
        Calls heston_model for each option
        Returns scalar error metric
        Includes regularization penalties
        """
        pass



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
        pass



    def _build_volatility_surface(self, symbol):
        """
        Aggregates implied vols across all expiries
        Creates regularly-spaced strike grid
        Interpolates missing points
        Handles irregular data
        Returns complete surface structure
        Required input for local vol calculation
        """
        pass



    def _interpolate_volatility(self, surface, K, T):
        """
        Bilinear interpolation on vol surface
        Handles strikes between grid points
        Handles times between grid points
        Returns interpolated implied vol
        Used for pricing at arbitrary strikes/expiries
        """
        pass


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
