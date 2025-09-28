import yfinance as yf
import datetime as dt
import threading as th
from queue import Queue as qu
import requests
import json
import time
import pytz
import pandas_market_calendars as mcal
import logging
import time
import traceback
from typing import Optional, Dict, Any

#service architecture for fetching and storing market data accessible across modules but not directly dependent on other modules


# ETFs for market data
SPY = 'SPY'  # S&P 500 ETF
QQQ = 'QQQ'  # Nasdaq-100 ETF
IWM = 'IWM'  # Russell 2000 ETF
DIA = 'DIA'  # Dow Jones Industrial Average ETF

# High-Volume Single Stocks

AAPL = 'AAPL'  # Apple Inc.
MSFT = 'MSFT'  # Microsoft Corporation
AMZN = 'AMZN'  # Amazon.com Inc.
GOOGL = 'GOOGL'  # Alphabet Inc. (Class A)
TSLA = 'TSLA'  # Tesla Inc.
NVDA = 'NVDA'  # NVIDIA Corporation
META = 'META'  # Meta Platforms Inc. (formerly Facebook)

# Financial Sector Stocks
JPM = 'JPM'  # JPMorgan Chase & Co.
BAC = 'BAC'  # Bank of America Corporation
GS = 'GS'    # The Goldman Sachs Group Inc.

# Volatility Special Cases
VIX = '^VIX'  # CBOE Volatility Index
UVXY = 'UVXY'  # ProShares Ultra VIX Short-Term Futures ETF
SVXY = 'SVXY'  # ProShares Short VIX Short-Term Futures ETF


# Sector Representative ETFs
XLF = 'XLF'  # Financial Select Sector SPDR Fund
XLE = 'XLE'  # Energy Select Sector SPDR Fund
XLK = "XLK"  # Technology Select Sector SPDR Fund
GLD = 'GLD'  # SPDR Gold Shares

#Time interval API Calls
SPOTS = 500 # 5 Second intervals
ATMS = 300 # 30 Second intervals
OTM = 60 * 5 * 100 # 5 Minute intervals
HISTORY = 100 * 60 * 60 # 1 Hour intervals or on demand


class MarketDataService:


    # Core Methods

    # Initialize with list of symbols and optional config
    def __init__(self, symbols, config = None):
        # Data Storage variables
        self.symbols = symbols
        self.spot_data = {}
        self.option_chain = {}
        self.historical_data = {}
        self.last_fetched = {}

        #Threading control variables
        self.running = False
        self.data_lock = th.Lock()
        self.spot_thread = th.Thread()
        self.option_threads = []
        self.option_update_queue = qu()
        self.update_frequency = {'spot': SPOTS, 'atm_options': ATMS, 'otm_options': OTM, 'history': HISTORY}
        self.priority_symbols = [SPY, QQQ, AAPL]
        self.stale_threshold = {'spot': 60, 'atm_options': 300, 'otm_options': 900, 'history': 3600 * 24}  # in seconds

    """
    - Performs initial data fetch for all symbols
    - Starts all update Threads
    - Sets running flag to True
    - Returns when service is ready
    """
    def start(self) -> bool:
        try:
            self.running = True
            self._initial_data_load()

            # Create and start spot price update thread
            self.spot_thread = th.Thread(
                target=self._spot_price_loop,
                name="SpotPriceUpdater",
                daemon=True  # Dies when main program exits
            )
            self.spot_thread.start()
            return True

        except Exception as e:
            self._handle_api_error(e, {'operation': 'start',
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})
            self.running = False
            return False



    """
    - Gracefully stops all update Threads
    - Saves any cached data
    - Cleans up resources
    - Sets running flag to False
    """
    def stop(self):
        self.running = False


    # Data Collection Methods

    """
    - One-time bulk fetch of spot prices for all symbols
    - Populates all data structures
    - Prioritizes spot prices first
    - Queues all options chains
    - Provides immediate data availability
    """
    def _initial_data_load(self):
        counter = 0
        self.spot_data = self._batch_fetch_spots(self.symbols)
        for symbol in self.symbols:

            try:
                for option_expiry in yf.Ticker(symbol).options:
                    if self.spot_data.get(symbol) and self.spot_data[symbol].get('price'):
                        self.option_update_queue.put((symbol, option_expiry))


                if counter < 10:
                    t = th.Thread(
                        target=self._option_chain_loop,
                        name=f"OptionChainUpdater-{counter}",
                        daemon=True  # Dies when main program exits
                    )
                    t.start()
                    self.option_threads.append(t)
                    counter += 1

            except Exception as e:
                self._handle_api_error(e, {'operation': '_initial_data_load',
                                           'symbol': symbol,
                                           'retry_count': 0,
                                           'max_retries': 3,
                                           'retry_delay': 1})



    """
    - Fetches current spot data from yahoo
    - Returns dict with price, bid, ask, volume
    - Handles errors gracefully
    - Single symbol fetch
    """
    def _fetch_spot_price(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            ticker_data = {
                        # Core Pricing Data
                        'price': ticker.info.get('regularMarketPrice', None),
                        'bid': ticker.info.get('bid', None),
                        'ask': ticker.info.get('ask', None),
                        'mid': (ticker.info.get('bid', 0) + ticker.info.get('ask', 0)) / 2 if ticker.info.get('bid') and ticker.info.get('ask') else None,

                        # Volume and Size
                        'volume': ticker.info.get('volume', None),
                        'bid_size': ticker.info.get('bidSize', None),
                        'ask_size': ticker.info.get('askSize', None),
                        'avg_volume': ticker.info.get('averageVolume', None),

                        # Price Movements
                        'open': ticker.info.get('open', None),
                        'high': ticker.info.get('dayHigh', None),
                        'low': ticker.info.get('dayLow', None),
                        'prev_close': ticker.info.get('previousClose', None),
                        'change': ticker.info.get('regularMarketChange', None),
                        'change_pct': ticker.info.get('regularMarketChangePercent', None),

                        # Timestamps
                        'timestamp': dt.datetime.now().timestamp(),
                        'market_time': ticker.info.get('regularMarketTime', None),
                        'quote_type': ticker.info.get('quoteType', None),

                        # Additional Info
                        'halted': ticker.info.get('halted', None),
                        'currency': ticker.info.get('currency', None),
                        'extchange': ticker.info.get('exchange', None),
                    }
        except Exception as e:
            self._handle_api_error(e, {'operation': '_fetch_spot_price',
                                       'symbol': symbol,
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})
        return ticker_data



    """
    - Retrieves complete option chain
    - Returns processed calls/puts data
    - Handles missing strikes
    - Validates data quality
    """
    def _fetch_option_chain(self, symbol, expiry):
        try:
            option = yf.Ticker(symbol).option_chain(expiry)
            options_data = {
                'calls': option.calls.to_dict(orient='records'),
                'puts': option.puts.to_dict(orient='records'),
                'last_updated': dt.datetime.now().timestamp()
            }
            return options_data
        except Exception as e:
            self._handle_api_error(e, {'operation': '_fetch_option_chain',
                                       'symbol': symbol,
                                       'expiry': expiry,
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})
            return {'calls': [], 'puts': [], 'last_updated': None}




    """
    - Fetches multiple spot prices efficiently
    - Uses Yahoo's batch capabilities
    - Reduces API calls
    - Returns dict of results
    """
    def _batch_fetch_spots(self, symbol_list):
        quotes = {}
        if not symbol_list:
            return quotes

        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={','.join(symbol_list)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            response = requests.get(url, headers=headers)
            data = response.json()

            for quote in data.get('quoteResponse', {}).get('result', []):
                symbol = quote.get('symbol')
                quotes[symbol] = {
                    # Core Pricing Data
                    'price': quote.get('regularMarketPrice'),
                    'bid': quote.get('bid'),
                    'ask': quote.get('ask'),
                    'mid': (quote.get('bid', 0) + quote.get('ask', 0)) / 2 if quote.get('bid') is not None and quote.get('ask') is not None else None,

                    # Volume and Size
                    'volume': quote.get('regularMarketVolume'),
                    'bid_size': quote.get('bidSize'),
                    'ask_size': quote.get('askSize'),
                    'avg_volume': quote.get('averageDailyVolume3Month'),

                    # Price Movements
                    'open': quote.get('regularMarketOpen'),
                    'high': quote.get('regularMarketDayHigh'),
                    'low': quote.get('regularMarketDayLow'),
                    'prev_close': quote.get('regularMarketPreviousClose'),
                    'change': quote.get('regularMarketChange'),
                    'change_pct': quote.get('regularMarketChangePercent'),

                    # Timestamps
                    'timestamp': dt.datetime.now().timestamp(),
                    'market_time': quote.get('regularMarketTime'),
                    'quote_type': quote.get('quoteType'),

                    # Additional Info
                    'halted': quote.get('halted'),
                    'currency': quote.get('currency'),
                    'extchange': quote.get('exchange')
                }

        except Exception as e:
            self._handle_api_error(e, {'operation': '_batch_fetch_spots',
                                       'symbol': ','.join(symbol_list),
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})
            # fallback: return empty dict
            return {}

        return quotes


    # Update Loops Methods

    """
    - Main spot price update loop
    - Runs every 5 seconds
    - Detects significant changes
    - Triggers option updates if needed
    """
    def _spot_price_loop(self):
        while self.running:
            try:
                # Update all symbols in batch (fast)
                batch_data = self._batch_fetch_spots(self.symbols)
                with self.data_lock:
                    for symbol, data in batch_data.items():
                        self.spot_data[symbol].update(data)
                        self.last_fetched["spot"][symbol] = dt.datetime.now().timestamp()

                # Update priority symbols individually (full details)
                for symbol in self.priority_symbols:
                    try:
                        detailed_data = self._fetch_spot_price(symbol)
                        with self.data_lock:
                            self.spot_data[symbol].update(detailed_data)
                            self.last_fetched["spot"][symbol] = dt.datetime.now().timestamp()
                    except Exception as e:
                        self._handle_api_error(e, {'operation': '_spot_price_loop',
                                                   'symbol': symbol,
                                                   'retry_count': 0,
                                                   'max_retries': 3,
                                                   'retry_delay': 1})

            except Exception as e:
                self._handle_api_error(e, {'operation': '_spot_price_loop',
                                           'retry_count': 0,
                                           'max_retries': 3,
                                           'retry_delay': 1})

            time.sleep(self.update_frequency['spot'])



    """
    - Worker method for option updates
    - Pulls from update queue
    - Fetches and stores chain data
    - Re-queues for next update
    - Multiple instances run in parallel
    """
    def _option_chain_loop(self):
        while self.running:
            try:
                symbol, expiry = self.option_update_queue.get(timeout=5)

                now = time.time()
                last_update = self.last_fetched['option'].get((symbol, expiry), 0)

                # Pick right interval (ATM vs OTM)
                interval = self.update_frequency['atm_options'] if expiry == 'nearest' else self.update_frequency['otm_options']

                if now - last_update >= interval:
                    # Fetch option chain
                    chain_data = self._fetch_option_chain(symbol, expiry)
                    with self.data_lock:
                        if symbol not in self.option_chain:
                            self.option_chain[symbol] = {}
                        self.option_chain[symbol][expiry] = chain_data

                    # Update timestamp
                    self.last_fetched['option'][(symbol, expiry)] = now

                # Always requeue for future updates
                self.option_update_queue.put((symbol, expiry))
                self.option_update_queue.task_done()

            except qu.Empty:
                continue
            except Exception as e:
                self._handle_api_error(e, {'operation': '_option_chain_loop',
                                        'symbol': symbol,
                                        'expiry': expiry,
                                        'retry_count': 0,
                                        'max_retries': 3,
                                        'retry_delay': 1})



    """
    - Updates historical price data
    - Runs less frequently (hourly)
    - Maintains rolling window
    - Used for volatility calculations
    """
    def _historical_data_loop(self):
        pass


    # Data Processing Methods

    """
    - Converts DataFrame to dict format
    - Extracts relevant fields
    - Handles missing/invalid data
    - Normalizes data structure
    """
    def _process_option_data(self, chain_df):
        pass



    """
    - Compares new vs. cached price
    - Calculates percentage change
    - Returns boolean for significant move
    - Triggers downstream updates
    """
    def _detect_spot_change(self, symbol, new_price):
        pass



    """
    - Determines which options need updating
    - Prioritizes by moneyness and expiry
    - Adds to update queue with priority
    - Balances update load
    """
    def _prioritize_option_updates(self, symbol):
        pass



    # Data Access Methods (Public Interface)

    """
    - Returns current spot price data
    - Thread-safe with lock
    - Returns copy to prevent modification
    - Includes timestamp
    """
    def get_spot_data(self, symbol):
        pass



    """
    - Returns specific option chain
    - Thread-safe access
    - Returns None if not available
    - Includes all strikes
    """
    def get_option_chain(self, symbol):
        pass


    """
    - Lists all expiries for symbol
    - Used by consumers to know options
    - Returns sorted list
    - Empty list if no options
    """
    def get_available_expiries(self, symbol):
        ticker = yf.Ticker(symbol)
        try:
            return sorted(ticker.options)
        except Exception as e:
            self._handle_api_error(e, {'operation': 'get_available_expiries',
                                       'symbol': symbol,
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})


    """"
    - Returns specific option contract data
    - Convenience method for single contract
    - Returns None if not found
    - Type is 'call' or 'put'
    """
    def get_option_contract(self, symbol, strike, expiry, option_type):
        pass



    """
    - Returns seconds since last update
    - Helps consumers check freshness
    - Returns None if never updated
    - Used for staleness detection
    """
    def get_data_age(self, symbol, data_type):
        pass



    """
    - Checks if market is currently open
    - Adjusts update behavior
    - Considers holidays
    - Returns boolean
    """
    def is_market_open(self):
        pass


    # Utility Methods

    """
    - Checks if symbol is valid
    - Adds to monitoring list if new
    - Handles symbol changes
    - Returns validation status
    """
    def _validate_symbol(self, symbol):
        pass


    """
    - Determines update urgency
    - Based on staleness and importance
    - Returns priority score
    - Used for queue ordering
    """
    def _calculate_update_priority(self, symbol, data_type):
        pass



    """
    - Removes very old data
    - Manages memory usage
    - Runs periodically
    - Configurable retention
    """
    def _clean_stale_data(self):
        pass



    """
    - Centralized error handling
    - Logs errors appropriately
    - Implements retry logic
    - Prevents service disruption
    """
    def _handle_api_error(self, error, context):
            operation = context.get('operation', 'Unknown operation')
            symbol = context.get('symbol', 'N/A')
            retry_count = context.get('retry_count', 0)
            max_retries = context.get('max_retries', 3)
            retry_delay = context.get('retry_delay', 1)

            # Log the error
            if retry_count == 0:
                logging.warning(f"API Error in {operation} for {symbol}: {type(error).__name__}: {str(error)}")
                if not isinstance(error, (ConnectionError, TimeoutError)):
                    logging.debug(f"Traceback:\n{traceback.format_exc()}")

            # Check if retryable
            retryable_errors = (ConnectionError, TimeoutError)
            is_retryable = isinstance(error, retryable_errors) or \
                        "HTTPError" in str(type(error)) or \
                        "URLError" in str(type(error)) or \
                        (hasattr(error, 'response') and
                            hasattr(error.response, 'status_code') and
                            500 <= error.response.status_code < 600)

            # Should we retry?
            if is_retryable and retry_count < max_retries:
                # Exponential backoff with jitter
                delay = retry_delay * (2 ** retry_count) + (time.time() % 1)
                logging.info(f"Retrying {operation} for {symbol} in {delay:.1f}s (attempt {retry_count + 1}/{max_retries})")
                time.sleep(delay)

                # Signal to caller to retry
                context['retry_count'] = retry_count + 1
                context['should_retry'] = True
                return None

            # Max retries exceeded or non-retryable
            if retry_count >= max_retries:
                logging.error(f"Max retries exceeded for {operation} (symbol: {symbol})")
            else:
                logging.error(f"Non-retryable error in {operation} for {symbol}")

            context['should_retry'] = False

            # Track error stats
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            self._error_count += 1

            if self._error_count % 100 == 0:
                logging.warning(f"Total API errors: {self._error_count}")

            return None


# Helper Functions

"""
- Returns default configuration dict
- Defines update frequencies
- Sets queue sizes
- Configurable defaults
"""
def create_default_config():
    pass


"""
- Checks if market is open
- Considers timezone
- Handles holidays
- Returns boolean
"""
def validate_market_hours():
    # Get current time in ET (market timezone)
    et_tz = pytz.timezone('America/New_York')
    now_et = dt.datetime.now(et_tz)

    # Quick check: weekends are always closed
    if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Get NYSE calendar for holiday checking
    nyse = mcal.get_calendar('NYSE')
    today_str = now_et.strftime('%Y-%m-%d')

    # Check if today is a trading day
    schedule = nyse.schedule(start_date=today_str, end_date=today_str)
    if schedule.empty:
        return False  # Holiday

    # Check if within market hours (9:30 AM - 4:00 PM ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close



"""
- Determines update priority
- Combines multiple factors
- Returns numeric score
- Used for intelligent scheduling
"""
def calculate_priority_score(symbol, staleness, volatility):
    pass
