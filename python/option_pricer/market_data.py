import yfinance as yf
import datetime as dt
import threading as th
from queue import Queue as qu, Empty
import requests
import json
import time
import pytz
import pandas_market_calendars as mcal
import logging
import time
import traceback
from typing import Optional, Dict, Any
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
SPOTS = 5 # 5 Second intervals
ATMS = 30 # 30 Second intervals
OTM = 60 * 5 # 5 Minute intervals
HISTORY = 60 * 60 # 1 Hour intervals or on demand


class MarketDataService:


    # Core Methods

    # Initialize with list of symbols and optional config
    def __init__(self, symbols, config = None):
        # Data Storage variables
        self.symbols = symbols
        self.spot_data = {}
        self.option_chain = {}
        self.historical_data = {}
        self.last_fetched = {
            'spot': {},
            'option': {},
            'history': {}
        }

        #Threading control variables
        self.running = False
        self.data_lock = th.Lock()
        self.spot_thread = th.Thread()
        self.option_threads = []
        self.option_update_queue = qu()
        self.update_frequency = {'spot': SPOTS, 'atm_options': ATMS, 'otm_options': OTM, 'history': HISTORY}
        self.priority_symbols = [SPY, QQQ, AAPL]
        self.stale_threshold = {'spot': 60, 'atm_options': 300, 'otm_options': 900, 'history': 3600 * 24}  # in seconds

        # Initialize error count
        self._error_count = 0

        # File saving configuration
        self.save_to_file = config.get('save_to_file', True) if config else True
        self.data_directory = config.get('data_directory', 'market_data') if config else 'market_data'
        self.save_interval = config.get('save_interval', 1) if config else 1  # Save every 1 second
        self.file_format = config.get('file_format', 'json') if config else 'json'

        # Create data directory if it doesn't exist
        if self.save_to_file:
            Path(self.data_directory).mkdir(parents=True, exist_ok=True)

        # File paths
        self.spot_file = os.path.join(self.data_directory, 'spot_data.json')
        self.option_file = os.path.join(self.data_directory, 'option_chains.json')
        self.history_file = os.path.join(self.data_directory, 'historical_data.json')
        self.metadata_file = os.path.join(self.data_directory, 'metadata.json')

    def start(self) -> bool:
        """
        - Performs initial data fetch for all symbols
        - Starts all update Threads
        - Sets running flag to True
        - Returns when service is ready
        """
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

            # Start historical data update thread
            self.history_thread = th.Thread(
                target=self._historical_data_loop,
                name="HistoricalDataUpdater",
                daemon=True
            )
            self.history_thread.start()

            # Start file saving thread if enabled
            if self.save_to_file:
                self.save_thread = th.Thread(
                    target=self._file_save_loop,
                    name="FileSaver",
                    daemon=True
                )
                self.save_thread.start()

            return True

        except Exception as e:
            self._handle_api_error(e, {'operation': 'start',
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})
            self.running = False
            return False

    def stop(self):
        """
        - Gracefully stops all update Threads
        - Saves any cached data
        - Cleans up resources
        - Sets running flag to False
        """
        for thread in self.option_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        self.running = False
        # Could add thread.join() calls here to wait for threads to finish
        # Could add data persistence here

    # Data Collection Methods

    def _initial_data_load(self):
        """
        - One-time bulk fetch of spot prices for all symbols
        - Populates all data structures
        - Prioritizes spot prices first
        - Queues all options chains
        - Provides immediate data availability
        """
        counter = 0
        self.spot_data = self._batch_fetch_spots(self.symbols)

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                expiries = ticker.options

                for option_expiry in expiries:
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

    def _fetch_spot_price(self, symbol):
        """
        - Fetches current spot data from yahoo
        - Returns dict with price, bid, ask, volume
        - Handles errors gracefully
        - Single symbol fetch
        """
        ticker_data = {}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            ticker_data = {
                # Core Pricing Data
                'price': info.get('regularMarketPrice', None),
                'bid': info.get('bid', None),
                'ask': info.get('ask', None),
                'mid': (info.get('bid', 0) + info.get('ask', 0)) / 2 if info.get('bid') and info.get('ask') else None,

                # Volume and Size
                'volume': info.get('volume', None),
                'bid_size': info.get('bidSize', None),
                'ask_size': info.get('askSize', None),
                'avg_volume': info.get('averageVolume', None),

                # Price Movements
                'open': info.get('open', None),
                'high': info.get('dayHigh', None),
                'low': info.get('dayLow', None),
                'prev_close': info.get('previousClose', None),
                'change': info.get('regularMarketChange', None),
                'change_pct': info.get('regularMarketChangePercent', None),

                # Timestamps
                'timestamp': dt.datetime.now().timestamp(),
                'market_time': info.get('regularMarketTime', None),
                'quote_type': info.get('quoteType', None),

                # Additional Info
                'halted': info.get('halted', None),
                'currency': info.get('currency', None),
                'exchange': info.get('exchange', None),
            }
        except Exception as e:
            self._handle_api_error(e, {'operation': '_fetch_spot_price',
                                       'symbol': symbol,
                                       'retry_count': 0,
                                       'max_retries': 3,
                                       'retry_delay': 1})
        return ticker_data

    def _fetch_option_chain(self, symbol, expiry):
        """
        - Retrieves complete option chain
        - Returns processed calls/puts data
        - Handles missing strikes
        - Validates data quality
        """
        try:
            option_chain = yf.Ticker(symbol).option_chain(expiry)
            options_data = {
                'calls': self._process_option_data(option_chain.calls, 'call'),
                'puts': self._process_option_data(option_chain.puts, 'put'),
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



    def _batch_fetch_spots(self, symbol_list):
        """
        Fetches spot prices using Yahoo's batch quote endpoint with proper auth
        """
        quotes = {}
        if not symbol_list:
            return quotes

        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={','.join(symbol_list)}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://finance.yahoo.com/'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            for item in data.get('quoteResponse', {}).get('result', []):
                symbol = item.get('symbol')
                quotes[symbol] = {
                    'price': item.get('regularMarketPrice'),
                    'bid': item.get('bid'),
                    'ask': item.get('ask'),
                    'mid': (item.get('bid', 0) + item.get('ask', 0)) / 2
                        if item.get('bid') and item.get('ask') else None,

                    'volume': item.get('regularMarketVolume'),
                    'bid_size': item.get('bidSize'),
                    'ask_size': item.get('askSize'),
                    'avg_volume': item.get('averageDailyVolume3Month'),

                    'open': item.get('regularMarketOpen'),
                    'high': item.get('regularMarketDayHigh'),
                    'low': item.get('regularMarketDayLow'),
                    'prev_close': item.get('regularMarketPreviousClose'),

                    'timestamp': item.get('regularMarketTime', dt.datetime.now().timestamp()),
                    'exchange': item.get('exchange'),
                    'currency': item.get('currency')
                }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching batch quotes: {e}")

        return quotes


    # Update Loops Methods

    def _spot_price_loop(self):
        """
        - Main spot price update loop
        - Runs every 5 seconds
        - Detects significant changes
        - Triggers option updates if needed
        """
        while self.running:
            if self.is_market_open():
                try:

                    # Trigger immediate save for real-time monitoring
                    if self.save_to_file:
                        self._save_spot_data()
                        self._save_metadata()

                    # Update priority symbols individually (full details)
                    for symbol in self.symbols:
                        try:
                            detailed_data = self._fetch_spot_price(symbol)
                            if detailed_data:  # Only update if we got data
                                with self.data_lock:
                                    if symbol not in self.spot_data:
                                        self.spot_data[symbol] = {}

                                    # Check for price change before updating
                                    old_price = self.spot_data[symbol].get('price', 0)
                                    new_price = detailed_data.get('price', 0)

                                    self.spot_data[symbol].update(detailed_data)
                                    self.last_fetched["spot"][symbol] = dt.datetime.now().timestamp()

                                    if old_price and new_price and self._detect_spot_change(symbol, old_price, new_price):
                                        self._prioritize_option_updates(symbol)

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
            else:
                time.sleep(60)  # Sleep longer when market is closed

    def _option_chain_loop(self):
        """
        - Worker method for option updates
        - Pulls from update queue
        - Fetches and stores chain data
        - Re-queues for next update
        - Multiple instances run in parallel
        """
        while self.running:
            if self.is_market_open():
                try:
                    symbol, expiry = self.option_update_queue.get(timeout=5)

                    now = time.time()
                    last_update = self.last_fetched['option'].get((symbol, expiry), 0)

                    # Pick right interval (ATM vs OTM)
                    # Determine if near the money by comparing days to expiry
                    expiry_date = dt.datetime.strptime(expiry, "%Y-%m-%d")
                    days_to_expiry = (expiry_date - dt.datetime.now()).days
                    is_near_term = days_to_expiry <= 7

                    interval = self.update_frequency['atm_options'] if is_near_term else self.update_frequency['otm_options']

                    if now - last_update >= interval:
                        # Fetch option chain
                        chain_data = self._fetch_option_chain(symbol, expiry)
                        with self.data_lock:
                            if symbol not in self.option_chain:
                                self.option_chain[symbol] = {}
                            self.option_chain[symbol][expiry] = chain_data

                        # Update timestamp
                        self.last_fetched['option'][(symbol, expiry)] = now

                        # Trigger immediate save for real-time monitoring
                        if self.save_to_file:
                            self._save_option_data()

                    # Always requeue for future updates
                    self.option_update_queue.put((symbol, expiry))
                    self.option_update_queue.task_done()

                except Empty:
                    continue
                except Exception as e:
                    self._handle_api_error(e, {'operation': '_option_chain_loop',
                                            'symbol': symbol if 'symbol' in locals() else 'unknown',
                                            'expiry': expiry if 'expiry' in locals() else 'unknown',
                                            'retry_count': 0,
                                            'max_retries': 3,
                                            'retry_delay': 1})
            else:
                time.sleep(60)  # Sleep longer when market is closed



    def _historical_data_loop(self):
        """
        - Updates historical price data
        - Runs less frequently (hourly)
        - Maintains rolling window
        - Used for volatility calculations
        """
        while self.running:
            try:
                if self.is_market_open():
                    now = time.time()

                    for symbol in self.symbols:
                        last_update = self.last_fetched['history'].get(symbol, 0)

                        if now - last_update >= self.update_frequency['history']:
                            try:
                                ticker = yf.Ticker(symbol)
                                hist = ticker.history(period="1y", interval="1d")  # Could use start/end for incremental fetch
                                with self.data_lock:
                                    self.historical_data[symbol] = hist
                                    self.last_fetched['history'][symbol] = dt.datetime.now().timestamp()

                                # Trigger immediate save for real-time monitoring
                                if self.save_to_file:
                                    self._save_historical_data()

                            except Exception as e:
                                self._handle_api_error(e, {
                                    'operation': '_historical_data_loop',
                                    'symbol': symbol,
                                    'retry_count': 0,
                                    'max_retries': 3,
                                    'retry_delay': 1
                                })

                    time.sleep(60)  # Check every minute whether updates are needed
                else:
                    time.sleep(300)  # Sleep longer when market is closed

            except Exception as e:
                self._handle_api_error(e, {'operation': '_historical_data_loop',
                                        'retry_count': 0,
                                        'max_retries': 3,
                                        'retry_delay': 1})

    # Data Processing Methods

    def _process_option_data(self, chain_df, option_type):
        """
        - Converts DataFrame to dict format
        - Extracts relevant fields
        - Handles missing/invalid data
        - Normalizes data structure
        """
        processed = []
        for _, row in chain_df.iterrows():
            try:
                processed.append({
                    'contractSymbol': row.get('contractSymbol'),
                    'strike': row.get('strike'),
                    'lastPrice': row.get('lastPrice'),
                    'bid': row.get('bid'),
                    'ask': row.get('ask'),
                    'mid': (row.get('bid', 0) + row.get('ask', 0)) / 2 if row.get('bid') and row.get('ask') else None,
                    'volume': row.get('volume'),
                    'openInterest': row.get('openInterest'),
                    'impliedVolatility': row.get('impliedVolatility'),
                    'expiration': row.get('expiration'),
                    'type': option_type,
                    'last_updated': dt.datetime.now().timestamp()
                })
            except Exception as e:
                continue
        return processed

    def _detect_spot_change(self, symbol, old_price, new_price):
        """
        - Compares new vs. cached price
        - Calculates percentage change
        - Returns boolean for significant move
        - Triggers downstream updates
        """
        if old_price == 0:
            return False
        if abs(new_price - old_price) / old_price > 0.01:  # 1% change
            return True
        return False

    def _prioritize_option_updates(self, symbol):
        """
        - Determines which options need updating
        - Prioritizes by moneyness and expiry
        - Adds to update queue with priority
        - Balances update load
        """
        try:
            spot_price = self.spot_data.get(symbol, {}).get('price')
            if not spot_price:
                return  # no spot data, skip

            ticker = yf.Ticker(symbol)
            expiries = ticker.options

            # Build a list of (expiry, priority_score)
            expiry_scores = []
            for expiry in expiries:
                try:
                    # Time to expiry in days
                    expiry_date = dt.datetime.strptime(expiry, "%Y-%m-%d")
                    days_to_expiry = max((expiry_date - dt.datetime.now()).days, 0)
                    expiry_score = 1 / (days_to_expiry + 1)  # sooner expiry = higher score

                    # For moneyness, we could fetch the chain but that's expensive
                    # Instead use time as proxy (near-term options are usually more important)
                    total_priority = expiry_score
                    expiry_scores.append((expiry, total_priority))
                except:
                    continue

            # Sort by highest priority first
            expiry_scores.sort(key=lambda x: x[1], reverse=True)

            # Enqueue updates in priority order
            for expiry, _ in expiry_scores:
                self.option_update_queue.put((symbol, expiry))

        except Exception as e:
            self._handle_api_error(e, {'operation': '_prioritize_option_updates',
                                    'symbol': symbol,
                                    'retry_count': 0,
                                    'max_retries': 3,
                                    'retry_delay': 1})

    # Utility Methods

    def _calculate_update_priority(self, symbol, data_type):
        """
        - Determines update urgency
        - Based on staleness and importance
        - Returns priority score
        - Used for queue ordering
        """
        now = time.time()

        # 1. Check last update time
        last_update = self.last_fetched.get(data_type, {}).get(symbol, 0)
        staleness = now - last_update

        # 2. Base priority on staleness vs threshold
        threshold = self.stale_threshold.get(data_type, 60)
        priority = staleness / threshold  # >1 means overdue

        # 3. Boost priority for important symbols
        if symbol in self.priority_symbols:
            priority *= 2  # double weight for key tickers

        # 4. Adjust for data type (spot updates might be more critical)
        if data_type == 'spot':
            priority *= 1.5
        elif data_type in ['atm_options', 'otm_options']:
            priority *= 1.2
        elif data_type == 'history':
            priority *= 0.8  # less frequent

        return priority

    def is_market_open(self):
        '''
        - Checks if market is currently open
        - Adjusts update behavior
        - Considers holidays
        - Returns boolean
        '''
        return validate_market_hours()

    def _clean_stale_data(self):
        """
        - Removes very old data
        - Manages memory usage
        - Runs periodically
        - Configurable retention
        """
        now = time.time()

        # Clean spot data
        for symbol in list(self.spot_data.keys()):
            last_time = self.last_fetched.get('spot', {}).get(symbol, 0)
            if now - last_time > self.stale_threshold['spot']:
                with self.data_lock:
                    self.spot_data.pop(symbol, None)
                    self.last_fetched['spot'].pop(symbol, None)

        # Clean option chains
        for symbol in list(self.option_chain.keys()):
            for expiry in list(self.option_chain[symbol].keys()):
                last_time = self.option_chain[symbol][expiry].get('last_updated', 0)
                if now - last_time > self.stale_threshold['otm_options']:
                    with self.data_lock:
                        self.option_chain[symbol].pop(expiry, None)
                        if not self.option_chain[symbol]:
                            self.option_chain.pop(symbol, None)

        # Clean historical data
        for symbol in list(self.historical_data.keys()):
            last_time = self.last_fetched.get('history', {}).get(symbol, 0)
            if now - last_time > self.stale_threshold['history']:
                with self.data_lock:
                    self.historical_data.pop(symbol, None)
                    self.last_fetched['history'].pop(symbol, None)

    def _handle_api_error(self, error, context):
        """
        - Centralized error handling
        - Logs errors appropriately
        - Implements retry logic
        - Prevents service disruption
        """
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
        self._error_count += 1

        if self._error_count % 100 == 0:
            logging.warning(f"Total API errors: {self._error_count}")

        return None

    def calculate_priority_score(self, symbol, staleness, volatility):
        """
        Determines update priority for a symbol
        Combines staleness, recent volatility, and importance into a numeric score
        Higher score = higher update priority
        """
        # Base priority from staleness (how overdue the data is)
        threshold = self.stale_threshold.get('spot', 60)  # default threshold for spot
        staleness_score = staleness / threshold  # >1 means data is stale

        # Volatility adjustment (higher volatility = higher priority)
        volatility_score = volatility * 2  # scale factor, can tune

        # Importance boost for key symbols
        importance_score = 1.5 if symbol in self.priority_symbols else 1.0

        # Combine scores
        total_score = (staleness_score + volatility_score) * importance_score

        return total_score

    # File Saving Methods

    def _file_save_loop(self):
        """
        - Continuously saves data to files
        - Runs on separate thread
        - Saves spot, options, and history data
        - Updates metadata with timestamps
        """
        while self.running:
            try:
                self._save_all_data()
                time.sleep(self.save_interval)
            except Exception as e:
                logging.error(f"Error in file save loop: {e}")
                time.sleep(5)  # Wait longer on error

    def _save_all_data(self):
        """
        - Saves all data types to their respective files
        - Thread-safe data access
        - Handles file I/O errors
        - Updates metadata
        """
        try:
            # Save spot data
            self._save_spot_data()

            # Save option chains
            self._save_option_data()

            # Save historical data
            self._save_historical_data()

            # Save metadata
            self._save_metadata()

        except Exception as e:
            logging.error(f"Error saving market data files: {e}")

    def _save_spot_data(self):
        """
        - Saves spot price data to JSON file
        - Includes all price, volume, and metadata
        - Human-readable formatting
        """
        try:
            with self.data_lock:
                spot_copy = self.spot_data.copy()

            # Convert timestamps to readable format
            for symbol, data in spot_copy.items():
                if 'timestamp' in data and data['timestamp']:
                    data['timestamp_readable'] = dt.datetime.fromtimestamp(data['timestamp']).isoformat()
                if 'market_time' in data and data['market_time']:
                    data['market_time_readable'] = dt.datetime.fromtimestamp(data['market_time']).isoformat()

            with open(self.spot_file, 'w') as f:
                json.dump(spot_copy, f, indent=2, default=str)

        except Exception as e:
            logging.error(f"Error saving spot data: {e}")

    def _save_option_data(self):
        """
        - Saves option chain data to JSON file
        - Organized by symbol and expiry
        - Includes calls and puts
        """
        try:
            with self.data_lock:
                option_copy = {}
                for symbol, expiries in self.option_chain.items():
                    option_copy[symbol] = {}
                    for expiry, data in expiries.items():
                        option_copy[symbol][expiry] = {
                            'calls': data.get('calls', []),
                            'puts': data.get('puts', []),
                            'last_updated': data.get('last_updated'),
                            'last_updated_readable': dt.datetime.fromtimestamp(data.get('last_updated', 0)).isoformat() if data.get('last_updated') else None
                        }

            with open(self.option_file, 'w') as f:
                json.dump(option_copy, f, indent=2, default=str)

        except Exception as e:
            logging.error(f"Error saving option data: {e}")

    def _save_historical_data(self):
        """
        - Saves historical price data to JSON file
        - Converts DataFrame to dict format
        - Preserves date indices
        """
        try:
            with self.data_lock:
                history_copy = {}
                for symbol, df in self.historical_data.items():
                    if not df.empty:
                        # Convert DataFrame to dict with date strings
                        df_dict = df.to_dict('index')
                        # Convert datetime index to strings
                        history_copy[symbol] = {
                            str(date): values for date, values in df_dict.items()
                        }
                    else:
                        history_copy[symbol] = {}

            with open(self.history_file, 'w') as f:
                json.dump(history_copy, f, indent=2, default=str)

        except Exception as e:
            logging.error(f"Error saving historical data: {e}")

    def _save_metadata(self):
        """
        - Saves metadata about the data collection
        - Includes update timestamps and statistics
        - Helps monitor service health
        """
        try:
            metadata = {
                'last_save_time': dt.datetime.now().isoformat(),
                'symbols_tracked': self.symbols,
                'priority_symbols': self.priority_symbols,
                'service_running': self.running,
                'total_errors': self._error_count,
                'last_fetched_times': {
                    'spot': {symbol: dt.datetime.fromtimestamp(ts).isoformat()
                            for symbol, ts in self.last_fetched['spot'].items()},
                    'options': {f"{symbol}_{expiry}": dt.datetime.fromtimestamp(ts).isoformat()
                               for (symbol, expiry), ts in self.last_fetched['option'].items()},
                    'history': {symbol: dt.datetime.fromtimestamp(ts).isoformat()
                               for symbol, ts in self.last_fetched['history'].items()}
                },
                'data_counts': {
                    'spot_symbols': len(self.spot_data),
                    'option_symbols': len(self.option_chain),
                    'total_option_chains': sum(len(expiries) for expiries in self.option_chain.values()),
                    'history_symbols': len(self.historical_data)
                },
                'update_frequencies': self.update_frequency,
                'market_open': self.is_market_open()
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logging.error(f"Error saving metadata: {e}")

    def get_data_files(self):
        """
        - Returns paths to all data files
        - Useful for external monitoring
        - Can be called to find where data is saved
        """
        return {
            'spot': self.spot_file,
            'options': self.option_file,
            'history': self.history_file,
            'metadata': self.metadata_file,
            'directory': self.data_directory
        }

    def enable_file_monitoring(self, enable=True):
        """
        - Enable or disable file saving at runtime
        - Useful for debugging or performance tuning
        """
        self.save_to_file = enable
        if enable and hasattr(self, 'save_thread') and not self.save_thread.is_alive():
            # Restart the save thread if it's not running
            self.save_thread = th.Thread(
                target=self._file_save_loop,
                name="FileSaver",
                daemon=True
            )
            self.save_thread.start()
            logging.info("File monitoring enabled")
        elif not enable:
            logging.info("File monitoring disabled")

    # Data Access Methods

    def get_spot_price(self, symbol):
        """
        - Get current spot price for a symbol
        - Returns dict with price data
        - Thread-safe access
        """
        with self.data_lock:
            return self.spot_data.get(symbol, {}).copy()

    def get_option_chain(self, symbol, expiry=None):
        """
        - Get option chain for a symbol
        - Optional expiry filter
        - Returns chain data or all expiries
        """
        with self.data_lock:
            if symbol not in self.option_chain:
                return {}

            if expiry:
                return self.option_chain[symbol].get(expiry, {}).copy()
            else:
                return self.option_chain[symbol].copy()

    def get_all_spot_prices(self):
        """
        - Get all current spot prices
        - Returns dict of all symbols
        - Thread-safe copy
        """
        with self.data_lock:
            return self.spot_data.copy()

    def get_data_summary(self):
        """
        - Get summary of all available data
        - Includes counts and last update times
        - Useful for monitoring
        """
        with self.data_lock:
            summary = {
                'spot_symbols': list(self.spot_data.keys()),
                'option_symbols': list(self.option_chain.keys()),
                'historical_symbols': list(self.historical_data.keys()),
                'total_option_chains': sum(len(expiries) for expiries in self.option_chain.values()),
                'last_updates': {
                    'spot': {symbol: dt.datetime.fromtimestamp(ts).isoformat()
                            for symbol, ts in self.last_fetched['spot'].items()},
                    'options': len(self.last_fetched['option']),
                    'history': len(self.last_fetched['history'])
                },
                'errors': self._error_count,
                'running': self.running,
                'market_open': self.is_market_open()
            }
        return summary


# Helper Functions

def validate_market_hours():
    """
    - Checks if market is open
    - Considers timezone
    - Handles holidays
    - Returns boolean
    """
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


# Example usage with file saving
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Define symbols to track
    symbols = [
                SPY, QQQ, IWM, DIA,
                AAPL, MSFT, AMZN, GOOGL, TSLA, NVDA, META,
                JPM, BAC, GS,
                VIX, UVXY, SVXY,
                XLF, XLE, XLK, GLD
            ]

    # Configuration with file saving options
    config = {
        'save_to_file': True,
        'data_directory': 'market_data_realtime',
        'save_interval': 1,  # Save every 1 second
        'file_format': 'json'
    }

    # Create and start the service
    service = MarketDataService(symbols, config)

    print(f"Starting market data service...")
    print(f"Data will be saved to: {service.get_data_files()['directory']}")
    print(f"Files:")
    for name, path in service.get_data_files().items():
        if name != 'directory':
            print(f"  {name}: {path}")

    if service.start():
        print("Service started successfully!")
        print("Press Ctrl+C to stop...")

        try:
            # Example of programmatic data access
            time.sleep(10)  # Wait for initial data

            # Get spot price for AAPL
            aapl_spot = service.get_spot_price('AAPL')
            if aapl_spot:
                print(f"\nAAPL Spot: ${aapl_spot.get('price', 'N/A')}")

            # Get summary
            summary = service.get_data_summary()
            print(f"\nTracking {len(summary['spot_symbols'])} symbols")
            print(f"Market is {'open' if summary['market_open'] else 'closed'}")

            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping service...")
            service.stop()
            print("Service stopped.")
    else:
        print("Failed to start service")
