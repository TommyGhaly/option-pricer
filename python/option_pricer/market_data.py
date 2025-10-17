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
import traceback
from typing import Optional, Dict, Any
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import tempfile

# Service architecture for fetching and storing market data accessible across modules but not directly dependent on other modules

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

# Time interval API Calls
SPOTS = 5  # 5 Second intervals
ATMS = 30  # 30 Second intervals
OTM = 60 * 5  # 5 Minute intervals
HISTORY = 60 * 60  # 1 Hour intervals or on demand


class MarketDataService:
    # Core Methods

    def __init__(self, symbols, config=None):
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

        # Threading control variables
        self.running = False
        self.data_lock = th.Lock()
        self.file_lock = th.Lock()  # Add file lock for coordinating file writes
        self.spot_thread = th.Thread()
        self.option_threads = []
        self.option_update_queue = qu()
        self.update_frequency = {'spot': SPOTS, 'atm_options': ATMS, 'otm_options': OTM, 'history': HISTORY}
        self.priority_symbols = [SPY, QQQ, AAPL]
        self.stale_threshold = {'spot': 60, 'atm_options': 300, 'otm_options': 900, 'history': 3600 * 24}  # in seconds

        # Initialize error count
        self._error_count = 0

        # Track which data needs saving
        self._dirty_flags = {
            'spot': False,
            'options': False,
            'history': False
        }

        # File saving configuration
        self.save_to_file = config.get('save_to_file', True) if config else True
        self.data_directory = config.get('data_directory', 'market_data') if config else 'market_data'
        self.save_interval = config.get('save_interval', 5) if config else 5  # Increased to 5 seconds for better performance
        self.file_format = config.get('file_format', 'json') if config else 'json'

        # Create data directory if it doesn't exist
        if self.save_to_file:
            Path(self.data_directory).mkdir(parents=True, exist_ok=True)

        # File paths
        self.spot_file = os.path.join(self.data_directory, 'spot_data.json')
        self.option_file = os.path.join(self.data_directory, 'option_chains.json')
        self.history_file = os.path.join(self.data_directory, 'historical_data.json')
        self.metadata_file = os.path.join(self.data_directory, 'metadata.json')

        # Load any existing data
        if self.save_to_file:
            self._load_existing_data()

    def _load_existing_data(self):
        """Load previously saved data if available"""
        try:
            loaded_data = self.load_saved_data()

            if 'spot' in loaded_data:
                self.spot_data = loaded_data['spot']
                logging.info(f"Loaded {len(self.spot_data)} spot prices from previous session")

            if 'options' in loaded_data:
                self.option_chain = loaded_data['options']
                logging.info(f"Loaded option chains for {len(self.option_chain)} symbols from previous session")

            if 'history' in loaded_data:
                # Note: Historical data needs conversion back to DataFrame
                # For now, we'll skip loading history and let it refresh
                logging.info("Historical data found but will be refreshed")

        except Exception as e:
            logging.warning(f"Could not load existing data: {e}")

    def start(self) -> bool:
        """
        Performs initial data fetch for all symbols
        Starts all update Threads
        Sets running flag to True
        Returns when service is ready
        """
        try:
            self.running = True
            self._initial_data_load()

            # Create and start spot price update thread
            self.spot_thread = th.Thread(
                target=self._spot_price_loop,
                name="SpotPriceUpdater",
                daemon=True
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
        Gracefully stops all update Threads
        Saves any cached data
        Cleans up resources
        Sets running flag to False
        """
        self.running = False

        # Final save of all data
        if self.save_to_file:
            self._save_all_data()

        # Wait for threads to finish
        for thread in self.option_threads:
            if thread.is_alive():
                thread.join(timeout=5)

    # Data Collection Methods

    def _initial_data_load(self):
        """Initial load using reliable yfinance Tickers method"""
        print("\n" + "="*60)
        print("STARTING INITIAL DATA LOAD")
        print("="*60)

        print("\n[1/3] Fetching spot prices...")
        self.spot_data = self._batch_fetch_spots(self.symbols)

        if not self.spot_data:
            print("\n⚠️  WARNING: No spot data loaded!")
            return

        print(f"\n✓ Loaded spot data for {len(self.spot_data)}/{len(self.symbols)} symbols")

        # Step 2: Queue option expiries
        print("\n[2/3] Loading option expiries...")
        total_expiries = 0
        successful_symbols = 0

        for i, symbol in enumerate(self.symbols):
            if symbol == '^VIX':
                print(f"  [{i+1}/{len(self.symbols)}] {symbol}: Skipped (index)")
                continue

            if not (self.spot_data.get(symbol) and self.spot_data[symbol].get('price')):
                print(f"  [{i+1}/{len(self.symbols)}] {symbol}: Skipped (no spot price)")
                continue

            try:
                ticker = yf.Ticker(symbol)
                expiries = ticker.options

                if not expiries:
                    print(f"  [{i+1}/{len(self.symbols)}] {symbol}: No options")
                    continue

                for option_expiry in expiries:
                    self.option_update_queue.put((symbol, option_expiry))
                    total_expiries += 1

                successful_symbols += 1
                print(f"  [{i+1}/{len(self.symbols)}] {symbol}: Queued {len(expiries)} expiries")

                time.sleep(0.3)

            except Exception as e:
                print(f"  [{i+1}/{len(self.symbols)}] {symbol}: Error - {e}")
                time.sleep(0.3)

        print(f"\n✓ Queued {total_expiries} option expiries from {successful_symbols} symbols")

        # Step 3: Start workers
        print("\n[3/3] Starting option worker threads...")
        num_workers = min(10, max(2, total_expiries // 20))

        for i in range(num_workers):
            t = th.Thread(
                target=self._option_chain_loop,
                name=f"OptionChainUpdater-{i}",
                daemon=True
            )
            t.start()
            self.option_threads.append(t)

        print(f"✓ Started {num_workers} option chain worker threads")
        print("\n" + "="*60)
        print("INITIAL DATA LOAD COMPLETE")
        print("="*60 + "\n")

    def _fetch_spot_price(self, symbol):
        """Fetches current spot data from yahoo"""
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
        """Retrieves complete option chain"""
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

    def _batch_fetch_spots(self, symbol_list, max_retries=2):
        """Fetches spot prices using yfinance.Tickers()"""
        quotes = {}

        if not symbol_list:
            return quotes

        for attempt in range(max_retries):
            try:
                print(f"Fetching {len(symbol_list)} symbols using Tickers method...")

                tickers = yf.Tickers(' '.join(symbol_list))

                for symbol in symbol_list:
                    try:
                        ticker = tickers.tickers[symbol]
                        info = ticker.info

                        if info and 'regularMarketPrice' in info:
                            quotes[symbol] = {
                                'price': info.get('regularMarketPrice'),
                                'bid': info.get('bid'),
                                'ask': info.get('ask'),
                                'mid': (info.get('bid', 0) + info.get('ask', 0)) / 2
                                    if info.get('bid') and info.get('ask') else info.get('regularMarketPrice'),
                                'volume': info.get('volume'),
                                'bid_size': info.get('bidSize'),
                                'ask_size': info.get('askSize'),
                                'avg_volume': info.get('averageVolume'),
                                'open': info.get('open'),
                                'high': info.get('dayHigh'),
                                'low': info.get('dayLow'),
                                'prev_close': info.get('previousClose'),
                                'timestamp': dt.datetime.now().timestamp(),
                                'exchange': info.get('exchange'),
                                'currency': info.get('currency')
                            }
                            print(f"  ✓ {symbol}: ${info.get('regularMarketPrice')}")

                        time.sleep(0.15)

                    except Exception as e:
                        print(f"  ✗ {symbol}: {e}")
                        continue

                if quotes:
                    print(f"Successfully fetched {len(quotes)}/{len(symbol_list)} spot prices")
                    return quotes
                else:
                    print(f"Attempt {attempt + 1}/{max_retries} returned no data")
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)

            except Exception as e:
                print(f"Error in batch fetch attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)

        return quotes

    # Update Loops Methods

    def _spot_price_loop(self):
        """Main spot price update loop using Tickers method"""
        loop_count = 0

        while self.running:
            if self.is_market_open():
                loop_count += 1
                try:
                    batch_data = self._batch_fetch_spots(self.symbols)

                    if batch_data:
                        updates = 0
                        price_changes = 0

                        with self.data_lock:
                            for symbol, data in batch_data.items():
                                old_price = self.spot_data.get(symbol, {}).get('price', 0)
                                new_price = data.get('price', 0)

                                self.spot_data[symbol] = data
                                self.last_fetched["spot"][symbol] = dt.datetime.now().timestamp()
                                updates += 1

                                if old_price and new_price and self._detect_spot_change(symbol, old_price, new_price):
                                    self._prioritize_option_updates(symbol)
                                    price_changes += 1

                            self._dirty_flags['spot'] = True

                        if loop_count % 12 == 0:  # Log every minute
                            print(f"Spot update: {updates} symbols updated, {price_changes} significant changes")

                except Exception as e:
                    print(f"Error in spot price loop: {e}")

                time.sleep(self.update_frequency['spot'])
            else:
                if loop_count % 60 == 0:
                    print("Market closed - spot updates paused")
                time.sleep(60)

    def _option_chain_loop(self):
        """Worker method for option updates"""
        while self.running:
            if self.is_market_open():
                try:
                    symbol, expiry = self.option_update_queue.get(timeout=5)

                    now = time.time()
                    last_update = self.last_fetched['option'].get((symbol, expiry), 0)

                    # Determine if near the money by comparing days to expiry
                    expiry_date = dt.datetime.strptime(expiry, "%Y-%m-%d")
                    days_to_expiry = (expiry_date - dt.datetime.now()).days
                    is_near_term = days_to_expiry <= 7

                    interval = self.update_frequency['atm_options'] if is_near_term else self.update_frequency['otm_options']

                    if now - last_update >= interval:
                        chain_data = self._fetch_option_chain(symbol, expiry)
                        with self.data_lock:
                            if symbol not in self.option_chain:
                                self.option_chain[symbol] = {}
                            self.option_chain[symbol][expiry] = chain_data
                            self._dirty_flags['options'] = True

                        self.last_fetched['option'][(symbol, expiry)] = now

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
                time.sleep(60)

    def _historical_data_loop(self):
        """Updates historical price data"""
        while self.running:
            try:
                if self.is_market_open():
                    now = time.time()

                    for symbol in self.symbols:
                        last_update = self.last_fetched['history'].get(symbol, 0)

                        if now - last_update >= self.update_frequency['history']:
                            try:
                                ticker = yf.Ticker(symbol)
                                hist = ticker.history(period="1y", interval="1d")
                                with self.data_lock:
                                    self.historical_data[symbol] = hist
                                    self.last_fetched['history'][symbol] = dt.datetime.now().timestamp()
                                    self._dirty_flags['history'] = True

                            except Exception as e:
                                self._handle_api_error(e, {
                                    'operation': '_historical_data_loop',
                                    'symbol': symbol,
                                    'retry_count': 0,
                                    'max_retries': 3,
                                    'retry_delay': 1
                                })

                    time.sleep(60)
                else:
                    time.sleep(300)

            except Exception as e:
                self._handle_api_error(e, {'operation': '_historical_data_loop',
                                        'retry_count': 0,
                                        'max_retries': 3,
                                        'retry_delay': 1})

    # Data Processing Methods

    def _process_option_data(self, chain_df, option_type):
        """Converts DataFrame to dict format"""
        processed = []
        for _, row in chain_df.iterrows():
            try:
                processed.append({
                    'contractSymbol': row.get('contractSymbol'),
                    'strike': row.get('strike'),
                    'lastPrice': row.get('lastPrice'),
                    'bid': row.get('bid') if self._not_NaN(row.get('bid')) else 0,
                    'ask': row.get('ask') if self._not_NaN(row.get('ask')) else 0,
                    'mid': (row.get('bid', 0) + row.get('ask', 0)) / 2 if self._not_NaN(row.get('bid')) and self._not_NaN(row.get('ask')) else 0,
                    'volume': row.get('volume', 0) if self._not_NaN(row.get('volume')) else 0,
                    'openInterest': row.get('openInterest') if self._not_NaN(row.get('openInterest')) else 0,
                    'impliedVolatility': row.get('impliedVolatility') if self._not_NaN(row.get('impliedVolatility')) else 0,
                    'type': option_type,
                    'last_updated': dt.datetime.now().timestamp()
                })
            except Exception as e:
                print(f'problem extracting data. Code failed with exception {e}')
                continue
        return processed

    def _not_NaN(self, value):
        """Determines if input is valid"""
        if not value:
            return False
        elif math.isnan(value):
            return False
        else:
            return True

    def _detect_spot_change(self, symbol, old_price, new_price):
        """Compares new vs. cached price"""
        if old_price == 0:
            return False
        if abs(new_price - old_price) / old_price > 0.01:  # 1% change
            return True
        return False

    def _prioritize_option_updates(self, symbol):
        """Determines which options need updating"""
        try:
            spot_price = self.spot_data.get(symbol, {}).get('price')
            if not spot_price:
                return

            ticker = yf.Ticker(symbol)
            expiries = ticker.options

            expiry_scores = []
            for expiry in expiries:
                try:
                    expiry_date = dt.datetime.strptime(expiry, "%Y-%m-%d")
                    days_to_expiry = max((expiry_date - dt.datetime.now()).days, 0)
                    expiry_score = 1 / (days_to_expiry + 1)
                    total_priority = expiry_score
                    expiry_scores.append((expiry, total_priority))
                except:
                    continue

            expiry_scores.sort(key=lambda x: x[1], reverse=True)

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
        """Determines update urgency"""
        now = time.time()
        last_update = self.last_fetched.get(data_type, {}).get(symbol, 0)
        staleness = now - last_update
        threshold = self.stale_threshold.get(data_type, 60)
        priority = staleness / threshold

        if symbol in self.priority_symbols:
            priority *= 2

        if data_type == 'spot':
            priority *= 1.5
        elif data_type in ['atm_options', 'otm_options']:
            priority *= 1.2
        elif data_type == 'history':
            priority *= 0.8

        return priority

    def is_market_open(self):
        """Checks if market is currently open"""
        return validate_market_hours()

    def _clean_stale_data(self):
        """Removes very old data"""
        now = time.time()

        for symbol in list(self.spot_data.keys()):
            last_time = self.last_fetched.get('spot', {}).get(symbol, 0)
            if now - last_time > self.stale_threshold['spot']:
                with self.data_lock:
                    self.spot_data.pop(symbol, None)
                    self.last_fetched['spot'].pop(symbol, None)

        for symbol in list(self.option_chain.keys()):
            for expiry in list(self.option_chain[symbol].keys()):
                last_time = self.option_chain[symbol][expiry].get('last_updated', 0)
                if now - last_time > self.stale_threshold['otm_options']:
                    with self.data_lock:
                        self.option_chain[symbol].pop(expiry, None)
                        if not self.option_chain[symbol]:
                            self.option_chain.pop(symbol, None)

        for symbol in list(self.historical_data.keys()):
            last_time = self.last_fetched.get('history', {}).get(symbol, 0)
            if now - last_time > self.stale_threshold['history']:
                with self.data_lock:
                    self.historical_data.pop(symbol, None)
                    self.last_fetched['history'].pop(symbol, None)

    def _handle_api_error(self, error, context):
        """Centralized error handling"""
        operation = context.get('operation', 'Unknown operation')
        symbol = context.get('symbol', 'N/A')
        retry_count = context.get('retry_count', 0)
        max_retries = context.get('max_retries', 3)
        retry_delay = context.get('retry_delay', 1)

        if retry_count == 0:
            logging.warning(f"API Error in {operation} for {symbol}: {type(error).__name__}: {str(error)}")
            if not isinstance(error, (ConnectionError, TimeoutError)):
                logging.debug(f"Traceback:\n{traceback.format_exc()}")

        retryable_errors = (ConnectionError, TimeoutError)
        is_retryable = isinstance(error, retryable_errors) or \
                    "HTTPError" in str(type(error)) or \
                    "URLError" in str(type(error)) or \
                    (hasattr(error, 'response') and
                        hasattr(error.response, 'status_code') and
                        500 <= error.response.status_code < 600)

        if is_retryable and retry_count < max_retries:
            delay = retry_delay * (2 ** retry_count) + (time.time() % 1)
            logging.info(f"Retrying {operation} for {symbol} in {delay:.1f}s (attempt {retry_count + 1}/{max_retries})")
            time.sleep(delay)
            context['retry_count'] = retry_count + 1
            context['should_retry'] = True
            return None

        if retry_count >= max_retries:
            logging.error(f"Max retries exceeded for {operation} (symbol: {symbol})")
        else:
            logging.error(f"Non-retryable error in {operation} for {symbol}")

        context['should_retry'] = False
        self._error_count += 1

        if self._error_count % 100 == 0:
            logging.warning(f"Total API errors: {self._error_count}")

        return None

    def calculate_priority_score(self, symbol, staleness, volatility):
        """Determines update priority for a symbol"""
        threshold = self.stale_threshold.get('spot', 60)
        staleness_score = staleness / threshold
        volatility_score = volatility * 2
        importance_score = 1.5 if symbol in self.priority_symbols else 1.0
        total_score = (staleness_score + volatility_score) * importance_score
        return total_score

    # Thread-Safe File Saving Methods

    def _atomic_json_save(self, filepath, data):
        """
        Atomically saves JSON data to file using temp file + rename pattern.
        This prevents corruption from partial writes or concurrent access.
        """
        directory = os.path.dirname(filepath)
        with tempfile.NamedTemporaryFile(mode='w',
                                         delete=False,
                                         dir=directory,
                                         prefix='.tmp_',
                                         suffix='.json') as temp_file:
            try:
                json.dump(data, temp_file, indent=2, default=str)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
                temp_path = temp_file.name
            except Exception as e:
                temp_path = temp_file.name
                os.unlink(temp_path)  # Clean up temp file on error
                raise e

        try:
            # On Windows, need to remove target first
            if os.name == 'nt' and os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_path, filepath)
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def _file_save_loop(self):
        """Continuously saves data to files"""
        while self.running:
            try:
                self._save_all_data()
                time.sleep(self.save_interval)
            except Exception as e:
                logging.error(f"Error in file save loop: {e}")
                time.sleep(5)

    def _save_all_data(self):
        """Saves all data types to their respective files"""
        try:
            # Only save data that has changed
            if self._dirty_flags.get('spot', False):
                self._save_spot_data()
                self._dirty_flags['spot'] = False

            if self._dirty_flags.get('options', False):
                self._save_option_data()
                self._dirty_flags['options'] = False

            if self._dirty_flags.get('history', False):
                self._save_historical_data()
                self._dirty_flags['history'] = False

            # Always save metadata
            self._save_metadata()

        except Exception as e:
            logging.error(f"Error saving market data files: {e}")

    def _save_spot_data(self):
        """Thread-safe save of spot price data"""
        try:
            with self.data_lock:
                spot_copy = self.spot_data.copy()

            for symbol, data in spot_copy.items():
                if 'timestamp' in data and data['timestamp']:
                    data['timestamp_readable'] = dt.datetime.fromtimestamp(
                        data['timestamp']).isoformat()
                if 'market_time' in data and data['market_time']:
                    data['market_time_readable'] = dt.datetime.fromtimestamp(
                        data['market_time']).isoformat()

            with self.file_lock:
                self._atomic_json_save(self.spot_file, spot_copy)

        except Exception as e:
            logging.error(f"Error saving spot data: {e}")

    def _save_option_data(self):
        """Thread-safe save of option chain data"""
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
                            'last_updated_readable': dt.datetime.fromtimestamp(
                                data.get('last_updated', 0)).isoformat()
                                if data.get('last_updated') else None
                        }

            with self.file_lock:
                self._atomic_json_save(self.option_file, option_copy)

        except Exception as e:
            logging.error(f"Error saving option data: {e}")

    def _save_historical_data(self):
        """Thread-safe save of historical price data"""
        try:
            with self.data_lock:
                history_copy = {}
                for symbol, df in self.historical_data.items():
                    if not df.empty:
                        df_dict = df.to_dict('index')
                        history_copy[symbol] = {
                            str(date): values for date, values in df_dict.items()
                        }
                    else:
                        history_copy[symbol] = {}

            with self.file_lock:
                self._atomic_json_save(self.history_file, history_copy)

        except Exception as e:
            logging.error(f"Error saving historical data: {e}")

    def _save_metadata(self):
        """Thread-safe save of metadata"""
        try:
            with self.data_lock:
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

            with self.file_lock:
                self._atomic_json_save(self.metadata_file, metadata)

        except Exception as e:
            logging.error(f"Error saving metadata: {e}")

    def load_saved_data(self):
        """Safely loads previously saved data from JSON files"""
        loaded_data = {}

        if os.path.exists(self.spot_file):
            try:
                with self.file_lock:
                    with open(self.spot_file, 'r') as f:
                        loaded_data['spot'] = json.load(f)
                logging.info(f"Loaded {len(loaded_data['spot'])} spot prices from file")
            except Exception as e:
                logging.error(f"Error loading spot data: {e}")

        if os.path.exists(self.option_file):
            try:
                with self.file_lock:
                    with open(self.option_file, 'r') as f:
                        loaded_data['options'] = json.load(f)
                logging.info(f"Loaded option data for {len(loaded_data['options'])} symbols from file")
            except Exception as e:
                logging.error(f"Error loading option data: {e}")

        if os.path.exists(self.history_file):
            try:
                with self.file_lock:
                    with open(self.history_file, 'r') as f:
                        loaded_data['history'] = json.load(f)
                logging.info(f"Loaded historical data for {len(loaded_data['history'])} symbols from file")
            except Exception as e:
                logging.error(f"Error loading historical data: {e}")

        return loaded_data

    def get_data_files(self):
        """Returns paths to all data files"""
        return {
            'spot': self.spot_file,
            'options': self.option_file,
            'history': self.history_file,
            'metadata': self.metadata_file,
            'directory': self.data_directory
        }

    def enable_file_monitoring(self, enable=True):
        """Enable or disable file saving at runtime"""
        self.save_to_file = enable
        if enable and hasattr(self, 'save_thread') and not self.save_thread.is_alive():
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
        """Get current spot price for a symbol"""
        with self.data_lock:
            return self.spot_data.get(symbol, {}).copy()

    def get_option_chain(self, symbol, expiry=None):
        """Get option chain for a symbol"""
        with self.data_lock:
            if symbol not in self.option_chain:
                return {}
            if expiry:
                return self.option_chain[symbol].get(expiry, {}).copy()
            else:
                return self.option_chain[symbol].copy()

    def get_all_spot_prices(self):
        """Get all current spot prices"""
        with self.data_lock:
            return self.spot_data.copy()

    def get_data_summary(self):
        """Get summary of all available data"""
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
    """Checks if market is open"""
    et_tz = pytz.timezone('America/New_York')
    now_et = dt.datetime.now(et_tz)

    if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    nyse = mcal.get_calendar('NYSE')
    today_str = now_et.strftime('%Y-%m-%d')

    schedule = nyse.schedule(start_date=today_str, end_date=today_str)
    if schedule.empty:
        return False

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
        'save_interval': 5,  # Increased to 5 seconds for better performance
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
            time.sleep(10)  # Wait for initial data

            aapl_spot = service.get_spot_price('AAPL')
            if aapl_spot:
                print(f"\nAAPL Spot: ${aapl_spot.get('price', 'N/A')}")

            summary = service.get_data_summary()
            print(f"\nTracking {len(summary['spot_symbols'])} symbols")
            print(f"Market is {'open' if summary['market_open'] else 'closed'}")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping service...")
            service.stop()
            print("Service stopped.")
    else:
        print("Failed to start service")
