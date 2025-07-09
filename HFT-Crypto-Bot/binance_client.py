import requests
import pandas as pd
import time
from typing import Optional, List, Dict
import os

class BinanceClient:
    """Binance API client for fetching market data"""
    
    def __init__(self):
        self.base_url = "https://api.binance.us/api/v3"
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'CryptoTradingBot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to Binance API with error handling"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit exceeded, wait and retry
                time.sleep(1)
                return self._make_request(endpoint, params)
            else:
                print(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            print(f"Request error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Validate symbol format
        if not self._validate_symbol_format(symbol):
            print(f"Invalid symbol format: {symbol}")
            return None
            
        endpoint = "/ticker/price"
        params = {"symbol": symbol.upper().strip()}
        
        data = self._make_request(endpoint, params)
        if data and 'price' in data:
            return float(data['price'])
        
        return None
    
    def _validate_symbol_format(self, symbol: str) -> bool:
        """Validate symbol format for Binance API"""
        import re
        if not symbol:
            return False
        # Binance symbol pattern: uppercase letters, numbers, hyphens, underscores, dots
        pattern = r'^[A-Z0-9\-_.]{1,20}$'
        return bool(re.match(pattern, symbol.upper().strip()))
    
    def get_24hr_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker statistics for a symbol"""
        if not self._validate_symbol_format(symbol):
            print(f"Invalid symbol format: {symbol}")
            return None
            
        endpoint = "/ticker/24hr"
        params = {"symbol": symbol.upper().strip()}
        
        return self._make_request(endpoint, params)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[List]:
        """
        Get kline/candlestick data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to return (max 1000, default 100)
        """
        if not self._validate_symbol_format(symbol):
            print(f"Invalid symbol format: {symbol}")
            return None
            
        endpoint = "/klines"
        params = {
            "symbol": symbol.upper().strip(),
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        return self._make_request(endpoint, params)
    
    def get_historical_klines(self, symbol: str, interval: str, start_time: int, end_time: int = None) -> Optional[List]:
        """
        Get historical kline data for a specific time range
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_time: Start time in milliseconds
            end_time: End time in milliseconds (optional)
        """
        endpoint = "/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": 1000
        }
        
        if end_time:
            params["endTime"] = end_time
        
        return self._make_request(endpoint, params)
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get order book depth for a symbol"""
        endpoint = "/depth"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        return self._make_request(endpoint, params)
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> Optional[List]:
        """Get recent trades for a symbol"""
        endpoint = "/trades"
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000)
        }
        
        return self._make_request(endpoint, params)
    
    def get_exchange_info(self, symbol: str = None) -> Optional[Dict]:
        """Get exchange trading rules and symbol information"""
        endpoint = "/exchangeInfo"
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        return self._make_request(endpoint, params)
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all available trading symbols"""
        exchange_info = self.get_exchange_info()
        
        if exchange_info and 'symbols' in exchange_info:
            symbols = []
            for symbol_info in exchange_info['symbols']:
                if symbol_info['status'] == 'TRADING' and symbol_info['quoteAsset'] == 'USDT':
                    symbols.append(symbol_info['symbol'])
            return sorted(symbols)
        
        return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a specific symbol"""
        exchange_info = self.get_exchange_info(symbol)
        
        if exchange_info and 'symbols' in exchange_info:
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
        
        return None
    
    def get_server_time(self) -> Optional[int]:
        """Get server time from Binance"""
        endpoint = "/time"
        data = self._make_request(endpoint)
        
        if data and 'serverTime' in data:
            return data['serverTime']
        
        return None
    
    def ping(self) -> bool:
        """Test connectivity to Binance API"""
        endpoint = "/ping"
        data = self._make_request(endpoint)
        
        return data is not None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and is tradeable"""
        symbol_info = self.get_symbol_info(symbol)
        
        if symbol_info:
            return symbol_info['status'] == 'TRADING'
        
        return False
    
    def get_price_change_24h(self, symbol: str) -> Optional[Dict]:
        """Get 24h price change statistics"""
        ticker_data = self.get_24hr_ticker(symbol)
        
        if ticker_data:
            return {
                'symbol': ticker_data['symbol'],
                'price_change': float(ticker_data['priceChange']),
                'price_change_percent': float(ticker_data['priceChangePercent']),
                'high_price': float(ticker_data['highPrice']),
                'low_price': float(ticker_data['lowPrice']),
                'volume': float(ticker_data['volume']),
                'quote_volume': float(ticker_data['quoteVolume'])
            }
        
        return None
    
    def is_market_open(self) -> bool:
        """Check if the market is open (crypto markets are always open)"""
        return self.ping()
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        endpoint = "/ticker/price"
        data = self._make_request(endpoint)
        
        prices = {}
        if data:
            for item in data:
                if item['symbol'] in symbols:
                    prices[item['symbol']] = float(item['price'])
        
        return prices
    
    def get_historical_data_for_backtest(self, symbol: str, interval: str = '1h', days: int = 365) -> Optional[List]:
        """
        Get historical data for backtesting purposes
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1h, 4h, 1d recommended for backtesting)
            days: Number of days of historical data (default 365 for 1 year)
        
        Returns:
            List of kline data or None if error
        """
        try:
            import time
            from datetime import datetime, timedelta
            
            # Calculate timestamps
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago
            
            all_data = []
            current_start = start_time
            
            # Fetch data in chunks (max 1000 per request)
            while current_start < end_time:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_time,
                    "limit": 1000
                }
                
                self._rate_limit()  # Respect rate limits
                data = self._make_request("/klines", params)
                
                if not data:
                    break
                
                all_data.extend(data)
                
                if len(data) < 1000:
                    break
                
                # Update start time for next batch
                current_start = data[-1][6] + 1  # Close time + 1ms
                
                # Small delay between requests
                time.sleep(0.1)
            
            return all_data
            
        except Exception as e:
            print(f"Error fetching historical data for backtesting: {e}")
            return None
