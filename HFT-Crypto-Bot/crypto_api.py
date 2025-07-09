import requests
import pandas as pd
import time
from typing import Optional, List, Dict
import json
from datetime import datetime, timedelta

class CryptoAPI:
    """Cryptocurrency API client using CoinGecko for fetching market data"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'CryptoTradingBot/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting for CoinGecko (50 calls per minute for free tier)
        self.last_request_time = 0
        self.min_request_interval = 1.2  # 1.2 seconds between requests
        
        # Symbol mapping
        self.symbol_map = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'ADAUSDT': 'cardano',
            'DOTUSDT': 'polkadot',
            'LINKUSDT': 'chainlink',
            'BNBUSDT': 'binancecoin'
        }
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to CoinGecko API with error handling"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API request failed: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"Request error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        coin_id = self.symbol_map.get(symbol)
        if not coin_id:
            return None
        
        endpoint = "/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd"
        }
        
        data = self._make_request(endpoint, params)
        if data and coin_id in data:
            return float(data[coin_id]['usd'])
        
        return None
    
    def get_24hr_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24hr ticker statistics for a symbol"""
        coin_id = self.symbol_map.get(symbol)
        if not coin_id:
            return None
        
        endpoint = f"/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        
        data = self._make_request(endpoint, params)
        if data and 'market_data' in data:
            market_data = data['market_data']
            return {
                'symbol': symbol,
                'price': market_data['current_price']['usd'],
                'price_change_24h': market_data['price_change_24h'],
                'price_change_percentage_24h': market_data['price_change_percentage_24h'],
                'high_24h': market_data['high_24h']['usd'],
                'low_24h': market_data['low_24h']['usd'],
                'volume': market_data['total_volume']['usd']
            }
        
        return None
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[List]:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (not used in CoinGecko, defaults to daily)
            limit: Number of data points to return
        """
        coin_id = self.symbol_map.get(symbol)
        if not coin_id:
            return None
        
        # CoinGecko uses days parameter
        days = min(limit, 365)  # Max 365 days for free tier
        
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "hourly" if days <= 90 else "daily"
        }
        
        data = self._make_request(endpoint, params)
        if data and 'prices' in data:
            klines = []
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            for i, price_data in enumerate(prices):
                timestamp = int(price_data[0])
                price = float(price_data[1])
                volume = float(volumes[i][1]) if i < len(volumes) else 0
                
                # Create OHLC data (simplified - using price as all OHLC values)
                klines.append([
                    timestamp,  # timestamp
                    price,      # open
                    price * 1.002,  # high (small variation)
                    price * 0.998,  # low (small variation)
                    price,      # close
                    volume,     # volume
                    timestamp,  # close_time
                    volume,     # quote_asset_volume
                    100,        # number_of_trades
                    volume * 0.5,  # taker_buy_base_asset_volume
                    volume * 0.5,  # taker_buy_quote_asset_volume
                    0           # ignore
                ])
            
            return klines[-limit:] if limit < len(klines) else klines
        
        return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        coin_ids = []
        symbol_to_id = {}
        
        for symbol in symbols:
            coin_id = self.symbol_map.get(symbol)
            if coin_id:
                coin_ids.append(coin_id)
                symbol_to_id[coin_id] = symbol
        
        if not coin_ids:
            return {}
        
        endpoint = "/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd"
        }
        
        data = self._make_request(endpoint, params)
        prices = {}
        
        if data:
            for coin_id, price_data in data.items():
                symbol = symbol_to_id.get(coin_id)
                if symbol:
                    prices[symbol] = float(price_data['usd'])
        
        return prices
    
    def ping(self) -> bool:
        """Test connectivity to CoinGecko API"""
        endpoint = "/ping"
        data = self._make_request(endpoint)
        return data is not None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported"""
        return symbol in self.symbol_map
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all supported trading symbols"""
        return list(self.symbol_map.keys())
    
    def is_market_open(self) -> bool:
        """Check if the market is open (crypto markets are always open)"""
        return self.ping()