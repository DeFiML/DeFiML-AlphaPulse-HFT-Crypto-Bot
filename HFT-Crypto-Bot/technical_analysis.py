import pandas as pd
import numpy as np
import ta
from typing import Optional, Union, List

class TechnicalAnalysis:
    """Technical analysis calculations for trading strategies"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI calculation period
            column: Column to calculate RSI on (default: 'close')
        """
        df = df.copy()
        
        # Use ta library for RSI calculation
        df['rsi'] = ta.momentum.rsi(df[column], window=period)
        
        return df
    
    def calculate_ma(self, df: pd.DataFrame, period: int, ma_type: str = 'sma', column: str = 'close') -> pd.DataFrame:
        """
        Calculate Moving Average
        
        Args:
            df: DataFrame with OHLCV data
            period: Moving average period
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            column: Column to calculate MA on
        """
        df = df.copy()
        
        if ma_type.lower() == 'sma':
            df[f'{ma_type}_{period}'] = ta.trend.sma_indicator(df[column], window=period)
        elif ma_type.lower() == 'ema':
            df[f'{ma_type}_{period}'] = ta.trend.ema_indicator(df[column], window=period)
        elif ma_type.lower() == 'wma':
            df[f'{ma_type}_{period}'] = ta.trend.wma_indicator(df[column], window=period)
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2, column: str = 'close') -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = df.copy()
        
        df['bb_upper'] = ta.volatility.bollinger_hband(df[column], window=period)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df[column], window=period) 
        df['bb_lower'] = ta.volatility.bollinger_lband(df[column], window=period)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'close') -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        df = df.copy()
        
        df['macd'] = ta.trend.macd(df[column], window_fast=fast_period, window_slow=slow_period)
        df['macd_signal'] = ta.trend.macd_signal(df[column], window_fast=fast_period, window_slow=slow_period, window_sign=signal_period)
        df['macd_histogram'] = ta.trend.macd_diff(df[column], window_fast=fast_period, window_slow=slow_period, window_sign=signal_period)
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df = df.copy()
        
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period)
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR)"""
        df = df.copy()
        
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        
        return df
    
    def calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 20, level_type: str = 'both') -> Union[float, tuple, None]:
        """
        Calculate dynamic support and resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of periods to look back
            level_type: 'support', 'resistance', or 'both'
        """
        if len(df) < lookback:
            return None
        
        recent_data = df.tail(lookback)
        
        if level_type == 'support':
            return recent_data['low'].min()
        elif level_type == 'resistance':
            return recent_data['high'].max()
        elif level_type == 'both':
            support = recent_data['low'].min()
            resistance = recent_data['high'].max()
            return support, resistance
        
        return None
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pivot Points"""
        df = df.copy()
        
        # Use previous day's high, low, close for pivot calculation
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        df['pivot'] = (prev_high + prev_low + prev_close) / 3
        df['r1'] = 2 * df['pivot'] - prev_low
        df['s1'] = 2 * df['pivot'] - prev_high
        df['r2'] = df['pivot'] + (prev_high - prev_low)
        df['s2'] = df['pivot'] - (prev_high - prev_low)
        df['r3'] = prev_high + 2 * (df['pivot'] - prev_low)
        df['s3'] = prev_low - 2 * (prev_high - df['pivot'])
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df = df.copy()
        
        # Volume Moving Average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=10) * 100
        
        # On Balance Volume (OBV)
        df['price_change'] = df['close'].diff()
        df['obv'] = 0
        
        for i in range(1, len(df)):
            if df.iloc[i]['price_change'] > 0:
                df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1]['obv'] + df.iloc[i]['volume']
            elif df.iloc[i]['price_change'] < 0:
                df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1]['obv'] - df.iloc[i]['volume']
            else:
                df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1]['obv']
        
        df.drop('price_change', axis=1, inplace=True)
        
        return df
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect common candlestick patterns"""
        df = df.copy()
        
        # Manual pattern detection (simplified)
        body_size = abs(df['close'] - df['open'])
        candle_range = df['high'] - df['low']
        
        # Doji pattern (small body relative to range)
        df['doji'] = (body_size / candle_range < 0.1).astype(int) * 100
        
        # Initialize other patterns as zeros
        df['hammer'] = 0
        df['shooting_star'] = 0
        df['engulfing_bullish'] = 0
        df['morning_star'] = 0
        df['evening_star'] = 0
        
        return df
    
    def calculate_trend_strength(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate trend strength indicators"""
        df = df.copy()
        
        # Simplified trend strength calculation
        price_change = df['close'].diff()
        df['trend_strength'] = price_change.rolling(window=period).mean() / df['close'].rolling(window=period).std()
        df['adx'] = abs(df['trend_strength']) * 100
        df['plus_di'] = 0
        df['minus_di'] = 0
        
        return df
    
    def get_signal_strength(self, df: pd.DataFrame, rsi_buy: float, rsi_sell: float) -> pd.DataFrame:
        """Calculate signal strength based on multiple indicators"""
        df = df.copy()
        
        df['signal_strength'] = 0
        
        # RSI signals
        df.loc[df['rsi'] < rsi_buy, 'signal_strength'] += 1
        df.loc[df['rsi'] > rsi_sell, 'signal_strength'] -= 1
        
        # MA crossover signals
        if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
            df.loc[df['fast_ma'] > df['slow_ma'], 'signal_strength'] += 1
            df.loc[df['fast_ma'] < df['slow_ma'], 'signal_strength'] -= 1
        
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df.loc[df['macd'] > df['macd_signal'], 'signal_strength'] += 1
            df.loc[df['macd'] < df['macd_signal'], 'signal_strength'] -= 1
        
        # Volume confirmation
        if 'volume_ma' in df.columns:
            df.loc[df['volume'] > df['volume_ma'], 'signal_strength'] += 0.5
        
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Calculate all technical indicators with given configuration"""
        if config is None:
            config = {
                'rsi_period': 6,
                'fast_ma_period': 3,
                'slow_ma_period': 33,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2,
                'atr_period': 14,
                'stoch_k': 14,
                'stoch_d': 3
            }
        
        df = self.calculate_rsi(df, config['rsi_period'])
        df = self.calculate_ma(df, config['fast_ma_period'], 'fast_ma')
        df = self.calculate_ma(df, config['slow_ma_period'], 'slow_ma')
        df = self.calculate_macd(df, config['macd_fast'], config['macd_slow'], config['macd_signal'])
        df = self.calculate_bollinger_bands(df, config['bb_period'], config['bb_std'])
        df = self.calculate_atr(df, config['atr_period'])
        df = self.calculate_stochastic(df, config['stoch_k'], config['stoch_d'])
        df = self.calculate_volume_indicators(df)
        df = self.calculate_trend_strength(df)
        df = self.detect_patterns(df)
        
        return df
