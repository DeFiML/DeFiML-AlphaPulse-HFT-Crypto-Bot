"""
Backtesting strategy implementation using the backtesting library
"""
from backtesting import Strategy
import pandas as pd


class RSIMAStrategy(Strategy):
    """RSI and Moving Average strategy for the backtesting library"""
    
    # Strategy parameters
    rsi_buy = 30
    rsi_sell = 70
    fast_ma = 20
    slow_ma = 50
    
    def init(self):
        """Initialize indicators"""
        # Calculate RSI
        self.rsi = self.I(self.calculate_rsi, self.data.Close, 14)
        
        # Calculate moving averages
        self.fast_ma_line = self.I(self.sma, self.data.Close, self.fast_ma)
        self.slow_ma_line = self.I(self.sma, self.data.Close, self.slow_ma)
    
    def next(self):
        """Execute trading logic on each bar"""
        # Get current values
        current_rsi = self.rsi[-1]
        current_fast_ma = self.fast_ma_line[-1]
        current_slow_ma = self.slow_ma_line[-1]
        
        # Buy conditions: RSI oversold and fast MA above slow MA
        if (current_rsi < self.rsi_buy and 
            current_fast_ma > current_slow_ma and 
            not self.position):
            self.buy()
        
        # Sell conditions: RSI overbought or fast MA below slow MA
        elif (current_rsi > self.rsi_sell or 
              current_fast_ma < current_slow_ma) and self.position:
            self.sell()
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    @staticmethod
    def sma(values, period):
        """Calculate Simple Moving Average"""
        return pd.Series(values).rolling(window=period).mean().fillna(method='bfill').values