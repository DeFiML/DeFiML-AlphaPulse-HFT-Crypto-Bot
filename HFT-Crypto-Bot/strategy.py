import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class TradingStrategy:
    """Implementation of RSI/MA trading strategy with risk management"""
    
    def __init__(self):
        self.name = "RSI_MA_Strategy"
        self.version = "1.0"
        self.signals_history = []
        self.trades_history = []
        self.performance_metrics = {}
    
    def generate_signals(self, df: pd.DataFrame, rsi_buy_threshold: float = 20, 
                        rsi_sell_threshold: float = 80) -> pd.DataFrame:
        """
        Generate trading signals based on RSI and MA strategy
        
        Strategy Rules:
        BUY:
        - RSI below buy threshold (10-30)
        - Fast MA > Slow MA (trend confirmation)
        
        SELL:
        - RSI above sell threshold (70-100)
        - Fast MA < Slow MA (trend confirmation)
        """
        signals_df = df.copy()
        signals_df['signal'] = 0
        signals_df['signal_strength'] = 0.0
        signals_df['signal_reason'] = ''
        
        # Ensure required columns exist
        required_columns = ['rsi', 'fast_ma', 'slow_ma', 'close']
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            return signals_df
        
        # Remove rows with NaN values in critical columns
        signals_df = signals_df.dropna(subset=required_columns)
        
        if len(signals_df) == 0:
            return signals_df
        
        # Generate BUY signals
        buy_condition = (
            (signals_df['rsi'] < rsi_buy_threshold) &
            (signals_df['fast_ma'] > signals_df['slow_ma'])
        )
        
        # Generate SELL signals
        sell_condition = (
            (signals_df['rsi'] > rsi_sell_threshold) &
            (signals_df['fast_ma'] < signals_df['slow_ma'])
        )
        
        # Apply signals
        signals_df.loc[buy_condition, 'signal'] = 1
        signals_df.loc[sell_condition, 'signal'] = -1
        
        # Calculate signal strength (0-100)
        if buy_condition.any():
            buy_strength = self._calculate_buy_strength(
                signals_df.loc[buy_condition], rsi_buy_threshold
            )
            signals_df.loc[buy_condition, 'signal_strength'] = buy_strength.astype(float)
        
        if sell_condition.any():
            sell_strength = self._calculate_sell_strength(
                signals_df.loc[sell_condition], rsi_sell_threshold
            )
            signals_df.loc[sell_condition, 'signal_strength'] = sell_strength.astype(float)
        
        # Add signal reasons
        signals_df.loc[buy_condition, 'signal_reason'] = 'RSI Oversold + MA Bullish'
        signals_df.loc[sell_condition, 'signal_reason'] = 'RSI Overbought + MA Bearish'
        
        # Add signal confirmation with additional filters
        signals_df = self._add_signal_confirmation(signals_df)
        
        # Store signals history
        self._update_signals_history(signals_df)
        
        return signals_df
    
    def _calculate_buy_strength(self, buy_signals: pd.DataFrame, threshold: float) -> pd.Series:
        """Calculate strength of buy signals (0-100)"""
        if len(buy_signals) == 0:
            return pd.Series()
        
        # Base strength from RSI distance from threshold
        rsi_strength = (threshold - buy_signals['rsi']) / threshold * 50
        
        # MA divergence strength
        ma_divergence = (buy_signals['fast_ma'] - buy_signals['slow_ma']) / buy_signals['slow_ma'] * 100
        ma_strength = np.clip(ma_divergence * 10, 0, 30)
        
        # Volume confirmation (if available)
        volume_strength = 0
        if 'volume' in buy_signals.columns and 'volume_ma' in buy_signals.columns:
            volume_ratio = buy_signals['volume'] / buy_signals['volume_ma']
            volume_strength = np.clip((volume_ratio - 1) * 20, 0, 20)
        
        total_strength = rsi_strength + ma_strength + volume_strength
        return np.clip(total_strength, 0, 100)
    
    def _calculate_sell_strength(self, sell_signals: pd.DataFrame, threshold: float) -> pd.Series:
        """Calculate strength of sell signals (0-100)"""
        if len(sell_signals) == 0:
            return pd.Series()
        
        # Base strength from RSI distance from threshold
        rsi_strength = (sell_signals['rsi'] - threshold) / (100 - threshold) * 50
        
        # MA divergence strength
        ma_divergence = (sell_signals['slow_ma'] - sell_signals['fast_ma']) / sell_signals['fast_ma'] * 100
        ma_strength = np.clip(ma_divergence * 10, 0, 30)
        
        # Volume confirmation (if available)
        volume_strength = 0
        if 'volume' in sell_signals.columns and 'volume_ma' in sell_signals.columns:
            volume_ratio = sell_signals['volume'] / sell_signals['volume_ma']
            volume_strength = np.clip((volume_ratio - 1) * 20, 0, 20)
        
        total_strength = rsi_strength + ma_strength + volume_strength
        return np.clip(total_strength, 0, 100)
    
    def _add_signal_confirmation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional signal confirmation filters"""
        df['confirmed_signal'] = df['signal'].copy()
        
        # Minimum signal strength threshold
        min_strength = 60
        weak_signals = df['signal_strength'] < min_strength
        df.loc[weak_signals, 'confirmed_signal'] = 0
        
        # Avoid signals in consolidation (low volatility)
        if 'atr' in df.columns:
            # Calculate ATR percentile
            atr_percentile = df['atr'].rolling(window=50).rank(pct=True)
            low_volatility = atr_percentile < 0.3
            df.loc[low_volatility, 'confirmed_signal'] = 0
        
        # Trend strength confirmation (if ADX available)
        if 'adx' in df.columns:
            weak_trend = df['adx'] < 25
            df.loc[weak_trend, 'confirmed_signal'] = 0
        
        return df
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float,
                              entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            account_balance: Total account balance
            risk_per_trade: Risk percentage per trade (0.1% = 0.1)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
        """
        if stop_loss <= 0 or entry_price <= 0:
            return 0
        
        # Calculate risk amount in currency
        risk_amount = account_balance * (risk_per_trade / 100)
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Position size should not exceed certain percentage of account
        max_position_value = account_balance * 0.1  # Max 10% of account per trade
        max_position_size = max_position_value / entry_price
        
        return min(position_size, max_position_size)
    
    def calculate_stop_loss(self, df: pd.DataFrame, signal_type: int, lookback: int = 20,
                           atr_multiplier: float = 2.0) -> Optional[float]:
        """
        Calculate dynamic stop loss based on recent price action and ATR
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 1 for buy, -1 for sell
            lookback: Number of periods to look back for support/resistance
            atr_multiplier: ATR multiplier for dynamic stop loss
        """
        if len(df) < lookback:
            return None
        
        recent_data = df.tail(lookback)
        current_price = df.iloc[-1]['close']
        
        if signal_type == 1:  # BUY signal - stop loss below support
            support_level = recent_data['low'].min()
            
            # Use ATR for dynamic stop loss if available
            if 'atr' in df.columns:
                current_atr = df.iloc[-1]['atr']
                atr_stop = current_price - (current_atr * atr_multiplier)
                stop_loss = min(support_level, atr_stop)
            else:
                stop_loss = support_level
                
        else:  # SELL signal - stop loss above resistance
            resistance_level = recent_data['high'].max()
            
            # Use ATR for dynamic stop loss if available
            if 'atr' in df.columns:
                current_atr = df.iloc[-1]['atr']
                atr_stop = current_price + (current_atr * atr_multiplier)
                stop_loss = max(resistance_level, atr_stop)
            else:
                stop_loss = resistance_level
        
        return stop_loss
    
    def calculate_take_profit(self, df: pd.DataFrame, signal_type: int, entry_price: float,
                             stop_loss: float, risk_reward_ratio: float = 2.0,
                             lookback: int = 20) -> Optional[float]:
        """
        Calculate take profit levels based on risk-reward ratio and resistance/support
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 1 for buy, -1 for sell
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_reward_ratio: Risk to reward ratio (2.0 = 1:2 risk/reward)
            lookback: Number of periods to look back
        """
        if len(df) < lookback or stop_loss <= 0:
            return None
        
        recent_data = df.tail(lookback)
        risk_amount = abs(entry_price - stop_loss)
        
        if signal_type == 1:  # BUY signal - take profit above resistance
            # Calculate based on risk-reward ratio
            rr_take_profit = entry_price + (risk_amount * risk_reward_ratio)
            
            # Find next resistance level
            resistance_level = recent_data['high'].max()
            
            # Use the more conservative (closer) target
            take_profit = min(rr_take_profit, resistance_level) if resistance_level > entry_price else rr_take_profit
            
        else:  # SELL signal - take profit below support
            # Calculate based on risk-reward ratio
            rr_take_profit = entry_price - (risk_amount * risk_reward_ratio)
            
            # Find next support level
            support_level = recent_data['low'].min()
            
            # Use the more conservative (closer) target
            take_profit = max(rr_take_profit, support_level) if support_level < entry_price else rr_take_profit
        
        return take_profit
    
    def implement_trailing_stop(self, current_price: float, entry_price: float,
                               signal_type: int, current_stop: float,
                               trailing_percent: float = 2.0) -> float:
        """
        Implement trailing stop loss logic
        
        Args:
            current_price: Current market price
            entry_price: Original entry price
            signal_type: 1 for buy, -1 for sell
            current_stop: Current stop loss level
            trailing_percent: Trailing percentage
        """
        trailing_distance = current_price * (trailing_percent / 100)
        
        if signal_type == 1:  # Long position
            # Only move stop loss up, never down
            new_stop = current_price - trailing_distance
            return max(current_stop, new_stop)
        else:  # Short position
            # Only move stop loss down, never up
            new_stop = current_price + trailing_distance
            return min(current_stop, new_stop)
    
    def implement_breakeven_logic(self, current_price: float, entry_price: float,
                                 signal_type: int, current_stop: float,
                                 breakeven_trigger: float = 1.5) -> float:
        """
        Implement breakeven stop loss logic
        
        Args:
            current_price: Current market price
            entry_price: Original entry price
            signal_type: 1 for buy, -1 for sell
            current_stop: Current stop loss level
            breakeven_trigger: Multiple of initial risk to trigger breakeven
        """
        initial_risk = abs(entry_price - current_stop)
        
        if signal_type == 1:  # Long position
            profit = current_price - entry_price
            if profit >= initial_risk * breakeven_trigger:
                # Move stop to breakeven (small profit)
                return max(current_stop, entry_price + (initial_risk * 0.1))
        else:  # Short position
            profit = entry_price - current_price
            if profit >= initial_risk * breakeven_trigger:
                # Move stop to breakeven (small profit)
                return min(current_stop, entry_price - (initial_risk * 0.1))
        
        return current_stop
    
    def backtest_strategy(self, df: pd.DataFrame, initial_balance: float = 10000,
                         risk_per_trade: float = 0.1, lookback: int = 20) -> Dict:
        """
        Backtest the strategy on historical data
        
        Args:
            df: Historical OHLCV data with indicators
            initial_balance: Starting balance
            risk_per_trade: Risk percentage per trade
            lookback: Lookback period for stop loss calculation
        """
        balance = initial_balance
        positions = []
        trades = []
        equity_curve = []
        
        signals = self.generate_signals(df)
        
        for i in range(len(signals)):
            current_row = signals.iloc[i]
            current_price = current_row['close']
            current_time = current_row.get('timestamp', i)
            
            # Update equity curve
            equity_curve.append({
                'timestamp': current_time,
                'equity': balance,
                'price': current_price
            })
            
            # Check for new signals
            if current_row['confirmed_signal'] != 0:
                signal_type = current_row['confirmed_signal']
                
                # Calculate stop loss
                stop_loss = self.calculate_stop_loss(
                    signals.iloc[:i+1], signal_type, lookback
                )
                
                if stop_loss is not None:
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        balance, risk_per_trade, current_price, stop_loss
                    )
                    
                    if position_size > 0:
                        # Calculate take profit
                        take_profit = self.calculate_take_profit(
                            signals.iloc[:i+1], signal_type, current_price, stop_loss
                        )
                        
                        # Create position
                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'signal_type': signal_type,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_index': i
                        }
                        positions.append(position)
            
            # Check existing positions for exits
            positions_to_remove = []
            for pos_idx, position in enumerate(positions):
                exit_triggered = False
                exit_reason = ""
                exit_price = current_price
                
                # Check stop loss
                if position['signal_type'] == 1:  # Long
                    if current_price <= position['stop_loss']:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    elif position['take_profit'] and current_price >= position['take_profit']:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                else:  # Short
                    if current_price >= position['stop_loss']:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    elif position['take_profit'] and current_price <= position['take_profit']:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                
                if exit_triggered:
                    # Calculate P&L
                    if position['signal_type'] == 1:  # Long
                        pnl = (exit_price - position['entry_price']) * position['position_size']
                    else:  # Short
                        pnl = (position['entry_price'] - exit_price) * position['position_size']
                    
                    balance += pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'signal_type': position['signal_type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'duration': i - position['entry_index']
                    }
                    trades.append(trade)
                    positions_to_remove.append(pos_idx)
            
            # Remove closed positions
            for idx in reversed(positions_to_remove):
                positions.pop(idx)
        
        # Calculate performance metrics
        performance = self._calculate_backtest_performance(
            trades, equity_curve, initial_balance
        )
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'performance': performance,
            'final_balance': balance
        }
    
    def _calculate_backtest_performance(self, trades: List[Dict], 
                                      equity_curve: List[Dict],
                                      initial_balance: float) -> Dict:
        """Calculate comprehensive performance metrics from backtest results"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        total_return = (total_pnl / initial_balance) * 100
        
        # Calculate returns for Sharpe ratio
        equity_values = [eq['equity'] for eq in equity_curve]
        returns = []
        for i in range(1, len(equity_values)):
            daily_return = (equity_values[i] / equity_values[i-1]) - 1
            returns.append(daily_return)
        
        # Annual return (assuming daily data)
        if len(equity_curve) > 365:
            periods_per_year = 365
            years = len(equity_curve) / periods_per_year
            annual_return = ((equity_values[-1] / equity_values[0]) ** (1/years) - 1) * 100
        else:
            annual_return = total_return
        
        # Maximum drawdown
        peak = initial_balance
        max_drawdown = 0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': gross_profit / len(winning_trades) if winning_trades else 0,
            'avg_loss': gross_loss / len(losing_trades) if losing_trades else 0
        }
    
    def _update_signals_history(self, signals_df: pd.DataFrame):
        """Update signals history for analysis"""
        new_signals = signals_df[signals_df['signal'] != 0].copy()
        if len(new_signals) > 0:
            self.signals_history.extend(new_signals.to_dict('records'))
            
        # Keep only recent signals (last 1000)
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]
    
    def get_strategy_performance(self) -> Dict:
        """Get current strategy performance metrics"""
        return self.performance_metrics
    
    def reset_strategy(self):
        """Reset strategy state"""
        self.signals_history = []
        self.trades_history = []
        self.performance_metrics = {}
