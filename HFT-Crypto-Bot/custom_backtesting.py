"""
Comprehensive backtesting module for trading strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Backtester:
    """Advanced backtesting engine for trading strategies"""
    
    def __init__(self, initial_balance: float = 10000):
        """
        Initialize backtester
        
        Args:
            initial_balance: Starting capital in USDT
        """
        self.initial_balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.results = {}
        self.performance_metrics = {}
        
    def run_backtest(self, df: pd.DataFrame, strategy, strategy_params: Dict = None) -> Dict:
        """
        Run comprehensive backtest on historical data
        
        Args:
            df: Historical OHLCV data with technical indicators
            strategy: Trading strategy instance
            strategy_params: Strategy parameters
            
        Returns:
            Dictionary containing detailed backtest results
        """
        if strategy_params is None:
            strategy_params = {
                'rsi_buy_threshold': 30,
                'rsi_sell_threshold': 70,
                'risk_per_trade': 0.02,  # 2% risk per trade
                'stop_loss_pct': 0.03,   # 3% stop loss
                'take_profit_pct': 0.06, # 6% take profit (1:2 risk/reward)
                'max_positions': 3,      # Maximum concurrent positions
                'commission': 0.001      # 0.1% commission
            }
        
        # Initialize backtest variables
        balance = self.initial_balance
        positions = []
        trades = []
        equity_curve = []
        max_equity = self.initial_balance
        max_drawdown = 0
        
        # Generate signals
        signals = strategy.generate_signals(
            df, 
            strategy_params['rsi_buy_threshold'], 
            strategy_params['rsi_sell_threshold']
        )
        
        # Iterate through historical data
        for i in range(1, len(signals)):
            current_row = signals.iloc[i]
            current_price = current_row['close']
            current_signal = current_row['signal']
            current_time = current_row.name if hasattr(current_row, 'name') else i
            
            # Update existing positions
            positions, closed_trades = self._update_positions(
                positions, current_price, current_time, strategy_params
            )
            trades.extend(closed_trades)
            
            # Calculate current portfolio value
            portfolio_value = balance
            for pos in positions:
                if pos['side'] == 'BUY':
                    portfolio_value += pos['quantity'] * (current_price - pos['entry_price'])
                else:  # SELL/SHORT
                    portfolio_value += pos['quantity'] * (pos['entry_price'] - current_price)
            
            # Record equity curve
            equity_curve.append({
                'timestamp': current_time,
                'equity': portfolio_value,
                'balance': balance,
                'unrealized_pnl': portfolio_value - balance
            })
            
            # Update drawdown
            if portfolio_value > max_equity:
                max_equity = portfolio_value
            current_drawdown = (max_equity - portfolio_value) / max_equity
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # Process new signals
            if current_signal != 0 and len(positions) < strategy_params['max_positions']:
                trade_result = self._execute_signal(
                    current_signal, current_price, current_time, 
                    balance, strategy_params
                )
                
                if trade_result:
                    positions.append(trade_result['position'])
                    balance -= trade_result['capital_used']
        
        # Close remaining positions at final price
        final_price = signals.iloc[-1]['close']
        final_time = signals.iloc[-1].name if hasattr(signals.iloc[-1], 'name') else len(signals) - 1
        
        for pos in positions:
            closed_trade = self._close_position(pos, final_price, final_time, 'end_of_data')
            trades.append(closed_trade)
            balance += closed_trade['pnl'] + pos['capital_used']
        
        # Store results
        self.trades = trades
        self.equity_curve = equity_curve
        
        # Calculate comprehensive metrics
        self.performance_metrics = self._calculate_performance_metrics(
            trades, equity_curve, max_drawdown, strategy_params
        )
        
        return self.performance_metrics
    
    def _execute_signal(self, signal: int, price: float, timestamp, 
                       balance: float, params: Dict) -> Optional[Dict]:
        """Execute trading signal"""
        try:
            # Calculate position size based on risk management
            risk_amount = balance * params['risk_per_trade']
            
            if signal == 1:  # BUY signal
                stop_loss_price = price * (1 - params['stop_loss_pct'])
                take_profit_price = price * (1 + params['take_profit_pct'])
                side = 'BUY'
            else:  # SELL signal
                stop_loss_price = price * (1 + params['stop_loss_pct'])
                take_profit_price = price * (1 - params['take_profit_pct'])
                side = 'SELL'
            
            # Calculate quantity based on risk
            risk_per_share = abs(price - stop_loss_price)
            if risk_per_share > 0:
                quantity = risk_amount / risk_per_share
                capital_used = quantity * price * (1 + params['commission'])
                
                if capital_used <= balance:
                    position = {
                        'id': len(self.trades) + len([]),
                        'side': side,
                        'entry_price': price,
                        'quantity': quantity,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'entry_time': timestamp,
                        'capital_used': capital_used
                    }
                    
                    return {
                        'position': position,
                        'capital_used': capital_used
                    }
            
            return None
            
        except Exception as e:
            print(f"Error executing signal: {e}")
            return None
    
    def _update_positions(self, positions: List[Dict], current_price: float, 
                         current_time, params: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Update existing positions and check for exits"""
        remaining_positions = []
        closed_trades = []
        
        for pos in positions:
            # Check stop loss and take profit
            should_close = False
            exit_reason = ""
            
            if pos['side'] == 'BUY':
                if current_price <= pos['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price >= pos['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # SELL/SHORT
                if current_price >= pos['stop_loss']:
                    should_close = True
                    exit_reason = "stop_loss"
                elif current_price <= pos['take_profit']:
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                closed_trade = self._close_position(pos, current_price, current_time, exit_reason)
                closed_trades.append(closed_trade)
            else:
                remaining_positions.append(pos)
        
        return remaining_positions, closed_trades
    
    def _close_position(self, position: Dict, exit_price: float, 
                       exit_time, exit_reason: str) -> Dict:
        """Close a position and calculate P&L"""
        if position['side'] == 'BUY':
            pnl = position['quantity'] * (exit_price - position['entry_price'])
        else:  # SELL/SHORT
            pnl = position['quantity'] * (position['entry_price'] - exit_price)
        
        # Apply commission
        commission = position['quantity'] * exit_price * 0.001  # 0.1% commission
        pnl -= commission
        
        # Calculate return percentage
        return_pct = pnl / position['capital_used'] * 100
        
        # Calculate holding period
        if hasattr(exit_time, 'timestamp'):
            holding_period = (exit_time - position['entry_time']).total_seconds() / 3600  # hours
        else:
            holding_period = exit_time - position['entry_time']  # periods
        
        return {
            'position_id': position['id'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'return_pct': return_pct,
            'holding_period': holding_period,
            'capital_used': position['capital_used']
        }
    
    def _calculate_performance_metrics(self, trades: List[Dict], 
                                     equity_curve: List[Dict], 
                                     max_drawdown: float, 
                                     params: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'message': 'No trades executed'
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in trades)
        final_equity = self.initial_balance + total_pnl
        total_return = (final_equity / self.initial_balance - 1) * 100
        
        # Win rate and profit factor
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        returns = [eq['equity'] for eq in equity_curve]
        if len(returns) > 1:
            daily_returns = [(returns[i] - returns[i-1]) / returns[i-1] for i in range(1, len(returns))]
            volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
            sharpe_ratio = (total_return / 100) / volatility if volatility > 0 else 0
        else:
            sharpe_ratio = 0
            volatility = 0
        
        # Holding period analysis
        avg_holding_period = np.mean([t['holding_period'] for t in trades])
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_return': round(total_return, 2),
            'total_pnl': round(total_pnl, 2),
            'final_equity': round(final_equity, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'volatility': round(volatility * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade': round(avg_trade, 2),
            'avg_holding_period': round(avg_holding_period, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'best_trade': round(max([t['pnl'] for t in trades]) if trades else 0, 2),
            'worst_trade': round(min([t['pnl'] for t in trades]) if trades else 0, 2),
            'consecutive_wins': self._calculate_consecutive_wins(trades),
            'consecutive_losses': self._calculate_consecutive_losses(trades),
            'recovery_factor': round(total_pnl / (max_drawdown * self.initial_balance) if max_drawdown > 0 else 0, 2),
            'calmar_ratio': round((total_return / 100) / max_drawdown if max_drawdown > 0 else 0, 2)
        }
    
    def _calculate_consecutive_wins(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive winning trades"""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive losing trades"""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

    def create_backtest_report(self) -> go.Figure:
        """Create comprehensive backtest report visualization"""
        if not self.equity_curve or not self.trades:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(text="No backtest data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 
                          'Monthly Returns', 'Trade Distribution',
                          'Rolling Sharpe Ratio', 'Trade P&L'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Equity curve
        equity_data = pd.DataFrame(self.equity_curve)
        fig.add_trace(
            go.Scatter(x=equity_data.index, y=equity_data['equity'],
                      mode='lines', name='Portfolio Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add initial balance line
        fig.add_hline(y=self.initial_balance, line_dash="dash", 
                     line_color="gray", row=1, col=1)
        
        # Drawdown
        max_equity = equity_data['equity'].expanding().max()
        drawdown = (equity_data['equity'] - max_equity) / max_equity * 100
        fig.add_trace(
            go.Scatter(x=equity_data.index, y=drawdown,
                      mode='lines', name='Drawdown %',
                      line=dict(color='red', width=1),
                      fill='tonexty'),
            row=1, col=2
        )
        
        # Trade P&L scatter
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
            fig.add_trace(
                go.Scatter(x=trades_df.index, y=trades_df['pnl'],
                          mode='markers', name='Trade P&L',
                          marker=dict(color=colors, size=6)),
                row=3, col=2
            )
            
            # Win/Loss distribution
            wins = len([t for t in self.trades if t['pnl'] > 0])
            losses = len([t for t in self.trades if t['pnl'] <= 0])
            fig.add_trace(
                go.Pie(labels=['Wins', 'Losses'], values=[wins, losses],
                      name="Trade Distribution"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Comprehensive Backtest Report",
            showlegend=True,
            height=1000
        )
        
        return fig

    def get_trade_details(self) -> pd.DataFrame:
        """Get detailed trade information as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)

    def export_results(self) -> Dict:
        """Export complete backtest results"""
        return {
            'performance_metrics': self.performance_metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'initial_balance': self.initial_balance
        }