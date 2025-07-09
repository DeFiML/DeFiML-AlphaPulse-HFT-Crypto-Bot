import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

class Portfolio:
    """Portfolio management class for tracking positions, P&L, and performance"""
    
    def __init__(self, initial_balance: float = 10000):
        """
        Initialize portfolio
        
        Args:
            initial_balance: Starting balance in USDT
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # Active positions
        self.trade_history = []  # Historical trades
        self.daily_pnl_history = []  # Daily P&L tracking
        self.equity_curve = []  # Portfolio value over time
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_balance
        self.current_drawdown = 0.0
        
        # Holdings (asset quantities)
        self.holdings = {'USDT': initial_balance}
        
        # Daily tracking
        self.today_date = datetime.now().date()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # Performance tracking
        self.monthly_returns = []
        self.daily_returns = []
        
    def add_position(self, symbol: str, quantity: float, entry_price: float,
                    stop_loss: float = None, take_profit: float = None,
                    position_type: str = 'long') -> bool:
        """
        Add a new position to the portfolio
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            quantity: Quantity of the asset
            entry_price: Entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            position_type: 'long' or 'short'
        """
        try:
            base_asset = symbol.replace('USDT', '')
            position_value = quantity * entry_price
            
            # Check if we have enough balance for long positions
            if position_type == 'long' and position_value > self.holdings.get('USDT', 0):
                return False
            
            # Update holdings
            if position_type == 'long':
                self.holdings['USDT'] -= position_value
                self.holdings[base_asset] = self.holdings.get(base_asset, 0) + quantity
            else:
                # For short positions, we need margin (simplified)
                margin_required = position_value * 0.1  # 10% margin
                if margin_required > self.holdings.get('USDT', 0):
                    return False
                self.holdings['USDT'] -= margin_required
            
            # Create position record with proper ID generation
            position_id = f"pos_{len(self.positions)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            position = {
                'id': position_id,
                'symbol': symbol,
                'base_asset': base_asset,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_type': position_type,
                'status': 'open',
                'unrealized_pnl': 0.0,
                'position_value': position_value
            }
            
            self.positions[position_id] = position
            
            # Record trade
            self._record_trade_entry(position)
            
            return True
            
        except Exception as e:
            print(f"Error adding position: {str(e)}")
            return False
    
    def remove_position(self, symbol: str, quantity: float, exit_price: float,
                       exit_reason: str = 'manual') -> bool:
        """
        Remove/close a position
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to close
            exit_price: Exit price
            exit_reason: Reason for closing
        """
        try:
            base_asset = symbol.replace('USDT', '')
            
            # Find matching position(s)
            positions_to_close = []
            remaining_quantity = quantity
            
            for pos_id, position in self.positions.items():
                if position['symbol'] == symbol and position['status'] == 'open':
                    if remaining_quantity <= 0:
                        break
                    
                    close_quantity = min(remaining_quantity, position['quantity'])
                    positions_to_close.append((pos_id, close_quantity))
                    remaining_quantity -= close_quantity
            
            if not positions_to_close:
                return False
            
            total_pnl = 0.0
            
            # Close positions
            for pos_id, close_qty in positions_to_close:
                position = self.positions[pos_id]
                pnl = self._calculate_position_pnl(position, exit_price, close_qty)
                total_pnl += pnl
                
                # Update holdings
                if position['position_type'] == 'long':
                    # Return USDT from sale
                    self.holdings['USDT'] += close_qty * exit_price
                    self.holdings[base_asset] -= close_qty
                else:
                    # Short position closing
                    margin_return = position['position_value'] * 0.1
                    profit_loss = (position['entry_price'] - exit_price) * close_qty
                    self.holdings['USDT'] += margin_return + profit_loss
                
                # Record trade exit
                self._record_trade_exit(position, close_qty, exit_price, exit_reason, pnl)
                
                # Update or remove position
                if close_qty >= position['quantity']:
                    position['status'] = 'closed'
                    position['exit_price'] = exit_price
                    position['exit_time'] = datetime.now()
                    position['realized_pnl'] = pnl
                else:
                    position['quantity'] -= close_qty
                    position['position_value'] = position['quantity'] * position['entry_price']
            
            # Update portfolio metrics
            self.realized_pnl += total_pnl
            self.total_pnl += total_pnl
            self.daily_pnl += total_pnl
            
            # Update trade statistics
            if total_pnl > 0:
                self.winning_trades += 1
            elif total_pnl < 0:
                self.losing_trades += 1
            
            self.total_trades += 1
            self.daily_trades += 1
            
            return True
            
        except Exception as e:
            print(f"Error removing position: {str(e)}")
            return False
    
    def update_positions(self, market_prices: Dict[str, float]):
        """
        Update all positions with current market prices
        
        Args:
            market_prices: Dictionary of current market prices
        """
        total_unrealized_pnl = 0.0
        
        for position in self.positions.values():
            if position['status'] == 'open':
                symbol = position['symbol']
                current_price = market_prices.get(symbol, position['entry_price'])
                
                # Calculate unrealized P&L
                if position['position_type'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                position['unrealized_pnl'] = pnl
                position['current_price'] = current_price
                total_unrealized_pnl += pnl
        
        self.unrealized_pnl = total_unrealized_pnl
        
        # Update equity curve
        current_portfolio_value = self.get_total_value_from_prices(market_prices)
        self._update_equity_curve(current_portfolio_value)
        
        # Update drawdown metrics
        self._update_drawdown_metrics(current_portfolio_value)
    
    def get_total_value(self, current_price: float = None, symbol: str = None) -> float:
        """
        Get total portfolio value
        
        Args:
            current_price: Current price for position valuation
            symbol: Symbol for the current price
        """
        total_value = self.holdings.get('USDT', 0)
        
        # Add value of crypto holdings
        for asset, quantity in self.holdings.items():
            if asset != 'USDT' and quantity > 0:
                if symbol and asset in symbol:
                    price = current_price or 0
                else:
                    price = 0  # We'd need market data for other assets
                
                total_value += quantity * price
        
        # Add unrealized P&L from open positions
        total_value += self.unrealized_pnl
        
        return total_value
    
    def get_total_value_from_prices(self, market_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value using provided market prices
        """
        total_value = self.holdings.get('USDT', 0)
        
        # Add value of crypto holdings
        for asset, quantity in self.holdings.items():
            if asset != 'USDT' and quantity > 0:
                # Find price for this asset
                asset_price = 0
                for symbol, price in market_prices.items():
                    if asset in symbol:
                        asset_price = price
                        break
                
                total_value += quantity * asset_price
        
        # Add unrealized P&L
        total_value += self.unrealized_pnl
        
        return total_value
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        current_date = datetime.now().date()
        
        # Reset daily tracking if new day
        if current_date != self.today_date:
            self.daily_pnl_history.append({
                'date': self.today_date,
                'pnl': self.daily_pnl,
                'trades': self.daily_trades
            })
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.today_date = current_date
        
        return self.daily_pnl
    
    def get_holdings(self) -> Dict[str, float]:
        """Get current holdings"""
        return self.holdings.copy()
    
    def get_open_positions(self) -> Dict:
        """Get all open positions grouped by symbol"""
        open_positions = {}
        for pos_id, pos in self.positions.items():
            if pos['status'] == 'open':
                symbol = pos['symbol'].replace('USDT', '')  # Use base asset as key
                if symbol not in open_positions:
                    open_positions[symbol] = []
                open_positions[symbol].append(pos)
        return open_positions
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        open_positions = self.get_open_positions()
        
        summary = {
            'total_positions': len(open_positions),
            'long_positions': len([p for p in open_positions.values() if p['position_type'] == 'long']),
            'short_positions': len([p for p in open_positions.values() if p['position_type'] == 'short']),
            'total_position_value': sum(p['position_value'] for p in open_positions.values()),
            'total_unrealized_pnl': self.unrealized_pnl,
            'largest_position': max(open_positions.values(), key=lambda x: x['position_value']) if open_positions else None,
            'most_profitable': max(open_positions.values(), key=lambda x: x['unrealized_pnl']) if open_positions else None,
            'least_profitable': min(open_positions.values(), key=lambda x: x['unrealized_pnl']) if open_positions else None
        }
        
        return summary
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        total_portfolio_value = self.balance + self.unrealized_pnl
        total_return = (total_portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        # Win rate
        total_closed_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # Average win/loss
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = sum(abs(t['pnl']) for t in losing_trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        if len(self.daily_returns) > 1:
            returns_std = np.std(self.daily_returns)
            mean_return = np.mean(self.daily_returns)
            sharpe_ratio = (mean_return / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Annual return estimate
        if len(self.equity_curve) > 30:  # At least 30 data points
            days = len(self.equity_curve)
            annual_return = ((total_portfolio_value / self.initial_balance) ** (365 / days) - 1) * 100
        else:
            annual_return = total_return
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_portfolio_value': total_portfolio_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        return sorted(self.trade_history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_equity_curve(self) -> List[Dict]:
        """Get equity curve data"""
        return self.equity_curve
    
    def export_data(self) -> Dict:
        """Export portfolio data for backup/analysis"""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'holdings': self.holdings,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'performance_metrics': self.get_performance_metrics(),
            'equity_curve': self.equity_curve,
            'daily_pnl_history': self.daily_pnl_history
        }
    
    def import_data(self, data: Dict):
        """Import portfolio data from backup"""
        try:
            self.initial_balance = data.get('initial_balance', self.initial_balance)
            self.balance = data.get('current_balance', self.balance)
            self.holdings = data.get('holdings', self.holdings)
            self.positions = data.get('positions', {})
            self.trade_history = data.get('trade_history', [])
            self.equity_curve = data.get('equity_curve', [])
            self.daily_pnl_history = data.get('daily_pnl_history', [])
            
            # Recalculate metrics
            self._recalculate_metrics()
            
            return True
        except Exception as e:
            print(f"Error importing data: {str(e)}")
            return False
    
    # Private helper methods
    
    def _calculate_position_pnl(self, position: Dict, current_price: float, quantity: float) -> float:
        """Calculate P&L for a position"""
        if position['position_type'] == 'long':
            return (current_price - position['entry_price']) * quantity
        else:
            return (position['entry_price'] - current_price) * quantity
    
    def _record_trade_entry(self, position: Dict):
        """Record trade entry in history"""
        trade_record = {
            'timestamp': position['entry_time'],
            'type': 'entry',
            'symbol': position['symbol'],
            'side': 'buy' if position['position_type'] == 'long' else 'sell',
            'quantity': position['quantity'],
            'price': position['entry_price'],
            'position_id': position['id']
        }
        self.trade_history.append(trade_record)
    
    def _record_trade_exit(self, position: Dict, quantity: float, exit_price: float,
                          exit_reason: str, pnl: float):
        """Record trade exit in history"""
        trade_record = {
            'timestamp': datetime.now(),
            'type': 'exit',
            'symbol': position['symbol'],
            'side': 'sell' if position['position_type'] == 'long' else 'buy',
            'quantity': quantity,
            'price': exit_price,
            'position_id': position['id'],
            'pnl': pnl,
            'exit_reason': exit_reason,
            'entry_price': position['entry_price'],
            'duration': (datetime.now() - position['entry_time']).total_seconds() / 3600  # hours
        }
        self.trade_history.append(trade_record)
    
    def _update_equity_curve(self, portfolio_value: float):
        """Update equity curve with current portfolio value"""
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': portfolio_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        })
        
        # Keep only recent data (last 1000 points)
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
        
        # Calculate daily returns
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            if prev_equity > 0:
                daily_return = (portfolio_value / prev_equity) - 1
                self.daily_returns.append(daily_return)
                
                # Keep only recent returns
                if len(self.daily_returns) > 252:  # One year of daily returns
                    self.daily_returns = self.daily_returns[-252:]
    
    def _update_drawdown_metrics(self, portfolio_value: float):
        """Update drawdown metrics"""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
        
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
    
    def _recalculate_metrics(self):
        """Recalculate all metrics from trade history"""
        self.winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        self.losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        self.total_trades = self.winning_trades + self.losing_trades
        self.realized_pnl = sum(t.get('pnl', 0) for t in self.trade_history if 'pnl' in t)
        
        # Recalculate peak and drawdown from equity curve
        if self.equity_curve:
            equity_values = [point['equity'] for point in self.equity_curve]
            self.peak_portfolio_value = max(equity_values)
            
            max_dd = 0
            peak = equity_values[0]
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                max_dd = max(max_dd, drawdown)
            
            self.max_drawdown = max_dd
            self.current_drawdown = (self.peak_portfolio_value - equity_values[-1]) / self.peak_portfolio_value * 100
