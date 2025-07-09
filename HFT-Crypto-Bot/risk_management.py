import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class RiskManager:
    """Comprehensive risk management system for trading operations"""
    
    def __init__(self, max_daily_loss: float = 5.0, max_portfolio_risk: float = 20.0):
        """
        Initialize Risk Manager
        
        Args:
            max_daily_loss: Maximum daily loss percentage (default 5%)
            max_portfolio_risk: Maximum portfolio risk percentage (default 20%)
        """
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_risk = max_portfolio_risk
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 10
        self.risk_metrics = {}
        self.position_limits = {}
        self.correlation_matrix = None
        
        # Risk thresholds
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_correlation = 0.7    # Maximum correlation between positions
        self.max_drawdown_limit = 10.0  # Maximum drawdown before stopping
        
        # Performance tracking
        self.performance_history = []
        self.risk_events = []
        
        # Logger setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_trade(self, trade_params: Dict, portfolio_value: float, 
                      current_positions: Dict) -> Tuple[bool, str]:
        """
        Validate if a trade meets risk management criteria
        
        Args:
            trade_params: Dictionary containing trade parameters
            portfolio_value: Current portfolio value
            current_positions: Current open positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check daily loss limit
        if self._check_daily_loss_limit():
            return False, "Daily loss limit exceeded"
        
        # Check daily trade limit
        if self._check_daily_trade_limit():
            return False, "Daily trade limit exceeded"
        
        # Check position size limits
        if not self._check_position_size_limit(trade_params, portfolio_value):
            return False, "Position size exceeds limit"
        
        # Check portfolio concentration
        if not self._check_concentration_limit(trade_params, current_positions, portfolio_value):
            return False, "Portfolio concentration limit exceeded"
        
        # Check correlation limits
        if not self._check_correlation_limit(trade_params, current_positions):
            return False, "Correlation limit exceeded"
        
        # Check volatility limits
        if not self._check_volatility_limit(trade_params):
            return False, "Volatility limit exceeded"
        
        # Check drawdown limits
        if not self._check_drawdown_limit(portfolio_value):
            return False, "Maximum drawdown limit exceeded"
        
        return True, "Trade approved"
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float,
                              entry_price: float, stop_loss: float,
                              volatility: float = None) -> float:
        """
        Calculate optimal position size using multiple risk metrics
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            volatility: Asset volatility (optional)
        """
        # Basic position sizing using fixed risk
        basic_size = self._calculate_fixed_risk_size(
            account_balance, risk_per_trade, entry_price, stop_loss
        )
        
        # Volatility-adjusted position sizing
        if volatility:
            volatility_adjusted_size = self._calculate_volatility_adjusted_size(
                basic_size, volatility
            )
        else:
            volatility_adjusted_size = basic_size
        
        # Kelly criterion adjustment (if sufficient data)
        kelly_adjusted_size = self._apply_kelly_criterion(volatility_adjusted_size)
        
        # Apply maximum position size limit
        max_position_value = account_balance * self.max_position_size
        max_size = max_position_value / entry_price
        
        final_size = min(kelly_adjusted_size, max_size)
        
        self.logger.info(f"Position sizing: Basic={basic_size:.4f}, "
                        f"Volatility Adjusted={volatility_adjusted_size:.4f}, "
                        f"Kelly Adjusted={kelly_adjusted_size:.4f}, "
                        f"Final={final_size:.4f}")
        
        return final_size
    
    def calculate_dynamic_stop_loss(self, df: pd.DataFrame, entry_price: float,
                                  position_type: int, method: str = 'atr') -> float:
        """
        Calculate dynamic stop loss using various methods
        
        Args:
            df: Price data DataFrame
            entry_price: Entry price
            position_type: 1 for long, -1 for short
            method: 'atr', 'volatility', 'support_resistance'
        """
        if method == 'atr':
            return self._calculate_atr_stop_loss(df, entry_price, position_type)
        elif method == 'volatility':
            return self._calculate_volatility_stop_loss(df, entry_price, position_type)
        elif method == 'support_resistance':
            return self._calculate_sr_stop_loss(df, entry_price, position_type)
        else:
            # Default to ATR method
            return self._calculate_atr_stop_loss(df, entry_price, position_type)
    
    def monitor_portfolio_risk(self, positions: Dict, market_data: Dict) -> Dict:
        """
        Monitor real-time portfolio risk metrics
        
        Args:
            positions: Current portfolio positions
            market_data: Current market data
            
        Returns:
            Dictionary of risk metrics
        """
        total_portfolio_value = sum(
            pos['quantity'] * market_data.get(pos['symbol'], {}).get('price', 0)
            for pos in positions.values()
        )
        
        # Calculate Value at Risk (VaR)
        var_95 = self._calculate_var(positions, market_data, confidence=0.95)
        var_99 = self._calculate_var(positions, market_data, confidence=0.99)
        
        # Calculate portfolio beta (if benchmark data available)
        portfolio_beta = self._calculate_portfolio_beta(positions, market_data)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk(positions, market_data)
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(positions)
        
        # Calculate maximum potential loss
        max_potential_loss = self._calculate_max_potential_loss(positions, market_data)
        
        risk_metrics = {
            'total_portfolio_value': total_portfolio_value,
            'var_95': var_95,
            'var_99': var_99,
            'portfolio_beta': portfolio_beta,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'max_potential_loss': max_potential_loss,
            'risk_score': self._calculate_overall_risk_score(
                var_95, concentration_risk, correlation_risk
            ),
            'timestamp': datetime.now()
        }
        
        # Store for historical analysis
        self.risk_metrics = risk_metrics
        
        return risk_metrics
    
    def implement_portfolio_hedge(self, positions: Dict, market_data: Dict) -> List[Dict]:
        """
        Suggest hedging strategies based on current portfolio risk
        
        Returns:
            List of suggested hedge trades
        """
        hedge_suggestions = []
        
        # Calculate portfolio exposure
        net_exposure = self._calculate_net_exposure(positions, market_data)
        
        # If exposure exceeds threshold, suggest hedge
        if abs(net_exposure) > 0.5:  # 50% net exposure threshold
            hedge_suggestions.append({
                'type': 'index_hedge',
                'action': 'sell' if net_exposure > 0 else 'buy',
                'amount': abs(net_exposure) * 0.5,  # Hedge 50% of excess exposure
                'reason': 'Excessive directional exposure'
            })
        
        # Check for sector concentration
        sector_exposure = self._calculate_sector_exposure(positions)
        for sector, exposure in sector_exposure.items():
            if exposure > 0.3:  # 30% sector concentration threshold
                hedge_suggestions.append({
                    'type': 'sector_hedge',
                    'sector': sector,
                    'action': 'hedge',
                    'amount': exposure * 0.3,
                    'reason': f'High {sector} sector concentration'
                })
        
        return hedge_suggestions
    
    def emergency_risk_control(self, portfolio_value: float, initial_value: float) -> Dict:
        """
        Emergency risk control measures when critical thresholds are breached
        
        Returns:
            Dictionary of emergency actions
        """
        current_drawdown = (initial_value - portfolio_value) / initial_value * 100
        
        emergency_actions = {
            'triggered': False,
            'actions': [],
            'severity': 'low'
        }
        
        if current_drawdown > self.max_drawdown_limit:
            emergency_actions['triggered'] = True
            emergency_actions['severity'] = 'critical'
            emergency_actions['actions'].extend([
                'STOP_ALL_TRADING',
                'CLOSE_LOSING_POSITIONS',
                'REVIEW_STRATEGY',
                'REDUCE_POSITION_SIZES'
            ])
        elif current_drawdown > self.max_drawdown_limit * 0.75:
            emergency_actions['triggered'] = True
            emergency_actions['severity'] = 'high'
            emergency_actions['actions'].extend([
                'REDUCE_POSITION_SIZES',
                'TIGHTEN_STOP_LOSSES',
                'PAUSE_NEW_TRADES'
            ])
        elif current_drawdown > self.max_drawdown_limit * 0.5:
            emergency_actions['triggered'] = True
            emergency_actions['severity'] = 'medium'
            emergency_actions['actions'].extend([
                'REVIEW_OPEN_POSITIONS',
                'TIGHTEN_RISK_LIMITS'
            ])
        
        if emergency_actions['triggered']:
            self.risk_events.append({
                'timestamp': datetime.now(),
                'type': 'emergency_control',
                'severity': emergency_actions['severity'],
                'drawdown': current_drawdown,
                'actions': emergency_actions['actions']
            })
        
        return emergency_actions
    
    # Private helper methods
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        return self.daily_pnl < -self.max_daily_loss
    
    def _check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is exceeded"""
        return self.daily_trades >= self.max_daily_trades
    
    def _check_position_size_limit(self, trade_params: Dict, portfolio_value: float) -> bool:
        """Check if position size exceeds limits"""
        position_value = trade_params.get('quantity', 0) * trade_params.get('price', 0)
        position_percentage = position_value / portfolio_value
        return position_percentage <= self.max_position_size
    
    def _check_concentration_limit(self, trade_params: Dict, positions: Dict, 
                                 portfolio_value: float) -> bool:
        """Check portfolio concentration limits"""
        symbol = trade_params.get('symbol', '')
        base_asset = symbol.replace('USDT', '')
        
        # Calculate current exposure to this asset
        current_exposure = 0
        for pos in positions.values():
            if pos.get('symbol', '').replace('USDT', '') == base_asset:
                current_exposure += pos.get('value', 0)
        
        # Add new trade exposure
        new_exposure = trade_params.get('quantity', 0) * trade_params.get('price', 0)
        total_exposure = (current_exposure + new_exposure) / portfolio_value
        
        return total_exposure <= 0.25  # 25% maximum exposure per asset
    
    def _check_correlation_limit(self, trade_params: Dict, positions: Dict) -> bool:
        """Check correlation limits between positions"""
        # Simplified correlation check
        # In practice, this would use historical correlation data
        return True
    
    def _check_volatility_limit(self, trade_params: Dict) -> bool:
        """Check if asset volatility exceeds acceptable limits"""
        volatility = trade_params.get('volatility', 0)
        max_volatility = 0.05  # 5% maximum daily volatility
        return volatility <= max_volatility
    
    def _check_drawdown_limit(self, portfolio_value: float) -> bool:
        """Check if portfolio drawdown exceeds limits"""
        if not hasattr(self, 'peak_portfolio_value'):
            self.peak_portfolio_value = portfolio_value
        
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
        return current_drawdown <= self.max_drawdown_limit
    
    def _calculate_fixed_risk_size(self, balance: float, risk_pct: float,
                                 entry_price: float, stop_loss: float) -> float:
        """Calculate position size using fixed risk method"""
        risk_amount = balance * (risk_pct / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        return risk_amount / price_risk
    
    def _calculate_volatility_adjusted_size(self, base_size: float, volatility: float) -> float:
        """Adjust position size based on volatility"""
        target_volatility = 0.02  # 2% target volatility
        adjustment_factor = target_volatility / max(volatility, 0.001)
        
        # Cap adjustment factor to reasonable range
        adjustment_factor = max(0.5, min(adjustment_factor, 2.0))
        
        return base_size * adjustment_factor
    
    def _apply_kelly_criterion(self, base_size: float) -> float:
        """Apply Kelly Criterion for position sizing"""
        # Simplified Kelly implementation
        # Requires historical win rate and average win/loss data
        
        if len(self.performance_history) < 10:
            return base_size
        
        wins = [p for p in self.performance_history if p > 0]
        losses = [p for p in self.performance_history if p < 0]
        
        if not wins or not losses:
            return base_size
        
        win_rate = len(wins) / len(self.performance_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return base_size
        
        kelly_percentage = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%
        
        return base_size * kelly_percentage / 0.1  # Assuming 10% base risk
    
    def _calculate_atr_stop_loss(self, df: pd.DataFrame, entry_price: float,
                               position_type: int, multiplier: float = 2.0) -> float:
        """Calculate ATR-based stop loss"""
        if 'atr' not in df.columns or len(df) == 0:
            # Fallback to percentage-based stop loss
            return entry_price * (0.98 if position_type == 1 else 1.02)
        
        current_atr = df.iloc[-1]['atr']
        
        if position_type == 1:  # Long position
            return entry_price - (current_atr * multiplier)
        else:  # Short position
            return entry_price + (current_atr * multiplier)
    
    def _calculate_volatility_stop_loss(self, df: pd.DataFrame, entry_price: float,
                                      position_type: int) -> float:
        """Calculate volatility-based stop loss"""
        if len(df) < 20:
            return entry_price * (0.98 if position_type == 1 else 1.02)
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Use 2 standard deviations as stop loss
        stop_distance = entry_price * volatility * 2 / np.sqrt(252)
        
        if position_type == 1:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance
    
    def _calculate_sr_stop_loss(self, df: pd.DataFrame, entry_price: float,
                              position_type: int, lookback: int = 20) -> float:
        """Calculate support/resistance-based stop loss"""
        if len(df) < lookback:
            return entry_price * (0.98 if position_type == 1 else 1.02)
        
        recent_data = df.tail(lookback)
        
        if position_type == 1:  # Long position - use support
            return recent_data['low'].min()
        else:  # Short position - use resistance
            return recent_data['high'].max()
    
    def _calculate_var(self, positions: Dict, market_data: Dict, confidence: float) -> float:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        # In practice, this would use historical simulation or Monte Carlo
        
        portfolio_values = []
        for pos in positions.values():
            symbol = pos.get('symbol', '')
            quantity = pos.get('quantity', 0)
            current_price = market_data.get(symbol, {}).get('price', 0)
            value = quantity * current_price
            portfolio_values.append(value)
        
        if not portfolio_values:
            return 0
        
        total_value = sum(portfolio_values)
        
        # Assume 2% daily volatility for simplified calculation
        daily_volatility = 0.02
        z_score = 1.645 if confidence == 0.95 else 2.326  # For 95% and 99% confidence
        
        var = total_value * daily_volatility * z_score
        return var
    
    def _calculate_portfolio_beta(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio beta"""
        # Simplified beta calculation
        # In practice, this would use regression against market benchmark
        return 1.0  # Default beta
    
    def _calculate_concentration_risk(self, positions: Dict, market_data: Dict) -> float:
        """Calculate concentration risk score"""
        if not positions:
            return 0
        
        total_value = sum(
            pos.get('quantity', 0) * market_data.get(pos.get('symbol', ''), {}).get('price', 0)
            for pos in positions.values()
        )
        
        if total_value == 0:
            return 0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        hhi = sum(
            ((pos.get('quantity', 0) * market_data.get(pos.get('symbol', ''), {}).get('price', 0)) / total_value) ** 2
            for pos in positions.values()
        )
        
        return hhi * 100  # Convert to percentage
    
    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """Calculate correlation risk score"""
        # Simplified correlation risk
        # In practice, this would use correlation matrix of assets
        return 30.0 if len(positions) > 5 else 10.0
    
    def _calculate_max_potential_loss(self, positions: Dict, market_data: Dict) -> float:
        """Calculate maximum potential loss"""
        total_loss = 0
        
        for pos in positions.values():
            symbol = pos.get('symbol', '')
            quantity = pos.get('quantity', 0)
            entry_price = pos.get('entry_price', 0)
            stop_loss = pos.get('stop_loss', 0)
            
            if stop_loss and entry_price:
                loss_per_unit = abs(entry_price - stop_loss)
                position_loss = loss_per_unit * quantity
                total_loss += position_loss
        
        return total_loss
    
    def _calculate_overall_risk_score(self, var: float, concentration: float,
                                    correlation: float) -> float:
        """Calculate overall risk score (0-100)"""
        # Weighted risk score
        risk_score = (var * 0.4 + concentration * 0.3 + correlation * 0.3)
        return min(risk_score, 100)
    
    def _calculate_net_exposure(self, positions: Dict, market_data: Dict) -> float:
        """Calculate net market exposure"""
        long_exposure = 0
        short_exposure = 0
        
        for pos in positions.values():
            symbol = pos.get('symbol', '')
            quantity = pos.get('quantity', 0)
            current_price = market_data.get(symbol, {}).get('price', 0)
            value = quantity * current_price
            
            if pos.get('side', 'long') == 'long':
                long_exposure += value
            else:
                short_exposure += value
        
        total_exposure = long_exposure + short_exposure
        if total_exposure == 0:
            return 0
        
        return (long_exposure - short_exposure) / total_exposure
    
    def _calculate_sector_exposure(self, positions: Dict) -> Dict[str, float]:
        """Calculate exposure by sector"""
        # Simplified sector mapping
        sector_mapping = {
            'BTC': 'crypto',
            'ETH': 'crypto',
            'ADA': 'crypto',
            'DOT': 'crypto',
            'LINK': 'defi',
            'BNB': 'exchange'
        }
        
        sector_exposure = {}
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        
        if total_value == 0:
            return sector_exposure
        
        for pos in positions.values():
            symbol = pos.get('symbol', '').replace('USDT', '')
            sector = sector_mapping.get(symbol, 'other')
            value = pos.get('value', 0)
            
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            
            sector_exposure[sector] += value / total_value
        
        return sector_exposure
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        
    def update_daily_trades(self):
        """Update daily trade count"""
        self.daily_trades += 1
        
    def reset_daily_counters(self):
        """Reset daily counters (call at start of each trading day)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'current_metrics': self.risk_metrics,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'risk_events': self.risk_events[-10:],  # Last 10 events
            'limits': {
                'max_daily_loss': self.max_daily_loss,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'max_drawdown_limit': self.max_drawdown_limit
            }
        }
