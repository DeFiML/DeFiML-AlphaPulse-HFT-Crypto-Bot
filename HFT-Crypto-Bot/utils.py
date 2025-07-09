import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import math

def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    Format monetary amounts with proper currency symbols and formatting
    
    Args:
        amount: The monetary amount to format
        currency: Currency code (USD, BTC, ETH, etc.)
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if amount is None or math.isnan(amount):
        return f"${0:.{decimals}f}"
    
    # Handle very large numbers
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.{max(1, decimals-3)}f}M"
    elif abs(amount) >= 1_000:
        return f"${amount/1_000:.{max(1, decimals-1)}f}K"
    
    # Currency symbols
    symbols = {
        "USD": "$",
        "USDT": "$",
        "BTC": "₿",
        "ETH": "Ξ",
        "EUR": "€",
        "GBP": "£"
    }
    
    symbol = symbols.get(currency.upper(), "$")
    
    # Handle negative amounts
    if amount < 0:
        return f"-{symbol}{abs(amount):.{decimals}f}"
    else:
        return f"{symbol}{amount:.{decimals}f}"

def format_percentage(percentage: float, decimals: int = 2, show_sign: bool = True) -> str:
    """
    Format percentage values with proper signs and formatting
    
    Args:
        percentage: The percentage value to format
        decimals: Number of decimal places
        show_sign: Whether to show + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    if percentage is None or math.isnan(percentage):
        return "0.00%"
    
    sign = ""
    if show_sign and percentage > 0:
        sign = "+"
    elif percentage < 0:
        sign = "-"
        percentage = abs(percentage)
    
    return f"{sign}{percentage:.{decimals}f}%"

def format_number(number: float, decimals: int = 4, compact: bool = False) -> str:
    """
    Format numbers with proper decimal places and optional compact notation
    
    Args:
        number: The number to format
        decimals: Number of decimal places
        compact: Whether to use compact notation (K, M, B)
        
    Returns:
        Formatted number string
    """
    if number is None or math.isnan(number):
        return f"0.{'0' * decimals}"
    
    if compact:
        if abs(number) >= 1_000_000_000:
            return f"{number/1_000_000_000:.{max(1, decimals-6)}f}B"
        elif abs(number) >= 1_000_000:
            return f"{number/1_000_000:.{max(1, decimals-3)}f}M"
        elif abs(number) >= 1_000:
            return f"{number/1_000:.{max(1, decimals-1)}f}K"
    
    return f"{number:.{decimals}f}"

def calculate_performance_metrics(trades_history: List[Dict]) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from trade history
    
    Args:
        trades_history: List of trade dictionaries
        
    Returns:
        Dictionary containing performance metrics
    """
    if not trades_history:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'net_profit': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'expectancy': 0.0,
            'recovery_factor': 0.0,
            'payoff_ratio': 0.0
        }
    
    # Filter trades with P&L data
    pnl_trades = [trade for trade in trades_history if 'pnl' in trade and trade['pnl'] is not None]
    
    if not pnl_trades:
        return calculate_performance_metrics([])  # Return empty metrics
    
    # Basic trade statistics
    total_trades = len(pnl_trades)
    winning_trades = [trade for trade in pnl_trades if trade['pnl'] > 0]
    losing_trades = [trade for trade in pnl_trades if trade['pnl'] < 0]
    breakeven_trades = [trade for trade in pnl_trades if trade['pnl'] == 0]
    
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    num_breakeven = len(breakeven_trades)
    
    # Win rate
    win_rate = (num_winning / total_trades) * 100 if total_trades > 0 else 0
    
    # P&L calculations
    gross_profit = sum(trade['pnl'] for trade in winning_trades)
    gross_loss = sum(abs(trade['pnl']) for trade in losing_trades)
    net_profit = gross_profit - gross_loss
    
    # Average metrics
    avg_win = gross_profit / num_winning if num_winning > 0 else 0
    avg_loss = gross_loss / num_losing if num_losing > 0 else 0
    
    # Maximum metrics
    max_win = max((trade['pnl'] for trade in winning_trades), default=0)
    max_loss = max((abs(trade['pnl']) for trade in losing_trades), default=0)
    
    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Payoff ratio (avg win / avg loss)
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
    
    # Expectancy
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)
    
    # Calculate returns and drawdown metrics
    returns_metrics = _calculate_returns_metrics(pnl_trades)
    
    # Combine all metrics
    metrics = {
        'total_trades': total_trades,
        'winning_trades': num_winning,
        'losing_trades': num_losing,
        'breakeven_trades': num_breakeven,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'net_profit': net_profit,
        'expectancy': expectancy,
        'payoff_ratio': payoff_ratio
    }
    
    # Add returns-based metrics
    metrics.update(returns_metrics)
    
    return metrics

def _calculate_returns_metrics(trades: List[Dict], initial_balance: float = 10000) -> Dict[str, float]:
    """
    Calculate returns-based performance metrics
    
    Args:
        trades: List of trades with P&L data
        initial_balance: Initial account balance
        
    Returns:
        Dictionary of returns metrics
    """
    if not trades:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'recovery_factor': 0.0
        }
    
    # Create equity curve
    equity_curve = [initial_balance]
    current_balance = initial_balance
    
    for trade in sorted(trades, key=lambda x: x.get('timestamp', datetime.now())):
        current_balance += trade['pnl']
        equity_curve.append(current_balance)
    
    # Calculate returns
    daily_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] > 0:
            daily_return = (equity_curve[i] / equity_curve[i-1]) - 1
            daily_returns.append(daily_return)
    
    # Total return
    total_return = ((equity_curve[-1] / initial_balance) - 1) * 100 if initial_balance > 0 else 0
    
    # Annual return
    if len(trades) > 0:
        # Estimate trading period
        first_trade = min(trades, key=lambda x: x.get('timestamp', datetime.now()))
        last_trade = max(trades, key=lambda x: x.get('timestamp', datetime.now()))
        
        first_date = first_trade.get('timestamp', datetime.now())
        last_date = last_trade.get('timestamp', datetime.now())
        
        if isinstance(first_date, str):
            first_date = datetime.fromisoformat(first_date.replace('Z', '+00:00'))
        if isinstance(last_date, str):
            last_date = datetime.fromisoformat(last_date.replace('Z', '+00:00'))
        
        trading_days = (last_date - first_date).days
        
        if trading_days > 0:
            years = trading_days / 365
            annual_return = ((equity_curve[-1] / initial_balance) ** (1/years) - 1) * 100
        else:
            annual_return = total_return
    else:
        annual_return = total_return
    
    # Maximum drawdown
    max_drawdown = 0
    peak = initial_balance
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    # Sharpe ratio
    if len(daily_returns) > 1:
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Sortino ratio (downside deviation)
    if len(daily_returns) > 1:
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns, ddof=1)
            mean_return = np.mean(daily_returns)
            sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = float('inf') if np.mean(daily_returns) > 0 else 0
    else:
        sortino_ratio = 0
    
    # Calmar ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # Recovery factor
    net_profit = equity_curve[-1] - initial_balance
    recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'recovery_factor': recovery_factor
    }

def calculate_risk_metrics(portfolio_value: float, trades: List[Dict], 
                          benchmark_return: float = 0.20) -> Dict[str, Any]:
    """
    Calculate risk-adjusted performance metrics
    
    Args:
        portfolio_value: Current portfolio value
        trades: Historical trades
        benchmark_return: Benchmark return for comparison (default 20% annual)
        
    Returns:
        Dictionary of risk metrics
    """
    if not trades:
        return {
            'var_95': 0.0,
            'var_99': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'beta': 1.0,
            'alpha': 0.0,
            'information_ratio': 0.0,
            'tracking_error': 0.0,
            'treynor_ratio': 0.0
        }
    
    # Calculate returns
    returns = []
    for trade in trades:
        if 'pnl' in trade and trade['pnl'] is not None:
            return_pct = trade['pnl'] / portfolio_value if portfolio_value > 0 else 0
            returns.append(return_pct)
    
    if not returns:
        return calculate_risk_metrics(portfolio_value, [], benchmark_return)
    
    returns = np.array(returns)
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5) * portfolio_value  # 5th percentile (95% VaR)
    var_99 = np.percentile(returns, 1) * portfolio_value  # 1st percentile (99% VaR)
    
    # Conditional Value at Risk (CVaR) - Expected Shortfall
    cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * portfolio_value
    cvar_99 = np.mean(returns[returns <= np.percentile(returns, 1)]) * portfolio_value
    
    # Beta (simplified - assume market correlation of 0.8)
    beta = 0.8 if len(returns) > 10 else 1.0
    
    # Alpha
    portfolio_return = np.mean(returns) * 252  # Annualized
    alpha = portfolio_return - benchmark_return
    
    # Information ratio
    excess_returns = returns - (benchmark_return / 252)  # Daily benchmark
    if len(excess_returns) > 1:
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
        information_ratio = (np.mean(excess_returns) * 252) / tracking_error if tracking_error > 0 else 0
    else:
        tracking_error = 0
        information_ratio = 0
    
    # Treynor ratio
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    treynor_ratio = (portfolio_return - risk_free_rate) / beta if beta != 0 else 0
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'beta': beta,
        'alpha': alpha,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'treynor_ratio': treynor_ratio
    }

def validate_trading_parameters(params: Dict) -> Tuple[bool, str]:
    """
    Validate trading parameters for safety and reasonableness
    
    Args:
        params: Dictionary of trading parameters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # RSI parameters
    rsi_period = params.get('rsi_period', 14)
    if not (3 <= rsi_period <= 50):
        errors.append("RSI period must be between 3 and 50")
    
    rsi_buy = params.get('rsi_buy_threshold', 30)
    rsi_sell = params.get('rsi_sell_threshold', 70)
    
    if not (10 <= rsi_buy <= 40):
        errors.append("RSI buy threshold must be between 10 and 40")
    
    if not (60 <= rsi_sell <= 90):
        errors.append("RSI sell threshold must be between 60 and 90")
    
    if rsi_buy >= rsi_sell:
        errors.append("RSI buy threshold must be less than sell threshold")
    
    # Moving average parameters
    fast_ma = params.get('fast_ma_period', 3)
    slow_ma = params.get('slow_ma_period', 33)
    
    if not (1 <= fast_ma <= 20):
        errors.append("Fast MA period must be between 1 and 20")
    
    if not (10 <= slow_ma <= 200):
        errors.append("Slow MA period must be between 10 and 200")
    
    if fast_ma >= slow_ma:
        errors.append("Fast MA period must be less than slow MA period")
    
    # Risk parameters
    risk_per_trade = params.get('risk_per_trade', 0.1)
    if not (0.01 <= risk_per_trade <= 5.0):
        errors.append("Risk per trade must be between 0.01% and 5.0%")
    
    lookback = params.get('lookback_period', 20)
    if not (5 <= lookback <= 100):
        errors.append("Lookback period must be between 5 and 100")
    
    # Position size validation
    max_position_size = params.get('max_position_size', 0.1)
    if not (0.01 <= max_position_size <= 0.5):
        errors.append("Maximum position size must be between 1% and 50%")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "Parameters are valid"

def calculate_position_metrics(position: Dict, current_price: float) -> Dict[str, float]:
    """
    Calculate metrics for a specific position
    
    Args:
        position: Position dictionary
        current_price: Current market price
        
    Returns:
        Dictionary of position metrics
    """
    if not position or 'entry_price' not in position:
        return {}
    
    entry_price = position['entry_price']
    quantity = position.get('quantity', 0)
    position_type = position.get('position_type', 'long')
    
    # Unrealized P&L
    if position_type == 'long':
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
    else:  # short
        unrealized_pnl = (entry_price - current_price) * quantity
        unrealized_pnl_pct = ((entry_price / current_price) - 1) * 100
    
    # Position value
    position_value = quantity * current_price
    
    # Risk metrics
    stop_loss = position.get('stop_loss')
    if stop_loss:
        if position_type == 'long':
            risk_amount = (entry_price - stop_loss) * quantity
            risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        else:
            risk_amount = (stop_loss - entry_price) * quantity
            risk_pct = ((stop_loss - entry_price) / entry_price) * 100
    else:
        risk_amount = 0
        risk_pct = 0
    
    # Distance to stop loss
    if stop_loss:
        distance_to_stop = abs(current_price - stop_loss) / current_price * 100
    else:
        distance_to_stop = 0
    
    # Take profit metrics
    take_profit = position.get('take_profit')
    if take_profit:
        if position_type == 'long':
            potential_profit = (take_profit - entry_price) * quantity
            distance_to_tp = (take_profit - current_price) / current_price * 100
        else:
            potential_profit = (entry_price - take_profit) * quantity
            distance_to_tp = (current_price - take_profit) / current_price * 100
    else:
        potential_profit = 0
        distance_to_tp = 0
    
    # Risk/reward ratio
    risk_reward_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
    
    return {
        'unrealized_pnl': unrealized_pnl,
        'unrealized_pnl_pct': unrealized_pnl_pct,
        'position_value': position_value,
        'risk_amount': risk_amount,
        'risk_pct': risk_pct,
        'potential_profit': potential_profit,
        'distance_to_stop': distance_to_stop,
        'distance_to_tp': distance_to_tp,
        'risk_reward_ratio': risk_reward_ratio,
        'entry_price': entry_price,
        'current_price': current_price,
        'price_change_pct': ((current_price / entry_price) - 1) * 100
    }

def get_market_status() -> Dict[str, Any]:
    """
    Get current market status and trading session information
    
    Returns:
        Dictionary with market status information
    """
    now = datetime.now()
    
    # Crypto markets are always open
    return {
        'is_open': True,
        'session': 'Continuous',
        'next_close': None,
        'next_open': None,
        'current_time': now,
        'market_type': 'Cryptocurrency',
        'timezone': 'UTC'
    }

def calculate_correlation_matrix(symbols: List[str], price_data: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple symbols
    
    Args:
        symbols: List of trading symbols
        price_data: Dictionary with price data for each symbol
        
    Returns:
        Correlation matrix as DataFrame
    """
    if not symbols or not price_data:
        return pd.DataFrame()
    
    # Create DataFrame from price data
    df_data = {}
    min_length = float('inf')
    
    for symbol in symbols:
        if symbol in price_data and price_data[symbol]:
            df_data[symbol] = price_data[symbol]
            min_length = min(min_length, len(price_data[symbol]))
    
    if not df_data or min_length == 0:
        return pd.DataFrame()
    
    # Trim all data to same length
    for symbol in df_data:
        df_data[symbol] = df_data[symbol][-min_length:]
    
    df = pd.DataFrame(df_data)
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns.corr()
    
    return correlation_matrix

def generate_trade_report(trades: List[Dict], start_date: datetime = None, 
                         end_date: datetime = None) -> Dict[str, Any]:
    """
    Generate comprehensive trading report
    
    Args:
        trades: List of trade records
        start_date: Start date for report period
        end_date: End date for report period
        
    Returns:
        Comprehensive trading report
    """
    if not trades:
        return {'error': 'No trades available for report'}
    
    # Filter trades by date if specified
    filtered_trades = trades
    if start_date or end_date:
        filtered_trades = []
        for trade in trades:
            trade_date = trade.get('timestamp')
            if isinstance(trade_date, str):
                trade_date = datetime.fromisoformat(trade_date.replace('Z', '+00:00'))
            
            if start_date and trade_date < start_date:
                continue
            if end_date and trade_date > end_date:
                continue
            
            filtered_trades.append(trade)
    
    if not filtered_trades:
        return {'error': 'No trades in specified date range'}
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(filtered_trades)
    
    # Trade distribution by symbol
    symbol_distribution = {}
    for trade in filtered_trades:
        symbol = trade.get('symbol', 'Unknown')
        if symbol not in symbol_distribution:
            symbol_distribution[symbol] = {'count': 0, 'pnl': 0}
        symbol_distribution[symbol]['count'] += 1
        symbol_distribution[symbol]['pnl'] += trade.get('pnl', 0)
    
    # Trade distribution by time
    hourly_distribution = {}
    for trade in filtered_trades:
        trade_time = trade.get('timestamp')
        if isinstance(trade_time, str):
            trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
        
        hour = trade_time.hour if trade_time else 0
        if hour not in hourly_distribution:
            hourly_distribution[hour] = 0
        hourly_distribution[hour] += 1
    
    # Monthly performance
    monthly_performance = {}
    for trade in filtered_trades:
        trade_time = trade.get('timestamp')
        if isinstance(trade_time, str):
            trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
        
        month_key = f"{trade_time.year}-{trade_time.month:02d}" if trade_time else "Unknown"
        if month_key not in monthly_performance:
            monthly_performance[month_key] = {'trades': 0, 'pnl': 0}
        monthly_performance[month_key]['trades'] += 1
        monthly_performance[month_key]['pnl'] += trade.get('pnl', 0)
    
    # Best and worst trades
    pnl_trades = [t for t in filtered_trades if 'pnl' in t and t['pnl'] is not None]
    best_trade = max(pnl_trades, key=lambda x: x['pnl']) if pnl_trades else None
    worst_trade = min(pnl_trades, key=lambda x: x['pnl']) if pnl_trades else None
    
    return {
        'report_period': {
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': len(filtered_trades)
        },
        'performance_metrics': performance,
        'symbol_distribution': symbol_distribution,
        'hourly_distribution': hourly_distribution,
        'monthly_performance': monthly_performance,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'generated_at': datetime.now()
    }

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or math.isnan(denominator) or math.isinf(denominator):
            return default
        
        result = numerator / denominator
        
        if math.isnan(result) or math.isinf(result):
            return default
            
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum bounds
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))

def round_to_precision(value: float, precision: int) -> float:
    """
    Round a value to a specific number of decimal places
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    
    try:
        return round(value, precision)
    except (TypeError, ValueError):
        return 0.0

def get_time_until_next_period(period: str = 'hour') -> timedelta:
    """
    Get time remaining until the next time period
    
    Args:
        period: Time period ('minute', 'hour', 'day')
        
    Returns:
        Time remaining as timedelta
    """
    now = datetime.now()
    
    if period == 'minute':
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return next_minute - now
    elif period == 'hour':
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour - now
    elif period == 'day':
        next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return next_day - now
    else:
        return timedelta(0)
