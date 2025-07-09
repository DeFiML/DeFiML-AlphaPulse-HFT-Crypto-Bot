import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import asyncio

from binance_client import BinanceClient
from technical_analysis import TechnicalAnalysis
from strategy import TradingStrategy
from risk_management import RiskManager
from portfolio import Portfolio
from backtesting import Backtest
from bt_strategy import RSIMAStrategy
from custom_backtesting import Backtester
from utils import format_currency, format_percentage, calculate_performance_metrics

# Configure Streamlit page
st.set_page_config(page_title="HFT Crypto Bot",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio(
        initial_balance=10000000)  # $10 million portfolio
if 'binance_client' not in st.session_state:
    st.session_state.binance_client = BinanceClient()
if 'technical_analysis' not in st.session_state:
    st.session_state.technical_analysis = TechnicalAnalysis()
if 'strategy' not in st.session_state:
    st.session_state.strategy = TradingStrategy()
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager()
if 'trades_history' not in st.session_state:
    st.session_state.trades_history = []
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
if 'last_portfolio_update' not in st.session_state:
    st.session_state.last_portfolio_update = time.time()


def update_portfolio_with_current_prices():
    """Update portfolio with current market prices and check stop loss/take profit"""
    try:
        current_time = time.time()
        # Update every 10 seconds to avoid rate limits
        if current_time - st.session_state.last_portfolio_update < 10:
            return

        st.session_state.last_portfolio_update = current_time

        # Get current positions
        positions = st.session_state.portfolio.get_open_positions()
        if not positions:
            return

        # Update prices for all positions
        market_prices = {}
        for symbol in positions.keys():
            if symbol != 'USDT':
                # Ensure clean symbol format without any timestamp corruption
                clean_symbol = symbol.replace(
                    'USDT', '') if symbol.endswith('USDT') else symbol
                trading_pair = f"{clean_symbol}USDT"
                price = st.session_state.binance_client.get_current_price(
                    trading_pair)
                if price:
                    market_prices[clean_symbol] = price

        # Update portfolio with current prices
        st.session_state.portfolio.update_positions(market_prices)

        # Check for stop loss/take profit triggers
        for symbol, position_data in positions.items():
            if symbol == 'USDT':
                continue

            current_price = market_prices.get(symbol)
            if not current_price or not isinstance(position_data, list):
                continue

            for position in position_data:
                if 'stop_loss' in position and position['stop_loss']:
                    if (position['position_type'] == 'long' and current_price <= position['stop_loss']) or \
                       (position['position_type'] == 'short' and current_price >= position['stop_loss']):
                        # Execute stop loss
                        execute_stop_loss_take_profit(symbol, position,
                                                      current_price,
                                                      'stop_loss')

                elif 'take_profit' in position and position['take_profit']:
                    if (position['position_type'] == 'long' and current_price >= position['take_profit']) or \
                       (position['position_type'] == 'short' and current_price <= position['take_profit']):
                        # Execute take profit
                        execute_stop_loss_take_profit(symbol, position,
                                                      current_price,
                                                      'take_profit')

    except Exception as e:
        st.error(f"Error updating portfolio: {str(e)}")


def execute_stop_loss_take_profit(symbol, position, current_price,
                                  trigger_type):
    """Execute stop loss or take profit order"""
    try:
        quantity = position['quantity']
        exit_reason = "Stop Loss" if trigger_type == 'stop_loss' else "Take Profit"

        # Remove position from portfolio
        success = st.session_state.portfolio.remove_position(
            symbol, quantity, current_price, exit_reason)

        if success:
            # Calculate P&L
            entry_price = position['entry_price']
            if position['position_type'] == 'long':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity

            # Record trade in history
            trade = {
                'timestamp': datetime.now(),
                'symbol': f"{symbol}USDT",
                'side':
                'SELL' if position['position_type'] == 'long' else 'BUY',
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'type': 'Auto',
                'exit_reason': exit_reason
            }
            st.session_state.trades_history.append(trade)

            # Show notification
            pnl_color = "🟢" if pnl > 0 else "🔴"
            st.success(
                f"{pnl_color} {exit_reason} executed for {symbol}: ${pnl:,.2f} P&L"
            )

    except Exception as e:
        st.error(f"Error executing {trigger_type}: {str(e)}")


def show_documentation():
    """Display the documentation page"""
    st.title("HFT Crypto Bot Documentation")
    st.markdown("**Complete Guide to High-Frequency Trading System**")

    st.header("Overview")
    st.markdown("""
    The HFT Crypto Bot is a sophisticated high-frequency trading system designed for cryptocurrency markets. 
    It combines technical analysis, risk management, and automated trading strategies to execute trades 
    with precision and speed in the volatile crypto market.
    """)

    st.header("Core Features")

    st.subheader("1. RSI/MA Trading Strategy")
    st.markdown("""
    **Relative Strength Index (RSI) and Moving Average Strategy:**
    - **RSI Signals**: Uses RSI overbought (>70) and oversold (<30) levels to identify entry/exit points
    - **Moving Average Crossover**: Employs fast MA (20-period) and slow MA (50-period) crossovers for trend confirmation
    - **Signal Confirmation**: Combines both indicators to reduce false signals and improve accuracy
    - **Customizable Parameters**: Adjustable RSI periods, thresholds, and MA periods for strategy optimization
    """)

    st.subheader("2. Advanced Risk Management")
    st.markdown("""
    **Comprehensive Risk Control System:**
    - **Position Sizing**: Risk-based position sizing using portfolio percentage allocation
    - **Stop Loss Management**: Automatic stop-loss orders to limit downside risk
    - **Take Profit Targets**: Predefined profit-taking levels with risk-reward ratios
    - **Maximum Drawdown Limits**: Portfolio-level drawdown protection
    - **Portfolio Diversification**: Multi-asset trading with correlation analysis
    """)

    st.subheader("3. Real-Time Portfolio Management")
    st.markdown("""
    **Dynamic Portfolio Tracking:**
    - **Live P&L Calculation**: Real-time profit and loss tracking for open and closed positions
    - **Portfolio Valuation**: Continuous portfolio value updates with market prices
    - **Holdings Management**: Track asset quantities and allocations across multiple cryptocurrencies
    - **Performance Metrics**: Sharpe ratio, win rate, profit factor, and drawdown analysis
    """)

    st.subheader("4. Strategy Backtesting")
    st.markdown("""
    **Historical Performance Analysis:**
    - **Multi-Timeframe Testing**: Test strategies on 1-minute to 1-hour intervals
    - **Historical Data**: Access to up to 1 year of historical market data
    - **Performance Metrics**: Comprehensive analysis including total return, max drawdown, and trade statistics
    - **Parameter Optimization**: Test different RSI/MA parameters to optimize strategy performance
    """)

    st.header("Technical Architecture")

    st.subheader("Data Sources")
    st.markdown("""
    - **Binance API**: Primary data source for real-time prices and historical data
    - **Rate Limiting**: Intelligent request management to comply with API limits
    - **Error Handling**: Robust error recovery and failover mechanisms
    """)

    st.subheader("Trading Engine")
    st.markdown("""
    - **Signal Generation**: Real-time technical indicator calculations
    - **Order Management**: Automated buy/sell order execution
    - **Position Tracking**: Multi-position management with individual P&L tracking
    - **Risk Controls**: Pre-trade risk checks and position limits
    """)

    st.header("How It Works")

    st.subheader("1. Market Analysis")
    st.markdown("""
    The system continuously monitors cryptocurrency markets, calculating technical indicators in real-time:
    - Fetches 1-minute candlestick data from Binance
    - Calculates RSI using configurable periods (default: 14)
    - Computes moving averages for trend analysis
    - Generates trading signals based on indicator crossovers
    """)

    st.subheader("2. Signal Processing")
    st.markdown("""
    Trading signals are generated when specific conditions are met:
    - **Buy Signal**: RSI < 30 (oversold) AND fast MA > slow MA (uptrend)
    - **Sell Signal**: RSI > 70 (overbought) OR fast MA < slow MA (downtrend)
    - **Confirmation**: Multiple indicator alignment reduces false signals
    """)

    st.subheader("3. Risk Assessment")
    st.markdown("""
    Before executing trades, the system performs comprehensive risk checks:
    - Portfolio exposure limits
    - Position size calculations based on account risk percentage
    - Stop-loss and take-profit level determination
    - Available capital verification
    """)

    st.subheader("4. Order Execution")
    st.markdown("""
    Automated trade execution with the following features:
    - Market order placement for immediate execution
    - Stop-loss and take-profit order management
    - Position monitoring and automatic exit triggers
    - Trade logging and performance tracking
    """)

    st.header("Performance Monitoring")

    st.subheader("Real-Time Metrics")
    st.markdown("""
    The system provides continuous performance monitoring:
    - **Portfolio Value**: Live portfolio valuation with unrealized P&L
    - **Daily P&L**: Today's profit/loss including all trading activity
    - **Win Rate**: Percentage of profitable trades
    - **Profit Factor**: Ratio of gross profit to gross loss
    - **Maximum Drawdown**: Largest peak-to-trough decline
    """)

    st.subheader("Trade Analysis")
    st.markdown("""
    Detailed trade tracking and analysis:
    - Individual trade P&L calculation
    - Entry and exit price tracking
    - Hold time analysis
    - Exit reason classification (profit target, stop loss, manual)
    """)

    st.header("Getting Started")

    st.subheader("Configuration")
    st.markdown("""
    1. **Select Trading Pair**: Choose from major cryptocurrency pairs (BTC, ETH, ADA, etc.)
    2. **Set Strategy Parameters**: Configure RSI periods, thresholds, and MA periods
    3. **Define Risk Settings**: Set risk per trade percentage and position limits
    4. **Enable Auto Trading**: Activate automated trading or use manual mode
    """)

    st.subheader("Monitoring")
    st.markdown("""
    1. **Dashboard Overview**: Monitor real-time prices, portfolio value, and key metrics
    2. **Trade History**: Review recent trades and performance
    3. **Portfolio Holdings**: Track asset allocations and positions
    4. **Performance Charts**: Analyze trading performance over time
    """)

    st.header("Risk Disclaimer")
    st.markdown("""
    **Important Notice:**
    - Cryptocurrency trading involves substantial risk and may result in significant losses
    - Past performance does not guarantee future results
    - This system is for educational and research purposes
    - Always conduct thorough testing before live trading
    - Never risk more than you can afford to lose
    """)


def main():
    st.title("HFT Crypto Bot")
    st.markdown(
        "**High-Frequency Trading with RSI/MA Strategy and Advanced Risk Management**"
    )

    # Add navigation
    page = st.sidebar.selectbox("Navigation",
                                ["Trading Dashboard", "Documentation"],
                                index=0)

    if page == "Documentation":
        show_documentation()
        return

    # Update portfolio with current prices and check stop loss/take profit
    update_portfolio_with_current_prices()

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")

        # Symbol Selection
        symbol = st.selectbox("Trading Pair", [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT"
        ],
                              index=0)

        # Strategy Parameters
        st.subheader("Strategy Parameters")
        rsi_period = st.slider("RSI Period", 3, 9, 6)
        rsi_buy_threshold = st.slider("RSI Buy Threshold", 10, 30, 20)
        rsi_sell_threshold = st.slider("RSI Sell Threshold", 70, 100, 80)

        fast_ma_period = st.slider("Fast MA Period", 3, 10, 3)
        slow_ma_period = st.slider("Slow MA Period", 20, 50, 33)

        # Risk Management
        st.subheader("Risk Management")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.05, 0.5, 0.1)
        lookback_period = st.slider("Lookback Period", 10, 50, 20)

        # Auto Trading
        st.subheader("Trading Mode")
        if st.button("🤖 Toggle Auto Trading"):
            st.session_state.auto_trading = not st.session_state.auto_trading

        st.write(
            f"Auto Trading: {'🟢 ON' if st.session_state.auto_trading else '🔴 OFF'}"
        )

        # Manual Trading
        st.subheader("Manual Trading")
        trade_amount = st.number_input("Trade Amount (USDT)",
                                       min_value=10.0,
                                       value=100.0)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 BUY", use_container_width=True):
                execute_manual_trade(symbol, "BUY", trade_amount)
        with col2:
            if st.button("📉 SELL", use_container_width=True):
                execute_manual_trade(symbol, "SELL", trade_amount)

    # Main Dashboard
    col1, col2, col3, col4 = st.columns(4)

    # Get current market data
    try:
        current_price = st.session_state.binance_client.get_current_price(
            symbol)
        klines = st.session_state.binance_client.get_klines(symbol, "1m", 100)

        if klines is not None and len(klines) > 0:
            df = pd.DataFrame(klines,
                              columns=[
                                  'timestamp', 'open', 'high', 'low', 'close',
                                  'volume', 'close_time', 'quote_asset_volume',
                                  'number_of_trades',
                                  'taker_buy_base_asset_volume',
                                  'taker_buy_quote_asset_volume', 'ignore'
                              ])

            # Convert to proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate technical indicators
            df = st.session_state.technical_analysis.calculate_rsi(
                df, rsi_period)
            df = st.session_state.technical_analysis.calculate_ma(
                df, fast_ma_period, 'sma', 'close')
            df['fast_ma'] = df[f'sma_{fast_ma_period}']
            df = st.session_state.technical_analysis.calculate_ma(
                df, slow_ma_period, 'sma', 'close')
            df['slow_ma'] = df[f'sma_{slow_ma_period}']

            # Get trading signals
            signals = st.session_state.strategy.generate_signals(
                df, rsi_buy_threshold, rsi_sell_threshold)

            # Portfolio metrics with real-time P&L calculation
            portfolio_value = st.session_state.portfolio.get_total_value(
                current_price, symbol.replace('USDT', ''))
            daily_pnl = st.session_state.portfolio.get_daily_pnl()

            # Calculate total P&L including both realized and unrealized
            realized_pnl = sum(
                trade.get('pnl', 0)
                for trade in st.session_state.trades_history
                if trade.get('pnl') is not None)
            unrealized_pnl = portfolio_value - st.session_state.portfolio.initial_balance - realized_pnl
            total_pnl = realized_pnl + unrealized_pnl

            with col1:
                st.metric(
                    f"{symbol} Price",
                    f"${current_price:.4f}",
                    delta=
                    f"{((current_price / df.iloc[-2]['close'] - 1) * 100):.2f}%"
                )

            with col2:
                st.metric("Portfolio Value",
                          format_currency(portfolio_value),
                          delta=format_currency(daily_pnl))

            with col3:
                st.metric(
                    "Total P&L",
                    format_currency(total_pnl),
                    delta=format_percentage(
                        total_pnl /
                        st.session_state.portfolio.initial_balance * 100))

            with col4:
                current_rsi = df.iloc[-1]['rsi']
                rsi_color = "🟢" if current_rsi < rsi_buy_threshold else "🔴" if current_rsi > rsi_sell_threshold else "🟡"
                st.metric("RSI",
                          f"{current_rsi:.2f} {rsi_color}",
                          delta=f"{(current_rsi - df.iloc[-2]['rsi']):.2f}")

            # Trading Chart
            st.subheader("Trading Chart")
            fig = create_trading_chart(df, signals, symbol)
            st.plotly_chart(fig, use_container_width=True)

            # Current Signals
            st.subheader("Current Signals")
            col1, col2 = st.columns(2)

            with col1:
                latest_signal = signals.iloc[-1] if len(signals) > 0 else None
                if latest_signal is not None:
                    signal_type = "BUY" if latest_signal[
                        'signal'] == 1 else "SELL" if latest_signal[
                            'signal'] == -1 else "HOLD"
                    signal_color = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "🟡"
                    st.info(
                        f"**Current Signal:** {signal_color} {signal_type}")

                    if signal_type != "HOLD":
                        st.write(f"**RSI:** {latest_signal['rsi']:.2f}")
                        st.write(
                            f"**Fast MA:** {latest_signal['fast_ma']:.4f}")
                        st.write(
                            f"**Slow MA:** {latest_signal['slow_ma']:.4f}")
                else:
                    st.info("**Current Signal:** 🟡 No Signal")

            with col2:
                # Auto Trading Logic
                if st.session_state.auto_trading and latest_signal is not None:
                    if latest_signal['signal'] != 0:
                        auto_trade_result = execute_auto_trade(
                            symbol, latest_signal['signal'], current_price, df,
                            risk_per_trade, lookback_period)
                        if auto_trade_result:
                            st.success(
                                f"Auto trade executed: {auto_trade_result}")

            # Performance Metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)

            performance_metrics = calculate_performance_metrics(
                st.session_state.trades_history)

            with col1:
                st.metric("Win Rate",
                          f"{performance_metrics['win_rate']:.1f}%")
                st.metric("Total Trades", performance_metrics['total_trades'])

            with col2:
                st.metric("Sharpe Ratio",
                          f"{performance_metrics['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown",
                          f"{performance_metrics['max_drawdown']:.2f}%")

            with col3:
                st.metric("Annual Return",
                          f"{performance_metrics['annual_return']:.2f}%")
                st.metric("Profit Factor",
                          f"{performance_metrics['profit_factor']:.2f}")

            # Recent Trades
            st.subheader("Recent Trades")
            if st.session_state.trades_history:
                trades_df = pd.DataFrame(st.session_state.trades_history)
                trades_df = trades_df.sort_values('timestamp',
                                                  ascending=False).head(10)

                # Format the trades display
                display_trades = trades_df.copy()
                display_trades['Time'] = display_trades[
                    'timestamp'].dt.strftime('%H:%M:%S')
                display_trades['Symbol'] = display_trades['symbol']
                display_trades['Side'] = display_trades['side']
                display_trades['Quantity'] = display_trades['quantity'].round(
                    6)
                display_trades['Price'] = display_trades['price'].round(4)
                display_trades['Amount'] = display_trades['amount'].round(2)
                display_trades['P&L'] = display_trades['pnl'].fillna(0).round(
                    2)
                display_trades['Type'] = display_trades['type']

                # Add color coding for P&L
                def color_pnl(val):
                    if val > 0:
                        return f"🟢 ${val:,.2f}"
                    elif val < 0:
                        return f"🔴 ${val:,.2f}"
                    else:
                        return f"⚪ ${val:,.2f}"

                display_trades['P&L Formatted'] = display_trades['P&L'].apply(
                    color_pnl)

                # Select columns to display
                columns_to_show = [
                    'Time', 'Symbol', 'Side', 'Quantity', 'Price', 'Amount',
                    'P&L Formatted', 'Type'
                ]
                if 'exit_reason' in display_trades.columns:
                    display_trades['Exit Reason'] = display_trades[
                        'exit_reason'].fillna('-')
                    columns_to_show.append('Exit Reason')

                st.dataframe(display_trades[columns_to_show],
                             use_container_width=True)

                # Show P&L summary with real-time calculations
                valid_pnl_trades = trades_df[trades_df['pnl'].notna()]
                total_realized_pnl = valid_pnl_trades['pnl'].sum() if len(
                    valid_pnl_trades) > 0 else 0
                profitable_trades = len(
                    valid_pnl_trades[valid_pnl_trades['pnl'] > 0])
                losing_trades = len(
                    valid_pnl_trades[valid_pnl_trades['pnl'] < 0])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Realized P&L",
                              f"${total_realized_pnl:,.2f}")
                with col2:
                    st.metric("Profitable Trades", profitable_trades)
                with col3:
                    st.metric("Losing Trades", losing_trades)

            else:
                st.info("No trades executed yet.")

            # Portfolio Holdings
            st.subheader("Portfolio Holdings")
            holdings = st.session_state.portfolio.get_holdings()
            if holdings:
                holdings_data = []
                for asset, amount in holdings.items():
                    if asset == 'USDT':
                        value = amount
                    else:
                        asset_price = st.session_state.binance_client.get_current_price(
                            f"{asset}USDT")
                        value = amount * asset_price if asset_price else 0

                    holdings_data.append({
                        'Asset': asset,
                        'Amount': f"{amount:.6f}",
                        'Value (USDT)': f"{value:.2f}"
                    })

                holdings_df = pd.DataFrame(holdings_data)
                st.dataframe(holdings_df, use_container_width=True)
            else:
                st.info("No holdings in portfolio.")

            # Strategy Backtesting Section
            st.header("Strategy Backtesting")
            st.markdown(
                "Test your RSI/MA strategy with **$10,000,000** starting equity"
            )

            backtest_col1, backtest_col2 = st.columns([2, 1])

            with backtest_col2:
                st.subheader("Backtest Parameters")

                # Backtest symbol selection
                backtest_symbol = st.selectbox("Backtest Symbol", [
                    "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
                    "BNBUSDT"
                ],
                                               index=0,
                                               key="backtest_symbol")

                # Time period
                backtest_days = st.selectbox(
                    "Historical Period", [30, 90, 180, 365],
                    index=3,
                    format_func=lambda x: f"{x} days ({x//30} months)"
                    if x < 365 else "1 year")

                # Interval selection - default to 1 minute for high-frequency testing
                backtest_interval = st.selectbox("Time Interval",
                                                 ["1m", "5m", "15m", "1h"],
                                                 index=0,
                                                 format_func=lambda x: {
                                                     "1m": "1 Minute",
                                                     "5m": "5 Minutes",
                                                     "15m": "15 Minutes",
                                                     "1h": "1 Hour"
                                                 }[x])

                # Strategy parameters for backtesting
                st.write("**Strategy Settings**")
                bt_rsi_period = st.slider("RSI Period",
                                          3,
                                          9,
                                          6,
                                          key="bt_rsi_period")
                bt_rsi_buy = st.slider("RSI Buy",
                                       10,
                                       30,
                                       rsi_buy_threshold,
                                       key="bt_rsi_buy")
                bt_rsi_sell = st.slider("RSI Sell",
                                        70,
                                        90,
                                        rsi_sell_threshold,
                                        key="bt_rsi_sell")
                bt_risk_per_trade = st.slider("Risk per Trade (%)",
                                              0.1,
                                              2.0,
                                              0.1,
                                              step=0.1,
                                              key="bt_risk")
                bt_lookback_period = st.slider("Lookback Period",
                                               10,
                                               50,
                                               20,
                                               key="bt_lookback")

                # Run backtest button
                run_backtest = st.button("Run Backtest",
                                         use_container_width=True)

            with backtest_col1:
                if run_backtest:
                    with st.spinner(
                            f"Running backtest on {backtest_days} days of {backtest_symbol} data..."
                    ):
                        try:
                            # Fetch historical data from Binance
                            historical_data = st.session_state.binance_client.get_historical_data_for_backtest(
                                backtest_symbol, backtest_interval,
                                backtest_days)

                            if historical_data and len(historical_data) > 50:
                                # Convert to DataFrame
                                backtest_df = pd.DataFrame(
                                    historical_data,
                                    columns=[
                                        'timestamp', 'open', 'high', 'low',
                                        'close', 'volume', 'close_time',
                                        'quote_asset_volume',
                                        'number_of_trades',
                                        'taker_buy_base_asset_volume',
                                        'taker_buy_quote_asset_volume',
                                        'ignore'
                                    ])

                                # Convert data types
                                for col in [
                                        'open', 'high', 'low', 'close',
                                        'volume'
                                ]:
                                    backtest_df[col] = pd.to_numeric(
                                        backtest_df[col])

                                backtest_df['timestamp'] = pd.to_datetime(
                                    backtest_df['timestamp'], unit='ms')
                                backtest_df.set_index('timestamp',
                                                      inplace=True)

                                # Calculate technical indicators using backtest parameters
                                backtest_df = st.session_state.technical_analysis.calculate_rsi(
                                    backtest_df, bt_rsi_period)
                                backtest_df = st.session_state.technical_analysis.calculate_ma(
                                    backtest_df, fast_ma_period, 'sma',
                                    'close')
                                backtest_df['fast_ma'] = backtest_df[
                                    f'sma_{fast_ma_period}']
                                backtest_df = st.session_state.technical_analysis.calculate_ma(
                                    backtest_df, slow_ma_period, 'sma',
                                    'close')
                                backtest_df['slow_ma'] = backtest_df[
                                    f'sma_{slow_ma_period}']
                                backtest_df = st.session_state.technical_analysis.calculate_atr(
                                    backtest_df, 14)

                                # Run backtest with $10,000,000 equity
                                backtester = Backtester(
                                    initial_balance=10000000)

                                strategy_params = {
                                    'rsi_buy_threshold': bt_rsi_buy,
                                    'rsi_sell_threshold': bt_rsi_sell,
                                    'risk_per_trade': bt_risk_per_trade / 100,
                                    'lookback_period': bt_lookback_period,
                                    'stop_loss_pct': 0.02,
                                    'take_profit_pct': 0.04,
                                    'max_positions': 5,
                                    'commission': 0.001
                                }

                                results = backtester.run_backtest(
                                    backtest_df, st.session_state.strategy,
                                    strategy_params)

                                # Display results
                                st.success(
                                    "✅ Backtest completed successfully!")

                                # Key metrics
                                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(
                                    4)

                                with metrics_col1:
                                    st.metric(
                                        "Total Return",
                                        f"{results['total_return']:.2f}%")
                                    st.metric("Total Trades",
                                              results['total_trades'])

                                with metrics_col2:
                                    st.metric("Win Rate",
                                              f"{results['win_rate']:.1f}%")
                                    st.metric(
                                        "Profit Factor",
                                        f"{results['profit_factor']:.2f}")

                                with metrics_col3:
                                    st.metric(
                                        "Max Drawdown",
                                        f"{results['max_drawdown']:.2f}%")
                                    st.metric(
                                        "Sharpe Ratio",
                                        f"{results['sharpe_ratio']:.2f}")

                                with metrics_col4:
                                    st.metric(
                                        "Final Balance",
                                        f"${results['final_equity']:,.2f}")
                                    st.metric("Avg Trade",
                                              f"${results['avg_trade']:,.2f}")

                                # Performance analysis
                                st.subheader("📊 Performance Analysis")

                                perf_col1, perf_col2 = st.columns(2)

                                with perf_col1:
                                    # Buy and Hold comparison
                                    first_price = backtest_df.iloc[0]['close']
                                    last_price = backtest_df.iloc[-1]['close']
                                    buy_hold_return = (
                                        last_price / first_price - 1) * 100

                                    st.write("**Strategy vs Buy & Hold**")
                                    comparison_data = {
                                        'Metric': [
                                            'Total Return', 'Max Drawdown',
                                            'Win Rate'
                                        ],
                                        'Strategy': [
                                            f"{results['total_return']:.2f}%",
                                            f"{results['max_drawdown']:.2f}%",
                                            f"{results['win_rate']:.1f}%"
                                        ],
                                        'Buy & Hold': [
                                            f"{buy_hold_return:.2f}%", "N/A",
                                            "N/A"
                                        ]
                                    }
                                    st.dataframe(pd.DataFrame(comparison_data),
                                                 use_container_width=True)

                                with perf_col2:
                                    # Trade distribution
                                    if results['winning_trades'] > 0 or results[
                                            'losing_trades'] > 0:
                                        trade_labels = [
                                            'Winning Trades', 'Losing Trades'
                                        ]
                                        trade_values = [
                                            results['winning_trades'],
                                            results['losing_trades']
                                        ]

                                        fig_pie = go.Figure(data=[
                                            go.Pie(
                                                labels=trade_labels,
                                                values=trade_values,
                                                hole=0.3,
                                                marker_colors=['green', 'red'])
                                        ])
                                        fig_pie.update_layout(
                                            title="Trade Distribution",
                                            height=300)
                                        st.plotly_chart(
                                            fig_pie, use_container_width=True)

                                # Equity curve
                                if backtester.equity_curve:
                                    st.subheader("💹 Equity Curve")
                                    equity_data = pd.DataFrame(
                                        backtester.equity_curve)

                                    fig_equity = go.Figure()
                                    fig_equity.add_trace(
                                        go.Scatter(x=list(
                                            range(len(equity_data))),
                                                   y=equity_data['equity'],
                                                   mode='lines',
                                                   name='Portfolio Value',
                                                   line=dict(color='blue',
                                                             width=2)))

                                    fig_equity.add_hline(
                                        y=10000000,
                                        line_dash="dash",
                                        line_color="gray",
                                        annotation_text="Initial Balance ($10M)"
                                    )

                                    fig_equity.update_layout(
                                        title=
                                        f"Portfolio Performance - {backtest_symbol} ({backtest_days} days)",
                                        xaxis_title="Time Period",
                                        yaxis_title="Portfolio Value (USDT)",
                                        height=400)
                                    st.plotly_chart(fig_equity,
                                                    use_container_width=True)

                                # Detailed trade history
                                if backtester.trades:
                                    st.subheader("📋 Trade Details")
                                    trades_df = pd.DataFrame(backtester.trades)

                                    # Format the trades DataFrame for display
                                    display_trades = trades_df.copy()
                                    display_trades[
                                        'entry_price'] = display_trades[
                                            'entry_price'].round(4)
                                    display_trades[
                                        'exit_price'] = display_trades[
                                            'exit_price'].round(4)
                                    display_trades['pnl'] = display_trades[
                                        'pnl'].round(2)
                                    display_trades[
                                        'return_pct'] = display_trades[
                                            'return_pct'].round(2)

                                    st.dataframe(display_trades.head(20),
                                                 use_container_width=True)

                                    if len(trades_df) > 20:
                                        st.info(
                                            f"Showing first 20 trades out of {len(trades_df)} total trades"
                                        )

                            else:
                                st.error(
                                    "❌ Unable to fetch sufficient historical data for backtesting"
                                )
                                st.info(
                                    "This might be due to API limits or network issues. Please try again or contact support if the issue persists."
                                )

                        except Exception as e:
                            st.error(f"❌ Backtest failed: {str(e)}")
                            st.info(
                                "Please check your connection and try again")

                elif 'backtest_results' not in st.session_state:
                    st.info(
                        "👆 Configure your backtest parameters and click 'Run Backtest' to test your strategy on real historical data"
                    )

                    # Show sample information about what backtesting will do
                    st.markdown("""
                    **What this backtest will do:**
                    - Fetch real historical price data from Binance
                    - Apply your RSI/MA strategy with the selected parameters
                    - Simulate trades with realistic commission fees
                    - Calculate comprehensive performance metrics
                    - Compare against buy-and-hold strategy
                    - Show detailed trade history and equity curve
                    """)

    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        st.info("Please check your internet connection and try again.")

    # Auto-refresh
    if st.session_state.auto_trading:
        time.sleep(5)
        st.rerun()


def create_trading_chart(df, signals, symbol):
    """Create interactive trading chart with technical indicators"""
    fig = make_subplots(rows=2,
                        cols=1,
                        subplot_titles=(f'{symbol} Price Chart', 'RSI'),
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name='Price'),
                  row=1,
                  col=1)

    # Moving averages
    fig.add_trace(go.Scatter(x=df['timestamp'],
                             y=df['fast_ma'],
                             line=dict(color='blue', width=1),
                             name='Fast MA'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=df['timestamp'],
                             y=df['slow_ma'],
                             line=dict(color='red', width=1),
                             name='Slow MA'),
                  row=1,
                  col=1)

    # Buy/Sell signals
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]

    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(x=buy_signals['timestamp'],
                                 y=buy_signals['close'],
                                 mode='markers',
                                 marker=dict(symbol='triangle-up',
                                             size=10,
                                             color='green'),
                                 name='Buy Signal'),
                      row=1,
                      col=1)

    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(x=sell_signals['timestamp'],
                                 y=sell_signals['close'],
                                 mode='markers',
                                 marker=dict(symbol='triangle-down',
                                             size=10,
                                             color='red'),
                                 name='Sell Signal'),
                      row=1,
                      col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['timestamp'],
                             y=df['rsi'],
                             line=dict(color='purple', width=2),
                             name='RSI'),
                  row=2,
                  col=1)

    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(title=f"{symbol} Trading Chart",
                      xaxis_title="Time",
                      yaxis_title="Price",
                      height=600,
                      xaxis_rangeslider_visible=False,
                      showlegend=True)

    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig


def execute_manual_trade(symbol, side, amount):
    """Execute manual trade"""
    try:
        current_price = st.session_state.binance_client.get_current_price(
            symbol)
        if current_price is None:
            st.error("Unable to fetch current price")
            return

        base_asset = symbol.replace('USDT', '')
        quantity = amount / current_price

        if side == "BUY":
            # Calculate stop loss and take profit for buy orders
            stop_loss = current_price * 0.97  # 3% stop loss
            take_profit = current_price * 1.06  # 6% take profit

            success = st.session_state.portfolio.add_position(
                base_asset,
                quantity,
                current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_type='long')
        else:
            # For sell orders, check if we have positions to close
            success = st.session_state.portfolio.remove_position(
                base_asset, quantity, current_price, exit_reason='manual')

        if success:
            # Calculate P&L for sell orders
            pnl = 0
            if side == "SELL":
                # This would be calculated in the portfolio class
                positions = st.session_state.portfolio.get_open_positions(
                ).get(base_asset, [])
                if positions:
                    # Simple P&L calculation for display
                    avg_entry = sum(p['entry_price'] * p['quantity']
                                    for p in positions) / sum(
                                        p['quantity'] for p in positions)
                    pnl = (current_price - avg_entry) * quantity

            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': current_price if side == "BUY" else None,
                'exit_price': current_price if side == "SELL" else None,
                'price': current_price,
                'amount': amount,
                'pnl': pnl,
                'type': 'Manual'
            }
            st.session_state.trades_history.append(trade)

            pnl_text = f" (P&L: ${pnl:,.2f})" if side == "SELL" and pnl != 0 else ""
            st.success(
                f"Manual {side} order executed: {quantity:.6f} {base_asset} at ${current_price:.4f}{pnl_text}"
            )
        else:
            st.error(f"Failed to execute {side} order")

    except Exception as e:
        st.error(f"Error executing manual trade: {str(e)}")


def execute_auto_trade(symbol, signal, current_price, df, risk_per_trade,
                       lookback_period):
    """Execute automatic trade based on signal"""
    try:
        base_asset = symbol.replace('USDT', '')
        portfolio_value = st.session_state.portfolio.get_total_value(
            current_price)

        # Calculate position size based on risk management
        risk_amount = portfolio_value * (risk_per_trade / 100)

        # Calculate stop loss and take profit levels
        if signal == 1:  # BUY
            stop_loss = st.session_state.technical_analysis.calculate_support_resistance(
                df, lookback_period, 'support')
            take_profit = st.session_state.technical_analysis.calculate_support_resistance(
                df, lookback_period, 'resistance')

            if stop_loss and current_price > stop_loss:
                risk_per_share = current_price - stop_loss
                quantity = risk_amount / risk_per_share
                amount = quantity * current_price

                success = st.session_state.portfolio.add_position(
                    base_asset, quantity, current_price)
                side = "BUY"
            else:
                return None

        else:  # SELL
            stop_loss = st.session_state.technical_analysis.calculate_support_resistance(
                df, lookback_period, 'resistance')
            take_profit = st.session_state.technical_analysis.calculate_support_resistance(
                df, lookback_period, 'support')

            if stop_loss and current_price < stop_loss:
                risk_per_share = stop_loss - current_price
                quantity = risk_amount / risk_per_share
                amount = quantity * current_price

                success = st.session_state.portfolio.remove_position(
                    base_asset, quantity, current_price)
                side = "SELL"
            else:
                return None

        if success:
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'amount': amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'type': 'Auto'
            }
            st.session_state.trades_history.append(trade)
            return f"{side} {quantity:.6f} {base_asset} at ${current_price:.4f}"

        return None

    except Exception as e:
        st.error(f"Error executing auto trade: {str(e)}")
        return None


if __name__ == "__main__":
    main()
