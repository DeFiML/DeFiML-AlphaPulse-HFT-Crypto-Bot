# HFT Crypto Bot

A sophisticated high-frequency trading (HFT) system for cryptocurrency markets built with Python and Streamlit. This bot combines RSI/Moving Average strategies with advanced risk management and real-time portfolio tracking.

## Features

- **High-Frequency Trading**: Real-time trading with RSI and Moving Average signals
- **Advanced Risk Management**: Stop-loss, take-profit, and position sizing controls
- **Live Portfolio Tracking**: Real-time P&L calculation and performance metrics
- **Strategy Backtesting**: Historical performance analysis with customizable parameters
- **Interactive Dashboard**: Web-based interface for monitoring and control
- **Multiple Trading Pairs**: Support for major cryptocurrency pairs (BTC, ETH, ADA, etc.)

## Requirements

- Python 3.8 or higher
- Internet connection for real-time market data
- Minimum 4GB RAM recommended

## Installation

### Option 1: Quick Start (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hft-crypto-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000 --server.address 0.0.0.0
   ```

### Option 2: Using Virtual Environment

1. **Clone and navigate to directory**
   ```bash
   git clone <repository-url>
   cd hft-crypto-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000 --server.address 0.0.0.0
   ```

### Option 3: Using Docker

1. **Build Docker image**
   ```bash
   docker build -t hft-crypto-bot .
   ```

2. **Run container**
   ```bash
   docker run -p 5000:5000 hft-crypto-bot
   ```

## Dependencies

The application requires the following Python packages:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
requests>=2.31.0
ta>=0.10.2
backtesting>=0.3.3
vectorbt>=0.25.0
```

## Configuration

### Streamlit Configuration

Create a `.streamlit/config.toml` file in your project directory:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
base = "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

### Trading Parameters

Default configuration can be modified in the sidebar:

- **RSI Period**: 3-9 (default: 6)
- **RSI Buy Threshold**: 10-30 (default: 20)
- **RSI Sell Threshold**: 70-90 (default: 80)
- **Risk per Trade**: 0.1-2.0% (default: 0.1%)
- **Moving Average Periods**: Fast (20), Slow (50)

## Usage

### Starting the Application

1. Open your terminal/command prompt
2. Navigate to the project directory
3. Run: `streamlit run app.py --server.port 5000 --server.address 0.0.0.0`
4. Open your browser and go to `http://localhost:5000`

### Navigation

The application has two main sections:

1. **Trading Dashboard**: Main interface for live trading and portfolio management
2. **Documentation**: Comprehensive guide to the system architecture and features

### Trading Dashboard Features

#### Portfolio Overview
- Real-time portfolio value tracking
- Daily P&L calculations
- Win rate and performance metrics
- Current holdings display

#### Manual Trading
- Buy/Sell buttons for immediate execution
- Real-time price display
- Position size calculator

#### Automated Trading
- Enable/disable automatic trading
- RSI-based signal generation
- Moving average trend confirmation
- Risk-managed position sizing

#### Recent Trades
- Trade history with P&L tracking
- Entry/exit prices and timestamps
- Trade performance analysis

#### Strategy Backtesting
- Historical performance testing
- Customizable parameters
- Multi-timeframe analysis (1m, 5m, 15m, 1h)
- Performance visualization

### Key Trading Strategies

#### RSI/MA Strategy
- **Buy Signal**: RSI < threshold AND fast MA > slow MA
- **Sell Signal**: RSI > threshold OR fast MA < slow MA
- **Risk Management**: Stop-loss and take-profit levels

#### Risk Management
- Position sizing based on account percentage
- Maximum drawdown protection
- Stop-loss orders for downside protection
- Take-profit targets for profit realization

## File Structure

```
hft-crypto-bot/
├── app.py                 # Main Streamlit application
├── portfolio.py           # Portfolio management and P&L tracking
├── binance_client.py      # Binance API client for market data
├── crypto_api.py          # Alternative crypto API client
├── strategy.py            # Trading strategy implementation
├── bt_strategy.py         # Backtesting strategy adapter
├── custom_backtesting.py  # Custom backtesting engine
├── technical_analysis.py  # Technical indicator calculations
├── risk_management.py     # Risk management utilities
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## API Configuration

The application uses free APIs for market data:

### Binance API (Default)
- **Endpoint**: `https://api.binance.com`
- **Rate Limit**: 1200 requests per minute
- **Data**: Real-time prices, historical data, order book

### CoinGecko API (Fallback)
- **Endpoint**: `https://api.coingecko.com`
- **Rate Limit**: 10-50 requests per minute
- **Data**: Price data, market statistics

No API keys are required for basic functionality.

## Performance Optimization

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Stable internet connection (low latency preferred)

### Configuration Tips
1. **Reduce Update Frequency**: Lower real-time update intervals for better performance
2. **Limit Historical Data**: Use shorter backtesting periods for faster analysis
3. **Close Unused Tabs**: Web browser performance affects UI responsiveness

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Kill process using port 5000
sudo lsof -ti:5000 | xargs kill -9
# Or use a different port
streamlit run app.py --server.port 8501
```

#### 2. API Rate Limiting
- Reduce update frequency in the sidebar
- Wait 60 seconds before retrying
- Check network connection

#### 3. Data Loading Issues
- Verify internet connection
- Check API endpoints are accessible
- Restart the application

#### 4. Performance Issues
- Close other applications
- Reduce browser tab count
- Clear browser cache

### Error Messages

#### "Invalid symbol format"
- Restart the application
- Check selected trading pair
- Verify symbol exists in the market

#### "Insufficient balance"
- Check portfolio balance
- Reduce position size
- Verify trading calculations

## Development

### Adding New Features

1. **New Trading Strategies**
   - Modify `strategy.py` for signal generation
   - Update `bt_strategy.py` for backtesting
   - Add parameters in `app.py` sidebar

2. **Additional Indicators**
   - Add calculations in `technical_analysis.py`
   - Import in strategy modules
   - Update UI components

3. **New Data Sources**
   - Create new API client similar to `binance_client.py`
   - Update data fetching logic
   - Add fallback mechanisms

### Testing

Run the application in development mode:
```bash
streamlit run app.py --server.runOnSave true
```

## Security Considerations

- **No API Keys**: Application uses public APIs only
- **Local Execution**: All trading is simulated locally
- **Data Privacy**: No personal data is transmitted
- **Open Source**: Full code transparency

## Limitations

- **Simulated Trading**: No real money trading capability
- **Market Data**: Limited to free API rate limits
- **Historical Data**: Maximum 1 year of historical data
- **Real-time Updates**: Subject to API rate limiting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in the console
3. Restart the application
4. Check network connectivity

## Disclaimer

**Important**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk and may result in significant losses. Past performance does not guarantee future results. Always conduct thorough testing before any live trading activities.

- This is a simulation environment only
- No real money trading is performed
- Use at your own risk
- Not financial advice

## Changelog

### v1.0.0
- Initial release with RSI/MA strategy
- Real-time portfolio tracking
- Strategy backtesting
- Interactive web dashboard
- Comprehensive documentation