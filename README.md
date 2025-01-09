# Stock Market Dashboard

A comprehensive stock market dashboard built with Python and Streamlit that provides real-time stock data, historical analysis, and news tracking.

## Features

- Real-time stock market data (price, volume, market cap)
- Historical stock data visualization with interactive charts
- Candlestick charts for technical analysis
- Company news feed
- Watchlist functionality
- User-friendly interface

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application using Streamlit:
```bash
streamlit run stock_app.py
```

## How to Use

1. **Search & Analysis**
   - Enter a stock symbol (e.g., AAPL, GOOGL) in the search box
   - View real-time stock data and historical charts
   - Add stocks to your watchlist
   - Read latest news about the company

2. **Watchlist**
   - View all your saved stocks in one place
   - Monitor their current prices and performance
   - Remove stocks from your watchlist

## Dependencies

- streamlit==1.28.0
- yfinance==0.2.31
- plotly==5.18.0
- pandas==2.1.3
- python-dotenv==1.0.0
