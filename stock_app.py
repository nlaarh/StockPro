import streamlit as st

# Set page config at the very top
st.set_page_config(
    page_title="StockPro",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import logging
import traceback
from datetime import datetime, timedelta
import time
import re

from technical_analysis import technical_analysis_tab, plot_daily_candlestick, plot_stock_history
from buffett_analysis import buffett_analysis_tab
from options_analysis import (
    options_analysis_tab, get_options_chain, get_options_strategy, 
    get_options_analyst_letter
)
from utils import calculate_rsi, calculate_macd
from company_profile import company_profile_tab
from market_movers import market_movers_tab
import stock_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px;
        color: #000000;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)

def get_stock_data(ticker, period="1y"):
    """Get stock data with technical indicators"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            st.error(f"No historical data available for {ticker}")
            return None
            
        # Calculate technical indicators
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        rsi_data = calculate_rsi(data['Close'])
        data['RSI'] = rsi_data
        
        # Calculate MACD
        macd_data = calculate_macd(data['Close'])
        data['MACD'] = macd_data['MACD']
        data['Signal'] = macd_data['Signal']
        data['MACD_Hist'] = macd_data['Histogram']
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        logger.error(f"Error fetching stock data: {str(e)}")
        return None

def predict_stock_price(data, prediction_days=7, model_type="Random Forest"):
    """Predict stock prices using various models"""
    try:
        # Prepare features
        data = data.copy()
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data)
        macd_data = calculate_macd(data)
        data['MACD'] = macd_data['MACD']
        data['Signal'] = macd_data['Signal']
        
        # Drop any NaN values
        data = data.dropna()
        
        if len(data) < 60:  # Need enough historical data
            return None, None, None
            
        # Prepare features and target
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
        X = data[features].values
        y = data['Close'].values
        
        # Scale the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        sequence_length = 60  # Use last 60 days to predict next day
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:(i + sequence_length)])
            y_sequences.append(y[i + sequence_length])
            
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Train the selected model
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_sequences.reshape(X_sequences.shape[0], -1), y_sequences)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.fit(X_sequences.reshape(X_sequences.shape[0], -1), y_sequences)
        elif model_type == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42)
            model.fit(X_sequences.reshape(X_sequences.shape[0], -1), y_sequences)
        else:
            return None, None, None
            
        # Generate predictions
        predictions = []
        confidence_lower = []
        confidence_upper = []
        last_sequence = X_scaled[-sequence_length:]
        
        for _ in range(prediction_days):
            # Predict next day
            next_pred = model.predict(last_sequence.reshape(1, -1))
            predictions.append(next_pred[0])
            
            # Calculate confidence bands (using model's feature importances)
            std = np.std(y_sequences) * (1 - model.score(X_sequences.reshape(X_sequences.shape[0], -1), y_sequences))
            confidence_lower.append(next_pred[0] - 2 * std)
            confidence_upper.append(next_pred[0] + 2 * std)
            
            # Update sequence for next prediction
            new_row = np.zeros_like(X_scaled[0])
            new_row[3] = next_pred[0]  # Set Close price
            last_sequence = np.vstack((last_sequence[1:], new_row))
            
        return predictions, (confidence_lower, confidence_upper), features
        
    except Exception as e:
        logger.error(f"Error in predict_stock_price: {str(e)}")
        return None, None, None

def prediction_tab():
    """Stock Prediction tab"""
    try:
        st.header("Stock Price Prediction")
        
        # Stock input
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Enter Stock Symbol:", "AAPL", key="prediction_ticker").upper()
            
        if ticker:
            # Import prediction module only when needed
            from prediction import display_prediction
            display_prediction(ticker)
            
    except ImportError as e:
        st.error("Failed to load prediction module. Please check that prediction.py exists in the same directory.")
    except Exception as e:
        st.error(f"Error in Prediction tab: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    try:
        # Initialize session state
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 'company'
        
        if 'stock_symbol' not in st.session_state:
            st.session_state.stock_symbol = 'AAPL'
        
        # Header section with clean layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.title("📈 StockPro")
            st.markdown("Advanced Stock Analysis and Prediction Platform")
        
        # Single stock input field
        st.session_state.stock_symbol = st.text_input(
            "Enter Stock Symbol:",
            value=st.session_state.stock_symbol,
            key="global_stock_symbol"
        ).upper()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Company Profile",
            "Technical Analysis",
            "Stock Prediction",
            "Buffett Analysis",
            "Options Analysis",
            "Stock Recommendations",
            "Market Movers"
        ])
        
        # Company Profile Tab
        with tab1:
            company_profile_tab(st.session_state.stock_symbol)
        
        # Technical Analysis Tab
        with tab2:
            technical_analysis_tab()
        
        # Stock Prediction Tab
        with tab3:
            prediction_tab()
        
        # Buffett Analysis Tab
        with tab4:
            buffett_analysis_tab()
        
        # Options Analysis Tab
        with tab5:
            options_analysis_tab(st.session_state.stock_symbol)
        
        # Stock Recommendations Tab
        with tab6:
            stock_recommendations.recommendations_tab()
        
        # Market Movers Tab
        with tab7:
            market_movers_tab()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")

def market_movers_tab():
    """Market Movers tab with subtabs"""
    try:
        st.header("Market Movers")
        
        # Create subtabs
        subtab1, subtab2, subtab3 = st.tabs(["Top Gainers", "Top Losers", "Most Active"])
        
        with subtab1:
            st.subheader("Top Gainers")
            gainers = get_market_movers('gainers')
            if gainers is not None:
                display_movers_table(gainers, 'gainers')
            else:
                st.error("Unable to fetch top gainers data")
            
        with subtab2:
            st.subheader("Top Losers")
            losers = get_market_movers('losers')
            if losers is not None:
                display_movers_table(losers, 'losers')
            else:
                st.error("Unable to fetch top losers data")
            
        with subtab3:
            st.subheader("Most Active")
            active = get_market_movers('active')
            if active is not None:
                display_movers_table(active, 'active')
            else:
                st.error("Unable to fetch most active stocks data")
            
    except Exception as e:
        st.error(f"Error in Market Movers tab: {str(e)}")

def get_market_movers(category):
    """Get market movers data using yfinance"""
    try:
        # Define tickers for each category
        tickers = {
            'gainers': ['^GSPC', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V'],
            'losers': ['^DJI', 'WMT', 'JNJ', 'PG', 'XOM', 'BAC', 'HD', 'CVX', 'PFE', 'KO'],
            'active': ['^IXIC', 'AMD', 'INTC', 'MU', 'CSCO', 'QCOM', 'PYPL', 'NFLX', 'DIS', 'ADBE']
        }
        
        # Get data for selected category
        selected_tickers = tickers.get(category, tickers['active'])
        data = []
        
        for ticker in selected_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='1d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Open'].iloc[0]
                    price_change = current_price - prev_price
                    pct_change = (price_change / prev_price) * 100
                    
                    data.append({
                        'symbol': ticker,
                        'name': info.get('longName', ticker),
                        'price': current_price,
                        'change': price_change,
                        '% change': pct_change,
                        'volume': hist['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap', 0)
                    })
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Sort based on category
        if category == 'gainers':
            df = df.sort_values('% change', ascending=False)
        elif category == 'losers':
            df = df.sort_values('% change', ascending=True)
        else:  # active
            df = df.sort_values('volume', ascending=False)
        
        return df.head(5)  # Return top 5
        
    except Exception as e:
        logger.error(f"Error getting market movers data: {str(e)}")
        return None

def display_movers_table(df, category):
    """Display market movers table"""
    try:
        if df is None or df.empty:
            st.error(f"No data available for {category}")
            return
            
        # Format numeric columns
        df = df.copy()
        df['price'] = df['price'].apply(lambda x: f"${x:,.2f}")
        df['change'] = df['change'].apply(lambda x: f"${x:,.2f}")
        df['% change'] = df['% change'].apply(lambda x: f"{x:,.2f}%")
        df['volume'] = df['volume'].apply(lambda x: f"{x:,.0f}")
        df['market_cap'] = df['market_cap'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
        
        # Rename columns for display
        df.columns = ['Symbol', 'Company', 'Price', 'Change', '% Change', 'Volume', 'Market Cap']
        
        # Display table
        st.dataframe(
            df,
            column_config={
                "Symbol": st.column_config.TextColumn(
                    "Symbol",
                    help="Stock symbol",
                    width="small",
                ),
                "Company": st.column_config.TextColumn(
                    "Company",
                    help="Company name",
                    width="medium",
                ),
                "Price": st.column_config.TextColumn(
                    "Price",
                    help="Current stock price",
                    width="small",
                ),
                "Change": st.column_config.TextColumn(
                    "Change",
                    help="Price change",
                    width="small",
                ),
                "% Change": st.column_config.TextColumn(
                    "% Change",
                    help="Percentage change",
                    width="small",
                ),
                "Volume": st.column_config.TextColumn(
                    "Volume",
                    help="Trading volume",
                    width="medium",
                ),
                "Market Cap": st.column_config.TextColumn(
                    "Market Cap",
                    help="Market capitalization",
                    width="medium",
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
    except Exception as e:
        logger.error(f"Error displaying market movers table: {str(e)}")
        st.error("Error displaying data table")

def technical_analysis_tab():
    """Technical Analysis tab"""
    try:
        # Get user input
        ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL").upper()
        
        if ticker:
            # Get stock data
            data = get_stock_data(ticker)
            
            if data is not None:
                # Call technical analysis function
                from technical_analysis import technical_analysis_tab as show_technical_analysis
                show_technical_analysis(ticker, data)
            else:
                st.error(f"Could not fetch data for {ticker}")
    except Exception as e:
        st.error(f"Error in technical analysis tab: {str(e)}")
        logger.error(f"Error in technical analysis tab: {str(e)}\n{traceback.format_exc()}")

def buffett_analysis_tab():
    """Buffett Analysis tab"""
    try:
        st.header("Buffett Analysis")
        
        # Stock input
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Enter Stock Symbol:", "AAPL", key="buffett_ticker").upper()
            
        if ticker:
            # Get stock data
            data = get_stock_data(ticker)
            
            if data is not None:
                # Get stock info
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Call buffett analysis function with data
                import buffett_analysis
                buffett_analysis.display_analysis(ticker)
            else:
                st.error(f"Could not fetch data for {ticker}")
    except Exception as e:
        st.error(f"Error in Buffett Analysis tab: {str(e)}")
        logger.error(f"Error in Buffett Analysis tab: {str(e)}\n{traceback.format_exc()}")
        
if __name__ == "__main__":
    main()