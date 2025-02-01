import streamlit as st
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
from prediction import predict_stock_price, prediction_tab
from buffett_analysis import buffett_analysis_tab
from options_analysis import (
    options_analysis_tab, get_options_chain, get_options_strategy, 
    highlight_options, get_best_options, analyze_market_conditions, 
    get_options_analyst_letter
)
from utils import calculate_rsi, calculate_macd
from company_profile import company_profile_tab
from market_movers import market_movers_tab
import stock_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def main():
    """Main function for the stock analysis app"""
    try:
        st.title("StockPro Analytics")
        
        # Initialize session state for ticker if not exists
        if 'ticker' not in st.session_state:
            st.session_state.ticker = 'AAPL'
            
        # Move stock selection to top
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input(
                "Enter Stock Ticker:",
                value=st.session_state.ticker,
                key="main_ticker"
            ).upper()
            
        # Update session state
        st.session_state.ticker = ticker
        
        if ticker:
            try:
                # Validate ticker
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info:
                    st.error(f"Invalid ticker: {ticker}")
                    return
                    
                with col2:
                    st.success(f"Analyzing {info.get('longName', ticker)}")
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        delta=f"{((current_price / stock.history(period='2d')['Close'].iloc[0]) - 1) * 100:+.2f}%"
                    )
                
                # Get stock data once
                data = get_stock_data(ticker)
                if data is None:
                    return
                
                # Create tabs
                tabs = st.tabs([
                    "🏢 Company Profile",
                    "📈 Technical Analysis",
                    "🔮 Price Prediction",
                    "💰 Buffett Analysis",
                    "📊 Market Movers",
                    "🎯 Options Analysis",
                    "📋 Stock Recommendations"
                ])
                
                # Company Profile Tab
                with tabs[0]:
                    company_profile_tab(ticker)
                    
                # Technical Analysis Tab
                with tabs[1]:
                    technical_analysis_tab(ticker, data)
                    
                # Price Prediction Tab
                with tabs[2]:
                    prediction_tab(ticker, data)
                    
                # Buffett Analysis Tab
                with tabs[3]:
                    buffett_analysis_tab()
                    
                # Market Movers Tab
                with tabs[4]:
                    market_movers_tab()
                    
                # Options Analysis Tab
                with tabs[5]:
                    options_analysis_tab(ticker)
                    
                # Stock Recommendations Tab
                with tabs[6]:
                    stock_recommendations.recommendations_tab()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
        else:
            st.info("Please enter a stock ticker to begin analysis")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()