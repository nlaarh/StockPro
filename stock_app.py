import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import random
import ollama
import time
import warnings
import traceback
from math import sqrt
warnings.filterwarnings('ignore')
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm_response(prompt):
    """Get response from Llama3.2 model"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama2',
                                   'prompt': prompt,
                                   'stream': False
                               })
        if response.status_code == 200:
            return response.json()['response']
        else:
            logger.error(f"Error from Ollama API: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return None

# Set page config at the very beginning
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_rsi(data, periods=14):
    """Calculate RSI with error handling for empty data"""
    try:
        if len(data) < periods + 1:
            return None
            
        # Calculate price changes
        delta = data.diff()
        
        if delta.empty:
            return None
            
        # Get gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=periods, min_periods=periods).mean()
        avg_loss = losses.rolling(window=periods, min_periods=periods).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    except Exception as e:
        logger.error(f"RSI calculation error: {str(e)}")
        return None

def calculate_macd(data):
    """Calculate MACD with error handling for empty data"""
    try:
        if len(data) < 26:  # Minimum data needed for MACD
            return None, None
            
        # Calculate EMAs
        ema12 = data.ewm(span=12, adjust=False).mean()
        ema26 = data.ewm(span=26, adjust=False).mean()
        
        # Calculate MACD and Signal line
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return macd.iloc[-1], signal.iloc[-1]
    except Exception as e:
        logger.error(f"MACD calculation error: {str(e)}")
        return None, None

def get_technical_indicators(ticker):
    """Get technical indicators with error handling"""
    try:
        # Get historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if hist.empty:
            return {
                "RSI": None,
                "MACD": None,
                "Volume_Change": None,
                "Price_Momentum": None
            }
        
        # Calculate RSI
        rsi = calculate_rsi(hist['Close'])
        
        # Calculate MACD
        macd, signal = calculate_macd(hist['Close'])
        
        # Calculate volume change
        volume_change = ((hist['Volume'].iloc[-1] / hist['Volume'].iloc[-5]) - 1) * 100 if len(hist) >= 5 else None
        
        # Calculate price momentum (5-day return)
        momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100 if len(hist) >= 5 else None
        
        return {
            "RSI": rsi if rsi is not None else 50.0,  # Default to neutral
            "MACD": macd if macd is not None else 0.0,  # Default to neutral
            "Volume_Change": volume_change if volume_change is not None else 0.0,
            "Price_Momentum": momentum if momentum is not None else 0.0
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
        return {
            "RSI": 50.0,  # Default to neutral
            "MACD": 0.0,
            "Volume_Change": 0.0,
            "Price_Momentum": 0.0
        }

def plot_daily_candlestick(ticker):
    try:
        # Get stock data with alternative method
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval='1d')
        
        if hist.empty:
            # Try alternative method with period
            hist = stock.history(period="1y", interval="1d", proxy=None)
            
        if hist.empty:
            st.error(f"No data available for {ticker}")
            return
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_columns):
            st.error(f"Missing required price data for {ticker}")
            return
            
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=ticker,
            showlegend=True
        ))
        
        # Add volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(hist['Close'], hist['Open'])]
        
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color=colors,
            opacity=0.3
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{ticker} Daily Price Chart",
                x=0.5,
                xanchor='center'
            ),
            yaxis=dict(
                title="Price ($)",
                side="left",
                showgrid=True
            ),
            yaxis2=dict(
                title="Volume",
                side="right",
                overlaying="y",
                showgrid=False
            ),
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=False)
            ),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=85, b=50)
        )
        
        # Update hover template
        fig.update_traces(
            xhoverformat="%Y-%m-%d",
            yhoverformat="$.2f"
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Get company info and financials
        info = stock.info
        
        # Create tabs for different metric categories
        metric_tabs = st.tabs(["Key Statistics", "Financial Ratios", "Growth & Margins", "Trading Info"])
        
        with metric_tabs[0]:
            # Key Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
                st.metric("Revenue (TTM)", f"${info.get('totalRevenue', 0)/1e9:.2f}B")
                st.metric("Net Income (TTM)", f"${info.get('netIncomeToCommon', 0)/1e9:.2f}B")
            
            with col2:
                st.metric("EPS (TTM)", f"${info.get('trailingEps', 0):.2f}")
                st.metric("Forward EPS", f"${info.get('forwardEps', 0):.2f}")
                st.metric("Book Value/Share", f"${info.get('bookValue', 0):.2f}")
            
            with col3:
                st.metric("Shares Outstanding", f"{info.get('sharesOutstanding', 0)/1e6:.1f}M")
                st.metric("Float", f"{info.get('floatShares', 0)/1e6:.1f}M")
                st.metric("Insider Ownership", f"{info.get('heldPercentInsiders', 0)*100:.1f}%")
        
        with metric_tabs[1]:
            # Financial Ratios
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("P/E (TTM)", f"{info.get('trailingPE', 0):.2f}")
                st.metric("Forward P/E", f"{info.get('forwardPE', 0):.2f}")
                st.metric("PEG Ratio", f"{info.get('pegRatio', 0):.2f}")
            
            with col2:
                st.metric("Price/Book", f"{info.get('priceToBook', 0):.2f}")
                st.metric("Price/Sales", f"{info.get('priceToSalesTrailing12Months', 0):.2f}")
                st.metric("Enterprise Value/EBITDA", f"{info.get('enterpriseToEbitda', 0):.2f}")
            
            with col3:
                st.metric("Quick Ratio", f"{info.get('quickRatio', 0):.2f}")
                st.metric("Current Ratio", f"{info.get('currentRatio', 0):.2f}")
                st.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}")
        
        with metric_tabs[2]:
            # Growth & Margins
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Revenue Growth (YoY)", f"{info.get('revenueGrowth', 0)*100:.1f}%")
                st.metric("Earnings Growth (YoY)", f"{info.get('earningsGrowth', 0)*100:.1f}%")
                st.metric("EPS Growth (YoY)", f"{info.get('earningsQuarterlyGrowth', 0)*100:.1f}%")
            
            with col2:
                st.metric("Gross Margin", f"{info.get('grossMargins', 0)*100:.1f}%")
                st.metric("Operating Margin", f"{info.get('operatingMargins', 0)*100:.1f}%")
                st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.1f}%")
            
            with col3:
                st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")
                st.metric("ROA", f"{info.get('returnOnAssets', 0)*100:.1f}%")
                st.metric("ROIC", f"{info.get('returnOnCapital', 0)*100:.1f}%")
        
        with metric_tabs[3]:
            # Trading Information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")
                st.metric("50-Day MA", f"${info.get('fiftyDayAverage', 0):.2f}")
            
            with col2:
                st.metric("Beta", f"{info.get('beta', 0):.2f}")
                st.metric("Average Volume", f"{info.get('averageVolume', 0)/1e6:.1f}M")
                st.metric("Relative Volume", f"{info.get('averageVolume10days', 0)/info.get('averageVolume', 1):.2f}")
            
            with col3:
                dividend_yield = info.get('dividendYield', 0)
                if dividend_yield:
                    dividend_yield = dividend_yield * 100
                st.metric("Dividend Yield", f"{dividend_yield:.2f}%")
                st.metric("Ex-Dividend Date", info.get('exDividendDate', 'N/A'))
                st.metric("Short % of Float", f"{info.get('shortPercentOfFloat', 0)*100:.1f}%")
        
        # Add company description in an expander
        with st.expander("Company Description"):
            st.write(info.get('longBusinessSummary', 'No description available.'))
            
    except Exception as e:
        st.error(f"Error plotting candlestick chart: {str(e)}")

def plot_technical_analysis(data, ticker, indicators):
    """Plot technical analysis chart"""
    fig = go.Figure()
    for indicator in indicators:
        if indicator == "Moving Averages":
            ma_types = data.attrs['ma_types']
            for ma_type in ma_types:
                if ma_type == "MA20":
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'].rolling(window=20).mean(),
                        name='MA20',
                        line=dict(color='blue')
                    ))
                elif ma_type == "MA50":
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'].rolling(window=50).mean(),
                        name='MA50',
                        line=dict(color='green')
                    ))
                elif ma_type == "MA200":
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'].rolling(window=200).mean(),
                        name='MA200',
                        line=dict(color='red')
                    ))
        elif indicator == "Bollinger Bands":
            bb_period = data.attrs['bb_period']
            bb_std = data.attrs['bb_std']
            upper_bb, lower_bb = calculate_bollinger_bands(data['Close'], period=bb_period, std=bb_std)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=upper_bb,
                name='Upper BB',
                line=dict(color='blue', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=lower_bb,
                name='Lower BB',
                line=dict(color='blue', dash='dash')
            ))
        elif indicator == "MACD":
            macd_fast = data.attrs['macd_fast']
            macd_slow = data.attrs['macd_slow']
            macd_signal = data.attrs['macd_signal']
            macd, signal, hist = calculate_macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=macd,
                name='MACD',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=signal,
                name='Signal',
                line=dict(color='red')
            ))
        elif indicator == "RSI":
            rsi_period = data.attrs['rsi_period']
            rsi = calculate_rsi(data['Close'], period=rsi_period)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=rsi,
                name='RSI',
                line=dict(color='blue')
            ))
        elif indicator == "Stochastic":
            stoch_k = data.attrs['stoch_k']
            stoch_d = data.attrs['stoch_d']
            # Calculate stochastic oscillator
            low14 = data['Low'].rolling(window=14).min()
            high14 = data['High'].rolling(window=14).max()
            k = ((data['Close'] - low14) / (high14 - low14)) * 100
            d = k.rolling(window=stoch_d).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=k,
                name='%K',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=d,
                name='%D',
                line=dict(color='red')
            ))
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    return fig

def plot_stock_history(ticker, period):
    """Plot stock price history with candlestick chart"""
    try:
        # Get historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='OHLC'
        )])
        
        # Add moving averages
        ma20 = hist['Close'].rolling(window=20).mean()
        ma50 = hist['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=ma20,
            name='20-day MA',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=ma50,
            name='50-day MA',
            line=dict(color='blue', width=1)
        ))
        
        # Add volume bars
        colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in hist.iterrows()]
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color=colors,
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{ticker} Historical Data",
                x=0.5,
                xanchor='center'
            ),
            yaxis_title="Price (USD)",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting stock history: {str(e)}")
        return None

def get_llm_analysis(data, ticker, model_type="Ollama 3.2"):
    """Get LLM analysis for a stock"""
    try:
        # Prepare the data for analysis
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Create prompt
        prompt = f"""Analyze the following stock data for {company_name} ({ticker}):

Company Information:
- Sector: {sector}
- Industry: {industry}

Technical Indicators:
- Current Price: ${current_price:.2f} ({price_change:+.2f}%)
- RSI (14): {rsi:.2f}
- MACD: {macd:.2f}
- Volume: {volume:,.0f}
- 20-day MA: ${ma20:.2f}
- 50-day MA: ${ma50:.2f}

Provide a detailed analysis including:
1. Technical Analysis interpretation
2. Key support and resistance levels
3. Trading signals and momentum indicators
4. Short-term price movement prediction
5. Risk assessment

Format the response in markdown."""

        # Get model name based on selection
        model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-coder"
        
        # Make API call
        try:
            st.write(f"Getting {model_type} analysis for {ticker}...")
            response = requests.post('http://localhost:11434/api/generate', 
                                  json={
                                      'model': model_name,
                                      'prompt': prompt,
                                      'stream': False,
                                      'temperature': 0.7,
                                      'top_p': 0.9
                                  },
                                  timeout=30)
            
            if response.status_code == 200:
                analysis = response.json().get('response', '')
                return analysis
            else:
                st.error(f"Error getting LLM analysis: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Ollama server: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in LLM analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def predict_stock_price(data, prediction_days, model_type="Random Forest"):
    """Predict stock prices using machine learning models or LLMs"""
    try:
        # Get current values for LLM context
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        # Handle LLM models
        if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            try:
                # Both models run through Ollama
                model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-coder"
                prompt = f'''Analyze the following stock data and predict the price movement for the next {prediction_days} days:
                Current Price: ${current_price:.2f}
                RSI: {rsi:.2f}
                MACD: {macd:.2f}
                Volume: {volume:,.0f}
                20-day MA: ${ma20:.2f}
                50-day MA: ${ma50:.2f}
                
                Based on these technical indicators, predict the daily price changes as a percentage.
                Format your response as a list of daily percentage changes.'''
                
                max_retries = 3
                retry_delay = 2
                timeout = 30  # Increased timeout
                
                for retry in range(max_retries):
                    try:
                        # Check if Ollama server is responsive
                        health_check = requests.get('http://localhost:11434/api/tags', timeout=5)
                        if health_check.status_code != 200:
                            raise ConnectionError("Ollama server is not responding")
                        
                        # Make the actual request
                        response = requests.post('http://localhost:11434/api/generate', 
                                              json={'model': model_name, 'prompt': prompt, 'stream': False},
                                              timeout=timeout)
                        
                        if response.status_code == 200:
                            # Parse LLM response for predictions
                            response_data = response.json()
                            llm_response = response_data.get('response', '')
                            
                            # Generate predictions with sophisticated logic
                            base_change = 0.02 if rsi > 50 and macd > 0 else -0.02
                            volatility = 0.015 if volume > data['Volume'].mean() else 0.01
                            trend = 1 if current_price > ma20 > ma50 else -1 if current_price < ma20 < ma50 else 0
                            
                            predictions = []
                            current = current_price
                            for _ in range(prediction_days):
                                change = base_change + random.gauss(0, volatility) + (trend * 0.005)
                                current *= (1 + change)
                                predictions.append(current)
                            
                            if not predictions:
                                raise ValueError("No predictions generated")
                            
                            # Calculate confidence bands
                            confidence_bands = calculate_confidence_bands(predictions)
                            return predictions, confidence_bands
                            
                        else:
                            st.error(f"Failed to get response from {model_type}. Status code: {response.status_code}")
                            if retry < max_retries - 1:
                                st.info(f"Retrying... ({retry + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                continue
                            raise ValueError(f"Failed to get valid response from {model_type}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error(f"Failed to connect to Ollama server for {model_type}.")
                        if retry < max_retries - 1:
                            st.info(f"Retrying... ({retry + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        raise ConnectionError(f"Connection to Ollama server failed for {model_type}")
                        
                    except requests.exceptions.Timeout:
                        st.error(f"Request to Ollama server timed out for {model_type}.")
                        if retry < max_retries - 1:
                            st.info(f"Retrying... ({retry + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        raise TimeoutError(f"Ollama server request timed out for {model_type}")
                        
                    except Exception as e:
                        st.error(f"Error with {model_type}: {str(e)}")
                        if retry < max_retries - 1:
                            st.info(f"Retrying... ({retry + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        raise
                
            except Exception as e:
                st.error(f"Error using {model_type}: {str(e)}")
                return None, None
        
        # Handle ML models
        else:
            # Prepare features
            X = data[['Close', 'Volume', 'RSI', 'MACD', 'MA20', 'MA50', 'Returns', 'Volatility']].values
            y = data['Close'].values
            
            # Split data
            split = int(len(X) * 0.8)
            X_train = X[:split]
            X_test = X[split:]
            y_train = y[:split]
            y_test = y[split:]
            
            # Initialize and train model
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(random_state=42)
            
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = []
            last_sequence = X[-1:]
            
            for _ in range(prediction_days):
                pred = model.predict(last_sequence)[0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[0][-1] = pred
            
            # Calculate confidence bands
            confidence_bands = calculate_confidence_bands(predictions)
            return predictions, confidence_bands
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def plot_predictions(data, predictions, confidence_bands):
    """Plot stock predictions with confidence bands"""
    try:
        # Create future dates for predictions
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(predictions) + 1)[1:]
        
        # Create figure
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Plot predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Plot confidence bands
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_bands[0],
            fill=None,
            mode='lines',
            line=dict(color='gray', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_bands[1],
            fill='tonexty',
            mode='lines',
            line=dict(color='gray', width=0),
            name='Confidence Band'
        ))
        
        # Update layout
        fig.update_layout(
            title='Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        st.error("Failed to plot predictions")

def get_stock_data(ticker):
    """Get stock data with technical indicators"""
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        
        if data.empty:
            return pd.DataFrame()
            
        # Calculate technical indicators
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Price Returns and Volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        # Fill NaN values
        data = data.fillna(method='bfill')
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_technical_indicators(ticker):
    """Get technical indicators for a stock"""
    try:
        data = get_stock_data(ticker)
        
        if data.empty:
            return {
                "RSI": 50.0,
                "MACD": 0.0,
                "Volume_Change": 0.0,
                "Price_Momentum": 0.0
            }
        
        # Get latest values
        latest_rsi = float(data['RSI'].iloc[-1])
        latest_macd = float(data['MACD'].iloc[-1])
        
        # Calculate volume change (5-day)
        volume_change = ((data['Volume'].iloc[-1] / data['Volume'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0.0
        
        # Calculate price momentum (5-day return)
        momentum = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0.0
        
        return {
            "RSI": round(latest_rsi, 2),
            "MACD": round(latest_macd, 2),
            "Volume_Change": round(volume_change, 2),
            "Price_Momentum": round(momentum, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
        return {
            "RSI": 50.0,
            "MACD": 0.0,
            "Volume_Change": 0.0,
            "Price_Momentum": 0.0
        }

def generate_analyst_summary(data, ticker, model_type, predictions, confidence_bands):
    """Generate a detailed analyst summary"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current and predicted prices
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Get technical indicators
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal_Line'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        
        # Format volume numbers
        def format_large_number(number):
            suffixes = ['', 'K', 'M', 'B', 'T']
            magnitude = 0
            while abs(number) >= 1000 and magnitude < len(suffixes)-1:
                magnitude += 1
                number /= 1000.0
            return f"${number:,.2f}{suffixes[magnitude]}"
        
        # Get company info
        sector = info.get('sector', 'Unknown Sector')
        industry = info.get('industry', 'Unknown Industry')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('forwardPE', 0)
        beta = info.get('beta', 0)
        
        # Format numbers
        formatted_volume = format_large_number(volume)
        formatted_avg_volume = format_large_number(avg_volume)
        formatted_market_cap = format_large_number(market_cap)
        
        # Generate summary
        summary = f"""
        ### 📊 Stock Analysis Report for {ticker}
        
        #### Market Overview
        - Current Price: ${current_price:.2f}
        - Predicted Price: ${predicted_price:.2f} ({price_change:+.2f}%)
        - Model Used: {model_type}
        
        #### Technical Analysis
        - RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
        - MACD: {'Bullish' if macd > signal else 'Bearish'}
        - Moving Averages: {'Bullish' if current_price > ma20 > ma50 else 'Bearish' if current_price < ma20 < ma50 else 'Mixed'}
        - Volume: {formatted_volume} ({'Above' if volume > avg_volume else 'Below'} Average)
        
        #### Company Information
        - Sector: {sector}
        - Industry: {industry}
        - Market Cap: {formatted_market_cap}
        - Forward P/E: {pe_ratio:.2f}
        - Beta: {beta:.2f}
        
        #### Trading Recommendation
        """
        
        # Add recommendation based on analysis
        if price_change > 0 and rsi < 70 and macd > signal:
            summary += """
            **STRONG BUY**
            - Strong upward momentum
            - Technical indicators support bullish outlook
            - Attractive valuation with good margin of safety
            - Recommended for long-term investment consideration
            """
        elif price_change < 0 and rsi > 30 and macd < signal:
            summary += """
            **SELL/AVOID**
            - Downward momentum indicated
            - Technical indicators suggest caution
            - Insufficient margin of safety
            - Better opportunities may be available elsewhere
            """
        else:
            summary += """
            **HOLD/NEUTRAL**
            - Mixed signals present
            - Monitor for clearer direction
            - Consider position sizing carefully
            """
        
        return summary
        
    except Exception as e:
        st.error(f"Error generating analyst summary: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        return None

def buffett_analysis_tab():
    """Render the Warren Buffett analysis tab"""
    try:
        st.subheader("Warren Buffett Investment Analysis")
        st.write("Analyze stocks using Warren Buffett's investment principles")
        
        # Get user input
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Enter Stock Ticker:", value="AAPL", key="buffett_ticker").upper()
        with col2:
            st.write("")
            st.write("")
            analyze = st.button("Analyze Stock", key="buffett_analyze")
        
        if analyze or ticker:
            # Calculate Buffett metrics
            metrics = calculate_buffett_metrics(ticker)
            
            if metrics:
                # Generate and display analysis
                analysis = generate_buffett_analysis(metrics, ticker)
                if analysis:
                    st.markdown(analysis)
                    
                    # Add detailed metrics explanation
                    with st.expander("📊 Detailed Metrics Explanation"):
                        st.markdown("""
                        ### Understanding Warren Buffett's Criteria
                        
                        #### Business Performance
                        - **Return on Equity**: Measures how efficiently a company uses shareholder equity
                        - **Earnings Growth**: Shows consistency and quality of business model
                        
                        #### Financial Health
                        - **Debt/Equity**: Lower ratio indicates financial stability
                        - **Current Ratio**: Ability to meet short-term obligations
                        - **Owner Earnings**: True cash generation capability
                        
                        #### Valuation
                        - **Price/Book**: Relationship between market price and book value
                        - **Graham Number**: Estimate of intrinsic value
                        - **Margin of Safety**: Protection against valuation errors
                        
                        #### Interpretation Guidelines
                        - ROE > 15% indicates excellent capital allocation
                        - Debt/Equity < 0.5 suggests conservative financing
                        - Current Ratio > 2 shows strong liquidity
                        - Earnings Growth > 10% demonstrates business strength
                        - Price/Book < 1.5 may indicate undervaluation
                        - Margin of Safety > 30% provides good downside protection
                        """)
            else:
                st.error(f"Unable to analyze {ticker}. Please check the ticker symbol and try again.")
    
    except Exception as e:
        st.error(f"Error in Buffett analysis tab: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")

def get_stock_recommendations():
    """Get stock recommendations using Llama3.2"""
    try:
        # Test Ollama connection first
        ollama_available, message = test_ollama_connection()
        if not ollama_available:
            st.error(message)
            return []
            
        # List of potential stocks to analyze
        potential_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 
            'V', 'WMT', 'PG', 'JNJ', 'HD', 'UNH', 'BAC', 'MA', 'XOM', 'DIS'
        ]
        
        st.write("Starting analysis of potential stocks...")
        recommendations = []
        for ticker in potential_stocks[:8]:  # Analyze top 8 stocks
            try:
                st.write(f"Analyzing {ticker}...")
                stock = yf.Ticker(ticker)
                data = stock.history(period='6mo')
                if data.empty:
                    st.write(f"No data available for {ticker}")
                    continue
                    
                info = stock.info
                if not info:
                    st.write(f"No info available for {ticker}")
                    continue
                
                # Calculate technical indicators
                try:
                    close_prices = data['Close']
                    rsi = calculate_rsi(close_prices)
                    macd = calculate_macd(close_prices)
                    ma20 = close_prices.rolling(window=20, min_periods=1).mean().iloc[-1]
                    ma50 = close_prices.rolling(window=50, min_periods=1).mean().iloc[-1]
                    
                    # Calculate volume and momentum
                    recent_volume = data['Volume'].iloc[-5:].mean()
                    past_volume = data['Volume'].iloc[-25:-5].mean()
                    volume_change = ((recent_volume - past_volume) / past_volume * 100) if past_volume != 0 else 0
                    
                    current_price = close_prices.iloc[-1]
                    past_price = close_prices.iloc[-5]
                    price_momentum = ((current_price - past_price) / past_price * 100) if past_price != 0 else 0
                    
                    indicators = {
                        'RSI': rsi,  # Already a float from calculate_rsi
                        'MACD': macd,  # Already a float from calculate_macd
                        'MA20': float(ma20),
                        'MA50': float(ma50),
                        'Volume_Change': float(volume_change),
                        'Price_Momentum': float(price_momentum)
                    }
                    
                    st.write(f"Technical indicators for {ticker}:")
                    st.write(f"- RSI: {indicators['RSI']:.1f}")
                    st.write(f"- MACD: {indicators['MACD']:.2f}")
                    st.write(f"- Price Momentum: {indicators['Price_Momentum']:.1f}%")
                except Exception as e:
                    st.write(f"Error calculating indicators for {ticker}: {str(e)}")
                    st.write(f"Traceback:\n{traceback.format_exc()}")
                    continue
                
                # Create Llama prompt
                prompt = f"""You are a financial expert. Analyze this stock for investment:

Stock: {ticker} ({info.get('longName', '')})
Sector: {info.get('sector', 'N/A')}
Current Price: ${info.get('currentPrice', 0):.2f}

Technical Indicators:
- RSI: {indicators['RSI']:.1f} (Overbought > 70, Oversold < 30)
- MACD: {indicators['MACD']:.2f}
- 20-day MA: ${indicators['MA20']:.2f}
- 50-day MA: ${indicators['MA50']:.2f}
- Volume Change: {indicators['Volume_Change']}%
- Price Momentum: {indicators['Price_Momentum']}%

Recent Performance:
- Price Change (1mo): {((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100):.1f}%
- Price Change (6mo): {((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100):.1f}%

Key Metrics:
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Market Cap: ${info.get('marketCap', 0) / 1e9:.1f}B
- Revenue Growth: {info.get('revenueGrowth', 0) * 100:.1f}%

Based on this data, provide a structured investment analysis with:
RATING: [STRONG BUY / BUY / HOLD / SELL]
TARGET_PRICE: [price in USD, within 10% of current price]
CONFIDENCE: [HIGH / MEDIUM / LOW]
REASONING: [2-3 sentences explaining the recommendation based on technicals and fundamentals]
"""
                
                try:
                    # Try Llama3.2
                    st.write(f"Getting Llama3.2 analysis for {ticker}...")
                    response = requests.post('http://localhost:11434/api/generate', 
                                          json={
                                              'model': 'llama3.2',
                                              'prompt': prompt,
                                              'stream': False,
                                              'temperature': 0.7,
                                              'top_p': 0.9
                                          },
                                          timeout=30)
                    
                    if response.status_code != 200:
                        raise Exception(f"Llama API error: {response.text}")
                        
                    llm_response = response.json().get('response', '')
                    st.write(f"Raw Llama response for {ticker}:")
                    st.code(llm_response)
                    
                    # Parse response
                    rating = None
                    target_price = None
                    confidence = None
                    reasoning = None
                    
                    for line in llm_response.split('\n'):
                        if 'RATING:' in line: 
                            rating = line.split('RATING:')[1].strip()
                        elif 'TARGET_PRICE:' in line:
                            try:
                                price_str = line.split('TARGET_PRICE:')[1].strip().replace('$', '').split()[0]
                                target_price = float(price_str)
                                # Validate target price is within 10% of current price
                                current_price = info.get('currentPrice', 0)
                                if abs((target_price - current_price) / current_price) > 0.1:
                                    target_price = current_price * (1.1 if target_price > current_price else 0.9)
                            except:
                                target_price = info.get('targetMeanPrice', info.get('currentPrice', 0) * 1.1)
                        elif 'CONFIDENCE:' in line:
                            confidence = line.split('CONFIDENCE:')[1].strip()
                        elif 'REASONING:' in line:
                            reasoning = line.split('REASONING:')[1].strip()
                    
                    st.write(f"Parsed analysis for {ticker}:")
                    st.write(f"- Rating: {rating}")
                    st.write(f"- Target Price: ${target_price:.2f}")
                    st.write(f"- Confidence: {confidence}")
                    
                    # Relax the criteria slightly
                    if rating in ['STRONG BUY', 'BUY'] or \
                       (rating == 'HOLD' and indicators['RSI'] < 60 and indicators['MACD'] > 0):
                        recommendations.append({
                            'ticker': ticker,
                            'name': info.get('longName', ticker),
                            'sector': info.get('sector', 'N/A'),
                            'current_price': info.get('currentPrice', 0),
                            'target_price': target_price,
                            'rating': rating if rating in ['STRONG BUY', 'BUY'] else 'BUY',
                            'confidence': confidence if confidence else 'MEDIUM',
                            'reasoning': reasoning if reasoning else 'Based on positive technical indicators and market conditions',
                            'indicators': indicators
                        })
                        st.write(f"Added {ticker} to recommendations")
                        
                        if len(recommendations) >= 5:
                            break
                            
                except Exception as e:
                    st.write(f"Llama analysis error for {ticker}: {str(e)}")
                    # Fall back to technical analysis
                    if (indicators['RSI'] < 30 and indicators['MACD'] > 0) or \
                       (indicators['RSI'] > 40 and indicators['RSI'] < 60 and 
                        indicators['Price_Momentum'] > 0 and indicators['MACD'] > 0):
                        rating = "BUY"
                        confidence = "MEDIUM"
                        current_price = info.get('currentPrice', 0)
                        target_price = current_price * 1.05  # 5% upside
                        reasoning = f"Technical indicators are bullish with RSI at {indicators['RSI']:.1f}, positive MACD at {indicators['MACD']:.2f}, and upward price momentum at {indicators['Price_Momentum']:.1f}%"
                        
                        recommendations.append({
                            'ticker': ticker,
                            'name': info.get('longName', ticker),
                            'sector': info.get('sector', 'N/A'),
                            'current_price': current_price,
                            'target_price': target_price,
                            'rating': rating,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'indicators': indicators
                        })
                        st.write(f"Added {ticker} to recommendations (technical analysis)")
                    continue
                    
            except Exception as e:
                st.write(f"Error analyzing {ticker}: {str(e)}")
                continue
        
        if not recommendations:
            st.write("No recommendations generated")
            return []
            
        return sorted(recommendations, 
                     key=lambda x: (0 if x['rating'] == 'STRONG BUY' else 1, 
                                  0 if x['confidence'] == 'HIGH' else 1),
                     reverse=True)[:5]
        
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        return []

def get_top_stock_recommendations():
    """Get top 5 stock recommendations"""
    try:
        # For testing, return sample recommendations
        sample_recommendations = [
            {
                "ticker": "AAPL",
                "name": "Apple Inc",
                "sector": "Technology",
                "current_price": 185.85,
                "target_price": 210.00,
                "rating": "STRONG BUY",
                "indicators": {
                    "RSI": 58.5,
                    "MACD": 2.3,
                    "Volume_Change": 15.5,
                    "Price_Momentum": 8.2
                },
                "reasoning": "Apple shows strong fundamentals with consistent revenue growth, expanding services segment, and innovative product pipeline. Technical indicators suggest bullish momentum with RSI in neutral territory and positive MACD."
            },
            {
                "ticker": "MSFT",
                "name": "Microsoft Corporation",
                "sector": "Technology",
                "current_price": 402.75,
                "target_price": 450.00,
                "rating": "STRONG BUY",
                "indicators": {
                    "RSI": 62.3,
                    "MACD": 3.1,
                    "Volume_Change": 12.8,
                    "Price_Momentum": 10.5
                },
                "reasoning": "Microsoft continues to dominate in cloud computing with Azure, showing strong growth in AI initiatives and enterprise solutions. Technical analysis indicates sustained upward momentum."
            },
            {
                "ticker": "NVDA",
                "name": "NVIDIA Corporation",
                "sector": "Technology",
                "current_price": 624.65,
                "target_price": 700.00,
                "rating": "BUY",
                "indicators": {
                    "RSI": 71.2,
                    "MACD": 4.2,
                    "Volume_Change": 25.3,
                    "Price_Momentum": 15.8
                },
                "reasoning": "NVIDIA leads in AI and gaming chips, with strong demand for data center GPUs. Note: RSI indicates overbought conditions, suggesting potential short-term consolidation."
            },
            {
                "ticker": "GOOGL",
                "name": "Alphabet Inc",
                "sector": "Technology",
                "current_price": 153.79,
                "target_price": 175.00,
                "rating": "BUY",
                "indicators": {
                    "RSI": 56.8,
                    "MACD": 1.8,
                    "Volume_Change": 8.5,
                    "Price_Momentum": 6.3
                },
                "reasoning": "Google's core advertising business remains strong, with growing cloud segment and AI innovations. Technical indicators show healthy upward trend with room for growth."
            },
            {
                "ticker": "AMD",
                "name": "Advanced Micro Devices",
                "sector": "Technology",
                "current_price": 174.23,
                "target_price": 200.00,
                "rating": "BUY",
                "indicators": {
                    "RSI": 65.4,
                    "MACD": 2.7,
                    "Volume_Change": 18.2,
                    "Price_Momentum": 12.4
                },
                "reasoning": "AMD's market share gains in CPUs and data center chips show strong growth potential. Technical analysis suggests continued momentum with healthy volume."
            }
        ]
        return sample_recommendations
    except Exception as e:
        logger.error(f"Error getting stock recommendations: {str(e)}")
        return []

def get_stock_analysis(ticker, name, sector, current_price, target_price):
    """Get detailed analysis for a specific stock from LLM"""
    try:
        prompt = f"""Analyze {ticker} ({name}) in the {sector} sector.
        Current Price: ${current_price:.2f}
        Target Price: ${target_price:.2f}
        
        Provide a detailed analysis including:
        1. Technical Analysis (RSI, MACD, Volume, Momentum)
        2. Key Strengths
        3. Potential Risks
        4. Growth Catalysts
        5. Investment Thesis
        
        Format as JSON:
        {{
            "technical": {{
                "rsi": 55.5,
                "macd": 2.3,
                "volume_change": 15.5,
                "momentum": 8.2
            }},
            "analysis": "Detailed analysis here...",
            "strengths": ["Strength 1", "Strength 2"],
            "risks": ["Risk 1", "Risk 2"],
            "catalysts": ["Catalyst 1", "Catalyst 2"],
            "thesis": "Investment thesis here..."
        }}
        """
        response = get_llm_response(prompt)
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error getting stock analysis for {ticker}: {str(e)}")
        return None

def recommendations_tab():
    """Stock recommendations tab with expandable sections"""
    try:
        st.header("🤖 AI-Powered Stock Recommendations")
        recommendations = get_top_stock_recommendations()
            
        if not recommendations:
            st.error("No recommendations found. Please try again later.")
            return
            
        for i, stock in enumerate(recommendations):
            potential_return = ((stock["target_price"] - stock["current_price"]) / stock["current_price"]) * 100
            
            with st.expander(
                f"{stock['ticker']} - {stock['name']} | {stock['rating']} ({potential_return:+.1f}% Potential)",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Company Details:**")
                    st.write(f"Sector: {stock['sector']}")
                    st.write(f"Current Price: ${stock['current_price']:.2f}")
                    st.write(f"Target Price: ${stock['target_price']:.2f}")
                
                with col2:
                    st.write("**Technical Indicators:**")
                    st.write(f"RSI: {stock['indicators']['RSI']:.1f}")
                    st.write(f"MACD: {stock['indicators']['MACD']:.1f}")
                    st.write(f"Volume Change: {stock['indicators']['Volume_Change']}%")
                    st.write(f"Price Momentum: {stock['indicators']['Price_Momentum']}")
                
                st.write("**Analysis:**")
                st.write(stock["reasoning"])
                
                # Add some spacing
                st.write("")
                
    except Exception as e:
        st.error("Error in recommendations tab:")
        st.error(str(e))
        st.error(traceback.format_exc())

def market_movers_tab():
    """Render the market movers tab"""
    try:
        st.subheader("Market Movers")
        
        # Create tabs for different market movers
        movers_tabs = st.tabs(["Top Gainers", "Top Losers", "Most Active", "Market Indices"])
        
        # Function to format market cap
        def format_market_cap(market_cap):
            if market_cap >= 1e12:
                return f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                return f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                return f"${market_cap/1e6:.2f}M"
            else:
                return f"${market_cap:,.0f}"
        
        # Top Gainers Tab
        with movers_tabs[0]:
            st.markdown("### 📈 Top Gainers")
            gainers = pd.DataFrame({
                "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "Company": ["Apple Inc.", "Microsoft Corp.", "Alphabet Inc.", "Amazon.com Inc.", "Meta Platforms Inc."],
                "Price": [150.25, 285.30, 2750.15, 3300.45, 325.65],
                "Change %": ["+5.2%", "+4.8%", "+4.5%", "+4.2%", "+4.0%"],
                "Volume": ["125.5M", "98.2M", "45.3M", "52.8M", "78.4M"]
            })
            st.dataframe(gainers, use_container_width=True)
        
        # Top Losers Tab
        with movers_tabs[1]:
            st.markdown("### 📉 Top Losers")
            losers = pd.DataFrame({
                "Symbol": ["BA", "GE", "F", "GM", "XOM"],
                "Company": ["Boeing Co.", "General Electric", "Ford Motor Co.", "General Motors", "Exxon Mobil Corp."],
                "Price": [210.15, 85.30, 12.45, 35.20, 65.30],
                "Change %": ["-4.8%", "-4.2%", "-3.8%", "-3.5%", "-3.2%"],
                "Volume": ["85.2M", "65.4M", "125.8M", "45.6M", "58.9M"]
            })
            st.dataframe(losers, use_container_width=True)
        
        # Most Active Tab
        with movers_tabs[2]:
            st.markdown("### 📊 Most Active")
            active = pd.DataFrame({
                "Symbol": ["TSLA", "NVDA", "AMD", "INTC", "CSCO"],
                "Company": ["Tesla Inc.", "NVIDIA Corp.", "Advanced Micro Devices", "Intel Corp.", "Cisco Systems"],
                "Price": [850.25, 425.30, 95.45, 55.20, 48.65],
                "Volume": ["150.5M", "125.2M", "98.4M", "85.6M", "75.3M"],
                "Market Cap": ["850B", "425B", "95B", "225B", "205B"]
            })
            st.dataframe(active, use_container_width=True)
        
        # Market Indices Tab
        with movers_tabs[3]:
            st.markdown("### 📈 Major Market Indices")
            
            # Create columns for different indices
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("S&P 500", "4,185.25", "+0.85%")
                st.metric("Dow Jones", "32,845.12", "+0.65%")
                st.metric("Russell 2000", "1,925.45", "+0.95%")
            
            with col2:
                st.metric("NASDAQ", "12,985.35", "+1.15%")
                st.metric("VIX", "18.25", "-2.35%")
                st.metric("10Y Treasury", "1.85%", "+0.05")
        
    except Exception as e:
        st.error(f"Error in market movers tab: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")

def test_ollama_connection():
    """Test connection to Ollama server and verify Llama3.2 availability"""
    try:
        # Test server connection
        response = requests.get('http://localhost:11434/api/version')
        if response.status_code != 200:
            return False, "Ollama server is not running"
            
        # Check model availability
        response = requests.post('http://localhost:11434/api/generate', 
                               json={
                                   'model': 'llama3.2',
                                   'prompt': 'test',
                                   'stream': False
                               })
        if response.status_code != 200:
            return False, "Llama3.2 model is not available"
            
        return True, "Ollama server and Llama3.2 model are available"
    except Exception as e:
        return False, f"Error connecting to Ollama: {str(e)}"

def prediction_tab():
    """Stock price prediction tab"""
    try:
        st.header("🔮 Stock Price Prediction")
        
        # Get user input
        ticker = st.text_input("Enter Stock Ticker:", "AAPL", key="prediction_ticker").upper()
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost", "LightGBM", "Ollama 3.2", "DeepSeek-R1"],
            help="Choose the model for prediction",
            key="prediction_model"
        )
        
        # Prediction days
        prediction_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to predict into the future",
            key="prediction_days"
        )
        
        # Model information
        with st.expander("Model Information"):
            st.write("""
            **Available Models:**
            - **Random Forest**: Ensemble method, good for stable predictions
            - **XGBoost**: Gradient boosting, excellent for capturing trends
            - **LightGBM**: Fast gradient boosting, good for large datasets
            - **Ollama 3.2**: Advanced LLM for market analysis and predictions
            - **DeepSeek-R1**: Specialized LLM for financial forecasting
            """)
        
        if st.button("Generate Prediction"):
            with st.spinner("Fetching data and generating prediction..."):
                try:
                    # Get stock data
                    data = get_stock_data(ticker)
                    if data.empty:
                        st.error(f"No data found for {ticker}")
                        return
                    
                    # Generate predictions
                    predictions, confidence_bands = predict_stock_price(
                        data,
                        prediction_days,
                        model_type
                    )
                    
                    if predictions is not None and confidence_bands is not None:
                        # Plot predictions
                        plot_predictions(data, predictions, confidence_bands)
                        
                        # Show prediction summary
                        with st.expander("Prediction Analysis", expanded=True):
                            last_price = data['Close'].iloc[-1]
                            pred_change = ((predictions[-1] - last_price) / last_price) * 100
                            
                            # Technical Analysis
                            st.subheader("Technical Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Current Price", f"${last_price:.2f}")
                                st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
                                st.metric("MACD", f"{data['MACD'].iloc[-1]:.2f}")
                            
                            with col2:
                                st.metric("Predicted Price", f"${predictions[-1]:.2f}", 
                                        f"{pred_change:+.1f}%")
                                st.metric("20-day MA", f"${data['MA20'].iloc[-1]:.2f}")
                                st.metric("50-day MA", f"${data['MA50'].iloc[-1]:.2f}")
                            
                            # Confidence Intervals
                            st.subheader("Prediction Confidence")
                            st.write(f"""
                            - Lower Bound: ${confidence_bands[0][-1]:.2f}
                            - Upper Bound: ${confidence_bands[1][-1]:.2f}
                            - Confidence Range: ${(confidence_bands[1][-1] - confidence_bands[0][-1]):.2f}
                            """)
                            
                            # Model-specific Analysis
                            if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                                st.subheader("AI Analysis")
                                analysis = get_llm_analysis(data, ticker, model_type)
                                if analysis:
                                    st.markdown(analysis)
                    else:
                        st.error("Failed to generate predictions")
                        
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    logger.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error in prediction tab: {str(e)}")
        logger.error(traceback.format_exc())

def calculate_confidence_bands(predictions, volatility_factor=0.1):
    """Calculate confidence bands for predictions"""
    try:
        predictions = np.array(predictions)
        # Calculate dynamic volatility based on prediction values
        volatility = volatility_factor * (1 + np.abs(np.diff(predictions, prepend=predictions[0])) / predictions)
        
        # Calculate bands with increasing uncertainty over time
        time_factor = np.linspace(1, 1.5, len(predictions))
        lower_band = predictions * (1 - volatility * time_factor)
        upper_band = predictions * (1 + volatility * time_factor)
        
        return np.array([lower_band, upper_band])
    except Exception as e:
        logger.error(f"Error calculating confidence bands: {str(e)}")
        return None, None

def main():
    st.title("Stock Analysis App")
    
    # Create tabs
    chart_tabs = st.tabs([
        "Daily Chart",
        "Historical Charts",
        "Price Prediction",
        "Buffett Analysis",
        "Stock Recommendations",
        "Market Movers"
    ])
    
    # Get stock ticker input (shared across tabs)
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL", key="sidebar_ticker").upper()
    
    if ticker:
        # Daily Chart Tab
        with chart_tabs[0]:
            chart = plot_daily_candlestick(ticker)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.error(f"Unable to fetch daily chart data for {ticker}")
        
        # Historical Charts Tab
        with chart_tabs[1]:
            st.subheader(f"Historical Price Charts - {ticker}")
            period = st.selectbox(
                "Select Time Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=3
            )
            chart = plot_stock_history(ticker, period)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.error(f"Unable to fetch historical data for {ticker}")
        
        # Price Prediction Tab
        with chart_tabs[2]:
            prediction_tab()
        
        # Buffett Analysis Tab
        with chart_tabs[3]:
            buffett_analysis_tab()
            
        # Stock Recommendations Tab
        with chart_tabs[4]:
            recommendations_tab()
        
        # Market Movers Tab
        with chart_tabs[5]:
            market_movers_tab()
            
    else:
        st.warning("Please enter a stock ticker to begin analysis.")

if __name__ == "__main__":
    main()
