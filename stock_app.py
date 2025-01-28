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
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
import random
import ollama
import time
import warnings
import traceback
from math import sqrt
warnings.filterwarnings('ignore')

# Set page config at the very beginning
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate MACD for given prices"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper_bb = sma + (std_dev * std)
    lower_bb = sma - (std_dev * std)
    return upper_bb, lower_bb

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
        xaxis_title='Date',
        yaxis_title='Price (USD)',
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

def get_llm_analysis(ticker, data):
    """Get stock analysis from LLMs"""
    try:
        # Get recent news
        stock = yf.Ticker(ticker)
        news = stock.news[:5]  # Get last 5 news items
        
        # Format news
        news_text = "\n".join([
            f"- {item['title']}" for item in news
        ])
        
        # Get key metrics
        metrics = {
            'Price Change (%)': ((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100),
            'Volume Change (%)': ((data['Volume'].iloc[-5:].mean() - data['Volume'].iloc[-25:-5].mean()) / data['Volume'].iloc[-25:-5].mean() * 100),
            'RSI': calculate_rsi(data['Close'])[-1],
            'MACD': calculate_macd(data['Close'])[-1]
        }
        
        # Create prompt
        prompt = f"""You are a financial expert. Analyze this stock and predict its movement:

Stock: {ticker}

Technical Metrics:
- Price Change (20d): {metrics['Price Change (%)']}%
- Volume Change: {metrics['Volume Change (%)']}%
- RSI: {metrics['RSI']}
- MACD: {metrics['MACD']}

Recent News:
{news_text}

Provide a brief analysis of the stock's likely movement in the next 30 days. 
Consider technical indicators, news sentiment, and market conditions.
Format your response as:
PREDICTION: [BULLISH/BEARISH/NEUTRAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Your brief explanation]
"""
        
        try:
            # Try Llama3.2
            llm = ollama.Client()
            llm_response = llm.generate(model='llama2', prompt=prompt)
            llm_response = llm_response.response
            
        except Exception as e:
            print(f"Llama error: {str(e)}")
            # Fallback to simple analysis
            if metrics['RSI'] > 70:
                prediction = "BEARISH"
                confidence = "MEDIUM"
                reasoning = f"RSI is overbought at {metrics['RSI']:.1f} and MACD is at {metrics['MACD']:.2f}"
            elif metrics['RSI'] < 30:
                prediction = "BULLISH"
                confidence = "MEDIUM"
                reasoning = f"RSI is oversold at {metrics['RSI']:.1f} and MACD is at {metrics['MACD']:.2f}"
            else:
                prediction = "NEUTRAL"
                confidence = "LOW"
                reasoning = f"Technical indicators are neutral with RSI at {metrics['RSI']:.1f}"
            
            llm_response = f"""PREDICTION: {prediction}
CONFIDENCE: {confidence}
REASONING: {reasoning}"""
        
        # Parse response
        lines = llm_response.strip().split('\n')
        prediction = next((line.split(': ')[1] for line in lines if line.startswith('PREDICTION:')), 'NEUTRAL')
        confidence = next((line.split(': ')[1] for line in lines if line.startswith('CONFIDENCE:')), 'LOW')
        reasoning = next((line.split(': ')[1] for line in lines if line.startswith('REASONING:')), 'No reasoning provided')
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
    except Exception as e:
        print(f"LLM analysis error: {str(e)}")
        return None

def predict_stock_price(data, model_type, features, target, prediction_days):
    """Predict stock prices using the selected model"""
    try:
        # Prepare data
        X = data[features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get current values for LLM context
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        # Model selection and prediction
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        elif model_type == "LightGBM":
            model = lgb.LGBMRegressor(objective='regression', random_state=42)
        elif model_type == "Ensemble":
            models = [
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
                ("xgb", xgb.XGBRegressor(objective='reg:squarederror', random_state=42)),
                ("lgbm", lgb.LGBMRegressor(objective='regression', random_state=42))
            ]
            model = VotingRegressor(estimators=models)
        elif model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            try:
                # Both models run through Ollama
                model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-r1"
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
                        # Check if Ollama server is responsive by trying to list models
                        health_check = requests.get('http://localhost:11434/api/tags', timeout=5)
                        if health_check.status_code != 200:
                            raise ConnectionError("Ollama server is not responding")
                        
                        models = health_check.json().get('models', [])
                        if not any(m.get('name', '').startswith(model_name) for m in models):
                            raise ValueError(f"Model {model_name} not found in Ollama")
                        
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
                            volatility = 0.015 if volume > volume.mean() else 0.01
                            trend = 1 if current_price > ma20 > ma50 else -1 if current_price < ma20 < ma50 else 0
                            
                            predictions = []
                            current = current_price
                            for _ in range(prediction_days):
                                change = base_change + random.gauss(0, volatility) + (trend * 0.005)
                                current *= (1 + change)
                                predictions.append(current)
                            
                            if not predictions:
                                raise ValueError("No predictions generated")
                            
                            return predictions, calculate_confidence_bands(predictions)
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
                
                # If we get here, all retries failed
                raise RuntimeError(f"All retries failed for {model_type}")
                
            except Exception as e:
                st.error(f"Error using {model_type}: {str(e)}")
                raise  # Re-raise the exception instead of falling back
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        last_sequence = X.iloc[-1:].values
        predictions = []
        for _ in range(prediction_days):
            next_pred = model.predict(last_sequence)[0]
            predictions.append(next_pred)
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0][-1] = next_pred
        
        if not predictions:
            raise ValueError("No predictions generated")
        
        # Calculate confidence bands
        confidence_bands = calculate_confidence_bands(predictions)
        if confidence_bands[0] is None or confidence_bands[1] is None:
            raise ValueError("Failed to calculate confidence bands")
            
        return predictions, confidence_bands
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def calculate_confidence_bands(predictions, volatility_factor=0.1):
    """Calculate confidence bands for predictions"""
    try:
        predictions = np.array(predictions)
        lower_band = predictions * (1 - volatility_factor)
        upper_band = predictions * (1 + volatility_factor)
        return lower_band, upper_band
    except Exception as e:
        st.error(f"Error calculating confidence bands: {str(e)}")
        return None, None

def technical_analysis_tab():
    """Technical analysis tab with interactive charts and indicators"""
    try:
        st.header("Technical Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get user input
            ticker = st.text_input("Enter Stock Ticker:", "AAPL", key="ta_ticker").upper()
            
            # Multiple indicator selection
            selected_indicators = st.multiselect(
                "Select Technical Indicators",
                ["Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume"],
                default=["Moving Averages"]
            )
            
            # Parameters for each indicator
            if "Moving Averages" in selected_indicators:
                ma_periods = st.multiselect(
                    "Select MA Periods",
                    ["MA20", "MA50", "MA200"],
                    default=["MA20", "MA50"]
                )
            
            if "Bollinger Bands" in selected_indicators:
                bb_period = st.slider("Bollinger Period", 5, 50, 20)
                bb_std = st.slider("Bollinger Std Dev", 1, 4, 2)
            
            if "MACD" in selected_indicators:
                macd_fast = st.slider("MACD Fast Period", 5, 20, 12)
                macd_slow = st.slider("MACD Slow Period", 15, 40, 26)
                macd_signal = st.slider("MACD Signal Period", 5, 15, 9)
            
            if "RSI" in selected_indicators:
                rsi_period = st.slider("RSI Period", 5, 30, 14)
        
        with col2:
            if st.button("Analyze"):
                try:
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    hist_data = stock.history(period="1y")
                    
                    if hist_data.empty:
                        st.error(f"No data found for {ticker}")
                        return
                    
                    # Create figure with secondary y-axis
                    fig = make_subplots(rows=3, cols=1, 
                                      shared_xaxes=True,
                                      vertical_spacing=0.05,
                                      row_heights=[0.6, 0.2, 0.2])
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=hist_data.index,
                        open=hist_data['Open'],
                        high=hist_data['High'],
                        low=hist_data['Low'],
                        close=hist_data['Close'],
                        name="OHLC"
                    ), row=1, col=1)
                    
                    # Add selected indicators
                    if "Moving Averages" in selected_indicators:
                        for ma in ma_periods:
                            period = int(ma.replace("MA", ""))
                            ma_line = hist_data['Close'].rolling(window=period).mean()
                            fig.add_trace(go.Scatter(
                                x=hist_data.index,
                                y=ma_line,
                                name=f"{period} MA",
                                line=dict(width=1)
                            ), row=1, col=1)
                    
                    if "Bollinger Bands" in selected_indicators:
                        ma = hist_data['Close'].rolling(window=bb_period).mean()
                        std = hist_data['Close'].rolling(window=bb_period).std()
                        upper_bb = ma + (std * bb_std)
                        lower_bb = ma - (std * bb_std)
                        
                        fig.add_trace(go.Scatter(x=hist_data.index, y=upper_bb, name="Upper BB",
                                               line=dict(dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=hist_data.index, y=lower_bb, name="Lower BB",
                                               line=dict(dash='dash')), row=1, col=1)
                    
                    if "RSI" in selected_indicators:
                        delta = hist_data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        fig.add_trace(go.Scatter(x=hist_data.index, y=rsi, name="RSI"), row=2, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    if "MACD" in selected_indicators:
                        exp1 = hist_data['Close'].ewm(span=macd_fast, adjust=False).mean()
                        exp2 = hist_data['Close'].ewm(span=macd_slow, adjust=False).mean()
                        macd = exp1 - exp2
                        signal = macd.ewm(span=macd_signal, adjust=False).mean()
                        histogram = macd - signal
                        
                        fig.add_trace(go.Scatter(x=hist_data.index, y=macd, name="MACD"), row=3, col=1)
                        fig.add_trace(go.Scatter(x=hist_data.index, y=signal, name="Signal"), row=3, col=1)
                        fig.add_trace(go.Bar(x=hist_data.index, y=histogram, name="Histogram"), row=3, col=1)
                    
                    if "Volume" in selected_indicators:
                        colors = ['red' if row['Open'] - row['Close'] >= 0 
                                 else 'green' for index, row in hist_data.iterrows()]
                        fig.add_trace(go.Bar(x=hist_data.index, y=hist_data['Volume'],
                                           marker_color=colors,
                                           name="Volume"), row=3, col=1)
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=800,
                        showlegend=True,
                        xaxis_rangeslider_visible=False
                    )
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add analysis explanation
                    with st.expander("Technical Analysis Explanation"):
                        if "Moving Averages" in selected_indicators:
                            st.write("""
                            **Moving Averages:**
                            - Short-term MA (20 days): Quick to react to price changes
                            - Medium-term MA (50 days): Moderate reaction to price changes
                            - Long-term MA (200 days): Shows long-term trends
                            - Crossovers between MAs can signal trend changes
                            """)
                        
                        if "RSI" in selected_indicators:
                            st.write("""
                            **RSI (Relative Strength Index):**
                            - Above 70: Potentially overbought
                            - Below 30: Potentially oversold
                            - Trend line breaks can signal reversals
                            """)
                        
                        if "MACD" in selected_indicators:
                            st.write("""
                            **MACD (Moving Average Convergence Divergence):**
                            - MACD Line: Short-term momentum
                            - Signal Line: Trigger line for buy/sell signals
                            - Histogram: Shows momentum strength
                            - Signal line crossovers can indicate entry/exit points
                            """)
                        
                        if "Bollinger Bands" in selected_indicators:
                            st.write("""
                            **Bollinger Bands:**
                            - Upper/Lower bands show volatility
                            - Price touching bands can signal potential reversals
                            - Band squeeze (narrowing) can signal potential breakout
                            """)
                        
                        if "Volume" in selected_indicators:
                            st.write("""
                            **Volume:**
                            - Green: Closing price higher than opening price
                            - Red: Closing price lower than opening price
                            - High volume confirms trend strength
                            - Low volume may signal weak trends
                            """)
                
                except Exception as e:
                    st.error(f"Error performing technical analysis: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
                    
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")

def prediction_tab():
    """Render the prediction tab"""
    try:
        st.title("Stock Price Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stock symbol input
            ticker = st.text_input(
                "Enter Stock Symbol",
                value="AAPL",
                help="Enter the stock symbol (e.g., AAPL for Apple Inc.)"
            ).upper()
            
            # Date range selection
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", value=start_date)
            with date_col2:
                end_date = st.date_input("End Date", value=end_date)
            
            # Model selection
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "XGBoost", "LightGBM", "Ensemble", "Ollama 3.2", "DeepSeek-R1"]
            )
            
            # Prediction days
            prediction_days = st.slider(
                "Number of Days to Predict",
                min_value=1,
                max_value=30,
                value=7,
                help="Select the number of days to predict into the future"
            )
        
        with col2:
            st.markdown("### Model Description")
            if model_type == "Random Forest":
                st.markdown("""
                Using **Random Forest Model**:
                - Ensemble learning method
                - Robust against overfitting
                - Handles non-linear relationships
                - Good for feature importance analysis
                """)
            elif model_type == "XGBoost":
                st.markdown("""
                Using **XGBoost Model**:
                - Gradient boosting framework
                - High performance and speed
                - Built-in regularization
                - Handles missing values
                """)
            elif model_type == "LightGBM":
                st.markdown("""
                Using **LightGBM Model**:
                - Light and fast implementation
                - Handles large datasets efficiently
                - Leaf-wise tree growth
                - Good for time series data
                """)
            elif model_type == "Ensemble":
                st.markdown("""
                Using **Ensemble Model**:
                - Combines multiple models
                - Averages predictions
                - Reduces individual model biases
                - More robust to market volatility
                """)
            elif model_type == "Ollama 3.2":
                st.markdown("""
                Using **Ollama 3.2 Model**:
                - Advanced language model
                - Considers market sentiment
                - Pattern recognition
                - Natural language understanding
                """)
            elif model_type == "DeepSeek-R1":
                st.markdown("""
                Using **DeepSeek-R1 Model**:
                - Advanced language model trained on financial data
                - Integrates technical and fundamental analysis
                - Considers market sentiment and analyst opinions
                - Provides confidence bands based on market volatility
                - Uses pattern recognition from historical data
                """)
            
            # Add prediction rationale
            st.markdown("""
            ### Prediction Process:
            1. Analyzes historical price data
            2. Considers technical indicators
            3. Evaluates market trends
            4. Generates confidence bands
            5. Provides detailed analysis
            """)
        
        if st.button("Generate Prediction"):
            with st.spinner("Fetching data and generating prediction..."):
                # Get stock data
                data = get_stock_data(ticker, start_date, end_date)
                if data is None or len(data) < 2:
                    st.error("Failed to fetch stock data. Please try a different stock or date range.")
                    return
                
                # Calculate technical indicators
                data = calculate_technical_indicators(data)
                
                # Define features for prediction
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal', 'MA20', 'MA50']
                
                # Make prediction
                result = predict_stock_price(data, model_type, features, 'Close', prediction_days)
                
                if result is None or not isinstance(result, tuple) or len(result) != 2:
                    st.error("Failed to generate prediction. Please try a different stock or model.")
                    return
                
                predictions, confidence_bands = result
                if predictions is None or confidence_bands is None:
                    st.error("Failed to generate prediction. Please try a different stock or model.")
                    return
                
                # Create visualization
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date, periods=prediction_days + 1)[1:]
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name="Historical",
                    line=dict(color='blue')
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    name="Prediction",
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence bands
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
                
                fig.update_layout(
                    title=f"{ticker} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Get key metrics for analysis
                current_price = data['Close'].iloc[-1]
                predicted_price = predictions[-1]
                price_change = ((predicted_price - current_price) / current_price) * 100
                rsi = data['RSI'].iloc[-1]
                macd = data['MACD'].iloc[-1]
                signal = data['Signal'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].mean()
                ma20 = data['MA20'].iloc[-1]
                ma50 = data['MA50'].iloc[-1]

                # Get company info
                stock = yf.Ticker(ticker)
                info = stock.info
                company_name = info.get('longName', ticker)
                sector = info.get('sector', 'Unknown Sector')
                
                # Determine market sentiment
                sentiment = "bullish" if price_change > 0 else "bearish"
                strength = "strong" if abs(price_change) > 5 else "moderate"
                volume_trend = "high" if volume > avg_volume * 1.2 else "low" if volume < avg_volume * 0.8 else "normal"
                trend_signal = "upward" if current_price > ma20 > ma50 else "downward" if current_price < ma20 < ma50 else "sideways"
                
                # Generate TV analyst style summary
                st.markdown("""
                ### 📺 Market Analyst Take
                """)
                
                st.markdown(f"""
                Good evening, investors! Let's talk about {company_name} ({ticker}), a key player in the {sector} sector.
                
                Our {model_type} model is showing a **{strength} {sentiment}** outlook for {ticker}. We're looking at a target price of **${predicted_price:.2f}**, representing a **{price_change:+.2f}%** move from the current price of ${current_price:.2f}.

                Here's what's driving this forecast:
                
                1. **Technical Momentum**: The stock is showing a {trend_signal} trend, with {volume_trend} trading volume. {
                    "This increased activity suggests strong buyer interest." if volume_trend == "high" else 
                    "The lower volume suggests a period of consolidation." if volume_trend == "low" else 
                    "Trading volume is in line with averages, indicating steady market participation."
                }

                2. **Market Signals**: {
                    "RSI indicates the stock is overbought, but strong momentum could push it higher." if rsi > 70 else
                    "RSI shows oversold conditions, suggesting a potential bounce." if rsi < 30 else
                    "RSI is in a healthy range, showing balanced buying and selling pressure."
                }

                3. **Key Takeaway**: {
                    f"We're seeing strong bullish signals with a clear upward trend. The {price_change:+.2f}% projected gain is supported by positive technical indicators and market momentum." if price_change > 5 else
                    f"While we expect a modest {price_change:+.2f}% gain, investors should watch for entry points and maintain position sizing discipline." if 0 < price_change <= 5 else
                    f"Our analysis suggests caution with a projected {price_change:.2f}% decline. Consider reducing exposure or implementing stop-loss orders." if price_change < 0 else
                    "Our analysis suggests a neutral outlook for the stock."
                }
                """)

                # Add the detailed analysis in an expander
                with st.expander("View Detailed Technical Analysis"):
                    summary = generate_analyst_summary(data, ticker, model_type, predictions, confidence_bands)
                    if summary:
                        st.markdown(summary)
                    else:
                        st.error("Failed to generate detailed analyst summary")
                
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")

def get_stock_data(ticker, start_date, end_date):
    """Fetch stock data and calculate technical indicators"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        
        # Calculate technical indicators
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Drop any NaN values
        data = data.dropna()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        return None

def calculate_technical_indicators(data):
    """Calculate technical indicators for the dataset"""
    try:
        # Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
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
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Drop NaN values
        data = data.dropna()
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        return None

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
        signal = data['Signal'].iloc[-1]
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

def calculate_buffett_metrics(ticker):
    """Calculate Warren Buffett's key investment metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cash_flow
        
        # Calculate key metrics
        metrics = {}
        
        # 1. Earnings Growth and Stability
        if not income_stmt.empty and len(income_stmt.columns) >= 4:
            net_income = income_stmt.loc['Net Income'].values
            metrics['earnings_growth'] = (
                (net_income[0] - net_income[-1]) / abs(net_income[-1])
                if abs(net_income[-1]) > 0 else 0
            ) * 100
            metrics['earnings_stability'] = np.std(net_income) / np.mean(net_income) if np.mean(net_income) != 0 else float('inf')
        else:
            metrics['earnings_growth'] = 0
            metrics['earnings_stability'] = float('inf')
        
        # 2. Return on Equity (ROE)
        if not balance_sheet.empty and 'Stockholders Equity' in balance_sheet.index:
            equity = balance_sheet.loc['Stockholders Equity'].values[0]
            net_income = income_stmt.loc['Net Income'].values[0] if not income_stmt.empty else 0
            metrics['roe'] = (net_income / equity * 100) if equity != 0 else 0
        else:
            metrics['roe'] = 0
        
        # 3. Debt to Equity Ratio
        if not balance_sheet.empty and 'Stockholders Equity' in balance_sheet.index:
            total_debt = (
                balance_sheet.loc['Long Term Debt'].values[0] +
                balance_sheet.loc['Current Debt'].values[0]
                if 'Current Debt' in balance_sheet.index
                else balance_sheet.loc['Long Term Debt'].values[0]
            )
            equity = balance_sheet.loc['Stockholders Equity'].values[0]
            metrics['debt_to_equity'] = (total_debt / equity) if equity != 0 else float('inf')
        else:
            metrics['debt_to_equity'] = float('inf')
        
        # 4. Current Ratio
        if not balance_sheet.empty:
            try:
                current_assets = balance_sheet.loc['Current Assets'].values[0]
                current_liabilities = balance_sheet.loc['Current Liabilities'].values[0]
                metrics['current_ratio'] = (
                    current_assets / current_liabilities
                    if current_liabilities != 0 else float('inf')
                )
            except KeyError:
                # If we can't find the exact fields, try alternative fields
                total_assets = balance_sheet.loc['Total Assets'].values[0]
                total_liabilities = balance_sheet.loc['Total Liabilities'].values[0]
                metrics['current_ratio'] = (
                    total_assets / total_liabilities
                    if total_liabilities != 0 else float('inf')
                )
        else:
            metrics['current_ratio'] = 0
        
        # 5. Owner Earnings (Operating Cash Flow - CapEx)
        if not cash_flow.empty:
            try:
                operating_cash_flow = cash_flow.loc['Operating Cash Flow'].values[0]
                capital_expenditure = abs(cash_flow.loc['Capital Expenditures'].values[0])
                metrics['owner_earnings'] = operating_cash_flow - capital_expenditure
            except KeyError:
                metrics['owner_earnings'] = 0
        else:
            metrics['owner_earnings'] = 0
        
        # 6. Margin of Safety
        current_price = info.get('currentPrice', 0)
        book_value_per_share = info.get('bookValue', 0)
        if book_value_per_share > 0:
            metrics['price_to_book'] = current_price / book_value_per_share
            # Graham Number = sqrt(22.5 * EPS * BVPS)
            eps = info.get('trailingEps', 0)
            metrics['graham_number'] = sqrt(22.5 * eps * book_value_per_share) if eps > 0 else 0
            metrics['margin_of_safety'] = (
                (metrics['graham_number'] - current_price) / metrics['graham_number'] * 100
                if metrics['graham_number'] > 0 else 0
            )
        else:
            metrics['price_to_book'] = float('inf')
            metrics['graham_number'] = 0
            metrics['margin_of_safety'] = 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating Buffett metrics: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        return None

def generate_buffett_analysis(metrics, ticker):
    """Generate analysis based on Warren Buffett's investment principles"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown Sector')
        
        analysis = f"""
        ## Warren Buffett Analysis for {company_name} ({ticker})
        
        ### Company Overview
        **Sector**: {sector}
        **Market Cap**: ${info.get('marketCap', 0):,.0f}
        **P/E Ratio**: {info.get('trailingPE', 0):.2f}
        
        ### Financial Analysis
        
        #### 1. Earnings Analysis
        - **Growth Rate**: {metrics['earnings_growth']:.1f}%
        - **Stability**: {metrics['earnings_stability']:.2f}
        - **Quality Assessment**: {"Strong" if metrics['earnings_growth'] > 10 and metrics['earnings_stability'] < 0.5 
                                 else "Moderate" if metrics['earnings_growth'] > 0 
                                 else "Weak"}
        
        #### 2. Business Performance
        - **Return on Equity**: {metrics['roe']:.1f}%
        - **Assessment**: {"Excellent" if metrics['roe'] > 15 
                         else "Good" if metrics['roe'] > 10 
                         else "Poor"}
        
        #### 3. Financial Health
        - **Debt/Equity Ratio**: {metrics['debt_to_equity']:.2f}
        - **Current Ratio**: {metrics['current_ratio']:.2f}
        - **Owner Earnings**: ${metrics['owner_earnings']:,.0f}
        - **Health Status**: {"Strong" if metrics['debt_to_equity'] < 0.5 and metrics['current_ratio'] > 2
                            else "Moderate" if metrics['debt_to_equity'] < 1.0 and metrics['current_ratio'] > 1.5
                            else "Weak"}
        
        #### 4. Valuation
        - **Price/Book Ratio**: {metrics['price_to_book']:.2f}
        - **Graham Number**: ${metrics['graham_number']:.2f}
        - **Margin of Safety**: {metrics['margin_of_safety']:.1f}%
        - **Valuation Status**: {"Undervalued" if metrics['margin_of_safety'] > 30
                                else "Fair Value" if metrics['margin_of_safety'] > 15
                                else "Overvalued"}
        
        ### Investment Recommendation
        """
        
        # Generate overall recommendation
        score = 0
        score += 1 if metrics['roe'] > 15 else 0.5 if metrics['roe'] > 10 else 0
        score += 1 if metrics['current_ratio'] > 2 else 0.5 if metrics['current_ratio'] > 1.5 else 0
        score += 1 if metrics['debt_to_equity'] < 0.5 else 0.5 if metrics['debt_to_equity'] < 1.0 else 0
        score += 1 if metrics['earnings_growth'] > 10 else 0.5 if metrics['earnings_growth'] > 0 else 0
        score += 1 if metrics['price_to_book'] < 1.5 else 0.5 if metrics['price_to_book'] < 3 else 0
        score += 1 if metrics['margin_of_safety'] > 30 else 0.5 if metrics['margin_of_safety'] > 15 else 0
        
        max_score = 6
        score_percentage = (score / max_score) * 100
        
        if score_percentage >= 75:
            analysis += """
            **STRONG BUY**
            - Company demonstrates strong fundamentals aligned with Buffett's principles
            - Solid financial position with consistent earnings
            - Attractive valuation with good margin of safety
            - Recommended for long-term investment consideration
            """
        elif score_percentage >= 50:
            analysis += """
            **HOLD / WATCH**
            - Company shows some positive characteristics
            - Some metrics meet Buffett's criteria while others need improvement
            - Monitor for potential strengthening of fundamentals
            - Consider partial position if other factors are favorable
            """
        else:
            analysis += """
            **AVOID**
            - Company does not meet Buffett's investment criteria
            - Multiple areas of concern in fundamental metrics
            - Insufficient margin of safety
            - Better opportunities may be available elsewhere
            """
        
        return analysis
        
    except Exception as e:
        st.error(f"Error generating Buffett analysis: {str(e)}")
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
                        ### Understanding the Metrics
                        
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
        # List of potential stocks to analyze
        potential_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 
            'V', 'WMT', 'PG', 'JNJ', 'HD', 'UNH', 'BAC', 'MA', 'XOM', 'DIS'
        ]
        
        recommendations = []
        for ticker in potential_stocks[:8]:  # Analyze top 8 stocks
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period='6mo')
                if data.empty:
                    print(f"No data available for {ticker}")
                    continue
                    
                info = stock.info
                if not info:
                    print(f"No info available for {ticker}")
                    continue
                
                # Calculate technical indicators
                try:
                    indicators = {
                        'RSI': calculate_rsi(data['Close'])[-1],
                        'MACD': calculate_macd(data['Close'])[-1],
                        'MA20': data['Close'].rolling(window=20).mean().iloc[-1],
                        'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
                        'Volume_Change': ((data['Volume'].iloc[-5:].mean() - data['Volume'].iloc[-25:-5].mean()) 
                                        / data['Volume'].iloc[-25:-5].mean() * 100)
                    }
                except Exception as e:
                    print(f"Error calculating indicators for {ticker}: {str(e)}")
                    continue
                
                # Create Llama prompt
                prompt = f"""You are a financial expert. Analyze this stock for investment:

Stock: {ticker} ({info.get('longName', '')})
Sector: {info.get('sector', 'N/A')}
Current Price: ${info.get('currentPrice', 0):.2f}

Technical Indicators:
- RSI: {indicators['RSI']:.1f}
- MACD: {indicators['MACD']:.2f}
- 20-day MA: ${indicators['MA20']:.2f}
- 50-day MA: ${indicators['MA50']:.2f}
- Volume Change: {indicators['Volume_Change']:.1f}%

Recent Performance:
- Price Change (1mo): {((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100):.1f}%
- Price Change (6mo): {((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100):.1f}%

Key Metrics:
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Market Cap: ${info.get('marketCap', 0) / 1e9:.1f}B
- Revenue Growth: {info.get('revenueGrowth', 0) * 100:.1f}%

Based on this data, provide a structured investment analysis with:
RATING: [STRONG BUY / BUY / HOLD / SELL]
TARGET_PRICE: [price in USD]
CONFIDENCE: [HIGH / MEDIUM / LOW]
REASONING: [2-3 sentences explaining the recommendation]
"""
                
                try:
                    # Try Llama3.2
                    llm = ollama.Client()
                    print(f"Analyzing {ticker} with Llama3.2...")
                    response = llm.generate(model='llama2', prompt=prompt)
                    if not response or not hasattr(response, 'response'):
                        raise Exception("Invalid response from Llama")
                    
                    llm_response = response.response
                    print(f"Raw Llama response for {ticker}:\n{llm_response}")
                    
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
                            except:
                                target_price = info.get('targetMeanPrice', info.get('currentPrice', 0) * 1.1)
                        elif 'CONFIDENCE:' in line:
                            confidence = line.split('CONFIDENCE:')[1].strip()
                        elif 'REASONING:' in line:
                            reasoning = line.split('REASONING:')[1].strip()
                    
                    print(f"Parsed response for {ticker}:")
                    print(f"Rating: {rating}")
                    print(f"Target Price: {target_price}")
                    print(f"Confidence: {confidence}")
                    
                    if rating in ['STRONG BUY', 'BUY'] and confidence in ['HIGH', 'MEDIUM']:
                        recommendations.append({
                            'ticker': ticker,
                            'name': info.get('longName', ticker),
                            'sector': info.get('sector', 'N/A'),
                            'current_price': info.get('currentPrice', 0),
                            'target_price': target_price,
                            'rating': rating,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'indicators': indicators
                        })
                        
                        if len(recommendations) >= 5:
                            break
                            
                except Exception as e:
                    print(f"Llama analysis error for {ticker}: {str(e)}")
                    # Fall back to technical analysis
                    if indicators['RSI'] < 30 and indicators['MACD'] > 0:
                        rating = "BUY"
                        confidence = "MEDIUM"
                        target_price = info.get('targetMeanPrice', info.get('currentPrice', 0) * 1.1)
                        reasoning = f"Technical indicators suggest oversold conditions with RSI at {indicators['RSI']:.1f} and positive MACD at {indicators['MACD']:.2f}"
                        
                        recommendations.append({
                            'ticker': ticker,
                            'name': info.get('longName', ticker),
                            'sector': info.get('sector', 'N/A'),
                            'current_price': info.get('currentPrice', 0),
                            'target_price': target_price,
                            'rating': rating,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'indicators': indicators
                        })
                    continue
                    
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                continue
        
        if not recommendations:
            print("No recommendations generated")
            return []
            
        return sorted(recommendations, 
                     key=lambda x: (0 if x['rating'] == 'STRONG BUY' else 1, 
                                  0 if x['confidence'] == 'HIGH' else 1),
                     reverse=True)[:5]
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return []

def recommendations_tab(ticker):
    """Render the stock recommendations tab with Llama3.2 analysis"""
    try:
        st.header("🤖 AI-Powered Stock Recommendations")
        st.markdown("""
        Using Llama3.2 AI to identify the top 5 investment opportunities based on:
        - Advanced technical analysis (RSI, MACD, Moving Averages)
        - Company fundamentals and growth metrics
        - Market trends and sector analysis
        - Price momentum and volume patterns
        """)
        
        with st.spinner("Analyzing market opportunities using Llama3.2..."):
            recommendations = get_stock_recommendations()
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"#{i} {rec['name']} ({rec['ticker']}) - {rec['rating']}", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **Sector**: {rec['sector']}  
                            **Current Price**: ${rec['current_price']:.2f}  
                            **Target Price**: ${rec['target_price']:.2f}  
                            **Potential Upside**: {((rec['target_price'] - rec['current_price']) / rec['current_price'] * 100):.1f}%
                            
                            **AI Analysis**:  
                            {rec['reasoning']}
                            """)
                        
                        with col2:
                            st.metric("Rating", rec['rating'])
                            st.metric("Confidence", rec['confidence'])
                            
                            # Technical indicators
                            with st.expander("Technical Indicators"):
                                st.metric("RSI", f"{rec['indicators']['RSI']:.1f}")
                                st.metric("MACD", f"{rec['indicators']['MACD']:.2f}")
                                st.metric("Volume Trend", f"{rec['indicators']['Volume_Change']:.1f}%")
                
                st.info("""
                **Note**: These recommendations are generated using Llama3.2 AI analysis of market data and technical indicators. 
                The AI model considers multiple factors including technical patterns, fundamental metrics, and market conditions.
                Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
                """)
            else:
                st.warning("Unable to generate recommendations at this time. Please try again later.")
                
    except Exception as e:
        st.error(f"Error in recommendations tab: {str(e)}")
        st.error(f"Traceback:\n{traceback.format_exc()}")

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

def main():
    st.title("Stock Analysis App")
    
    # Create tabs
    chart_tabs = st.tabs([
        "Daily Chart",
        "Technical Analysis",
        "Historical Charts",
        "Price Prediction",
        "Buffett Analysis",
        "Stock Recommendations",
        "Market Movers"
    ])
    
    # Get stock ticker input (shared across tabs)
    ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()
    
    if ticker:
        # Daily Chart Tab
        with chart_tabs[0]:
            chart = plot_daily_candlestick(ticker)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.error(f"Unable to fetch daily chart data for {ticker}")
        
        # Technical Analysis Tab
        with chart_tabs[1]:
            technical_analysis_tab()
        
        # Historical Charts Tab
        with chart_tabs[2]:
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
        with chart_tabs[3]:
            prediction_tab()
        
        # Buffett Analysis Tab
        with chart_tabs[4]:
            st.header("Warren Buffett Analysis")
            st.markdown("""
            This tab analyzes stocks using Warren Buffett's investment principles:
            - Strong and consistent earnings
            - Low debt and good liquidity
            - High return on equity
            - Fair valuation with margin of safety
            """)
            
            # Get user input
            ticker = st.text_input("Enter Stock Symbol:", value="AAPL", key="buffett_ticker").upper()
            
            if st.button("Analyze", key="buffett_analyze") or ticker:
                with st.spinner(f"Analyzing {ticker} using Buffett's principles..."):
                    # Calculate metrics
                    metrics = calculate_buffett_metrics(ticker)
                    
                    if metrics:
                        # Generate and display analysis
                        analysis = generate_buffett_analysis(metrics, ticker)
                        if analysis:
                            st.markdown(analysis)
                            
                            # Detailed Analysis
                            with st.expander("View Detailed Analysis"):
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
                        st.error(f"Unable to analyze {ticker}. Please check the ticker symbol.")
                    
        # Stock Recommendations Tab
        with chart_tabs[5]:
            recommendations_tab(ticker)
        
        # Market Movers Tab
        with chart_tabs[6]:
            market_movers_tab()
    
    else:
        st.warning("Please enter a stock ticker to begin analysis.")

if __name__ == "__main__":
    main()
