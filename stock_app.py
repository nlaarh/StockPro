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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

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

def get_llm_response(prompt):
    """Get response from Llama3.2 model"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama3.2',
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

def test_ollama_connection(model_type="Ollama 3.2"):
    """Test connection to Ollama server and model availability"""
    try:
        # First test server connection
        try:
            health_check = requests.get('http://localhost:11434/api/tags', timeout=5)
            if health_check.status_code != 200:
                return False, "Ollama server is not responding. Please make sure it's running."
        except requests.exceptions.RequestException:
            return False, "Cannot connect to Ollama server. Please make sure it's installed and running."
            
        # Get model name based on selection
        model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-coder"
        
        # Test model availability
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': 'test',
                    'stream': False
                },
                timeout=10
            )
            
            if response.status_code != 200:
                return False, f"{model_type} model is not available. Please install it using 'ollama pull {model_name}'"
                
            return True, "Model is available and ready"
            
        except requests.exceptions.RequestException as e:
            return False, f"Error testing {model_type} model: {str(e)}"
            
    except Exception as e:
        return False, f"Unexpected error testing {model_type} connection: {str(e)}"

def predict_stock_price(data, prediction_days, model_type="Random Forest"):
    """Predict stock prices using various models"""
    try:
        # If LLM model is selected, test connection first
        if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            # Test model availability
            is_available, message = test_ollama_connection(model_type)
            if not is_available:
                st.error(message)
                st.warning("Falling back to Random Forest model...")
                model_type = "Random Forest"
            else:
                try:
                    # Get current metrics
                    current_price = data['Close'].iloc[-1]
                    rsi = data['RSI'].iloc[-1]
                    macd = data['MACD'].iloc[-1]
                    ma20 = data['MA20'].iloc[-1]
                    ma50 = data['MA50'].iloc[-1]
                    
                    # Create an improved prompt for better results
                    prompt = f"""You are a stock market expert. Analyze the following technical indicators and predict the stock price changes for the next {prediction_days} days:

Current Price: ${current_price:.2f}
Technical Analysis:
- RSI: {rsi:.2f} (>70 overbought, <30 oversold)
- MACD: {macd:.2f} (momentum indicator)
- 20-day MA: ${ma20:.2f}
- 50-day MA: ${ma50:.2f}

Based on these indicators, predict the daily price changes as a percentage for each of the next {prediction_days} days.
Format your response exactly like this example:
Day 1: +0.5%
Day 2: -0.3%
Day 3: +0.8%
...
Explanation: [Your brief technical analysis here]
"""

                    # Make API call to Ollama
                    response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': 'llama3.2' if model_type == "Ollama 3.2" else "deepseek-coder",
                            'prompt': prompt,
                            'stream': False
                        },
                        timeout=15
                    )

                    if response.status_code == 200:
                        # Parse the response
                        result = response.json()['response']
                        
                        # Extract daily predictions
                        predictions = []
                        lines = result.split('\n')
                        for line in lines:
                            if line.startswith('Day '):
                                try:
                                    pct = float(line.split(':')[1].strip().strip('%'))
                                    predictions.append(1 + (pct/100))
                                except:
                                    continue
                        
                        if predictions:
                            # Calculate cumulative price changes
                            y_test = data['Close'].values[-prediction_days:]
                            y_pred = [current_price]
                            for pct in predictions[:prediction_days]:
                                y_pred.append(y_pred[-1] * pct)
                            y_pred = np.array(y_pred[1:])
                            
                            # Calculate confidence bands (±1%)
                            confidence_lower = y_pred * 0.99
                            confidence_upper = y_pred * 1.01
                            
                            return y_pred, y_test, (confidence_lower, confidence_upper)
                            
                    # If we get here, something went wrong with the LLM
                    st.error("Error getting prediction from LLM model. Falling back to Random Forest...")
                    model_type = "Random Forest"
                    
                except Exception as e:
                    st.error(f"Error using LLM model: {str(e)}")
                    st.warning("Falling back to Random Forest model...")
                    model_type = "Random Forest"
        
        # Handle ML models
        if model_type in ["Random Forest", "XGBoost", "LightGBM"]:
            # Prepare the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[['Close', 'RSI', 'MACD', 'MA20', 'MA50', 'Volume']].values)
            
            # Create sequences
            X = []
            y = []
            sequence_length = 60  # Use last 60 days for prediction
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])  # 0 index is the Close price
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Select and train model
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(random_state=42)
                model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            else:  # LightGBM
                model = lgb.LGBMRegressor(random_state=42)
                model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            
            # Make predictions
            last_sequence = scaled_data[-sequence_length:]
            predictions = []
            
            for _ in range(prediction_days):
                # Reshape sequence for prediction
                current_pred = model.predict(last_sequence.reshape(1, -1))
                predictions.append(current_pred[0])
                
                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = np.append(current_pred, scaled_data[-1, 1:])
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            dummy = np.zeros((len(predictions), scaled_data.shape[1]-1))
            predictions = np.hstack((predictions, dummy))
            predictions = scaler.inverse_transform(predictions)[:, 0]
            
            return predictions
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prediction_tab():
    """Stock price prediction tab"""
    try:
        st.header("🔮 Stock Price Prediction")
        
        # Get user input
        col1, col2 = st.columns([3, 1])
        with col1:
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
        
        if st.button("Generate Prediction", key="generate_prediction"):
            with st.spinner("Fetching data and generating prediction..."):
                try:
                    # Get stock data safely
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Safely get current price and create initial metrics
                    current_price = float(info.get('currentPrice', 0))
                    if current_price == 0:
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                    
                    # Get historical data for technical analysis
                    data = stock.history(period="1y")
                    if data.empty:
                        st.error(f"No data found for {ticker}")
                        return
                    
                    # Process technical indicators
                    data['MA20'] = data['Close'].rolling(window=20).mean()
                    data['MA50'] = data['Close'].rolling(window=50).mean()
                    data['MA200'] = data['Close'].rolling(window=200).mean()
                    
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
                    
                    # Test LLM connection if selected
                    if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                        is_available = test_ollama_connection(model_type)
                        if not is_available:
                            st.warning(f"{model_type} not available. Falling back to Random Forest model...")
                            model_type = "Random Forest"
                    
                    # Generate predictions
                    if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                        # Use LLM prediction
                        prediction_data = {
                            'ticker': ticker,
                            'current_price': current_price,
                            'ma20': float(data['MA20'].iloc[-1]),
                            'ma50': float(data['MA50'].iloc[-1]),
                            'ma200': float(data['MA200'].iloc[-1]),
                            'rsi': float(data['RSI'].iloc[-1]),
                            'volume': float(data['Volume'].iloc[-1]),
                            'prediction_days': prediction_days
                        }
                        
                        predictions = predict_stock_price(prediction_data, model_type)
                        
                        if predictions:
                            # Display LLM prediction results
                            st.subheader("Price Prediction")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Direction",
                                    predictions.get('direction', 'UNKNOWN'),
                                    delta="↑" if predictions.get('direction') == "UP" else "↓"
                                )
                            
                            with col2:
                                change = float(predictions.get('percent_change', 0))
                                st.metric(
                                    "Expected Change",
                                    f"{change:+.2f}%",
                                    delta=change,
                                    delta_color="normal"
                                )
                            
                            with col3:
                                confidence = float(predictions.get('confidence', 0))
                                st.metric(
                                    "Confidence",
                                    f"{confidence:.1f}%"
                                )
                            
                            # Calculate price targets
                            predicted_change = float(predictions.get('percent_change', 0)) / 100
                            predicted_price = current_price * (1 + predicted_change)
                            
                            # Display price predictions
                            st.subheader("Price Targets")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                lower_bound = predicted_price * 0.95
                                st.metric(
                                    "Conservative Target",
                                    f"${lower_bound:.2f}",
                                    delta=f"{((lower_bound/current_price - 1) * 100):.1f}%",
                                    delta_color="inverse"
                                )
                            
                            with col2:
                                st.metric(
                                    "Base Target",
                                    f"${predicted_price:.2f}",
                                    delta=f"{((predicted_price/current_price - 1) * 100):.1f}%",
                                    delta_color="normal"
                                )
                            
                            with col3:
                                upper_bound = predicted_price * 1.05
                                st.metric(
                                    "Optimistic Target",
                                    f"${upper_bound:.2f}",
                                    delta=f"{((upper_bound/current_price - 1) * 100):.1f}%",
                                    delta_color="normal"
                                )
                    else:
                        # Use traditional ML models
                        predictions = predict_stock_price(data, prediction_days, model_type)
                        if predictions is not None:
                            plot_predictions(data, predictions)
                            
                            # Show prediction summary
                            last_price = data['Close'].iloc[-1]
                            pred_price = predictions[-1]
                            price_change = ((pred_price - last_price) / last_price) * 100
                            
                            st.subheader("Price Prediction")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Current Price",
                                    f"${last_price:.2f}"
                                )
                            
                            with col2:
                                st.metric(
                                    f"Predicted Price ({prediction_days} days)",
                                    f"${pred_price:.2f}",
                                    delta=f"{price_change:+.2f}%",
                                    delta_color="normal" if price_change > 0 else "inverse"
                                )
                    
                    # Display technical analysis
                    st.subheader("Technical Indicators")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ma_signal = "Bullish" if data['MA20'].iloc[-1] > data['MA50'].iloc[-1] else "Bearish"
                        st.metric(
                            "Moving Average Signal",
                            ma_signal,
                            delta="↑" if ma_signal == "Bullish" else "↓"
                        )
                    
                    with col2:
                        rsi = data['RSI'].iloc[-1]
                        rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                        st.metric(
                            "RSI Signal",
                            f"{rsi:.1f}",
                            delta=rsi_signal
                        )
                    
                    with col3:
                        volume_change = ((data['Volume'].iloc[-1] / data['Volume'].iloc[-2]) - 1) * 100
                        st.metric(
                            "Volume Change",
                            f"{volume_change:+.1f}%",
                            delta="Higher" if volume_change > 0 else "Lower"
                        )
                
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    st.info("Please verify the ticker symbol and try again.")
    
    except Exception as e:
        st.error(f"Error in prediction tab: {str(e)}")
        st.info("Please refresh the page and try again.")

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
        
        # Add ticker attribute
        data.attrs['ticker'] = ticker
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def plot_daily_candlestick(ticker):
    """Plot daily candlestick chart with technical indicators"""
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period='1y')
        
        if hist_data.empty:
            st.error(f"No data available for {ticker}")
            return None
            
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name=ticker
        )])
        
        # Add Moving Averages
        ma20 = hist_data['Close'].rolling(window=20).mean()
        ma50 = hist_data['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=ma20,
            line=dict(color='orange', width=1),
            name='20-day MA'
        ))
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=ma50,
            line=dict(color='blue', width=1),
            name='50-day MA'
        ))
        
        # Add Bollinger Bands
        bb_period = 20
        std_dev = 2
        
        bb_middle = hist_data['Close'].rolling(window=bb_period).mean()
        bb_std = hist_data['Close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=bb_upper,
            line=dict(color='gray', width=1, dash='dash'),
            name='Upper BB'
        ))
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=bb_lower,
            line=dict(color='gray', width=1, dash='dash'),
            name='Lower BB',
            fill='tonexty'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Daily Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating candlestick chart for {ticker}: {str(e)}")
        return None

def plot_stock_history(ticker, period='1y'):
    """Plot historical stock data with multiple timeframes and technical indicators"""
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        
        if hist_data.empty:
            st.error(f"No historical data available for {ticker}")
            return None
            
        # Calculate technical indicators
        # Moving Averages
        hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
        hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
        hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()
        
        # RSI
        delta = hist_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
        hist_data['MACD'] = exp1 - exp2
        hist_data['Signal'] = hist_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Create subplots
        fig = make_subplots(rows=3, cols=1, 
                          shared_xaxes=True,
                          vertical_spacing=0.05,
                          row_heights=[0.6, 0.2, 0.2])
        
        # Price and MA Plot
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name=ticker
        ), row=1, col=1)
        
        # Add Moving Averages
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['MA20'],
            line=dict(color='orange', width=1),
            name='MA20'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['MA50'],
            line=dict(color='blue', width=1),
            name='MA50'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['MA200'],
            line=dict(color='red', width=1),
            name='MA200'
        ), row=1, col=1)
        
        # Volume Plot
        colors = ['red' if row['Open'] - row['Close'] > 0 
                 else 'green' for index, row in hist_data.iterrows()]
        
        fig.add_trace(go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            marker_color=colors,
            name='Volume'
        ), row=2, col=1)
        
        # RSI Plot
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['RSI'],
            line=dict(color='purple'),
            name='RSI'
        ), row=3, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Historical Data ({period})',
            yaxis_title='Price (USD)',
            yaxis2_title='Volume',
            yaxis3_title='RSI',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=800
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating historical chart for {ticker}: {str(e)}")
        return None

def get_buffett_analysis(ticker, financials, info):
    """Generate Warren Buffett style analysis for a stock"""
    try:
        # Safely get values with defaults
        def safe_get(data, key, default=0):
            try:
                value = data.get(key, default)
                if isinstance(value, pd.Series):
                    return float(value.iloc[0])
                elif isinstance(value, list):
                    return float(value[0])
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return default
            except (TypeError, ValueError, IndexError):
                return default

        # Calculate key Buffett metrics with safe division
        def safe_divide(a, b, default=0):
            try:
                return float(a) / float(b) if float(b) != 0 else default
            except (TypeError, ValueError):
                return default

        # Get basic metrics
        market_cap = safe_get(info, 'marketCap')
        revenue = safe_get(financials, 'Total Revenue')
        net_income = safe_get(financials, 'Net Income')
        total_assets = safe_get(financials, 'Total Assets')
        total_debt = safe_get(financials, 'Total Debt')
        cash = safe_get(financials, 'Cash')
        operating_margin = safe_get(info, 'operatingMargins', 0) * 100
        current_price = safe_get(info, 'currentPrice')
        shares_outstanding = safe_get(info, 'sharesOutstanding', 1)
        
        # Calculate ratios safely
        pe_ratio = safe_get(info, 'forwardPE')
        profit_margin = safe_divide(net_income, revenue) * 100
        debt_to_equity = safe_get(info, 'debtToEquity')
        current_ratio = safe_get(info, 'currentRatio')
        roa = safe_divide(net_income, total_assets) * 100
        roe = safe_get(info, 'returnOnEquity', 0) * 100
        gross_margin = safe_get(info, 'grossMargins', 0) * 100
        
        # Calculate FCF and yield
        try:
            fcf = financials.get('Free Cash Flow', pd.Series([0])).iloc[0]
        except:
            fcf = 0
        fcf_yield = safe_divide(fcf, market_cap) * 100 if market_cap > 0 else 0
        
        # Calculate growth rates
        try:
            revenue_growth = safe_get(info, 'revenueGrowth', 0.05)
            if isinstance(revenue_growth, (int, float)):
                growth_rate = max(min(revenue_growth, 0.15), 0)  # Cap between 0% and 15%
            else:
                growth_rate = 0.05  # Default to 5% if growth rate is invalid
        except:
            growth_rate = 0.05
            
        # Valuation calculations
        discount_rate = 0.10  # 10% discount rate
        years = 10
        
        if fcf > 0:
            future_fcf = fcf * (1 + growth_rate) ** years
            terminal_value = future_fcf / (discount_rate - growth_rate) if discount_rate > growth_rate else fcf * 15
            present_value = terminal_value / (1 + discount_rate) ** years
            intrinsic_value = present_value / shares_outstanding if shares_outstanding > 0 else current_price
        else:
            # Fallback to a simple earnings-based valuation
            intrinsic_value = (net_income * 15) / shares_outstanding if shares_outstanding > 0 else current_price
        
        # Calculate margin of safety
        margin_of_safety = safe_divide((intrinsic_value - current_price), intrinsic_value) * 100 if intrinsic_value > 0 else -100
        
        # Get company info with defaults
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        business_summary = info.get('longBusinessSummary', 'No business summary available.')
        
        # Collect metrics for write-up
        metrics = {
            'market_cap': market_cap,
            'operating_margin': operating_margin,
            'roe': roe,
            'gross_margin': gross_margin,
            'current_ratio': current_ratio,
            'debt_to_equity': debt_to_equity,
            'fcf_yield': fcf_yield,
            'pe_ratio': pe_ratio,
            'margin_of_safety': margin_of_safety,
            'current_price': current_price,
            'profit_margin': profit_margin,
            'revenue_growth': growth_rate * 100,  # Convert to percentage
            'profit_growth': safe_get(info, 'netIncomeGrowth', 0) * 100  # Convert to percentage
        }
        
        # Generate the analysis
        analysis = f"""
### 🎩 Warren Buffett's Analysis of {ticker}

#### 💼 Business Overview
_{business_summary[:300]}..._

#### 📊 Key Buffett Metrics and Their Meaning

1. **Profitability Metrics**
   | Metric | Value | Meaning | Assessment |
   |--------|--------|---------|------------|
   | Return on Equity (ROE) | {roe:.1f}% | Measures how efficiently company uses shareholder money | {'Excellent' if roe > 15 else 'Good' if roe > 12 else 'Fair' if roe > 10 else 'Poor'} |
   | Operating Margin | {operating_margin:.1f}% | Indicates pricing power and operational efficiency | {'Excellent' if operating_margin > 25 else 'Good' if operating_margin > 15 else 'Fair' if operating_margin > 10 else 'Poor'} |
   | Gross Margin | {gross_margin:.1f}% | Shows pricing power and brand strength | {'Excellent' if gross_margin > 40 else 'Good' if gross_margin > 30 else 'Fair' if gross_margin > 20 else 'Poor'} |
   | Profit Margin | {profit_margin:.1f}% | Net profit per dollar of revenue | {'Excellent' if profit_margin > 20 else 'Good' if profit_margin > 15 else 'Fair' if profit_margin > 10 else 'Poor'} |

2. **Financial Health**
   | Metric | Value | Meaning | Assessment |
   |--------|--------|---------|------------|
   | Debt/Equity | {debt_to_equity:.1f}% | Measures financial leverage | {'Conservative' if debt_to_equity < 50 else 'Moderate' if debt_to_equity < 100 else 'High'} |
   | Current Ratio | {current_ratio:.2f} | Shows ability to pay short-term obligations | {'Excellent' if current_ratio > 2 else 'Good' if current_ratio > 1.5 else 'Fair' if current_ratio > 1 else 'Poor'} |
   | FCF Yield | {fcf_yield:.1f}% | Indicates cash generation relative to price | {'Excellent' if fcf_yield > 8 else 'Good' if fcf_yield > 5 else 'Fair' if fcf_yield > 3 else 'Poor'} |

3. **Growth and Value**
   | Metric | Value | Meaning | Assessment |
   |--------|--------|---------|------------|
   | Forward P/E | {pe_ratio:.1f}x | Price relative to earnings | {'Attractive' if pe_ratio < 15 else 'Fair' if pe_ratio < 20 else 'Premium' if pe_ratio < 25 else 'Expensive'} |
   | Growth Rate | {growth_rate*100:.1f}% | Expected annual growth | {'Strong' if growth_rate > 0.15 else 'Good' if growth_rate > 0.10 else 'Fair' if growth_rate > 0.05 else 'Low'} |
   | Market Cap | ${market_cap/1e9:.1f}B | Company size and stability | {'Large Cap' if market_cap > 10e9 else 'Mid Cap' if market_cap > 2e9 else 'Small Cap'} |

#### 💰 Valuation Analysis
- Current Price: ${current_price:.2f}
- Estimated Intrinsic Value: ${intrinsic_value:.2f}
- Margin of Safety: {margin_of_safety:.1f}%

#### 🎯 Investment Recommendation
"""
        # Add technical analysis
        try:
            hist_data = yf.Ticker(ticker).history(period='1y')
            if not hist_data.empty:
                # Calculate technical signals
                sma_50 = hist_data['Close'].rolling(window=50).mean().iloc[-1]
                sma_200 = hist_data['Close'].rolling(window=200).mean().iloc[-1]
                rsi = calculate_rsi(hist_data['Close'])[-1]
                macd, signal, _ = calculate_macd(hist_data['Close'])
                
                technical_signal = "Bullish" if sma_50 > sma_200 and rsi < 70 and macd[-1] > signal[-1] else \
                                 "Bearish" if sma_50 < sma_200 and rsi > 30 and macd[-1] < signal[-1] else "Neutral"
                
                analysis += f"""
#### 📊 Technical Analysis
- **50-Day MA**: ${sma_50:.2f}
- **200-Day MA**: ${sma_200:.2f}
- **RSI**: {rsi:.1f}
- **Technical Outlook**: {technical_signal}
"""
        except Exception as e:
            logger.warning(f"Could not generate technical analysis: {str(e)}")
            
        # Add risk assessment
        analysis += f"""
#### ⚠️ Risk Assessment
1. **Market Risk**
   - Beta: {safe_get(info, 'beta', 1.0):.2f}
   - Market Cap Stability: {'High' if market_cap > 10e9 else 'Moderate' if market_cap > 2e9 else 'Low'}

2. **Financial Risk**
   - Debt Level: {'High' if debt_to_equity > 100 else 'Moderate' if debt_to_equity > 50 else 'Low'}
   - Liquidity Risk: {'Low' if current_ratio > 2 else 'Moderate' if current_ratio > 1.5 else 'High'}

3. **Business Risk**
   - Revenue Concentration: {'Diversified' if operating_margin > 15 and gross_margin > 40 else 'Moderate' if operating_margin > 10 and gross_margin > 30 else 'Concentrated'}
   - Competitive Position: {'Strong' if operating_margin > 20 else 'Moderate' if operating_margin > 10 else 'Weak'}
"""
        
        return metrics, analysis
        
    except Exception as e:
        logger.error(f"Error in Buffett analysis: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"Could not complete Buffett analysis: {str(e)}")

def get_board_level_analysis(ticker, metrics, financials, info):
    """Generate board-level Warren Buffett style analysis"""
    try:
        # Get historical data for technical analysis
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        # Calculate technical indicators
        hist['RSI'] = calculate_rsi(hist)
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
        
        # Get current values
        current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
        current_rsi = float(hist['RSI'].iloc[-1])
        ma50 = float(hist['MA50'].iloc[-1])
        ma200 = float(hist['MA200'].iloc[-1])
        
        # Get key metrics with safe type conversion
        market_cap = float(info.get('marketCap', 0)) / 1e9  # Convert to billions
        pe_ratio = float(info.get('forwardPE', info.get('trailingPE', 0)))
        profit_margin = float(info.get('profitMargin', 0)) * 100
        revenue_growth = float(info.get('revenueGrowth', 0)) * 100
        debt_to_equity = float(info.get('debtToEquity', 0))
        
        # Generate analysis sections
        financial_health = f"""
### Financial Health Assessment
- Market Cap: ${market_cap:.1f}B
- P/E Ratio: {pe_ratio:.1f}x
- Profit Margin: {profit_margin:.1f}%
- Revenue Growth: {revenue_growth:+.1f}%
- Debt/Equity: {debt_to_equity:.1f}x
"""

        technical_signals = f"""
### Technical Signals
- Current Price: ${current_price:.2f}
- RSI ({current_rsi:.1f}): {"Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"}
- MA50/MA200: {"Bullish" if ma50 > ma200 else "Bearish"} Trend
"""

        # Risk assessment
        risk_factors = []
        if pe_ratio > 30:
            risk_factors.append("High valuation multiple")
        if debt_to_equity > 2:
            risk_factors.append("High debt levels")
        if profit_margin < 10:
            risk_factors.append("Low profit margins")
        if current_rsi > 70:
            risk_factors.append("Overbought conditions")
        
        risk_section = f"""
### Risk Assessment
{"- " + "\\n- ".join(risk_factors) if risk_factors else "- No significant risk factors identified"}
"""

        # Investment thesis
        if pe_ratio < 20 and profit_margin > 15 and revenue_growth > 10:
            recommendation = "Strong Buy"
            thesis = "Company shows strong fundamentals with reasonable valuation"
        elif pe_ratio < 25 and profit_margin > 10:
            recommendation = "Buy"
            thesis = "Solid performance with moderate growth potential"
        elif pe_ratio > 40 or profit_margin < 5:
            recommendation = "Sell"
            thesis = "Valuation concerns and weak fundamentals"
        else:
            recommendation = "Hold"
            thesis = "Monitor for better entry points"
        
        investment_thesis = f"""
### Investment Recommendation: {recommendation}
{thesis}
"""

        # Combine all sections
        analysis = f"""
# Board-Level Analysis for {ticker}
{financial_health}
{technical_signals}
{risk_section}
{investment_thesis}
"""
        
        return analysis
        
    except Exception as e:
        error_msg = f"Unable to generate complete board-level analysis: {str(e)}"
        st.error(error_msg)
        return error_msg

def get_buffett_writeup(ticker, metrics):
    """Generate Warren Buffett style investment write-up"""
    try:
        # Safely get metrics with defaults
        def safe_get(data, key, default=0):
            try:
                value = data.get(key, default)
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default
        
        # Extract metrics safely
        operating_margin = safe_get(metrics, 'operating_margin', 0)
        gross_margin = safe_get(metrics, 'gross_margin', 0)
        profit_margin = safe_get(metrics, 'profit_margin', 0)
        roe = safe_get(metrics, 'roe', 0)
        current_ratio = safe_get(metrics, 'current_ratio', 0)
        debt_to_equity = safe_get(metrics, 'debt_to_equity', 0)
        market_cap = safe_get(metrics, 'market_cap', 0) / 1e9  # Convert to billions
        pe_ratio = safe_get(metrics, 'pe_ratio', 0)
        current_price = safe_get(metrics, 'current_price', 0)
        revenue_growth = safe_get(metrics, 'revenue_growth', 0)
        profit_growth = safe_get(metrics, 'profit_growth', 0)
        
        # Calculate fair value
        if pe_ratio > 0 and pe_ratio < 100:
            annual_earnings = current_price / pe_ratio
        else:
            annual_earnings = current_price * (profit_margin / 100)
        
        growth_rate = min(max((revenue_growth + profit_growth) / 2 / 100, 0.02), 0.15)
        future_earnings = annual_earnings * (1 + growth_rate) ** 5
        pe_multiple = min(max(10 + growth_rate * 100, 12), 20)
        fair_value = max(future_earnings * pe_multiple, current_price * 0.5)
        
        # Calculate price targets
        strong_buy_price = fair_value * 0.7  # 30% discount
        buy_price = fair_value * 0.8  # 20% discount
        sell_price = fair_value * 1.2  # 20% premium
        stop_loss = current_price * 0.85  # 15% below current
        
        margin_of_safety = ((fair_value - current_price) / fair_value) * 100
        
        # Assess competitive advantages
        moat_strength = "Strong" if operating_margin > 20 and roe > 15 else \
                       "Moderate" if operating_margin > 15 and roe > 12 else "Limited"
        
        moat_factors = []
        if operating_margin > 20:
            moat_factors.append("high operating margins indicating pricing power")
        if roe > 15:
            moat_factors.append("excellent return on equity showing competitive advantages")
        if gross_margin > 40:
            moat_factors.append("strong gross margins suggesting brand value")
        if market_cap > 50:  # Already in billions
            moat_factors.append("significant market position")
        if revenue_growth > 15:
            moat_factors.append("strong revenue growth")
        if profit_growth > 15:
            moat_factors.append("impressive profit growth")
        
        # Assess financial health
        health = "Excellent" if current_ratio > 2 and debt_to_equity < 50 else \
                "Good" if current_ratio > 1.5 and debt_to_equity < 80 else "Fair"
        
        financial_strengths = []
        if current_ratio > 2:
            financial_strengths.append("strong liquidity position")
        if debt_to_equity < 50:
            financial_strengths.append("conservative debt management")
        if profit_margin > 15:
            financial_strengths.append("healthy profit margins")
        if revenue_growth > 10:
            financial_strengths.append("solid revenue growth")
        
        # Assess valuation
        value = "Attractive" if pe_ratio < 15 and margin_of_safety > 25 else \
                "Fair" if pe_ratio < 20 and margin_of_safety > 15 else "Expensive"
        
        value_reasons = []
        if pe_ratio < 15:
            value_reasons.append("reasonable P/E ratio")
        if margin_of_safety > 25:
            value_reasons.append("significant margin of safety")
        if profit_margin > 15:
            value_reasons.append("strong profitability")
        
        # Determine overall investment attractiveness
        is_compelling = (moat_strength == "Strong" and health == "Excellent" and value == "Attractive")
        is_potential = moat_strength != "Limited"
        
        # Generate summary and recommendation
        if is_compelling:
            action = "Strong Buy"
            summary = f"Strong investment opportunity with compelling moat, excellent financials, and attractive valuation"
            position_size = "5-7% of portfolio"
        elif is_potential and value != "Expensive":
            action = "Buy"
            summary = f"Good investment opportunity with some competitive advantages and reasonable valuation"
            position_size = "3-5% of portfolio"
        elif is_potential:
            action = "Hold/Monitor"
            summary = f"Decent company but wait for better entry price"
            position_size = "2-3% of portfolio"
        else:
            action = "Avoid"
            summary = f"Limited competitive advantages and/or unfavorable valuation"
            position_size = "0%"
        
        writeup = f"""
### 📝 Warren Buffett's Investment Write-up for {ticker}

#### 🎯 Executive Summary
{summary}

#### 💰 Price Analysis and Recommendations
| Price Point | Value | Notes |
|-------------|-------|-------|
| Current Price | ${current_price:.2f} | - |
| Fair Value | ${fair_value:.2f} | Based on fundamentals and growth |
| Strong Buy Below | ${strong_buy_price:.2f} | Significant margin of safety |
| Buy Below | ${buy_price:.2f} | Good entry point |
| Sell Above | ${sell_price:.2f} | Consider taking profits |
| Stop Loss | ${stop_loss:.2f} | Risk management level |

**Recommended Action**: {action}
**Position Size**: {position_size}
**Margin of Safety**: {margin_of_safety:.1f}%

#### 🏰 Economic Moat Analysis
The company demonstrates a {moat_strength.lower()} economic moat, evidenced by {', '.join(moat_factors) if moat_factors else 'Limited competitive advantages identified'}.
{'This sustainable competitive advantage positions the company well for long-term value creation.' if moat_strength == 'Strong' else 
'While showing some competitive strengths, the moat needs further development.' if moat_strength == 'Moderate' else
'The lack of a strong economic moat raises concerns about long-term profitability.'
}

#### 💪 Business Strengths
1. **Profitability Metrics**
   | Metric | Value | Assessment |
   |--------|--------|------------|
   | Return on Equity | {roe:.1f}% | {'Excellent' if roe > 15 else 'Good' if roe > 12 else 'Fair' if roe > 10 else 'Poor'} |
   | Operating Margin | {operating_margin:.1f}% | {'Excellent' if operating_margin > 25 else 'Good' if operating_margin > 15 else 'Fair' if operating_margin > 10 else 'Poor'} |
   | Gross Margin | {gross_margin:.1f}% | {'Excellent' if gross_margin > 40 else 'Good' if gross_margin > 30 else 'Fair' if gross_margin > 20 else 'Poor'} |

2. **Growth & Financial Health**
   | Metric | Value | Assessment |
   |--------|--------|------------|
   | Revenue Growth | {revenue_growth:.1f}% | {'Excellent' if revenue_growth > 15 else 'Good' if revenue_growth > 10 else 'Fair' if revenue_growth > 5 else 'Poor'} |
   | Profit Growth | {profit_growth:.1f}% | {'Excellent' if profit_growth > 15 else 'Good' if profit_growth > 10 else 'Fair' if profit_growth > 5 else 'Poor'} |
   | Current Ratio | {current_ratio:.2f} | {'Excellent' if current_ratio > 2 else 'Good' if current_ratio > 1.5 else 'Fair' if current_ratio > 1 else 'Poor'} |
   | Debt/Equity | {debt_to_equity:.1f}% | {'Excellent' if debt_to_equity < 30 else 'Good' if debt_to_equity < 60 else 'Fair' if debt_to_equity < 100 else 'Poor'} |
   | Market Cap | ${market_cap:.1f}B | {'Large Cap' if market_cap > 10 else 'Mid Cap' if market_cap > 2 else 'Small Cap'} |

#### 📈 Investment Strategy
1. **Entry Strategy**:
   - Current Price: ${current_price:.2f}
   - Ideal Entry: Below ${buy_price:.2f}
   - Strong Entry: Below ${strong_buy_price:.2f}
   - Position Size: {position_size}

2. **Exit Strategy**:
   - Take Profits: Above ${sell_price:.2f}
   - Stop Loss: ${stop_loss:.2f}
   - Review Position: If fundamentals deteriorate

3. **Monitoring Points**:
   - Quarterly: Check operating margins and ROE
   - Semi-Annual: Review competitive position
   - Annual: Full valuation reassessment

#### 💡 Key Investment Considerations
1. **Competitive Advantages**: {', '.join(moat_factors) if moat_factors else 'Limited competitive advantages identified'}
2. **Financial Strengths**: {', '.join(financial_strengths) if financial_strengths else 'Some financial concerns noted'}
3. **Valuation Factors**: {', '.join(value_reasons) if value_reasons else 'Valuation appears unfavorable'}

Remember: "Price is what you pay. Value is what you get." - Warren Buffett
"""
        return writeup
        
    except Exception as e:
        error_msg = f"Error generating Buffett analysis: {str(e)}"
        st.error(error_msg)
        return error_msg

def buffett_analysis_tab():
    """Display Warren Buffett style analysis tab"""
    try:
        st.header("🎩 Warren Buffett Analysis")
        
        # Get stock data and analysis
        ticker = st.session_state.get('ticker', '')
        if not ticker:
            st.warning("Please enter a stock ticker in the sidebar.")
            return
        
        # Get financial data
        with st.spinner("Analyzing stock fundamentals..."):
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                st.error(f"Could not fetch data for {ticker}. Please verify the ticker symbol.")
                return
                
            # Get financials with error handling
            try:
                financials = stock.financials
            except Exception as e:
                st.warning(f"Could not fetch complete financial data. Using available metrics only.")
                financials = pd.DataFrame()
            
            # Test Ollama connection
            llm_available, llm_message = test_ollama_connection("Ollama 3.2")
            if not llm_available:
                st.warning(f"LLM service not available: {llm_message}")
                st.info("Proceeding with traditional analysis...")
            
            # Get metrics and analysis
            metrics, analysis = get_buffett_analysis(ticker, financials, info)
            
            # Display analysis
            st.markdown(analysis)
            
            # Generate letter if LLM is available and add it in a collapsible section
            if llm_available:
                with st.expander("📝 Warren Buffett's Perspective", expanded=False):
                    letter = get_shareholder_letter(ticker, metrics)
                    st.markdown(letter)
            
            # Add technical analysis in expander
            with st.expander("📊 Technical Analysis Details"):
                # Get historical data
                hist = stock.history(period="1y")
                if not hist.empty:
                    # Calculate indicators
                    hist['RSI'] = calculate_rsi(hist['Close'])
                    hist['MA50'] = hist['Close'].rolling(window=50).mean()
                    hist['MA200'] = hist['Close'].rolling(window=200).mean()
                    
                    # Create technical analysis chart
                    fig = go.Figure()
                    
                    # Add price
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name='Price'
                    ))
                    
                    # Add moving averages
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['MA50'],
                        name='50-day MA',
                        line=dict(color='orange')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['MA200'],
                        name='200-day MA',
                        line=dict(color='blue')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} - Technical Analysis",
                        yaxis_title="Price",
                        xaxis_title="Date",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current technical signals
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "RSI (14)",
                            f"{hist['RSI'].iloc[-1]:.1f}",
                            help="Relative Strength Index. Values > 70 indicate overbought, < 30 oversold"
                        )
                        
                        st.metric(
                            "50-day MA",
                            f"${hist['MA50'].iloc[-1]:.2f}",
                            f"{((hist['Close'].iloc[-1] / hist['MA50'].iloc[-1] - 1) * 100):.1f}%",
                            help="50-day Moving Average"
                        )
                        
                    with col2:
                        st.metric(
                            "200-day MA",
                            f"${hist['MA200'].iloc[-1]:.2f}",
                            f"{((hist['Close'].iloc[-1] / hist['MA200'].iloc[-1] - 1) * 100):.1f}%",
                            help="200-day Moving Average"
                        )
                        
                        trend = "Bullish" if hist['MA50'].iloc[-1] > hist['MA200'].iloc[-1] else "Bearish"
                        st.metric(
                            "Trend",
                            trend,
                            help="Based on 50-day MA vs 200-day MA"
                        )
                else:
                    st.warning("No historical data available for technical analysis.")
                    
    except Exception as e:
        st.error(f"Error in Buffett analysis tab: {str(e)}")
        st.info("Please verify the ticker symbol and try again.")

def get_analyst_writeup(ticker, model_type="Ollama 3.2"):
    """Get detailed analyst writeup for a stock using LLM"""
    try:
        # Test model availability
        is_available, message = test_ollama_connection(model_type)
        if not is_available:
            st.error(message)
            return None
            
        # Get stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Create prompt for analysis
        prompt = f"""You are a senior stock analyst. Write a detailed analysis for {ticker} ({info.get('longName', '')}).
        Include:
        1. Investment Thesis
        2. Growth Catalysts
        3. Competitive Advantages
        4. Risk Factors
        5. Valuation Analysis
        6. Price Target Range
        
        Format your response in markdown with clear sections and bullet points.
        Be specific and use data where relevant.
        Keep the analysis concise but comprehensive.
        """
        
        # Make API call to Ollama
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2' if model_type == "Ollama 3.2" else "deepseek-coder",
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                return result
                
        except Exception as e:
            st.error(f"Error getting analyst writeup: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in analyst writeup: {str(e)}")
        return None

def get_stock_recommendations(model_type="Ollama 3.2"):
    """Get top 5 stock recommendations using LLM analysis"""
    try:
        # Test model availability
        is_available, message = test_ollama_connection(model_type)
        if not is_available:
            st.error(message)
            return None
            
        # Create prompt for stock recommendations
        prompt = """You are a senior portfolio manager. Recommend 5 stocks to analyze based on current market conditions.
        
        For each stock provide:
        1. Ticker symbol
        2. Company name
        3. Sector
        4. Key investment points (3-4 bullet points)
        5. Target price range (low and high estimates)
        6. Risk level (Low/Medium/High)
        
        Format response EXACTLY as JSON:
        {
            "recommendations": [
                {
                    "ticker": "AAPL",
                    "name": "Apple Inc.",
                    "sector": "Technology",
                    "points": [
                        "Strong market position",
                        "Growing services revenue",
                        "Solid balance sheet"
                    ],
                    "target_low": 180.50,
                    "target_high": 210.75,
                    "risk": "Low"
                }
            ]
        }
        
        Notes:
        - Use only real stock tickers
        - Include exactly 5 stocks
        - Format numbers as floats without currency symbols
        - Keep points concise and specific
        """
        
        # Make API call to Ollama
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2' if model_type == "Ollama 3.2" else "deepseek-coder",
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                
                # Extract JSON from response (handle potential text before/after JSON)
                try:
                    # Find JSON content between curly braces
                    json_str = result[result.find('{'):result.rfind('}')+1]
                    recommendations = json.loads(json_str)
                    
                    # Validate response structure
                    if not isinstance(recommendations, dict):
                        st.error("Invalid response format: not a dictionary")
                        return None
                        
                    if 'recommendations' not in recommendations:
                        st.error("Invalid response format: missing 'recommendations' key")
                        return None
                        
                    recs = recommendations['recommendations']
                    if not isinstance(recs, list) or len(recs) == 0:
                        st.error("Invalid response format: recommendations not a list or empty")
                        return None
                        
                    # Validate each recommendation
                    required_fields = ['ticker', 'name', 'sector', 'points', 'target_low', 'target_high', 'risk']
                    for rec in recs:
                        if not all(field in rec for field in required_fields):
                            st.error("Invalid response format: missing required fields")
                            return None
                        
                        # Ensure numeric fields are float
                        try:
                            rec['target_low'] = float(rec['target_low'])
                            rec['target_high'] = float(rec['target_high'])
                        except (ValueError, TypeError):
                            st.error("Invalid response format: price targets must be numbers")
                            return None
                            
                        # Ensure points is a list
                        if not isinstance(rec['points'], list):
                            st.error("Invalid response format: points must be a list")
                            return None
                    
                    return recs
                    
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing recommendations: Invalid JSON format - {str(e)}")
                    return None
                except Exception as e:
                    st.error(f"Error processing recommendations: {str(e)}")
                    return None
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to LLM service: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in stock recommendations: {str(e)}")
        return None

def recommendations_tab():
    """Stock recommendations tab with detailed analysis"""
    try:
        st.header("🎯 Top Stock Picks")
        
        # Model selection
        col1, col2 = st.columns([2, 3])
        with col1:
            model_type = st.selectbox(
                "Select Analysis Model",
                ["Ollama 3.2", "DeepSeek-R1"],
                key="rec_model"
            )
        
        # Get recommendations
        if st.button("Get Recommendations", key="get_rec"):
            with st.spinner("Generating stock recommendations..."):
                recommendations = get_stock_recommendations(model_type)
                
                if recommendations:
                    # Create tabs for different views
                    summary_tab, detail_tab = st.tabs(["Summary View", "Detailed Analysis"])
                    
                    with summary_tab:
                        # Create a summary table
                        summary_data = []
                        for rec in recommendations:
                            summary_data.append({
                                "Ticker": rec['ticker'],
                                "Company": rec['name'],
                                "Sector": rec['sector'],
                                "Risk": rec['risk'],
                                "Target Range": f"${rec['target_low']:.2f} - ${rec['target_high']:.2f}"
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(
                            summary_df,
                            column_config={
                                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                                "Company": st.column_config.TextColumn("Company", width="medium"),
                                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                                "Risk": st.column_config.TextColumn("Risk", width="small"),
                                "Target Range": st.column_config.TextColumn("Target Range", width="medium")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    with detail_tab:
                        # Create detailed analysis for each stock
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"#{i}: {rec['ticker']} - {rec['name']}", expanded=i==1):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.subheader("Investment Summary")
                                    st.write(f"**Sector:** {rec['sector']}")
                                    st.write(f"**Risk Level:** {rec['risk']}")
                                    st.write("**Key Points:**")
                                    for point in rec['points']:
                                        st.write(f"• {point}")
                                    
                                with col2:
                                    # Get real-time stock data
                                    stock = yf.Ticker(rec['ticker'])
                                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                                    
                                    # Calculate potential returns
                                    low_return = (rec['target_low'] / current_price - 1) * 100
                                    high_return = (rec['target_high'] / current_price - 1) * 100
                                    
                                    st.metric("Current Price", f"${current_price:.2f}")
                                    st.metric("Target Range", 
                                            f"${rec['target_low']:.2f} - ${rec['target_high']:.2f}",
                                            f"{low_return:.1f}% to {high_return:.1f}%")
                                
                                # Add technical chart
                                st.subheader("Technical Analysis")
                                chart = plot_stock_history(rec['ticker'], period="6mo")
                                if chart:
                                    st.plotly_chart(chart, use_container_width=True)
                                
                                # Get and display analyst writeup
                                st.subheader("Analyst Write-up")
                                writeup = get_analyst_writeup(rec['ticker'], model_type)
                                if writeup:
                                    st.markdown(writeup)
                    
                    # Add disclaimer
                    st.info("""
                    **Disclaimer:** These recommendations are generated using AI analysis of market data and should not be the sole basis for investment decisions. 
                    Always conduct your own research and consider consulting with a financial advisor before making investment choices.
                    """)
                    
    except Exception as e:
        st.error(f"Error in recommendations tab: {str(e)}")

def get_market_movers():
    """Get top gainers, losers, and most active stocks with SSL handling"""
    try:
        # Set up SSL context
        import ssl
        import certifi
        import requests
        
        # Create session with SSL verification and headers
        session = requests.Session()
        session.verify = certifi.where()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Function to safely get data
        def get_data(url):
            response = session.get(url)
            response.raise_for_status()
            df = pd.read_html(response.text)[0]
            
            # Select and rename required columns
            df = df[['Symbol', 'Name', 'Price (Intraday)', 'Change %', 'Volume', 'Market Cap']]
            df = df.rename(columns={
                'Change %': '% Change',
                'Price (Intraday)': 'Price'
            })
            return df
            
        # Helper functions to clean data
        def clean_percentage(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val).replace('%', '').replace('+', '').replace(',', '')
            try:
                return float(val)
            except:
                return 0.0
                
        def clean_price(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val).replace('$', '').replace('+', '').replace(',', '')
            try:
                return float(val)
            except:
                return 0.0
                
        def clean_volume(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val)
            try:
                val = val.replace(',', '')
                if 'K' in val:
                    return float(val.replace('K', '')) * 1000
                elif 'M' in val:
                    return float(val.replace('M', '')) * 1000000
                elif 'B' in val:
                    return float(val.replace('B', '')) * 1000000000
                else:
                    return float(val)
            except:
                return 0.0
        
        # Get market data from Yahoo Finance with updated URLs
        gainers = get_data('https://finance.yahoo.com/screener/predefined/day_gainers')
        losers = get_data('https://finance.yahoo.com/screener/predefined/day_losers')
        active = get_data('https://finance.yahoo.com/screener/predefined/most_actives')
        
        # Clean up data
        for df in [gainers, losers, active]:
            df['% Change'] = df['% Change'].apply(clean_percentage)
            df['Price'] = df['Price'].apply(clean_price)
            df['Volume'] = df['Volume'].apply(clean_volume)
            
        return gainers.head(), losers.head(), active.head()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching market data: {str(e)}")
        return None, None, None
    except ValueError as e:
        st.error(f"Error processing market data: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None, None, None

def display_movers_table(df, category):
    """Display market movers table with formatting"""
    try:
        if df is not None and not df.empty:
            # Format the display
            df_display = df.copy()
            df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:,.2f}")
            df_display['Volume'] = df_display['Volume'].apply(lambda x: f"{x:,.0f}")
            
            # Format % Change with appropriate colors and symbols
            def format_change(x):
                if x > 0:
                    return f"🔺 +{x:.2f}%"
                elif x < 0:
                    return f"🔻 {x:.2f}%"
                else:
                    return f"⚫ {x:.2f}%"
                    
            df_display['% Change'] = df_display['% Change'].apply(format_change)
            
            # Display table with styling
            st.dataframe(
                df_display,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Name": st.column_config.TextColumn("Company", width="medium"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "% Change": st.column_config.TextColumn("Change", width="small"),
                    "Volume": st.column_config.TextColumn("Volume", width="medium"),
                    "Market Cap": st.column_config.TextColumn("Market Cap", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                avg_change = df['% Change'].mean()
                color = "normal" if category == "active" else ("inverse" if category == "losers" else "normal")
                st.metric(
                    "Average Change",
                    f"{avg_change:+.2f}%",
                    delta=f"{avg_change:+.2f}%",
                    delta_color=color
                )
            
            with col2:
                total_volume = df['Volume'].sum()
                st.metric("Total Volume", f"{total_volume:,.0f}")
                
    except Exception as e:
        st.error(f"Error displaying {category}: {str(e)}")

def market_movers_tab():
    """Market movers tab with enhanced UI and analysis"""
    try:
        st.header("📊 Market Movers")
        
        # Add refresh button
        col1, col2 = st.columns([1, 5])
        with col1:
            refresh = st.button("🔄 Refresh", key="refresh_market")
        with col2:
            st.write("Last updated: " + datetime.now().strftime("%I:%M:%S %p"))
        
        if refresh:
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Get market data
        with st.spinner("Fetching market data..."):
            gainers, losers, active = get_market_movers()
            
            if gainers is not None and losers is not None and active is not None:
                # Create tabs for different categories
                tab1, tab2, tab3 = st.tabs(["🔺 Top Gainers", "🔻 Top Losers", "📊 Most Active"])
                
                with tab1:
                    st.subheader("Top Gainers")
                    display_movers_table(gainers, "Top Gainers")
                    with st.expander("Understanding Gainers"):
                        st.write("""
                        **Top Gainers Analysis:**
                        - Strong upward price movement
                        - May indicate positive news or market sentiment
                        - Consider volume to confirm trend strength
                        - Watch for potential overextension
                        """)
                
                with tab2:
                    st.subheader("Top Losers")
                    display_movers_table(losers, "Top Losers")
                    with st.expander("Understanding Losers"):
                        st.write("""
                        **Top Losers Analysis:**
                        - Significant price decline
                        - May indicate negative news or market concerns
                        - Potential oversold conditions
                        - Consider fundamental factors
                        """)
                
                with tab3:
                    st.subheader("Most Active")
                    display_movers_table(active, "Most Active")
                    with st.expander("Understanding Volume"):
                        st.write("""
                        **High Volume Analysis:**
                        - Indicates strong market interest
                        - Higher volume validates price movement
                        - May signal trend changes
                        - Important for momentum trading
                        """)
                
                # Market Summary
                with st.expander("📈 Market Summary"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_gain = gainers['% Change'].mean()
                        st.metric(
                            "Average Gain",
                            f"+{avg_gain:.2f}%"
                        )
                        
                    with col2:
                        avg_loss = losers['% Change'].mean()
                        st.metric(
                            "Average Loss",
                            f"{avg_loss:.2f}%"
                        )
                        
                    with col3:
                        total_vol = active['Volume'].sum()
                        st.metric("Total Active Volume", f"{total_vol:,.0f}")
                    
                    st.write("""
                    ### Market Insights
                    - Monitor volume for confirmation of price moves
                    - Large gains/losses may indicate sector rotation
                    - High volume often precedes major moves
                    - Consider market sentiment and sector trends
                    """)
            else:
                st.error("Unable to fetch market data. Please try again later.")
                
    except Exception as e:
        st.error(f"Error in market movers tab: {str(e)}")

def technical_analysis_tab():
    """Technical analysis tab with customizable indicators"""
    try:
        st.header("📈 Technical Analysis")
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Enter Stock Ticker:", "AAPL", key="tech_ticker").upper()
        
        with col2:
            period = st.selectbox(
                "Select Time Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=3,
                key="tech_period"
            )
        
        # Technical Indicators Selection
        st.subheader("Customize Technical Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_ma = st.checkbox("Moving Averages", value=True, key="show_ma")
            if show_ma:
                ma_periods = st.multiselect(
                    "MA Periods:",
                    [20, 50, 200],
                    default=[20, 50],
                    key="ma_periods"
                )
        
        with col2:
            show_bb = st.checkbox("Bollinger Bands", value=True, key="show_bb")
            if show_bb:
                bb_period = st.number_input(
                    "BB Period:",
                    min_value=5,
                    max_value=50,
                    value=20,
                    key="bb_period"
                )
                bb_std = st.number_input(
                    "BB Std Dev:",
                    min_value=1,
                    max_value=4,
                    value=2,
                    key="bb_std"
                )
        
        with col3:
            show_rsi = st.checkbox("RSI", value=True, key="show_rsi")
            if show_rsi:
                rsi_period = st.number_input(
                    "RSI Period:",
                    min_value=5,
                    max_value=50,
                    value=14,
                    key="rsi_period"
                )
        
        if ticker:
            try:
                # Fetch stock data
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period=period)
                
                if hist_data.empty:
                    st.error(f"No data available for {ticker}")
                    return
                
                # Calculate indicators
                if show_ma:
                    for ma in ma_periods:
                        hist_data[f'MA{ma}'] = hist_data['Close'].rolling(window=ma).mean()
                
                if show_bb:
                    bb_middle = hist_data['Close'].rolling(window=bb_period).mean()
                    bb_std_dev = hist_data['Close'].rolling(window=bb_period).std()
                    hist_data['BB_Upper'] = bb_middle + (bb_std * bb_std_dev)
                    hist_data['BB_Lower'] = bb_middle - (bb_std * bb_std_dev)
                    hist_data['BB_Middle'] = bb_middle
                
                if show_rsi:
                    hist_data['RSI'] = calculate_rsi(hist_data, rsi_period)
                
                # Create figure with secondary y-axis
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # Add candlestick
                fig.add_trace(go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name=ticker
                ), row=1, col=1)
                
                # Add Moving Averages
                if show_ma:
                    colors = ['orange', 'blue', 'red']
                    for i, ma in enumerate(ma_periods):
                        fig.add_trace(go.Scatter(
                            x=hist_data.index,
                            y=hist_data[f'MA{ma}'],
                            name=f'MA{ma}',
                            line=dict(color=colors[i % len(colors)])
                        ), row=1, col=1)
                
                # Add Bollinger Bands
                if show_bb:
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray', dash='dash')
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray', dash='dash'),
                        fill='tonexty'
                    ), row=1, col=1)
                
                # Add RSI
                if show_rsi:
                    fig.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['RSI'],
                        name='RSI',
                        line=dict(color='purple')
                    ), row=2, col=1)
                    
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Technical Analysis ({period})',
                    yaxis_title='Price',
                    yaxis2_title='RSI' if show_rsi else None,
                    xaxis_rangeslider_visible=False,
                    height=800
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add technical analysis summary
                with st.expander("Technical Analysis Summary"):
                    # Calculate current values
                    current_price = hist_data['Close'].iloc[-1]
                    ma_signals = []
                    if show_ma:
                        for ma in ma_periods:
                            ma_value = hist_data[f'MA{ma}'].iloc[-1]
                            signal = "ABOVE" if current_price > ma_value else "BELOW"
                            ma_signals.append(f"Price is {signal} MA{ma} (${ma_value:.2f})")
                    
                    bb_signal = ""
                    if show_bb:
                        upper_bb = hist_data['BB_Upper'].iloc[-1]
                        lower_bb = hist_data['BB_Lower'].iloc[-1]
                        if current_price > upper_bb:
                            bb_signal = "Price is ABOVE upper Bollinger Band - Potentially overbought"
                        elif current_price < lower_bb:
                            bb_signal = "Price is BELOW lower Bollinger Band - Potentially oversold"
                        else:
                            bb_signal = "Price is WITHIN Bollinger Bands - Normal volatility"
                    
                    rsi_signal = ""
                    if show_rsi:
                        current_rsi = hist_data['RSI'].iloc[-1]
                        if current_rsi > 70:
                            rsi_signal = f"RSI is {current_rsi:.2f} - Overbought"
                        elif current_rsi < 30:
                            rsi_signal = f"RSI is {current_rsi:.2f} - Oversold"
                        else:
                            rsi_signal = f"RSI is {current_rsi:.2f} - Neutral"
                    
                    # Display summary
                    st.write(f"**Current Price:** ${current_price:.2f}")
                    if ma_signals:
                        st.write("**Moving Averages:**")
                        for signal in ma_signals:
                            st.write(f"- {signal}")
                    if bb_signal:
                        st.write(f"**Bollinger Bands:** {bb_signal}")
                    if rsi_signal:
                        st.write(f"**RSI:** {rsi_signal}")
                
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error in technical analysis tab: {str(e)}")

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    try:
        # Convert data to Series if it's not already
        if isinstance(data, pd.DataFrame):
            close_prices = data['Close']
        else:
            close_prices = pd.Series(data)
            
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains (up) and losses (down)
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        avg_losses = losses.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        
        # Calculate relative strength
        rs = avg_gains / avg_losses
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        if isinstance(data, pd.DataFrame):
            return pd.Series(index=data.index)
        return pd.Series()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        # Convert data to Series if it's not already
        if isinstance(data, pd.DataFrame):
            close_prices = data['Close']
        else:
            close_prices = pd.Series(data)
            
        # Calculate EMAs
        fast_ema = close_prices.ewm(span=fast_period, min_periods=fast_period).mean()
        slow_ema = close_prices.ewm(span=slow_period, min_periods=slow_period).mean()
        
        # Calculate MACD line
        macd = fast_ema - slow_ema
        
        # Calculate signal line
        signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()
        
        # Calculate histogram
        hist = macd - signal
        
        return macd, signal, hist
        
    except Exception as e:
        st.error(f"Error calculating MACD: {str(e)}")
        if isinstance(data, pd.DataFrame):
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
        return pd.Series(), pd.Series(), pd.Series()

def get_shareholder_letter(ticker, metrics):
    """Generate a detailed letter to shareholders using LLM analysis"""
    try:
        # Extract key metrics
        current_price = metrics.get('current_price', 0)
        pe_ratio = metrics.get('pe_ratio', 0)
        market_cap = metrics.get('market_cap', 0) / 1e9  # Convert to billions
        revenue_growth = metrics.get('revenue_growth', 0)
        profit_margin = metrics.get('profit_margin', 0)
        roe = metrics.get('roe', 0)
        debt_to_equity = metrics.get('debt_to_equity', 0)
        
        # Create prompt for LLM
        prompt = f"""
        As Warren Buffett, write a detailed letter to shareholders about {ticker} stock analysis. 
        Include:
        1. Investment Thesis
        2. Growth Catalysts
        3. Competitive Advantages
        4. Risk Factors
        5. Valuation Analysis
        6. Price Target Range
        
        Format your response in markdown with clear sections and bullet points.
        Be specific and use data where relevant.
        Keep the analysis concise but comprehensive.
        """
        
        # Get LLM response
        response = get_llm_response(prompt)
        if not response:
            return "Error: Unable to generate letter to shareholders. LLM service unavailable."
            
        return f"""
### 📝 Letter to Shareholders: {ticker} Analysis
#### From the Desk of Warren Buffett

{response}

*Note: This analysis is AI-generated in the style of Warren Buffett's letters to shareholders.*
"""
    except Exception as e:
        return f"Error generating shareholder letter: {str(e)}"

def main():
    st.title("Stock Analysis App")
    
    # Initialize session state for ticker if not exists
    if 'ticker' not in st.session_state:
        st.session_state.ticker = 'AAPL'
    
    # Create tabs
    chart_tabs = st.tabs([
        "Daily Chart",
        "Historical Charts",
        "Price Prediction",
        "Buffett Analysis",
        "Stock Recommendations",
        "Market Movers",
        "Technical Analysis"
    ])
    
    # Get stock ticker input (shared across tabs)
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker:",
        value=st.session_state.ticker,
        key="sidebar_ticker"
    ).upper()
    
    # Update session state
    st.session_state.ticker = ticker
    
    if ticker:
        try:
            # Validate ticker
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                st.sidebar.error(f"Invalid ticker: {ticker}")
                return
                
            st.sidebar.success(f"Analyzing {ticker}: {info.get('longName', '')}")
            
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
                
            # Technical Analysis Tab
            with chart_tabs[6]:
                technical_analysis_tab()
                
        except Exception as e:
            st.sidebar.error(f"Error analyzing {ticker}: {str(e)}")
    else:
        st.warning("Please enter a stock ticker to begin analysis.")

if __name__ == "__main__":
    main()
