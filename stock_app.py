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
                    # Get stock data
                    data = get_stock_data(ticker)
                    if data.empty:
                        st.error(f"No data found for {ticker}")
                        return
                    
                    # Test LLM connection if selected
                    if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                        is_available, message = test_ollama_connection(model_type)
                        if not is_available:
                            st.error(message)
                            st.warning("Falling back to Random Forest model...")
                            model_type = "Random Forest"
                    
                    # Generate predictions
                    predictions = predict_stock_price(
                        data,
                        prediction_days,
                        model_type
                    )
                    
                    if predictions is not None:
                        # Plot predictions
                        confidence_bands = calculate_confidence_bands(predictions)
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

def calculate_confidence_bands(predictions, confidence=0.01):
    """Calculate confidence bands around predictions"""
    try:
        mean = np.mean(predictions)
        std = np.std(predictions)
        
        # Calculate confidence intervals
        upper_band = mean + (std * confidence)
        lower_band = mean - (std * confidence)
        
        return mean, upper_band, lower_band
    except Exception as e:
        st.error(f"Error calculating confidence bands: {str(e)}")
        return None, None, None

def predict_stock_price(ticker, model_type="Ollama 3.2"):
    """Get stock price prediction using LLM analysis"""
    try:
        # Test model availability
        is_available, message = test_ollama_connection(model_type)
        if not is_available:
            st.error(message)
            return None
            
        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        current_price = hist['Close'].iloc[-1]
        
        # Create prompt for price prediction
        prompt = f"""You are a stock market expert. Analyze {ticker} and predict:
        1. Price movement direction (UP/DOWN)
        2. Percentage change (-10% to +10%)
        3. Confidence level (0-100%)
        4. Key factors supporting prediction
        
        Current price: ${current_price:.2f}
        
        Format response exactly as JSON:
        {{
            "direction": "UP or DOWN",
            "percent_change": float between -10 and 10,
            "confidence": float between 0 and 100,
            "factors": ["factor1", "factor2", ...]
        }}
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
                # Parse JSON response
                try:
                    prediction = json.loads(result)
                    return prediction
                except json.JSONDecodeError:
                    st.error("Error parsing LLM response")
                    return None
                    
        except Exception as e:
            st.error(f"Error getting prediction: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in stock prediction: {str(e)}")
        return None

def prediction_tab():
    """Stock price prediction tab with LLM analysis"""
    try:
        st.header("🔮 Price Prediction")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Ollama 3.2", "DeepSeek-R1"],
            key="pred_model"
        )
        
        # Get prediction
        if st.button("Get Prediction", key="get_pred"):
            with st.spinner("Analyzing market data..."):
                prediction = predict_stock_price(ticker, model_type)
                
                if prediction:
                    # Display prediction summary
                    st.subheader("Price Prediction")
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        direction = prediction['direction']
                        st.metric(
                            "Direction",
                            direction,
                            delta="↑" if direction == "UP" else "↓"
                        )
                    
                    with col2:
                        change = prediction['percent_change']
                        st.metric(
                            "Expected Change",
                            f"{change:+.2f}%",
                            delta=change,
                            delta_color="normal"
                        )
                    
                    with col3:
                        confidence = prediction['confidence']
                        st.metric(
                            "Confidence",
                            f"{confidence:.1f}%"
                        )
                    
                    # Calculate confidence bands
                    stock = yf.Ticker(ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    predicted_change = prediction['percent_change'] / 100
                    predicted_price = current_price * (1 + predicted_change)
                    
                    # Generate range of predictions
                    predictions = np.random.normal(
                        predicted_price,
                        current_price * 0.01,  # 1% standard deviation
                        1000
                    )
                    
                    mean, upper, lower = calculate_confidence_bands(predictions)
                    
                    # Display price predictions
                    st.subheader("Price Range Prediction")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Lower Bound",
                            f"${lower:.2f}",
                            delta=f"{((lower/current_price - 1) * 100):.1f}%",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Target Price",
                            f"${mean:.2f}",
                            delta=f"{((mean/current_price - 1) * 100):.1f}%",
                            delta_color="normal"
                        )
                    
                    with col3:
                        st.metric(
                            "Upper Bound",
                            f"${upper:.2f}",
                            delta=f"{((upper/current_price - 1) * 100):.1f}%",
                            delta_color="normal"
                        )
                    
                    # Display supporting factors
                    st.subheader("Analysis Factors")
                    for factor in prediction['factors']:
                        st.write(f"• {factor}")
                    
                    # Add prediction chart
                    fig = go.Figure()
                    
                    # Add current price line
                    fig.add_hline(
                        y=current_price,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Current Price"
                    )
                    
                    # Add prediction range
                    fig.add_hrect(
                        y0=lower,
                        y1=upper,
                        fillcolor="lightblue",
                        opacity=0.2,
                        line_width=0
                    )
                    
                    # Add target price line
                    fig.add_hline(
                        y=mean,
                        line_color="blue",
                        annotation_text="Target Price"
                    )
                    
                    fig.update_layout(
                        title="Price Prediction Range",
                        yaxis_title="Price ($)",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    with st.expander("About Price Predictions"):
                        st.write("""
                        This prediction is based on:
                        1. **Technical Analysis**: Price patterns and indicators
                        2. **Market Sentiment**: News and social media analysis
                        3. **Historical Data**: Past price movements and volatility
                        4. **Confidence Bands**: Represent potential price range
                        
                        Note: All predictions are estimates and should not be the sole basis for investment decisions.
                        """)
                        
    except Exception as e:
        st.error(f"Error in prediction tab: {str(e)}")

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
                return float(value[0]) if isinstance(value, list) else float(value)
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
        fcf = safe_get(financials, 'Free Cash Flow')
        fcf_yield = safe_divide(fcf, market_cap) * 100 if market_cap > 0 else 0
        
        # Calculate intrinsic value using simple DCF with safety checks
        growth_rate = max(min(safe_get(info, 'revenueGrowth', 0.05), 0.15), 0)  # Cap between 0% and 15%
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
        
        # Calculate margin of safety with check for positive intrinsic value
        margin_of_safety = safe_divide((intrinsic_value - current_price), intrinsic_value) * 100 if intrinsic_value > 0 else -100
        
        # Get company info with defaults
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        business_summary = info.get('longBusinessSummary', 'No business summary available.')

        # Collect metrics for write-up
        metrics = {
            'operating_margin': float(operating_margin),
            'roe': float(roe),
            'gross_margin': float(gross_margin),
            'current_ratio': float(current_ratio),
            'debt_to_equity': float(debt_to_equity),
            'fcf_yield': float(fcf_yield),
            'pe_ratio': float(pe_ratio),
            'margin_of_safety': float(margin_of_safety),
            'current_price': float(current_price),
            'market_cap': float(market_cap),
            'profit_margin': float(profit_margin)
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
   | Gross Margin | {gross_margin:.1f}% | Shows pricing power and brand strength | {'Strong' if gross_margin > 40 else 'Good' if gross_margin > 30 else 'Fair' if gross_margin > 20 else 'Weak'} |
   | Profit Margin | {profit_margin:.1f}% | Net profit per dollar of revenue | {'Excellent' if profit_margin > 20 else 'Good' if profit_margin > 15 else 'Fair' if profit_margin > 10 else 'Poor'} |

2. **Financial Health**
   | Metric | Value | Meaning | Assessment |
   |--------|--------|---------|------------|
   | Debt/Equity | {debt_to_equity:.1f} | Measures financial leverage | {'Conservative' if debt_to_equity < 50 else 'Moderate' if debt_to_equity < 100 else 'High'} |
   | Current Ratio | {current_ratio:.1f} | Shows ability to pay short-term obligations | {'Strong' if current_ratio > 2 else 'Adequate' if current_ratio > 1.5 else 'Weak'} |
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

#### 💡 Investment Recommendation
"""
        # Generate recommendation and detailed analysis
        if roe > 15 and debt_to_equity < 50 and margin_of_safety > 30:
            analysis += f"""
**Strong Buy** 📈

{ticker} presents a compelling investment opportunity with strong fundamentals and significant upside potential:

1. **Superior Profitability**
   - Industry-leading ROE of {roe:.1f}%, indicating excellent capital allocation
   - Strong operating margins at {operating_margin:.1f}%, suggesting pricing power
   - Healthy gross margins of {gross_margin:.1f}%, reflecting brand strength

2. **Robust Financial Position**
   - Conservative debt levels with debt/equity at {debt_to_equity:.1f}
   - Strong liquidity with current ratio at {current_ratio:.1f}
   - Attractive FCF yield of {fcf_yield:.1f}%, indicating quality earnings

3. **Attractive Valuation**
   - Significant margin of safety at {margin_of_safety:.1f}%
   - Forward P/E of {pe_ratio:.1f}x, below industry average
   - Market cap of ${market_cap/1e9:.1f}B suggests stability with room for growth

#### 📊 Investment Strategy
1. Entry Strategy:
   - Current Price: ${current_price:.2f}
   - Recommended Entry: Up to ${current_price * 1.15:.2f}
   - Ideal Position Size: 5-7% of portfolio

2. Risk Management:
   - Set stop loss at ${current_price * 0.85:.2f}
   - Monitor quarterly for maintenance of ROE and margins
   - Review if debt/equity exceeds 80%"""

        elif roe > 12 and debt_to_equity < 80 and margin_of_safety > 15:
            analysis += f"""
**Buy** 🔼

{ticker} shows promising potential with solid fundamentals and reasonable valuation:

1. **Good Profitability**
   - Above-average ROE of {roe:.1f}%, showing good capital efficiency
   - Competitive operating margins at {operating_margin:.1f}%
   - Sustainable gross margins of {gross_margin:.1f}%

2. **Sound Financial Health**
   - Manageable debt with debt/equity at {debt_to_equity:.1f}
   - Adequate liquidity position with current ratio at {current_ratio:.1f}
   - Positive FCF yield of {fcf_yield:.1f}%

3. **Fair Valuation**
   - Reasonable margin of safety at {margin_of_safety:.1f}%
   - Forward P/E of {pe_ratio:.1f}x suggests fair value
   - Market cap of ${market_cap/1e9:.1f}B indicates established market position

#### 📊 Investment Strategy
1. Entry Strategy:
   - Current Price: ${current_price:.2f}
   - Recommended Entry: Up to ${current_price * 1.10:.2f}
   - Suggested Position Size: 3-5% of portfolio

2. Risk Management:
   - Set stop loss at ${current_price * 0.90:.2f}
   - Quarterly review of financial metrics
   - Monitor competitive position"""

        elif roe > 10 and debt_to_equity < 100 and margin_of_safety > 0:
            analysis += f"""
**Hold/Accumulate** ⏺

{ticker} presents a balanced risk-reward profile with some concerns:

1. **Moderate Profitability**
   - Acceptable ROE of {roe:.1f}%, but room for improvement
   - Operating margins of {operating_margin:.1f}% near industry average
   - Gross margins of {gross_margin:.1f}% suggest moderate competitive position

2. **Mixed Financial Health**
   - Elevated debt levels with debt/equity at {debt_to_equity:.1f}
   - Adequate liquidity with current ratio at {current_ratio:.1f}
   - FCF yield of {fcf_yield:.1f}% indicates moderate cash generation

3. **Neutral Valuation**
   - Limited margin of safety at {margin_of_safety:.1f}%
   - Forward P/E of {pe_ratio:.1f}x near fair value
   - Market cap of ${market_cap/1e9:.1f}B suggests established presence

#### 📊 Investment Strategy
1. Position Management:
   - Hold existing positions
   - Consider adding below ${current_price * 0.90:.2f}
   - Keep position size below 3% of portfolio

2. Risk Management:
   - Set stop loss at ${current_price * 0.85:.2f}
   - Regular review of fundamentals
   - Consider selling if metrics deteriorate"""

        else:
            analysis += f"""
**Sell/Avoid** 🔻

{ticker} shows several concerning factors that suggest elevated investment risk:

1. **Weak Profitability**
   - Below-average ROE of {roe:.1f}%, indicating poor capital efficiency
   - Operating margins of {operating_margin:.1f}% suggest competitive pressures
   - Gross margins of {gross_margin:.1f}% reflect limited pricing power

2. **Financial Concerns**
   - High debt levels with debt/equity at {debt_to_equity:.1f}
   - Liquidity concerns with current ratio at {current_ratio:.1f}
   - FCF yield of {fcf_yield:.1f}% indicates cash flow pressure

3. **Unattractive Valuation**
   - Negative margin of safety at {margin_of_safety:.1f}%
   - Forward P/E of {pe_ratio:.1f}x suggests overvaluation
   - Market cap of ${market_cap/1e9:.1f}B may limit downside protection

#### 📊 Investment Strategy
1. Action Items:
   - Avoid new positions
   - Consider selling existing holdings
   - Look for better opportunities

2. Exit Strategy:
   - Sell above ${current_price * 1.05:.2f}
   - Use any strength to reduce positions
   - Reallocate to stronger opportunities"""

        analysis += """

#### 📘 Key Investment Principles
1. **Circle of Competence**: Stay within businesses you understand
2. **Margin of Safety**: Always buy at a discount to intrinsic value
3. **Economic Moat**: Focus on sustainable competitive advantages
4. **Financial Strength**: Prefer conservative financial management
5. **Long-term Perspective**: Invest in businesses, not stocks
"""
        return analysis
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return """
### ⚠️ Analysis Error

Unable to generate complete analysis due to missing or invalid data. This could be due to:
1. Limited financial data availability
2. Recent IPO or spinoff
3. Complex corporate structure
4. Temporary data provider issues

Please verify the ticker symbol and try again later.
"""

def get_buffett_writeup(ticker, metrics):
    """Generate Warren Buffett style investment write-up"""
    
    def get_moat_assessment(metrics):
        moat_strength = "Strong" if metrics['operating_margin'] > 20 and metrics['roe'] > 15 else \
                       "Moderate" if metrics['operating_margin'] > 15 and metrics['roe'] > 12 else "Limited"
        
        moat_factors = []
        if metrics['operating_margin'] > 20:
            moat_factors.append("high operating margins indicating pricing power")
        if metrics['roe'] > 15:
            moat_factors.append("excellent return on equity showing competitive advantages")
        if metrics['gross_margin'] > 40:
            moat_factors.append("strong gross margins suggesting brand value")
        if metrics['market_cap'] > 50e9:
            moat_factors.append("significant market position")
            
        return moat_strength, moat_factors

    def get_financial_health(metrics):
        if metrics['current_ratio'] > 2 and metrics['debt_to_equity'] < 50:
            health = "Excellent"
        elif metrics['current_ratio'] > 1.5 and metrics['debt_to_equity'] < 80:
            health = "Good"
        else:
            health = "Fair"
                
        strengths = []
        if metrics['current_ratio'] > 2:
            strengths.append("strong liquidity position")
        if metrics['debt_to_equity'] < 50:
            strengths.append("conservative debt management")
        if metrics['fcf_yield'] > 5:
            strengths.append("healthy free cash flow generation")
            
        return health, strengths

    def get_valuation_assessment(metrics):
        if metrics['pe_ratio'] < 15 and metrics['margin_of_safety'] > 25:
            value = "Attractive"
        elif metrics['pe_ratio'] < 20 and metrics['margin_of_safety'] > 15:
            value = "Fair"
        else:
            value = "Expensive"
                
        reasons = []
        if metrics['pe_ratio'] < 15:
            reasons.append("reasonable P/E ratio")
        if metrics['margin_of_safety'] > 25:
            reasons.append("significant margin of safety")
        if metrics['fcf_yield'] > 5:
            reasons.append("attractive free cash flow yield")
            
        return value, reasons

    moat_strength, moat_factors = get_moat_assessment(metrics)
    health, financial_strengths = get_financial_health(metrics)
    value, value_reasons = get_valuation_assessment(metrics)
    
    # Determine overall investment attractiveness
    is_compelling = (moat_strength == "Strong" and 
                    health in ("Excellent", "Good") and 
                    value in ("Attractive", "Fair"))
    is_potential = moat_strength != "Limited"
    
    writeup = f"""
### 📝 Warren Buffett's Investment Write-up for {ticker}

#### 🎯 Executive Summary
As a value investor focused on high-quality businesses at reasonable prices, {ticker} presents a {'compelling' if is_compelling else 'potential' if is_potential else 'challenging'} investment opportunity.

#### 🏰 Economic Moat Analysis
The company demonstrates a {moat_strength.lower()} economic moat, evidenced by {', '.join(moat_factors) if moat_factors else 'limited competitive advantages'}. {
'This sustainable competitive advantage positions the company well for long-term value creation.' if moat_strength == 'Strong' else 
'While showing some competitive strengths, the moat needs further development.' if moat_strength == 'Moderate' else
'The lack of a strong economic moat raises concerns about long-term profitability.'
}

#### 💪 Business Strengths
1. **Profitability Metrics**
   | Metric | Value | Meaning | Assessment |
   |--------|--------|---------|------------|
   | Return on Equity (ROE) | {metrics['roe']:.1f}% | Measures how efficiently company uses shareholder money | {'Excellent' if metrics['roe'] > 15 else 'Good' if metrics['roe'] > 12 else 'Fair' if metrics['roe'] > 10 else 'Poor'} |
   | Operating Margin | {metrics['operating_margin']:.1f}% | Indicates pricing power and operational efficiency | {'Excellent' if metrics['operating_margin'] > 25 else 'Good' if metrics['operating_margin'] > 15 else 'Fair' if metrics['operating_margin'] > 10 else 'Poor'} |
   | Gross Margin | {metrics['gross_margin']:.1f}% | Shows pricing power and brand strength | {'Strong' if metrics['gross_margin'] > 40 else 'Good' if metrics['gross_margin'] > 30 else 'Fair' if metrics['gross_margin'] > 20 else 'Weak'} |
   | Profit Margin | {metrics['profit_margin']:.1f}% | Net profit per dollar of revenue | {'Excellent' if metrics['profit_margin'] > 20 else 'Good' if metrics['profit_margin'] > 15 else 'Fair' if metrics['profit_margin'] > 10 else 'Poor'} |

2. **Financial Position**
The company's {health.lower()} financial health is supported by {', '.join(financial_strengths) if financial_strengths else 'adequate financial metrics'}. {
'This strong financial foundation provides flexibility for future growth.' if health == 'Excellent' else
'The financial position is stable but could be improved.' if health == 'Good' else
'Financial metrics warrant careful monitoring.'
}

#### 💰 Valuation Assessment
At the current price of ${metrics['current_price']:.2f}, the valuation appears {value.lower()}, with {', '.join(value_reasons) if value_reasons else 'limited value characteristics'}. {
'This presents an attractive entry point for long-term investors.' if value == 'Attractive' else
'The price offers a reasonable balance of risk and reward.' if value == 'Fair' else
'The current valuation leaves limited room for error.'
}

#### 🎯 Investment Recommendation
"""
    # Generate specific recommendation based on metrics
    if moat_strength == "Strong" and health == "Excellent" and value == "Attractive":
        writeup += f"""
**Strong Buy** 📈
1. Entry Strategy:
   - Current Price: ${metrics['current_price']:.2f}
   - Recommended Entry: Up to ${metrics['current_price'] * 1.15:.2f}
   - Ideal Position Size: 5-7% of portfolio

2. Investment Thesis:
   - Strong economic moat with {', '.join(moat_factors)}
   - {health} financial health supported by {', '.join(financial_strengths)}
   - {value} valuation with {', '.join(value_reasons)}

3. Risk Management:
   - Set stop loss at ${metrics['current_price'] * 0.85:.2f}
   - Monitor quarterly for maintenance of operating margins and ROE
   - Review if debt/equity exceeds 80%"""

    elif moat_strength != "Limited" and health != "Fair" and value != "Expensive":
        writeup += f"""
**Buy with Caution** 🔼
1. Entry Strategy:
   - Current Price: ${metrics['current_price']:.2f}
   - Recommended Entry: Up to ${metrics['current_price'] * 1.10:.2f}
   - Suggested Position Size: 3-5% of portfolio

2. Investment Thesis:
   - {moat_strength} moat characteristics
   - {health} financial position
   - {value} current valuation

3. Risk Management:
   - Set stop loss at ${metrics['current_price'] * 0.90:.2f}
   - Quarterly review of competitive position
   - Monitor financial metrics for deterioration"""

    elif metrics['margin_of_safety'] > 0:
        writeup += f"""
**Hold/Accumulate** ⏺
1. Position Management:
   - Hold existing positions
   - Consider adding below ${metrics['current_price'] * 0.90:.2f}
   - Keep position size below 3% of portfolio

2. Monitoring Points:
   - Watch for moat strengthening
   - Monitor financial health improvements
   - Look for better entry points

3. Risk Management:
   - Set stop loss at ${metrics['current_price'] * 0.85:.2f}
   - Regular review of business fundamentals
   - Consider selling if metrics deteriorate"""

    else:
        writeup += f"""
**Sell/Avoid** 🔻
1. Action Items:
   - Avoid new positions
   - Consider selling existing holdings
   - Look for better opportunities

2. Concerns:
   - Limited economic moat
   - Challenging financial metrics
   - Unattractive valuation

3. Exit Strategy:
   - Sell above ${metrics['current_price'] * 1.05:.2f}
   - Use any strength to reduce positions
   - Reallocate to stronger opportunities"""

    writeup += """

#### 📘 Key Principles Applied
1. **Circle of Competence**: Stay within businesses you understand
2. **Margin of Safety**: Always buy at a discount to intrinsic value
3. **Economic Moat**: Focus on sustainable competitive advantages
4. **Financial Strength**: Prefer conservative financial management
5. **Long-term Perspective**: Invest in businesses, not stocks
"""
    
    return writeup

def get_buffett_analysis(ticker, financials, info):
    """Generate Warren Buffett style analysis for a stock"""
    try:
        # Safely get values with defaults
        def safe_get(data, key, default=0):
            try:
                value = data.get(key, default)
                return float(value[0]) if isinstance(value, list) else float(value)
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
        fcf = safe_get(financials, 'Free Cash Flow')
        fcf_yield = safe_divide(fcf, market_cap) * 100 if market_cap > 0 else 0
        
        # Calculate intrinsic value using simple DCF with safety checks
        growth_rate = max(min(safe_get(info, 'revenueGrowth', 0.05), 0.15), 0)  # Cap between 0% and 15%
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
        
        # Calculate margin of safety with check for positive intrinsic value
        margin_of_safety = safe_divide((intrinsic_value - current_price), intrinsic_value) * 100 if intrinsic_value > 0 else -100
        
        # Get company info with defaults
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        business_summary = info.get('longBusinessSummary', 'No business summary available.')

        # Collect metrics for write-up
        metrics = {
            'operating_margin': float(operating_margin),
            'roe': float(roe),
            'gross_margin': float(gross_margin),
            'current_ratio': float(current_ratio),
            'debt_to_equity': float(debt_to_equity),
            'fcf_yield': float(fcf_yield),
            'pe_ratio': float(pe_ratio),
            'margin_of_safety': float(margin_of_safety),
            'current_price': float(current_price),
            'market_cap': float(market_cap),
            'profit_margin': float(profit_margin)
        }
        
        # Generate main analysis and write-up
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
   | Gross Margin | {gross_margin:.1f}% | Shows pricing power and brand strength | {'Strong' if gross_margin > 40 else 'Good' if gross_margin > 30 else 'Fair' if gross_margin > 20 else 'Weak'} |
   | Profit Margin | {profit_margin:.1f}% | Net profit per dollar of revenue | {'Excellent' if profit_margin > 20 else 'Good' if profit_margin > 15 else 'Fair' if profit_margin > 10 else 'Poor'} |

2. **Financial Health**
   | Metric | Value | Meaning | Assessment |
   |--------|--------|---------|------------|
   | Debt/Equity | {debt_to_equity:.1f} | Measures financial leverage | {'Conservative' if debt_to_equity < 50 else 'Moderate' if debt_to_equity < 100 else 'High'} |
   | Current Ratio | {current_ratio:.1f} | Shows ability to pay short-term obligations | {'Strong' if current_ratio > 2 else 'Adequate' if current_ratio > 1.5 else 'Weak'} |
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
"""
        
        # Add detailed write-up
        analysis += get_buffett_writeup(ticker, metrics)
        
        return analysis
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return """
### ⚠️ Analysis Error

Unable to generate complete analysis due to missing or invalid data. This could be due to:
1. Limited financial data availability
2. Recent IPO or spinoff
3. Complex corporate structure
4. Temporary data provider issues

Please verify the ticker symbol and try again later.
"""

def buffett_analysis_tab():
    """Display Warren Buffett style analysis tab"""
    st.header("🎩 Warren Buffett Analysis")
    
    # Get user input
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
    with col2:
        st.write("")  # Spacing
        analyze_button = st.button("Analyze Stock", type="primary")
    with col3:
        st.write("")  # Spacing
        st.caption("'Price is what you pay. Value is what you get.' - Warren Buffett")
        
    if analyze_button and ticker:
        try:
            # Show loading message
            with st.spinner(f"Analyzing {ticker} through Warren Buffett's lens..."):
                # Get stock data
                stock = yf.Ticker(ticker)
                info = stock.info
                financials = stock.financials.to_dict()
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["📝 Analysis", "📈 Performance"])
                
                with tab1:
                    # Generate and display analysis
                    analysis = get_buffett_analysis(ticker, financials, info)
                    st.markdown(analysis)
                    
                with tab2:
                    # Show historical performance
                    st.subheader("Historical Performance")
                    hist = stock.history(period="5y")
                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            name='Stock Price',
                            line=dict(color='#2E86C1')
                        ))
                        fig.update_layout(
                            title=f"{ticker} 5-Year Price History",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            showlegend=True,
                            height=400,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add key statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                            st.metric("5-Year Return", f"{price_change:.1f}%")
                        with col2:
                            max_price = hist['High'].max()
                            st.metric("5-Year High", f"${max_price:.2f}")
                        with col3:
                            min_price = hist['Low'].min()
                            st.metric("5-Year Low", f"${min_price:.2f}")
                
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            st.info("Please check the ticker symbol and try again.")
            
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
                "Select Time Period:",
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
                    delta = hist_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                    rs = gain / loss
                    hist_data['RSI'] = 100 - (100 / (1 + rs))
                
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

def main():
    st.title("Stock Analysis App")
    
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
            
        # Technical Analysis Tab
        with chart_tabs[6]:
            technical_analysis_tab()
            
    else:
        st.warning("Please enter a stock ticker to begin analysis.")

if __name__ == "__main__":
    main()
