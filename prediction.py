"""Stock price prediction module for StockPro"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
import plotly.graph_objects as go

def test_ollama_connection(model_type="Ollama 3.2"):
    """Test connection to Ollama server"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            return True, "Connected to Ollama server"
        else:
            return False, "Could not connect to Ollama server"
    except Exception as e:
        return False, str(e)

def calculate_rsi(data):
    """Calculate RSI"""
    delta = data['Close'].diff(1)
    up_days = delta.copy()
    up_days[delta <= 0] = 0
    down_days = abs(delta.copy())
    down_days[delta > 0] = 0
    RS_up = up_days.ewm(com=13 - 1, min_periods=13).mean()
    RS_down = down_days.ewm(com=13 - 1, min_periods=13).mean().add(0.000001)
    return 100.0 - (100.0 / (1.0 + RS_up / RS_down))

def calculate_macd(data):
    """Calculate MACD"""
    macd_df = pd.DataFrame()
    macd_df['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    macd_df['Signal'] = macd_df['MACD'].ewm(span=9, adjust=False).mean()
    return macd_df

def predict_with_llm(data, prediction_days, model_type):
    """Predict with LLM"""
    current_price = data['Close'].iloc[-1]
    hist_prices = data['Close'].tail(30).tolist()
    
    prompt = f"""You are a stock market expert. Analyze this data and predict the stock price movement:
    Current Price: ${current_price:.2f}
    Last 30 days closing prices: {hist_prices}
    
    Based on this data, predict:
    1. Price movement direction (UP/DOWN)
    2. Percentage change (-10% to +10%)
    3. Confidence level (0-100%)
    
    Format your response as JSON:
    {{
        "direction": "UP or DOWN",
        "percentage": float,
        "confidence": int
    }}
    """
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = json.loads(response.json()['response'])
            pct_change = result['percentage']
            predictions = []
            
            for day in range(prediction_days):
                if day == 0:
                    pred = current_price * (1 + pct_change/100)
                else:
                    # Decay the effect for future days
                    decay = (prediction_days - day) / prediction_days
                    daily_change = (pct_change/100) * decay
                    pred = predictions[-1] * (1 + daily_change)
                predictions.append(pred)
            
            return pd.Series(predictions, index=pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='B'
            ))
        else:
            st.error("Error connecting to LLM service")
            return None
            
    except Exception as e:
        st.error(f"Error in LLM prediction: {str(e)}")
        return None

def predict_stock_price(data, prediction_days=7, model_type="Random Forest"):
    """Predict stock prices using various models"""
    try:
        # Prepare data
        df = data.copy()
        df['Target'] = df['Close'].shift(-prediction_days)
        
        # Add technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df)
        macd_data = calculate_macd(df)
        df['MACD'] = macd_data['MACD']
        df['Signal'] = macd_data['Signal']
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        # Create features
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA20', 'MA50', 'RSI', 'MACD', 'Signal', 'Volume_MA20'
        ]
        
        X = df[features].values
        y = df['Target'].values
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:-prediction_days]
        y_train, y_test = y[:split], y[split:-prediction_days]
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_pred = scaler.transform(X[-prediction_days:])
        
        # Select and train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
        elif model_type == "XGBoost":
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
        elif model_type == "LightGBM":
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
        elif model_type == "Prophet":
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close']
            })
            
            # Create and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True
            )
            model.fit(prophet_df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)
            
            # Return only the predictions
            return pd.Series(
                forecast['yhat'].tail(prediction_days).values,
                index=pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1),
                    periods=prediction_days,
                    freq='B'
                )
            )
            
        elif model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            # Use LLM for prediction
            return predict_with_llm(df, prediction_days, model_type)
        
        # Make predictions
        predictions = model.predict(X_pred)
        
        # Create prediction series with future dates
        return pd.Series(
            predictions,
            index=pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='B'
            )
        )
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def plot_predictions(data, predictions):
    """Plot stock predictions"""
    try:
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
            x=predictions.index,
            y=predictions,
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting predictions: {str(e)}")

def display_prediction_results(data, predictions, ticker, model_type):
    """Display prediction results"""
    try:
        # Display current stock info
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        price_change = ((current_price - prev_close) / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f}%"
            )
        
        with col2:
            st.metric(
                "52-Week High",
                f"${data['High'].max():.2f}"
            )
        
        with col3:
            st.metric(
                "52-Week Low",
                f"${data['Low'].min():.2f}"
            )
        
        # Plot predictions
        plot_predictions(data, predictions)
        
        # Show prediction details
        final_price = predictions.iloc[-1]
        price_change = ((final_price - current_price) / current_price) * 100
        
        st.subheader("Prediction Summary")
        st.write(f"Current Price: ${current_price:.2f}")
        st.write(f"Predicted Price ({len(predictions)} days): ${final_price:.2f}")
        st.write(f"Predicted Change: {price_change:+.2f}%")
        
        # Confidence level
        if model_type == "Random Forest":
            st.write("Model: Random Forest (based on technical analysis)")
        else:
            st.write(f"Model: {model_type} (based on technical and sentiment analysis)")
    
    except Exception as e:
        st.error(f"Error displaying prediction results: {str(e)}")

def prediction_tab():
    """Stock price prediction tab"""
    try:
        st.header("🔮 Stock Price Prediction")
        
        # Get ticker from session state
        ticker = st.session_state.get('current_ticker', '')
        if not ticker:
            st.warning("Please enter a stock ticker above.")
            return
            
        # User inputs
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            prediction_days = st.number_input(
                "Prediction Days:",
                min_value=1,
                max_value=30,
                value=7,
                key="pred_days"
            )
        
        with col3:
            model_type = st.selectbox(
                "Select Model:",
                [
                    "Random Forest",
                    "Ollama 3.2",
                    "DeepSeek-R1",
                    "XGBoost",
                    "LightGBM",
                    "Prophet"
                ],
                index=0,
                key="pred_model"
            )
            
        # Model explanation
        with st.expander("ℹ️ Model Information"):
            if model_type == "Random Forest":
                st.markdown("""
                    **Random Forest Model**
                    - Ensemble learning method for prediction
                    - Combines multiple decision trees
                    - Good at handling non-linear relationships
                    - Less prone to overfitting
                """)
            elif model_type == "Ollama 3.2":
                st.markdown("""
                    **Ollama 3.2 LLM**
                    - Large Language Model for market analysis
                    - Considers technical and fundamental factors
                    - Provides detailed price movement rationale
                    - Includes confidence bands for predictions
                """)
            elif model_type == "DeepSeek-R1":
                st.markdown("""
                    **DeepSeek-R1 Model**
                    - Advanced language model for financial analysis
                    - Specialized in market pattern recognition
                    - Considers global market conditions
                    - Provides comprehensive market analysis
                """)
            elif model_type == "XGBoost":
                st.markdown("""
                    **XGBoost Model**
                    - Gradient boosting algorithm
                    - Excellent performance on structured data
                    - Handles missing values well
                    - Regularization to prevent overfitting
                """)
            elif model_type == "LightGBM":
                st.markdown("""
                    **LightGBM Model**
                    - Light Gradient Boosting Machine
                    - Fast training speed
                    - High efficiency with large datasets
                    - Good handling of categorical features
                """)
            elif model_type == "Prophet":
                st.markdown("""
                    **Facebook Prophet**
                    - Specialized in time series forecasting
                    - Handles seasonality and holidays
                    - Robust to missing data and outliers
                    - Good for daily and weekly patterns
                """)
        
        if st.button("Generate Prediction", key="generate_prediction"):
            with st.spinner("Fetching data and generating prediction..."):
                try:
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="1y")
                    
                    if data.empty:
                        st.error(f"Could not fetch data for {ticker}")
                        return
                    
                    # Generate prediction
                    if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                        # Check if LLM is available
                        if not test_ollama_connection():
                            st.warning("LLM server not available. Falling back to Random Forest model.")
                            prediction = predict_stock_price(data, prediction_days, "Random Forest")
                        else:
                            prediction = predict_stock_price(data, prediction_days, model_type)
                    else:
                        prediction = predict_stock_price(data, prediction_days, model_type)
                    
                    # Display results
                    display_prediction_results(data, prediction, ticker, model_type)
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
