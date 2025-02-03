import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt
import requests
import json
import logging
import traceback
import time
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
import matplotlib.pyplot as plt
from utils import (
    calculate_rsi, format_large_number, format_percentage,
    calculate_delta, calculate_gamma, calculate_theta, calculate_vega,
    calculate_macd, calculate_bollinger_bands, safe_divide,
    calculate_growth_rate, calculate_intrinsic_value
)
from company_profile import company_profile_tab
from technical_analysis import technical_analysis_tab
from prediction import prediction_tab
from buffett_analysis import buffett_analysis_tab
from options_analysis import options_analysis_tab
from market_movers import market_movers_tab
from fundamental_analysis import fundamental_analysis_tab
from portfolio_analysis import portfolio_analysis_tab
from stock_recommendations import stock_recommendations_tab
from stock_chat import stock_chat_tab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        logger.error(f"Error fetching stock data: {str(e)}")
        return None

def predict_stock_price(data, prediction_days=7, model_type="Ollama 3.2"):
    """Predict stock prices using various models"""
    try:
        # Prepare data
        df = data.copy()
        df['Target'] = df['Close'].shift(-prediction_days)
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Volume MA
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < prediction_days:
            st.error("Not enough data for prediction")
            return None
            
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
            predictions = model.predict(X_pred)
            
        elif model_type == "XGBoost":
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_pred)
            
        elif model_type == "LightGBM":
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_pred)
            
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
            predictions = forecast['yhat'].tail(prediction_days).values
            
        elif model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            # Use LLM for prediction
            current_price = df['Close'].iloc[-1]
            hist_prices = df['Close'].tail(30).tolist()
            price_changes = df['Close'].pct_change().tail(30).tolist()
            avg_volume = df['Volume'].tail(30).mean()
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            signal = df['Signal'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            
            prompt = f"""You are a stock market expert. Analyze this data and predict the stock price movement:
            Current Price: ${current_price:.2f}
            Technical Indicators:
            - RSI ({rsi:.2f}): {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}
            - MACD: {macd:.2f} (Signal: {signal:.2f})
            - 20-day MA: ${ma20:.2f} ({'Above' if current_price > ma20 else 'Below'} price)
            - 50-day MA: ${ma50:.2f} ({'Above' if current_price > ma50 else 'Below'} price)
            - Average Volume: {avg_volume:,.0f}
            
            Recent Performance:
            - Last 30 days price changes: {[f'{x*100:.1f}%' for x in price_changes if not pd.isna(x)]}
            
            Based on this technical analysis, predict:
            1. Price movement direction (UP/DOWN)
            2. Percentage change (-10% to +10%)
            3. Confidence level (0-100%)
            
            Format your response EXACTLY as JSON:
            {{
                "direction": "UP or DOWN",
                "percentage": float,
                "confidence": int
            }}
            
            ONLY return the JSON object, nothing else.
            """
            
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        "model": "llama2" if model_type == "Ollama 3.2" else "deepseek-coder",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = json.loads(response.json()['response'])
                    pct_change = float(result['percentage'])
                    
                    # Clamp the percentage change to [-10, 10]
                    pct_change = max(min(pct_change, 10), -10)
                    
                    # Generate predictions
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
                    
                    predictions = np.array(predictions)
                    
                    # Show the LLM's analysis
                    st.info(f"""
                    **{model_type} Analysis**
                    - Direction: {result['direction']}
                    - Predicted Change: {pct_change:.1f}%
                    - Confidence: {result['confidence']}%
                    """)
                    
                else:
                    st.warning(f"Error connecting to {model_type} service. Falling back to Random Forest model.")
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_pred)
                    
            except Exception as e:
                st.warning(f"Error in {model_type} prediction: {str(e)}. Falling back to Random Forest model.")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_pred)
        
        # Create prediction series with future dates
        prediction_series = pd.Series(
            predictions,
            index=pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='B'
            )
        )
        
        return prediction_series
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

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
                    "Ollama 3.2",
                    "DeepSeek-R1",
                    "Random Forest",
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
                        try:
                            response = requests.get('http://localhost:11434/api/tags')
                            if response.status_code != 200:
                                st.warning("LLM server not available. Falling back to Random Forest model.")
                                prediction = predict_stock_price(data, prediction_days, "Random Forest")
                            else:
                                prediction = predict_stock_price(data, prediction_days, model_type)
                        except:
                            st.warning("LLM server not available. Falling back to Random Forest model.")
                            prediction = predict_stock_price(data, prediction_days, "Random Forest")
                    else:
                        prediction = predict_stock_price(data, prediction_days, model_type)
                    
                    if prediction is not None:
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
                        
                        # Plot predictions using Altair
                        # Prepare data for plotting
                        hist_df = pd.DataFrame({
                            'Date': data.index,
                            'Price': data['Close'].values,
                            'Type': 'Historical'
                        })
                        
                        pred_df = pd.DataFrame({
                            'Date': prediction.index,
                            'Price': prediction.values,
                            'Type': 'Prediction'
                        })
                        
                        # Combine historical and prediction data
                        plot_df = pd.concat([hist_df, pred_df])
                        
                        # Create base chart
                        base = alt.Chart(plot_df).encode(
                            x=alt.X('Date:T', axis=alt.Axis(title='Date', format='%Y-%m-%d')),
                            y=alt.Y('Price:Q', axis=alt.Axis(title='Price ($)', format='$.2f')),
                            color=alt.Color('Type:N', scale=alt.Scale(
                                domain=['Historical', 'Prediction'],
                                range=['#1f77b4', '#ff7f0e']
                            ))
                        )
                        
                        # Create line chart
                        lines = base.mark_line().encode(
                            strokeDash=alt.condition(
                                alt.datum.Type == 'Prediction',
                                alt.value([5, 5]),  # dashed line for predictions
                                alt.value([0])      # solid line for historical
                            )
                        )
                        
                        # Create points for better interactivity
                        points = base.mark_circle(size=100).encode(
                            opacity=alt.value(0),
                            tooltip=[
                                alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                                alt.Tooltip('Price:Q', title='Price', format='$.2f'),
                                alt.Tooltip('Type:N', title='Type')
                            ]
                        )
                        
                        # Combine line and points
                        chart = (lines + points).properties(
                            width=700,
                            height=400,
                            title=f"{ticker} Stock Price Prediction"
                        ).configure_axis(
                            labelFontSize=12,
                            titleFontSize=14
                        ).configure_title(
                            fontSize=16,
                            anchor='middle'
                        ).configure_legend(
                            labelFontSize=12,
                            titleFontSize=14
                        ).interactive()
                        
                        # Display chart
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Show prediction details
                        final_price = prediction.iloc[-1]
                        price_change = ((final_price - current_price) / current_price) * 100
                        
                        st.subheader("Prediction Summary")
                        st.write(f"Current Price: ${current_price:.2f}")
                        st.write(f"Predicted Price ({prediction_days} days): ${final_price:.2f}")
                        st.write(f"Predicted Change: {price_change:+.2f}%")
                        
                        # Generate CIO Letter
                        if prediction is not None and len(prediction) > 0 and model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                            st.subheader("🎯 CIO Investment Letter")
                            
                            try:
                                # Get current price and predicted price
                                current_price = float(data['Close'].iloc[-1])
                                final_price = float(prediction.iloc[-1])
                                price_change = ((final_price - current_price) / current_price) * 100
                                
                                # Calculate technical indicators
                                delta = data['Close'].diff()
                                gain = (delta.where(delta > 0, 0)).fillna(0)
                                loss = (-delta.where(delta < 0, 0)).fillna(0)
                                avg_gain = gain.rolling(window=14).mean()
                                avg_loss = loss.rolling(window=14).mean()
                                rs = avg_gain / avg_loss
                                data['RSI'] = 100 - (100 / (1 + rs))
                                
                                # Calculate MACD
                                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                                data['MACD'] = exp1 - exp2
                                data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                                
                                # Calculate Moving Averages
                                data['MA20'] = data['Close'].rolling(window=20).mean()
                                data['MA50'] = data['Close'].rolling(window=50).mean()
                                
                                # Get latest values
                                latest_data = {
                                    'rsi': float(data['RSI'].iloc[-1]),
                                    'macd': float(data['MACD'].iloc[-1]),
                                    'signal': float(data['Signal'].iloc[-1]),
                                    'ma20': float(data['MA20'].iloc[-1]),
                                    'ma50': float(data['MA50'].iloc[-1]),
                                    'volume': float(data['Volume'].iloc[-1]),
                                    'avg_volume': float(data['Volume'].mean()),
                                    'volume_trend': '+' if data['Volume'].tail(10).mean() > data['Volume'].tail(30).mean() else '-'
                                }
                                
                                cio_prompt = f"""You are the Chief Investment Officer of a prestigious investment firm. Write a professional investment analysis letter for {ticker} stock.

                                Current Analysis Data:
                                - Current Price: ${current_price:.2f}
                                - Predicted Price (in {prediction_days} days): ${final_price:.2f}
                                - Predicted Change: {price_change:+.2f}%
                                
                                Technical Indicators:
                                - RSI: {latest_data['rsi']:.2f} ({'Overbought' if latest_data['rsi'] > 70 else 'Oversold' if latest_data['rsi'] < 30 else 'Neutral'})
                                - MACD: {latest_data['macd']:.2f}
                                - Signal Line: {latest_data['signal']:.2f}
                                - 20-day MA: ${latest_data['ma20']:.2f}
                                - 50-day MA: ${latest_data['ma50']:.2f}
                                
                                Volume Analysis:
                                - Current Volume: {latest_data['volume']:,.0f}
                                - Average Volume: {latest_data['avg_volume']:,.0f}
                                - Volume Trend: {latest_data['volume_trend']}
                                
                                Write a detailed analysis covering:
                                1. Executive Summary
                                - Brief overview of current position
                                - Key recommendation
                                - Target price range
                                
                                2. Technical Analysis
                                - RSI interpretation
                                - MACD signal analysis
                                - Moving average trends
                                - Volume analysis
                                
                                3. Price Targets
                                - Entry points (with specific prices)
                                - Exit targets (with specific prices)
                                - Stop loss levels (with specific prices)
                                
                                4. Risk Assessment
                                - Market risks
                                - Technical risks
                                - Volatility analysis
                                
                                5. Investment Recommendation
                                - Clear BUY/HOLD/SELL recommendation
                                - Position sizing suggestion
                                - Time horizon
                                - Risk management rules
                                
                                Format with markdown for better readability.
                                Be specific with numbers and percentages.
                                Provide clear actionable recommendations.
                                """
                                
                                with st.spinner("Generating CIO Investment Letter..."):
                                    response = requests.post(
                                        'http://localhost:11434/api/generate',
                                        json={
                                            "model": "llama2" if model_type == "Ollama 3.2" else "deepseek-coder",
                                            "prompt": cio_prompt,
                                            "stream": False
                                        },
                                        timeout=60
                                    )
                                    
                                    if response.status_code == 200:
                                        cio_letter = response.json()['response']
                                        
                                        # Add a professional header
                                        st.markdown(f"""
                                        <div style='text-align: center; margin-bottom: 20px;'>
                                            <h3>Investment Analysis Report</h3>
                                            <p style='color: gray;'>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
                                            <p style='color: gray;'>For: {ticker}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Display the analysis
                                        st.markdown(cio_letter)
                                        
                                        # Add disclaimer
                                        st.markdown("""
                                        ---
                                        *Disclaimer: This analysis is generated using artificial intelligence and should not be considered as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
                                        """)
                                    else:
                                        st.error(f"Failed to generate CIO letter. Status code: {response.status_code}")
                                        
                            except Exception as e:
                                st.error(f"Error in CIO letter generation: {str(e)}")
                                st.write("Debug: Exception traceback:", traceback.format_exc())
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

def main():
    """Main function to run the stock analysis app"""
    
    # Set page config to make it wider
    st.set_page_config(
        page_title="StockPro - Advanced Stock Analysis",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide sidebar
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    .stApp {max-width: 100%;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {
        padding-left: 4px;
        padding-right: 4px;
        white-space: pre;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("StockPro - Advanced Stock Analysis")
    
    # Initialize session state
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'AAPL'
        st.session_state['current_ticker'] = 'AAPL'  # For backwards compatibility
    
    # Stock input in main page
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input(
            "Enter Stock Symbol",
            value=st.session_state['ticker'],
            placeholder="e.g., AAPL",
            help="Enter a valid stock symbol to analyze"
        ).upper()
    
    # Update session state if ticker changes
    if ticker != st.session_state['ticker']:
        st.session_state['ticker'] = ticker
        st.session_state['current_ticker'] = ticker  # For backwards compatibility
        if ticker:  # Only rerun if ticker is not empty
            st.rerun()
    
    # Show current stock info if available
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name = info.get('longName', ticker)
            current_price = info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', 0)
            price_change = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
            
            with col2:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.2f}%",
                    delta_color="normal"
                )
                
            with col3:
                market_cap = info.get('marketCap', 0)
                st.metric("Market Cap", format_large_number(market_cap))
            
            st.markdown(f"### {company_name}")
        except Exception as e:
            if ticker:
                st.error(f"Error loading stock data: {str(e)}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Company Profile",
        "Technical Analysis",
        "Stock Prediction",
        "Buffett Analysis",
        "Options Analysis",
        "Market Movers",
        "Stock Recommendations",
        "Stock Chat"
    ])
    
    with tab1:
        # Create sub-tabs for Company Profile
        subtab1, subtab2, subtab3 = st.tabs([
            "Company Info",
            "Portfolio Analysis",
            "Fundamental Analysis"
        ])
        
        with subtab1:
            company_profile_tab()
        
        with subtab2:
            portfolio_analysis_tab()
            
        with subtab3:
            fundamental_analysis_tab()
    
    with tab2:
        technical_analysis_tab()
    
    with tab3:
        prediction_tab()
    
    with tab4:
        buffett_analysis_tab()
    
    with tab5:
        options_analysis_tab()
    
    with tab6:
        market_movers_tab()
    
    with tab7:
        stock_recommendations_tab()
        
    with tab8:
        stock_chat_tab()

if __name__ == "__main__":
    main()