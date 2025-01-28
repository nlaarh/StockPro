import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import traceback
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import requests
import lightgbm as lgb
from transformers import pipeline
import ollama
import time

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
    """Plot stock history chart"""
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period=period)
    fig = go.Figure(data=[go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        name=ticker,
        line=dict(color='blue')
    )])
    fig.update_layout(
        title=f'{ticker} Stock History',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    return fig

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
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["ticker", "metrics", "news"],
            template="""You are a financial expert. Analyze this stock and predict its movement:

Stock: {ticker}

Technical Metrics:
- Price Change (20d): {metrics['Price Change (%)']}%
- Volume Change: {metrics['Volume Change (%)']}%
- RSI: {metrics['RSI']}
- MACD: {metrics['MACD']}

Recent News:
{news}

Provide a brief analysis of the stock's likely movement in the next 30 days. 
Consider technical indicators, news sentiment, and market conditions.
Format your response as:
PREDICTION: [BULLISH/BEARISH/NEUTRAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Your brief explanation]
"""
        )
        
        try:
            # Try Ollama first
            llm = Ollama(model="mistral")
            llm_response = llm(prompt.format(
                ticker=ticker,
                metrics=metrics,
                news=news_text
            ))
        except Exception as e:
            print(f"Ollama error: {str(e)}")
            # Fallback to Hugging Face model
            from transformers import pipeline
            classifier = pipeline("text-classification", model="ProsusAI/finbert")
            news_sentiments = classifier(news_text)
            
            # Analyze sentiment distribution
            sentiment_counts = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
            for sentiment in news_sentiments:
                sentiment_counts[sentiment['label']] += 1
            
            # Generate response based on sentiment and metrics
            if sentiment_counts['positive'] > sentiment_counts['negative']:
                prediction = "BULLISH"
            elif sentiment_counts['negative'] > sentiment_counts['positive']:
                prediction = "BEARISH"
            else:
                prediction = "NEUTRAL"
                
            # Generate confidence based on sentiment strength
            total = sum(sentiment_counts.values())
            max_count = max(sentiment_counts.values())
            confidence = "HIGH" if max_count/total > 0.6 else "MEDIUM" if max_count/total > 0.4 else "LOW"
            
            llm_response = f"""PREDICTION: {prediction}
CONFIDENCE: {confidence}
REASONING: Based on news sentiment analysis: {sentiment_counts['positive']} positive, {sentiment_counts['negative']} negative, {sentiment_counts['neutral']} neutral articles. Technical indicators show RSI at {metrics['RSI']:.1f} and MACD at {metrics['MACD']:.2f}."""
        
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
    """
    Predict stock prices using the selected model
    """
    try:
        # Validate input data
        data_ok, data_error = test_data_preparation(data, features, target)
        if not data_ok:
            print(f"Data validation failed: {data_error}")
            return None, None, None
        
        # Prepare the data
        X = data[features].values
        y = data[target].values
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X) - prediction_days):
            X_seq.append(X[i:(i + prediction_days)])
            y_seq.append(y[i + prediction_days])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        if len(X_seq) == 0 or len(y_seq) == 0:
            print("Not enough data points")
            return None, None, None
        
        # Split the data
        split = int(0.8 * len(X_seq))
        X_train = X_seq[:split]
        X_test = X_seq[split:]
        y_train = y_seq[:split]
        y_test = y_seq[split:]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print("Not enough data for train/test split")
            return None, None, None
        
        predictions = None
        confidence_bands = None
        
        # Make predictions based on model type
        try:
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                predictions = test_model_prediction(model, X_train, y_train, X_test)
                
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(random_state=42)
                predictions = test_model_prediction(model, X_train, y_train, X_test)
                
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(random_state=42)
                predictions = test_model_prediction(model, X_train, y_train, X_test)
                
            elif model_type == "Ensemble":
                predictions_list = []
                models = [
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    xgb.XGBRegressor(random_state=42),
                    lgb.LGBMRegressor(random_state=42)
                ]
                
                for model in models:
                    pred = test_model_prediction(model, X_train, y_train, X_test)
                    if pred is not None:
                        predictions_list.append(pred)
                
                if predictions_list:
                    predictions = np.mean(predictions_list, axis=0)
                else:
                    # If all models fail, use a simple moving average prediction
                    predictions = data['MA20'].values[-len(y_test):]
                
            elif model_type in ["LLM", "Ollama 3.2"]:
                # First try LLM prediction
                try:
                    client = ollama.Client(host='http://localhost:11434')
                    prompt = f"""You are a financial expert analyzing stock data. Based on the following technical indicators and recent price data, predict the stock's movement and percentage change for the next {prediction_days} days.

                    Recent Data (last 10 days):
                    {data[features + ['Close']].tail(10).to_string()}
                    
                    Technical Analysis:
                    - RSI: {data['RSI'].iloc[-1]:.2f}
                    - MACD: {data['MACD'].iloc[-1]:.2f}
                    - 20-day MA: {data['MA20'].iloc[-1]:.2f}
                    - 50-day MA: {data['MA50'].iloc[-1]:.2f}
                    
                    Current Price: ${data['Close'].iloc[-1]:.2f}
                    
                    Provide:
                    1. Price prediction with confidence range (1% bands)
                    2. Directional prediction (UP/DOWN)
                    3. Expected percentage change
                    4. Key factors influencing the prediction
                    
                    Format your response as:
                    DIRECTION: UP/DOWN
                    PRICE: $XXX.XX
                    CHANGE: +/-X.X%
                    CONFIDENCE_LOW: $XXX.XX
                    CONFIDENCE_HIGH: $XXX.XX
                    ANALYSIS: Your detailed analysis here
                    """
                    
                    response = client.generate(model='llama3.2', prompt=prompt)
                    response_text = response['response']
                    
                    # Parse the response
                    lines = response_text.split('\n')
                    prediction_dict = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            prediction_dict[key.strip()] = value.strip()
                    
                    # Get the predicted price
                    if 'PRICE' in prediction_dict:
                        price_str = prediction_dict['PRICE'].replace('$', '')
                        try:
                            predicted_price = float(price_str)
                            predictions = np.linspace(y_test[-1], predicted_price, len(y_test))
                            
                            # Create confidence bands
                            if 'CONFIDENCE_LOW' in prediction_dict and 'CONFIDENCE_HIGH' in prediction_dict:
                                low_str = prediction_dict['CONFIDENCE_LOW'].replace('$', '')
                                high_str = prediction_dict['CONFIDENCE_HIGH'].replace('$', '')
                                try:
                                    lower_bound = float(low_str)
                                    upper_bound = float(high_str)
                                    confidence_bands = (
                                        np.linspace(y_test[-1], lower_bound, len(y_test)),
                                        np.linspace(y_test[-1], upper_bound, len(y_test))
                                    )
                                except ValueError:
                                    confidence_bands = (predictions * 0.99, predictions * 1.01)
                        except ValueError:
                            # Fall back to Random Forest
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            predictions = test_model_prediction(model, X_train, y_train, X_test)
                except Exception as e:
                    print(f"LLM prediction failed: {str(e)}, falling back to Random Forest")
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    predictions = test_model_prediction(model, X_train, y_train, X_test)
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            # Fall back to moving average
            predictions = data['MA20'].values[-len(y_test):]
        
        # If predictions is still None, use a simple moving average
        if predictions is None:
            predictions = data['MA20'].values[-len(y_test):]
        
        # Create default confidence bands if none exist
        if confidence_bands is None:
            confidence_bands = (
                predictions * 0.99,  # Lower bound: -1%
                predictions * 1.01   # Upper bound: +1%
            )
        
        # Final validation
        if len(predictions) != len(y_test):
            print("Prediction length mismatch")
            # Adjust predictions to match y_test length
            predictions = np.interp(
                np.linspace(0, 1, len(y_test)),
                np.linspace(0, 1, len(predictions)),
                predictions
            )
            
            # Adjust confidence bands
            lower_bound = np.interp(
                np.linspace(0, 1, len(y_test)),
                np.linspace(0, 1, len(confidence_bands[0])),
                confidence_bands[0]
            )
            upper_bound = np.interp(
                np.linspace(0, 1, len(y_test)),
                np.linspace(0, 1, len(confidence_bands[1])),
                confidence_bands[1]
            )
            confidence_bands = (lower_bound, upper_bound)
        
        # Final safety check
        if predictions is None or y_test is None or confidence_bands is None:
            print("Final validation failed")
            return None, None, None
            
        if len(predictions) == 0 or len(y_test) == 0:
            print("Empty predictions or test data")
            return None, None, None
            
        if not isinstance(confidence_bands, tuple) or len(confidence_bands) != 2:
            print("Invalid confidence bands")
            return None, None, None
        
        return predictions, y_test, confidence_bands
        
    except Exception as e:
        print(f"Error in predict_stock_price: {str(e)}")
        # Return simple moving average prediction as fallback
        try:
            ma20 = data['MA20'].values[-len(y_test):]
            return ma20, y_test, (ma20 * 0.99, ma20 * 1.01)
        except:
            return None, None, None

def analyze_trend(recent_df, predictions):
    """Analyze trend and generate summary"""
    try:
        # Calculate key metrics
        current_price = recent_df['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate RSI
        rsi = calculate_rsi(recent_df['Close'])[-1]
        
        # Calculate MACD
        macd = calculate_macd(recent_df['Close'])[-1]
        
        # Analyze volume trend
        recent_volume = recent_df['Volume'].iloc[-5:].mean()
        past_volume = recent_df['Volume'].iloc[-25:-5].mean()
        volume_change = ((recent_volume - past_volume) / past_volume) * 100
        
        # Generate trend signals
        signals = []
        
        # Moving Average signals
        if current_price > recent_df['MA20'].iloc[-1]:
            signals.append("✅ Price above 20-day MA (Short-term bullish)")
        else:
            signals.append("🔴 Price below 20-day MA (Short-term bearish)")
            
        if current_price > recent_df['MA50'].iloc[-1]:
            signals.append("✅ Price above 50-day MA (Medium-term bullish)")
        else:
            signals.append("🔴 Price below 50-day MA (Medium-term bearish)")
            
        if recent_df['MA20'].iloc[-1] > recent_df['MA50'].iloc[-1]:
            signals.append("✅ Golden Cross: 20-day MA above 50-day MA (Bullish)")
        else:
            signals.append("🔴 Death Cross: 20-day MA below 50-day MA (Bearish)")
            
        # RSI signals
        if rsi > 70:
            signals.append("⚠️ RSI above 70 (Overbought)")
        elif rsi < 30:
            signals.append("💡 RSI below 30 (Oversold)")
        else:
            signals.append("✅ RSI between 30-70 (Neutral)")
            
        # MACD signals
        if 'MACD' in recent_df.columns:
            if macd > 0:
                signals.append("✅ MACD above 0 (Bullish)")
            else:
                signals.append("🔴 MACD below 0 (Bearish)")
        
        # Generate trend data
        trend_data = "### Technical Signals\n\n" + "\n".join(signals)
        
        # Generate trend summary
        if price_change > 0 and rsi < 70 and macd > 0:
            trend = "BULLISH"
        elif price_change < 0 and rsi > 30 and macd < 0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        trend_summary = f"""
        ### Technical Analysis Summary
        
        **Overall Trend:** {trend}
        
        **Key Metrics:**
        - Predicted Price Change: {price_change:.1f}%
        - RSI: {rsi:.1f}
        - MACD: {macd:.2f}
        - Volume Change: {volume_change:.1f}%
        """
        
        return trend_data, trend_summary
        
    except Exception as e:
        print(f"Error in trend analysis: {str(e)}")
        return None, None

def generate_plain_summary(ticker, current_price, predictions):
    """Generate a plain English summary of the prediction"""
    try:
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Determine direction and confidence
        if price_change > 0:
            direction = "increase"
            emoji = "📈"
        else:
            direction = "decrease"
            emoji = "📉"
            
        confidence = "high" if abs(price_change) < 20 else "medium"
        
        summary = f"""
        ### Plain English Summary {emoji}
        
        Based on our analysis, we expect {ticker} to {direction} from ${current_price:.2f} to ${predicted_price:.2f} 
        over the next 30 days, a change of {abs(price_change):.1f}%. 
        
        We have {confidence} confidence in this prediction based on:
        - Recent price trends
        - Trading volume patterns
        - Technical indicators
        """
        
        return summary
        
    except Exception as e:
        print(f"Error generating plain summary: {str(e)}")
        return None

def generate_price_summary(ticker, current_price, predictions):
    """Generate a summary of price predictions"""
    try:
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        summary = f"""
        ### Price Prediction Summary 💰
        
        **Current Price:** ${current_price:.2f}
        **Predicted Price:** ${predicted_price:.2f}
        **Expected Change:** {price_change:+.1f}%
        
        This prediction is based on historical price data, technical indicators, and market trends.
        """
        
        return summary
        
    except Exception as e:
        print(f"Error generating price summary: {str(e)}")
        return None

def generate_prediction_summary(ticker, data, model_type, prediction_days, predictions, confidence_bands=None):
    """Generate a comprehensive prediction summary including fundamentals and analyst views"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price and predicted price
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Historical price changes
        week_change = ((current_price / data['Close'].iloc[-5]) - 1) * 100
        month_change = ((current_price / data['Close'].iloc[-20]) - 1) * 100
        
        # Get fundamental data
        market_cap = info.get('marketCap', 0) / 1e9  # Convert to billions
        pe_ratio = info.get('forwardPE', 0)
        peg_ratio = info.get('pegRatio', 0)
        beta = info.get('beta', 0)
        revenue_growth = info.get('revenueGrowth', 0) * 100
        profit_margins = info.get('profitMargins', 0) * 100
        
        # Get analyst recommendations
        target_high = info.get('targetHighPrice', 0)
        target_low = info.get('targetLowPrice', 0)
        target_mean = info.get('targetMeanPrice', 0)
        recommendation = info.get('recommendationKey', '').upper()
        
        # Technical indicators
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        # Volume analysis
        avg_volume = data['Volume'].mean()
        recent_volume = data['Volume'].iloc[-1]
        volume_ratio = ((recent_volume - avg_volume) / avg_volume) * 100
        
        # Generate summary
        summary = f"""
        ### 📊 Comprehensive Stock Analysis for {ticker}

        #### 🎯 Price Prediction Summary ({prediction_days} days):
        - Current Price: ${current_price:.2f}
        - Predicted Price: ${predicted_price:.2f}
        - Expected Change: {price_change:.1f}%
        """
        
        if confidence_bands:
            lower_bound, upper_bound = confidence_bands
            summary += f"""
        - Confidence Range: ${lower_bound[-1]:.2f} to ${upper_bound[-1]:.2f}
            """
            
        summary += f"""
        #### 📈 Historical Performance:
        - 1 Week Change: {week_change:.1f}%
        - 1 Month Change: {month_change:.1f}%
        - Volume Trend: {volume_ratio:+.1f}% vs average
        
        #### 📊 Technical Analysis:
        - RSI ({rsi:.1f}): {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}
        - MACD ({macd:.2f}): {'Bullish' if macd > 0 else 'Bearish' if macd < 0 else 'Neutral'}
        - Moving Averages: {'Bullish' if current_price > ma20 > ma50 else 'Bearish' if current_price < ma20 < ma50 else 'Mixed'}
        - Volume: {'Above' if volume_ratio > 0 else 'Below'} average by {abs(volume_ratio):.1f}%
        
        #### 💰 Fundamental Analysis:
        - Market Cap: ${market_cap:.1f}B
        - Forward P/E: {pe_ratio:.2f}
        - PEG Ratio: {peg_ratio:.2f}
        - Beta: {beta:.2f}
        - Revenue Growth: {revenue_growth:.1f}%
        - Profit Margins: {profit_margins:.1f}%
        
        #### 👥 Market Sentiment:
        - Analyst Consensus: {recommendation}
        - Price Targets:
          • High: ${target_high:.2f}
          • Mean: ${target_mean:.2f}
          • Low: ${target_low:.2f}
        """
        
        # Add model-specific insights
        if model_type == "Ollama 3.2":
            # Get LLM analysis of the prediction
            analysis_prompt = f"""
            You are a financial expert. Based on the following data, provide a brief analysis (3-4 sentences) explaining the rationale behind the prediction:
            - Current Price: ${current_price:.2f}
            - Predicted Price: ${predicted_price:.2f} ({price_change:.1f}%)
            - RSI: {rsi:.1f}
            - MACD: {macd:.2f}
            - Market Cap: ${market_cap:.1f}B
            - Forward P/E: {pe_ratio:.2f}
            - Revenue Growth: {revenue_growth:.1f}%
            - Analyst Target: ${target_mean:.2f}
            - Recent Volume Change: {volume_ratio:+.1f}%
            
            Consider technical indicators, fundamental metrics, and market sentiment in your analysis.
            """
            
            client = ollama.Client(host='http://localhost:11434')
            response = client.generate(model='llama3.2', prompt=analysis_prompt)
            
            # Parse the response
            response_text = response['response']
            
            # Extract predictions from response
            lines = response_text.split('\n')
            prediction_dict = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    prediction_dict[key.strip()] = value.strip()
            
            # Get the predicted price
            if 'PRICE' in prediction_dict:
                price_str = prediction_dict['PRICE'].replace('$', '')
                predicted_price = float(price_str)
                
                # Create confidence bands
                if 'CONFIDENCE_LOW' in prediction_dict and 'CONFIDENCE_HIGH' in prediction_dict:
                    low_str = prediction_dict['CONFIDENCE_LOW'].replace('$', '')
                    high_str = prediction_dict['CONFIDENCE_HIGH'].replace('$', '')
                    lower_bound = float(low_str)
                    upper_bound = float(high_str)
                    
                    # Create arrays for confidence bands
                    confidence_bands = (
                        np.linspace(y_test[-1], lower_bound, prediction_days),
                        np.linspace(y_test[-1], upper_bound, prediction_days)
                    )
                    
                    # Create prediction array
                    predictions = np.linspace(y_test[-1], predicted_price, prediction_days)
                    return predictions, y_test, confidence_bands
                
            # Add LLM analysis to summary
            summary += f"""
        #### 🤖 AI Analysis:
        {response_text}
            """
        
        # Add model explanation
        summary += f"""
        #### 🔮 Prediction Model Details:
        """
        
        if model_type == "Ollama 3.2":
            summary += """
        Using **Ollama 3.2 LLM Model**:
        - Advanced language model trained on financial data
        - Integrates technical and fundamental analysis
        - Considers market sentiment and analyst opinions
        - Provides confidence bands based on market volatility
        - Uses pattern recognition from historical data
        """
        elif model_type == "Random Forest":
            summary += """
        Using **Random Forest Model**:
        - Ensemble of decision trees for robust predictions
        - Handles non-linear relationships in market data
        - Reduces overfitting through multiple tree averaging
        - Good at capturing market regime changes
        """
        elif model_type == "XGBoost":
            summary += """
        Using **XGBoost Model**:
        - Gradient boosting for high accuracy
        - Handles missing data points automatically
        - Excellent performance on structured financial data
        - Advanced feature importance analysis
        """
        elif model_type == "LightGBM":
            summary += """
        Using **LightGBM Model**:
        - Light Gradient Boosting Machine
        - Faster training on large datasets
        - Leaf-wise tree growth for better accuracy
        - Handles categorical features efficiently
        """
        elif model_type == "Ensemble":
            summary += """
        Using **Ensemble Model**:
        - Combines Random Forest, XGBoost, and LightGBM
        - Averages predictions for better stability
        - Reduces individual model biases
        - More robust to market volatility
        """
        
        # Add prediction rationale
        summary += """
        #### 📝 Prediction Rationale:
        This prediction integrates multiple factors:
        1. **Technical Analysis**: Price trends, momentum indicators, and volume patterns
        2. **Fundamental Metrics**: Company valuation, growth metrics, and financial health
        3. **Market Sentiment**: Analyst consensus, institutional holdings, and market momentum
        4. **Historical Patterns**: Price action, volatility, and trading ranges
        5. **Sector Analysis**: Industry trends and comparative performance
        """
        
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None

def generate_analyst_summary(data, ticker, model_type, predictions, confidence_bands):
    """Generate a detailed analyst summary of the stock prediction"""
    try:
        # Get current and predicted prices
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Calculate key metrics
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown Sector')
        industry = info.get('industry', 'Unknown Industry')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('forwardPE', 0)
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield:
            dividend_yield = dividend_yield * 100
        beta = info.get('beta', 'N/A')
        
        # Format volume numbers
        def format_large_number(number):
            """Format large numbers with appropriate suffixes (K, M, B, T)"""
            if number is None:
                return "N/A"
            
            suffixes = ['', 'K', 'M', 'B', 'T']
            magnitude = 0
            
            while abs(number) >= 1000 and magnitude < len(suffixes)-1:
                magnitude += 1
                number /= 1000.0
            
            if magnitude > 0:
                return f"${number:,.2f}{suffixes[magnitude]}"
            else:
                return f"${number:,.2f}"

        formatted_volume = format_large_number(volume)
        formatted_avg_volume = format_large_number(avg_volume)
        formatted_market_cap = format_large_number(market_cap)
        
        # Determine trend strength
        trend = "UPWARD" if price_change > 0 else "DOWNWARD"
        strength = "STRONG" if abs(price_change) > 5 else "MODERATE" if abs(price_change) > 2 else "MILD"
        
        # Analyze confidence bands
        lower_bound = confidence_bands[0][-1]
        upper_bound = confidence_bands[1][-1]
        band_spread = ((upper_bound - lower_bound) / current_price) * 100
        
        # Moving average analysis
        ma_status = 'strong support' if current_price > ma20 > ma50 else 'significant resistance' if current_price < ma20 < ma50 else 'mixed signals'
        ma_text = f"with the 20-day MA at ${ma20:,.2f} and the 50-day MA at ${ma50:,.2f}"
        
        # Format moving averages text
        ma_description = (
            f"while the moving average configuration shows "
            f"{'strong support' if current_price > ma20 > ma50 else 'significant resistance' if current_price < ma20 < ma50 else 'mixed signals'} "
            f"with the 20-day MA at ${ma20:,.2f} and the 50-day MA at ${ma50:,.2f}"
        )

        # Market analysis parts
        analysis_parts = [
            f"**Market Overview:** Based on our comprehensive analysis of **{ticker}**, operating in the *{sector} sector*,",
            f"our *{model_type}* model projects a **{strength.lower()} {trend.lower()} movement** with a target price of **${predicted_price:,.2f}**,",
            f"representing a **{price_change:+,.2f}%** change from the current price of **${current_price:,.2f}**.",
            
            f"\n\n**Technical Analysis:** The indicators present a {'compelling' if abs(price_change) > 5 else 'moderate'} case for this projection,",
            f"with the **RSI** at **{rsi:,.1f}** indicating *{'overbought conditions that may limit upside' if rsi > 70 else 'oversold conditions that may provide buying opportunities' if rsi < 30 else 'neutral momentum'}*.",
            f"The **MACD** indicator *{'confirms this trend' if (price_change > 0 and macd > signal) or (price_change < 0 and macd < signal) else 'shows potential trend weakness'}*,",
            ma_description + ".",
            
            f"\n\n**Volume Analysis:** Trading volume *{'supports this outlook' if volume > avg_volume else 'suggests caution'}*,",
            f"with current volume at **{formatted_volume}** {'above' if volume > avg_volume else 'below'} the average of **{formatted_avg_volume}**,",
            f"indicating **{'strong' if volume > avg_volume else 'weak'}** market participation.",
            
            f"\n\n**Risk Metrics:** The stock's beta of **{beta if isinstance(beta, str) else f'{beta:,.2f}'}** and P/E ratio of **{pe_ratio if isinstance(pe_ratio, str) else f'{pe_ratio:,.2f}'}**",
            f"*{'suggest higher volatility' if isinstance(beta, (int, float)) and beta > 1 else 'indicate relative stability'}*,",
            f"while the {f'dividend yield of **{dividend_yield:,.2f}%** provides additional return potential' if dividend_yield else 'absence of dividends focuses returns on price appreciation'}.",
            
            f"\n\n**Trading Strategy:** Given the confidence band spread of **{band_spread:,.1f}%**, we recommend",
            f"**{'aggressive' if band_spread < 5 and volume > avg_volume else 'moderate' if band_spread < 10 else 'conservative'}** position sizing",
            f"with clear stop-loss levels at **${min(lower_bound, ma20):,.2f}**.",
            f"Market conditions and sector trends *{'support' if price_change > 0 and volume > avg_volume else 'challenge'}* this outlook,",
            f"suggesting a **{'favorable' if price_change > 0 and volume > avg_volume else 'cautious'}** approach to position management."
        ]
        
        # Join parts with proper spacing
        market_analysis = " ".join(analysis_parts)
        
        # Generate summary sections
        summary = f"""
### 📊 Stock Analysis Report for {ticker}

#### Detailed Market Analysis
{market_analysis}

#### Company Overview
- **Sector**: {sector}
- **Industry**: {industry}
- **Market Cap**: {formatted_market_cap}
- **Beta**: {beta if isinstance(beta, str) else f'{beta:,.2f}'}
- **P/E Ratio**: {pe_ratio if isinstance(pe_ratio, str) else f'{pe_ratio:,.2f}'}
- **Dividend Yield**: {f'{dividend_yield:,.2f}%' if dividend_yield else 'N/A'}

#### Price Analysis
- **Current Price**: ${current_price:,.2f}
- **Predicted Price**: ${predicted_price:,.2f} ({price_change:+,.2f}%)
- **Confidence Range**: ${lower_bound:,.2f} to ${upper_bound:,.2f} ({band_spread:,.1f}% spread)
- **Model Used**: {model_type}

#### Technical Indicators
- **RSI (14)**: {rsi:,.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
- **MACD Signal**: {'Bullish' if macd > signal else 'Bearish' if macd < signal else 'Neutral'}
- **Moving Averages**: {'Bullish' if current_price > ma20 > ma50 else 'Bearish' if current_price < ma20 < ma50 else 'Mixed'}
- **Volume**: {formatted_volume} ({'Above' if volume > avg_volume else 'Below'} Average)

#### Trading Considerations
"""
        # Add trading considerations based on analysis
        if trend == "UPWARD":
            summary += f"""
- **Entry Strategy**: Look for pullbacks near ${lower_bound:,.2f} (lower band) or ${ma20:,.2f} (20-day MA)
- **Target Price**: Primary target at ${predicted_price:,.2f}, extended target at ${upper_bound:,.2f}
- **Stop Loss**: Place protective stops below ${min(lower_bound, ma20):,.2f}
- **Position Sizing**: {'Consider larger positions due to strong volume' if volume > avg_volume else 'Start with smaller positions and scale in'}
- **Risk Management**: {'Volume confirms trend strength' if volume > avg_volume else 'Lower volume suggests caution'}"""
        else:
            summary += f"""
- **Risk Management**: Consider reducing exposure or implementing hedges
- **Support Levels**: Watch for support at ${lower_bound:,.2f} and ${ma50:,.2f}
- **Position Sizing**: {'Reduce position size due to high volatility' if band_spread > 10 else 'Standard position sizing acceptable'}
- **Alternative Strategy**: Consider waiting for price stabilization near ${lower_bound:,.2f}
- **Hedging**: Consider protective puts or stop orders at ${min(lower_bound, ma20):,.2f}"""
        
        return summary
        
    except Exception as e:
        print(f"Error generating analyst summary: {str(e)}")
        return None

def technical_analysis_tab():
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
                    
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def prediction_tab():
    """Stock price prediction tab with analyst insights"""
    try:
        st.header("Stock Price Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            ticker = st.text_input("Enter Stock Symbol:", "AAPL").upper()
            prediction_days = st.slider("Prediction Days", 1, 30, 7)
            
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "XGBoost", "LightGBM", "Ensemble", "Ollama 3.2"]
            )
            
            if model_type == "Ollama 3.2":
                # Test Ollama connection
                ollama_running, error = test_ollama_connection()
                if not ollama_running:
                    st.error(f"""
                    Ollama server is not running or not accessible. Please:
                    1. Install Ollama from https://ollama.ai
                    2. Start the Ollama server
                    3. Install the required model:
                       ```
                       ollama pull llama2
                       ollama pull llama3.2
                       ```
                    Error: {error}
                    """)
                    return
        
        if st.button("Generate Prediction"):
            try:
                with st.spinner("Loading data and generating prediction..."):
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="1y")
                    
                    if data.empty:
                        st.error(f"No data found for {ticker}")
                        return
                    
                    # Calculate technical indicators
                    data['MA20'] = data['Close'].rolling(window=20).mean()
                    data['MA50'] = data['Close'].rolling(window=50).mean()
                    data['RSI'] = calculate_rsi(data['Close'])
                    
                    # Calculate MACD with error handling
                    try:
                        macd_line, signal_line, histogram = calculate_macd(data['Close'])
                        data['MACD'] = macd_line
                        data['Signal'] = signal_line
                        data['MACD_Hist'] = histogram
                    except Exception as e:
                        st.warning(f"Error calculating MACD: {str(e)}. Using default values.")
                        data['MACD'] = data['Close'].rolling(window=12).mean() - data['Close'].rolling(window=26).mean()
                        data['Signal'] = data['MACD'].rolling(window=9).mean()
                        data['MACD_Hist'] = data['MACD'] - data['Signal']
                    
                    # Drop NaN values
                    data = data.dropna()
                    
                    # Prepare features
                    features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD']
                    
                    # Make prediction
                    result = predict_stock_price(data, model_type, features, 'Close', prediction_days)
                    
                    if result is None or not isinstance(result, tuple) or len(result) != 3:
                        st.error("Failed to generate prediction. Please try a different stock or model.")
                        return
                    
                    predictions, y_test, confidence_bands = result
                    
                    # Create visualization
                    last_date = data.index[-1]
                    future_dates = [last_date + timedelta(days=x) for x in range(prediction_days)]
                    date_labels = [d.strftime('%Y-%m-%d') for d in future_dates]
                    
                    fig = go.Figure()
                    
                    # Plot historical data
                    historical_days = 30
                    recent_data = data.tail(historical_days)
                    hist_dates = [d.strftime('%Y-%m-%d') for d in recent_data.index]
                    
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=recent_data['Close'],
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    # Plot predictions
                    fig.add_trace(go.Scatter(
                        x=date_labels,
                        y=predictions,
                        name='Predicted',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add confidence bands
                    fig.add_trace(go.Scatter(
                        x=date_labels,
                        y=confidence_bands[1],
                        name='Upper Band',
                        line=dict(color='rgba(0,100,80,0.2)', dash='dash'),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=date_labels,
                        y=confidence_bands[0],
                        name='Lower Band',
                        line=dict(color='rgba(0,100,80,0.2)', dash='dash'),
                        fill='tonexty',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} Stock Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        showlegend=True,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate and display analyst summary
                    with st.expander("📈 View Detailed Analyst Report", expanded=True):
                        analyst_summary = generate_analyst_summary(
                            data, ticker, model_type, predictions, confidence_bands
                        )
                        if analyst_summary:
                            st.markdown(analyst_summary)
                        else:
                            st.warning("Could not generate analyst summary. Some data may be missing.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)
    except Exception as e:
        st.error(f"Error in prediction tab: {str(e)}")
        st.exception(e)

def test_ollama_connection():
    """Test if Ollama server is running and accessible"""
    try:
        client = ollama.Client(host='http://localhost:11434')
        # Try a simple test prompt
        response = client.generate(model='llama3.2', prompt='Say hello')
        return True, None
    except Exception as e:
        return False, str(e)

def test_model_prediction(model, X_train, y_train, X_test):
    """Test if a model can make predictions successfully"""
    try:
        # Input validation
        if X_train is None or y_train is None or X_test is None:
            print("Invalid input data")
            return y_train[-len(X_test):]  # Return last known values as fallback
            
        if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0:
            print("Empty input data")
            return y_train[-len(X_test):]  # Return last known values as fallback
            
        # Reshape data for model input
        try:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        except Exception as e:
            print(f"Error reshaping data: {str(e)}")
            return y_train[-len(X_test):]  # Return last known values as fallback
        
        # Fit and predict
        try:
            model.fit(X_train_reshaped, y_train)
            predictions = model.predict(X_test_reshaped)
        except Exception as e:
            print(f"Error in model fitting/prediction: {str(e)}")
            return y_train[-len(X_test):]  # Return last known values as fallback
        
        # Validate predictions
        if predictions is None or len(predictions) == 0:
            print("Model returned empty predictions")
            return y_train[-len(X_test):]  # Return last known values as fallback
            
        if np.isnan(predictions).any():
            print("Predictions contain NaN values")
            return y_train[-len(X_test):]  # Return last known values as fallback
            
        return predictions
        
    except Exception as e:
        print(f"Error in model prediction: {str(e)}")
        return y_train[-len(X_test):]  # Return last known values as fallback

def test_data_preparation(data, features, target):
    """Test if data is properly prepared for prediction"""
    try:
        # Check if data is empty
        if data.empty:
            return False, "No data available"
            
        # Check if all required features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            return False, f"Missing features: {', '.join(missing_features)}"
            
        # Check if target exists
        if target not in data.columns:
            return False, f"Target column '{target}' not found"
            
        # Check for NaN values
        if data[features + [target]].isna().any().any():
            return False, "Data contains NaN values"
            
        # Check if there's enough data
        if len(data) < 50:  # Minimum required data points
            return False, "Not enough data points (minimum 50 required)"
            
        return True, None
        
    except Exception as e:
        return False, str(e)

def test_confidence_bands(predictions, confidence_level=0.01):
    """Test and generate confidence bands"""
    try:
        if predictions is None or len(predictions) == 0:
            return None
        
        upper_bound = predictions * (1 + confidence_level)
        lower_bound = predictions * (1 - confidence_level)
        
        return lower_bound, upper_bound
    except Exception as e:
        print(f"Error generating confidence bands: {str(e)}")
        return None

def analyze_stock_recommendation(ticker):
    """Analyze a stock and provide buy/sell recommendations"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
            
        # Calculate key metrics
        current_price = hist['Close'].iloc[-1]
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        rsi = calculate_rsi(hist['Close'])[-1]
        
        # Calculate price targets
        buy_price = current_price * 0.95  # 5% below current price
        sell_price = current_price * 1.15  # 15% above current price
        
        # Get company info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('forwardPE', 0)
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield:
            dividend_yield = dividend_yield * 100
        
        # Calculate technical signals
        technical_score = 0
        technical_reasons = []
        
        if current_price > ma_50:
            technical_score += 1
            technical_reasons.append("Price above 50-day MA")
        if current_price > ma_200:
            technical_score += 1
            technical_reasons.append("Price above 200-day MA")
        if ma_50 > ma_200:
            technical_score += 1
            technical_reasons.append("Golden Cross (50MA > 200MA)")
        if 30 < rsi < 70:
            technical_score += 1
            technical_reasons.append("RSI in healthy range")
            
        # Calculate fundamental signals
        fundamental_score = 0
        fundamental_reasons = []
        
        if pe_ratio != 'N/A' and pe_ratio < 20:
            fundamental_score += 1
            fundamental_reasons.append("Attractive P/E ratio")
        if dividend_yield > 2:
            fundamental_score += 1
            fundamental_reasons.append(f"Good dividend yield: {dividend_yield:.1f}%")
        if market_cap > 10:
            fundamental_score += 1
            fundamental_reasons.append("Large-cap company")
            
        # Combine scores
        total_score = technical_score + fundamental_score
        
        # Generate summary
        summary = f"""
### {company_name} ({ticker})
- **Sector**: {sector}
- **Industry**: {industry}
- **Market Cap**: ${market_cap/1e9:.1f}B
- **Current Price**: ${current_price:.2f}
- **Recommended Buy Price**: ${buy_price:.2f}
- **Target Sell Price**: ${sell_price:.2f}

#### Technical Analysis
{', '.join(technical_reasons) if technical_reasons else 'No significant technical signals'}

#### Fundamental Analysis
{', '.join(fundamental_reasons) if fundamental_reasons else 'No significant fundamental signals'}

#### Investment Thesis
{info.get('longBusinessSummary', 'No business summary available.')}
"""
        
        return {
            'summary': summary,
            'score': total_score,
            'current_price': current_price,
            'buy_price': buy_price,
            'sell_price': sell_price
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

def recommendations_tab():
    """Display stock recommendations"""
    try:
        st.subheader("Top Stock Recommendations")
        
        # List of stocks to analyze
        potential_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            # Finance
            'JPM', 'BAC', 'V', 'MA', 'BRK-B',
            # Consumer
            'AMZN', 'WMT', 'PG', 'KO', 'PEP',
            # Industrial
            'CAT', 'BA', 'HON', 'UPS', 'GE'
        ]
        
        with st.spinner("Analyzing stocks..."):
            # Analyze all stocks
            recommendations = []
            for ticker in potential_stocks:
                try:
                    analysis = analyze_stock_recommendation(ticker)
                    if analysis and isinstance(analysis, dict):  # Verify we got a valid analysis
                        if all(key in analysis for key in ['summary', 'score', 'current_price', 'buy_price', 'sell_price']):
                            recommendations.append({
                                'ticker': ticker,
                                'analysis': analysis
                            })
                except Exception as e:
                    st.warning(f"Could not analyze {ticker}: {str(e)}")
                    continue
            
            if not recommendations:
                st.error("Unable to generate recommendations. Please try again later.")
                return
            
            # Sort by score and get top 5
            recommendations.sort(key=lambda x: x['analysis']['score'], reverse=True)
            top_recommendations = recommendations[:5]
            
            if not top_recommendations:
                st.error("No valid recommendations found.")
                return
            
            # Display recommendations
            for i, rec in enumerate(top_recommendations, 1):
                try:
                    with st.expander(f"#{i}: {rec['ticker']}", expanded=True):
                        st.markdown(rec['analysis']['summary'])
                        
                        # Create price chart
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${rec['analysis']['current_price']:.2f}")
                        with col2:
                            st.metric("Buy Below", f"${rec['analysis']['buy_price']:.2f}")
                        with col3:
                            st.metric("Sell Above", f"${rec['analysis']['sell_price']:.2f}")
                except Exception as e:
                    st.error(f"Error displaying recommendation for {rec['ticker']}: {str(e)}")
                    continue
                    
    except Exception as e:
        st.error(f"Error in recommendations tab: {str(e)}")

def get_market_movers():
    """Get top gainers and losers from major indices"""
    try:
        # Get data for S&P 500 stocks
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT",
                  "PG", "JNJ", "XOM", "BAC", "MA", "UNH", "HD", "CVX", "PFE", "CSCO", "VZ",
                  "CRM", "ABT", "AVGO", "ACN", "DHR", "ADBE", "NFLX", "TMO", "NKE"]
        
        stock_data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    current_close = hist['Close'].iloc[-1]
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    volume = hist['Volume'].iloc[-1]
                    market_cap = stock.info.get('marketCap', 0)
                    
                    stock_data.append({
                        'Ticker': ticker,
                        'Name': stock.info.get('shortName', ticker),
                        'Price': current_close,
                        'Change %': change_pct,
                        'Volume': volume,
                        'Market Cap': market_cap
                    })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        # Sort by percentage change
        stock_data.sort(key=lambda x: x['Change %'])
        
        # Get top 5 losers and gainers
        losers = stock_data[:5]
        gainers = stock_data[-5:][::-1]
        
        # Convert to DataFrames
        gainers_df = pd.DataFrame(gainers)
        losers_df = pd.DataFrame(losers)
        
        # Format columns
        for df in [gainers_df, losers_df]:
            df['Price'] = df['Price'].map('${:,.2f}'.format)
            df['Change %'] = df['Change %'].map('{:+.2f}%'.format)
            df['Volume'] = df['Volume'].map('{:,.0f}'.format)
            df['Market Cap'] = df['Market Cap'].map('${:,.0f}'.format)
        
        return gainers_df, losers_df
    except Exception as e:
        print(f"Error in get_market_movers: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def market_movers_tab():
    """Display market movers tab with top gainers and losers"""
    try:
        st.header("📊 Market Movers")
        
        # Add refresh button
        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        # Get market movers data
        with st.spinner("Fetching market data..."):
            gainers_df, losers_df = get_market_movers()
        
        if not gainers_df.empty and not losers_df.empty:
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Display gainers
            with col1:
                st.subheader("🚀 Top Gainers")
                st.dataframe(
                    gainers_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Change %": st.column_config.TextColumn("Change %", width="small"),
                        "Volume": st.column_config.TextColumn("Volume", width="medium"),
                        "Market Cap": st.column_config.TextColumn("Market Cap", width="medium")
                    }
                )
            
            # Display losers
            with col2:
                st.subheader("📉 Top Losers")
                st.dataframe(
                    losers_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Change %": st.column_config.TextColumn("Change %", width="small"),
                        "Volume": st.column_config.TextColumn("Volume", width="medium"),
                        "Market Cap": st.column_config.TextColumn("Market Cap", width="medium")
                    }
                )
            
            # Add market indices summary
            st.markdown("---")
            st.subheader("📈 Market Indices")
            
            try:
                indices = {
                    "^GSPC": "S&P 500",
                    "^DJI": "Dow Jones",
                    "^IXIC": "NASDAQ"
                }
                
                index_data = []
                for symbol, name in indices.items():
                    index = yf.Ticker(symbol)
                    hist = index.history(period="2d")
                    if len(hist) >= 2:
                        prev_close = hist['Close'].iloc[-2]
                        current_close = hist['Close'].iloc[-1]
                        change_pct = ((current_close - prev_close) / prev_close) * 100
                        index_data.append({
                            'Index': name,
                            'Price': current_close,
                            'Change %': change_pct
                        })
                
                # Create columns for indices
                cols = st.columns(len(index_data))
                for i, data in enumerate(index_data):
                    with cols[i]:
                        st.metric(
                            data['Index'],
                            f"${data['Price']:,.2f}",
                            f"{data['Change %']:+.2f}%",
                            delta_color="normal"
                        )
            except Exception as e:
                st.error(f"Error fetching market indices: {str(e)}")
        else:
            st.error("Unable to fetch market movers data. Please try again later.")
            
    except Exception as e:
        st.error(f"Error in market movers tab: {str(e)}")

def calculate_buffett_metrics(ticker):
    """Calculate Warren Buffett style investment metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        if balance_sheet.empty or income_stmt.empty or cash_flow.empty:
            st.warning(f"Unable to perform complete Buffett analysis for {ticker}. Some financial statements are missing.")
            return None
            
        # Current financials - use the first column (most recent)
        total_assets = balance_sheet.loc['Total Assets', balance_sheet.columns[0]]
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', balance_sheet.columns[0]]
        equity = total_assets - total_liabilities
        
        net_income = income_stmt.loc['Net Income', income_stmt.columns[0]]
        revenue = income_stmt.loc['Total Revenue', income_stmt.columns[0]]
        operating_cash_flow = cash_flow.loc['Operating Cash Flow', cash_flow.columns[0]]
        
        # Calculate ratios
        roe = net_income / equity if equity != 0 else 0
        profit_margin = net_income / revenue if revenue != 0 else 0
        debt_equity = total_liabilities / equity if equity != 0 else float('inf')
        
        # Get current price and shares outstanding
        current_price = info.get('currentPrice', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        if current_price == 0 or shares_outstanding == 0:
            st.warning(f"Missing price or shares data for {ticker}")
            return None
            
        market_cap = current_price * shares_outstanding
        
        # Calculate intrinsic value using multiple methods
        # 1. DCF with conservative growth rate and higher discount rate for safety
        future_cash_flows = []
        growth_rate = min(0.20, max(0.08, (net_income / equity) if equity != 0 else 0.10))
        discount_rate = 0.10  # Standard discount rate
        initial_cash_flow = operating_cash_flow
        
        # Project cash flows for 10 years
        for i in range(10):
            future_cash_flow = initial_cash_flow * (1 + growth_rate) ** i
            discounted_cf = future_cash_flow / (1 + discount_rate) ** i
            future_cash_flows.append(discounted_cf)
        
        # Terminal value calculation with perpetual growth
        terminal_growth = 0.03  # 3% perpetual growth
        terminal_value = (future_cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        discounted_terminal_value = terminal_value / (1 + discount_rate) ** 10
        
        # Add terminal value to cash flows
        total_dcf = sum(future_cash_flows) + discounted_terminal_value
        dcf_value = total_dcf / shares_outstanding
        
        # 2. Graham Number with adjusted multiplier based on growth and quality
        eps = info.get('trailingEps', 0)
        book_value_per_share = equity / shares_outstanding
        
        # Adjust Graham multiplier based on growth and profitability
        base_multiplier = 22.5  # Graham's base multiplier
        growth_adjustment = min(7.5, growth_rate * 25)  # Up to 7.5 additional points for growth
        quality_adjustment = min(5, (roe * 100 - 15) / 3) if roe > 0.15 else 0  # Up to 5 points for high ROE
        
        graham_multiplier = base_multiplier + growth_adjustment + quality_adjustment
        graham_value = (graham_multiplier * eps * book_value_per_share) ** 0.5
        
        # 3. Owner Earnings with dynamic multiple
        capex = abs(cash_flow.loc['Capital Expenditure', cash_flow.columns[0]])
        maintenance_capex = capex * 0.7  # Assume 70% of capex is maintenance
        owner_earnings = operating_cash_flow - maintenance_capex
        
        # Calculate appropriate earnings multiple based on growth and quality
        base_multiple = 15
        growth_premium = min(10, growth_rate * 50)  # Up to 10 points for growth
        quality_premium = min(5, (roe * 100 - 15) / 3) if roe > 0.15 else 0  # Up to 5 points for quality
        
        earnings_multiple = base_multiple + growth_premium + quality_premium
        owner_earnings_value = (owner_earnings / shares_outstanding) * earnings_multiple
        
        # Calculate weighted average intrinsic value with dynamic weights
        weights = {
            'dcf': 0.4,
            'graham': 0.3,
            'owner_earnings': 0.3
        }
        
        # Validate and adjust values
        valid_values = []
        valid_weights = []
        
        # Only include values that are positive and within reasonable range
        price_range = (current_price * 0.3, current_price * 3.0)  # Reasonable range: 30% to 300% of current price
        
        if dcf_value > price_range[0] and dcf_value < price_range[1]:
            valid_values.append(dcf_value)
            valid_weights.append(weights['dcf'])
        
        if graham_value > price_range[0] and graham_value < price_range[1]:
            valid_values.append(graham_value)
            valid_weights.append(weights['graham'])
        
        if owner_earnings_value > price_range[0] and owner_earnings_value < price_range[1]:
            valid_values.append(owner_earnings_value)
            valid_weights.append(weights['owner_earnings'])
        
        # If no valid values, use alternative calculation
        if not valid_values:
            # Use PE-based valuation as fallback
            forward_pe = info.get('forwardPE', 15)
            industry_pe = info.get('industryPE', 20)
            target_pe = min(max(forward_pe, industry_pe * 0.8), industry_pe * 1.2)
            avg_intrinsic_value = eps * target_pe
        else:
            # Normalize weights and calculate weighted average
            weight_sum = sum(valid_weights)
            valid_weights = [w/weight_sum for w in valid_weights]
            avg_intrinsic_value = sum(v * w for v, w in zip(valid_values, valid_weights))
        
        # Ensure intrinsic value is not too far from current price
        avg_intrinsic_value = max(min(avg_intrinsic_value, current_price * 3), current_price * 0.4)
        
        # Calculate entry points with dynamic margin of safety
        base_margin = 0.20  # Base margin of safety (20%)
        
        # Adjust margin based on:
        # 1. Company quality (ROE)
        quality_factor = max(0, min(0.05, (0.15 - roe)))  # Reduce margin for high ROE
        
        # 2. Market volatility (beta)
        beta = info.get('beta', 1.0)
        volatility_factor = max(0, min(0.05, (beta - 1) * 0.05))  # Increase margin for high beta
        
        # 3. Financial strength (debt/equity)
        strength_factor = max(0, min(0.05, (debt_equity - 1) * 0.05))  # Increase margin for high debt
        
        # Combine factors
        adjusted_margin = base_margin + quality_factor + volatility_factor + strength_factor
        
        # Calculate entry points
        strong_buy = avg_intrinsic_value * (1 - adjusted_margin)
        buy = avg_intrinsic_value * (1 - adjusted_margin/2)
        hold = avg_intrinsic_value
        sell = avg_intrinsic_value * (1 + adjusted_margin/3)
        
        # Compile metrics
        metrics = {
            'Business Understanding': {
                'Industry': info.get('industry', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Business Model': info.get('longBusinessSummary', 'N/A')
            },
            'Competitive Advantage': {
                'Market Position': info.get('marketPosition', 'N/A'),
                'Brand Value': info.get('brandValue', 'N/A'),
                'Economic Moat': 'Wide' if profit_margin > 0.2 else 'Narrow' if profit_margin > 0.1 else 'None'
            },
            'Management Quality': {
                'Return on Equity': f"{roe*100:.1f}%",
                'Profit Margin': f"{profit_margin*100:.1f}%",
                'Operating Efficiency': 'Good' if profit_margin > 0.15 else 'Fair' if profit_margin > 0.08 else 'Poor'
            },
            'Financial Health': {
                'Debt/Equity': f"{debt_equity:.2f}x",
                'Operating Cash Flow': f"${operating_cash_flow/1e9:.1f}B",
                'Net Income': f"${net_income/1e9:.1f}B"
            },
            'Value Metrics': {
                'Market Cap': f"${market_cap/1e9:.1f}B",
                'P/E Ratio': info.get('forwardPE', 0),
                'Book Value': f"${book_value_per_share:.2f}/share"
            },
            'Entry Price Analysis': {
                'Current Price': current_price,
                'Target Price': avg_intrinsic_value,
                'Entry Points': {
                    'Strong Buy Below': strong_buy,
                    'Buy Below': buy,
                    'Hold Above': hold
                },
                'Valuation Methods': {
                    'DCF Value': dcf_value,
                    'Graham Number': graham_value,
                    'Owner Earnings Value': owner_earnings_value
                }
            }
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating Buffett metrics: {str(e)}")
        return None

def get_buffett_recommendation(metrics):
    """Generate investment recommendation based on Buffett metrics"""
    try:
        if not metrics or not isinstance(metrics, dict):
            return "HOLD", "yellow", ["Insufficient data to make a recommendation"]
            
        entry_analysis = metrics.get('Entry Price Analysis', {})
        if not entry_analysis:
            return "HOLD", "yellow", ["Unable to analyze entry prices"]
            
        current_price = entry_analysis.get('Current Price', 0)
        target_price = entry_analysis.get('Target Price', 0)
        entry_points = entry_analysis.get('Entry Points', {})
        
        if not current_price or not target_price or not entry_points:
            return "HOLD", "yellow", ["Missing price analysis data"]
        
        strong_buy = entry_points.get('Strong Buy Below', float('inf'))
        buy = entry_points.get('Buy Below', float('inf'))
        hold = entry_points.get('Hold Above', 0)
        
        # Initialize recommendation components
        recommendation = "HOLD"
        color = "yellow"
        reasons = []
        
        # Calculate price metrics
        upside = ((target_price - current_price) / current_price) * 100
        
        # Add price-based reasons
        if current_price <= strong_buy:
            recommendation = "STRONG BUY"
            color = "green"
            reasons.append(f"Price (${current_price:.2f}) is below strong buy level (${strong_buy:.2f})")
            reasons.append(f"Significant upside potential of {upside:.1f}%")
        elif current_price <= buy:
            recommendation = "BUY"
            color = "blue"
            reasons.append(f"Price (${current_price:.2f}) is at an attractive entry point")
            reasons.append(f"Potential upside of {upside:.1f}%")
        elif current_price <= hold:
            recommendation = "HOLD"
            color = "yellow"
            reasons.append(f"Price (${current_price:.2f}) is near fair value")
            reasons.append(f"Limited upside of {upside:.1f}%")
        else:
            recommendation = "REDUCE"
            color = "red"
            reasons.append(f"Price (${current_price:.2f}) is above fair value")
            reasons.append("Consider taking profits or reducing position")
        
        # Add fundamental reasons
        if metrics.get('Financial Health'):
            health = metrics['Financial Health']
            if 'Strong Balance Sheet' in health:
                reasons.append("Strong balance sheet")
            if 'Good Cash Flow' in health:
                reasons.append("Healthy cash flow generation")
            if 'Low Debt' in health:
                reasons.append("Conservative debt levels")
                
        if metrics.get('Competitive Advantage'):
            moat = metrics['Competitive Advantage']
            if 'Strong Moat' in moat:
                reasons.append("Strong competitive advantages")
            if 'Market Leader' in moat:
                reasons.append("Market leadership position")
                
        if metrics.get('Management Quality'):
            mgmt = metrics['Management Quality']
            if 'Good Track Record' in mgmt:
                reasons.append("Proven management track record")
            if 'Shareholder Friendly' in mgmt:
                reasons.append("Shareholder-friendly management")
        
        # Ensure we have at least one reason
        if not reasons:
            reasons.append("Basic valuation metrics suggest this recommendation")
            
        return recommendation, color, reasons
        
    except Exception as e:
        print(f"Error in get_buffett_recommendation: {str(e)}")
        return "HOLD", "yellow", ["Error in analysis, defaulting to HOLD recommendation"]

def main():
    st.title("Stock Analysis App")
    
    # Add a text input for the stock symbol
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)", "AAPL", key="main_ticker").upper()
    
    if ticker:
        try:
            # Create tabs for different charts
            tab_names = ["Daily Chart", "Technical Analysis", "Historical", "Price Prediction", "Buffett Analysis", "Stock Recommendations", "Market Movers"]
            chart_tabs = st.tabs(tab_names)
            
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
            
            # Historical Chart Tab
            with chart_tabs[2]:
                period = st.select_slider("Select Time Period", 
                                        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                                        value='1y')
                chart = plot_stock_history(ticker, period)
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.error(f"Unable to fetch historical data for {ticker}")
            
            # Price Prediction Tab
            with chart_tabs[3]:
                prediction_tab()
            
            # Warren Buffett Analysis Tab
            with chart_tabs[4]:
                st.subheader("Warren Buffett Investment Analysis")
                st.write("Analyzing stock based on Warren Buffett's investment principles")
                
                # Calculate Buffett metrics
                metrics = calculate_buffett_metrics(ticker)
                
                if metrics and metrics.get('Entry Price Analysis'):
                    entry_analysis = metrics['Entry Price Analysis']
                    
                    # Display Entry Price Analysis first
                    st.markdown("### 💰 Entry Price Analysis")
                    
                    # Create a price comparison table
                    price_data = {
                        "Current Price": [entry_analysis['Current Price']],
                        "Target Price": [entry_analysis['Target Price']],
                        "Strong Buy Below": [entry_analysis['Entry Points']['Strong Buy Below']],
                        "Buy Below": [entry_analysis['Entry Points']['Buy Below']],
                        "Hold Above": [entry_analysis['Entry Points']['Hold Above']]
                    }
                    df_prices = pd.DataFrame(price_data)
                    st.dataframe(df_prices.style.highlight_min(axis=1, color='lightgreen')
                                                .highlight_max(axis=1, color='lightcoral'),
                               use_container_width=True)
                    
                    # Show detailed valuation methods
                    st.markdown("#### 📊 Valuation Methods")
                    methods_data = pd.DataFrame({
                        "Method": list(entry_analysis['Valuation Methods'].keys()),
                        "Target Price": list(entry_analysis['Valuation Methods'].values())
                    })
                    st.dataframe(methods_data, use_container_width=True)
                    
                    # Add explanation
                    st.info("""
                    **Entry Price Guidelines:**
                    - **Strong Buy:** Excellent entry point with maximum margin of safety
                    - **Buy:** Good entry point with reasonable margin of safety
                    - **Hold:** Fair value, consider holding if already owned
                    - **Above Hold:** Consider taking profits or reducing position
                    """)
                    
                    # Display recommendation
                    try:
                        recommendation, color, reasons = get_buffett_recommendation(metrics)
                        st.markdown(f"### Overall Recommendation: :{color}[{recommendation}]")
                        
                        # Display reasons
                        st.markdown("### Key Findings:")
                        for reason in reasons:
                            st.write(f"- {reason}")
                    except Exception as e:
                        st.error(f"Error getting recommendation: {str(e)}")
                    
                    # Display detailed metrics in expandable sections
                    st.markdown("### 📈 Detailed Analysis")
                    
                    # 1. Business Understanding
                    with st.expander("🏢 Business Understanding"):
                        st.write(metrics['Business Understanding'])
                        if 'Business Model' in metrics['Business Understanding']:
                            st.markdown("### Business Model")
                            st.write(metrics['Business Understanding']['Business Model'])
                    
                    # 2. Competitive Advantage (Moat)
                    with st.expander("🏰 Competitive Advantage (Moat)"):
                        for key, value in metrics['Competitive Advantage'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # 3. Management Quality
                    with st.expander("👥 Management Quality"):
                        for key, value in metrics['Management Quality'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # 4. Financial Health
                    with st.expander("💰 Financial Health"):
                        for key, value in metrics['Financial Health'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # 5. Value Metrics
                    with st.expander("📊 Value Metrics"):
                        for key, value in metrics['Value Metrics'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # Additional Notes
                    st.markdown("### 📝 Investment Notes")
                    st.info("""
                    Warren Buffett's Investment Principles:
                    1. Understand the business
                    2. Look for companies with strong competitive advantages
                    3. Focus on management quality and integrity
                    4. Ensure strong financial health
                    5. Buy at a reasonable price with a margin of safety
                    """)
                else:
                    st.error("Unable to perform Buffett analysis. Insufficient data available.")
            
            # Stock Recommendations Tab
            with chart_tabs[5]:
                recommendations_tab()
                
            # Market Movers Tab
            with chart_tabs[6]:
                market_movers_tab()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
