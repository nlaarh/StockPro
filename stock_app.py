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

def calculate_macd(prices, fast=12, slow=26):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_bollinger_bands(prices, period=20, std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper_bb = sma + (std_dev * std)
    lower_bb = sma - (std_dev * std)
    return upper_bb, lower_bb

def plot_daily_candlestick(ticker):
    """Plot daily candlestick chart"""
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period='1y')
    fig = go.Figure(data=[go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name=ticker
    )])
    fig.update_layout(
        title=f'{ticker} Daily Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    return fig

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
            macd = calculate_macd(data['Close'], fast=macd_fast, slow=macd_slow)
            signal = macd.ewm(span=macd_signal, adjust=False).mean()
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
        if price_change > 0:
            signals.append("✅ Price is predicted to increase")
        else:
            signals.append("🔻 Price is predicted to decrease")
            
        if rsi > 70:
            signals.append("⚠️ Stock may be overbought (RSI > 70)")
        elif rsi < 30:
            signals.append("💡 Stock may be oversold (RSI < 30)")
            
        if macd > 0:
            signals.append("✅ MACD indicates bullish momentum")
        else:
            signals.append("🔻 MACD indicates bearish momentum")
            
        if volume_change > 10:
            signals.append("📈 Trading volume is increasing")
        elif volume_change < -10:
            signals.append("📉 Trading volume is decreasing")
        
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
        volume_ratio = (recent_volume / avg_volume - 1) * 100
        
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
        - MACD ({macd:.2f}): {'Bullish' if macd > 0 else 'Bearish'}
        - Moving Averages: {'Bullish' if ma20 > ma50 else 'Bearish'} (MA20 vs MA50)
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
        # 1. DCF with 10% growth rate and 15% discount rate
        future_cash_flows = []
        growth_rate = 0.10
        discount_rate = 0.15
        initial_cash_flow = operating_cash_flow
        
        for i in range(10):
            future_cash_flow = initial_cash_flow * (1 + growth_rate) ** i
            discounted_cf = future_cash_flow / (1 + discount_rate) ** i
            future_cash_flows.append(discounted_cf)
        
        dcf_value = sum(future_cash_flows) / shares_outstanding
        
        # 2. Graham Number
        eps = info.get('trailingEps', 0)
        book_value_per_share = equity / shares_outstanding
        graham_value = (22.5 * eps * book_value_per_share) ** 0.5
        
        # 3. Owner Earnings (Buffett's preferred metric)
        capex = abs(cash_flow.loc['Capital Expenditure', cash_flow.columns[0]])
        owner_earnings = operating_cash_flow - capex
        owner_earnings_value = owner_earnings * 15 / shares_outstanding  # 15x multiple
        
        # Calculate average intrinsic value
        intrinsic_values = [dcf_value, graham_value, owner_earnings_value]
        avg_intrinsic_value = sum(v for v in intrinsic_values if v > 0) / len([v for v in intrinsic_values if v > 0])
        
        # Calculate entry points
        margin_of_safety = 0.25  # 25% margin of safety
        strong_buy = avg_intrinsic_value * (1 - margin_of_safety)
        buy = avg_intrinsic_value * (1 - margin_of_safety/2)
        hold = avg_intrinsic_value
        
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
                'P/E Ratio': info.get('forwardPE', 'N/A'),
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
            if health.get('Strong Balance Sheet'):
                reasons.append("Strong balance sheet")
            if health.get('Good Cash Flow'):
                reasons.append("Healthy cash flow generation")
            if health.get('Low Debt'):
                reasons.append("Conservative debt levels")
                
        if metrics.get('Competitive Advantage'):
            moat = metrics['Competitive Advantage']
            if moat.get('Strong Moat'):
                reasons.append("Strong competitive advantages")
            if moat.get('Market Leader'):
                reasons.append("Market leadership position")
                
        if metrics.get('Management Quality'):
            mgmt = metrics['Management Quality']
            if mgmt.get('Good Track Record'):
                reasons.append("Proven management track record")
            if mgmt.get('Shareholder Friendly'):
                reasons.append("Shareholder-friendly management")
        
        # Ensure we have at least one reason
        if not reasons:
            reasons.append("Basic valuation metrics suggest this recommendation")
            
        return recommendation, color, reasons
        
    except Exception as e:
        print(f"Error in get_buffett_recommendation: {str(e)}")
        return "HOLD", "yellow", ["Error in analysis, defaulting to HOLD recommendation"]

def technical_analysis_tab():
    try:
        st.header("Technical Analysis")
        
        # Get the stock data
        ticker = st.session_state.get('main_ticker', 'AAPL')
        if not ticker:
            st.warning("Please enter a stock symbol")
            return
            
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            st.error(f"No data available for {ticker}")
            return
            
        # Calculate technical indicators
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        macd = calculate_macd(hist['Close'])
        if macd is not None:
            hist['MACD'] = macd
        
        # Create technical analysis plots
        fig = make_subplots(rows=3, cols=1, shared_xaxis=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price and Moving Averages', 'RSI', 'MACD'))
        
        # Plot 1: Price and Moving Averages
        fig.add_trace(go.Candlestick(x=hist.index,
                                    open=hist['Open'],
                                    high=hist['High'],
                                    low=hist['Low'],
                                    close=hist['Close'],
                                    name='Price'),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'],
                                name='20-day MA',
                                line=dict(color='orange')),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'],
                                name='50-day MA',
                                line=dict(color='blue')),
                     row=1, col=1)
        
        # Plot 2: RSI
        fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'],
                                name='RSI',
                                line=dict(color='purple')),
                     row=2, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                     annotation_text="Overbought (70)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                     annotation_text="Oversold (30)", row=2, col=1)
        
        # Plot 3: MACD
        if 'MACD' in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'],
                                    name='MACD',
                                    line=dict(color='blue')),
                         row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Analysis Summary
        st.subheader("Technical Analysis Summary")
        
        # Current values
        current_price = hist['Close'].iloc[-1]
        ma20 = hist['MA20'].iloc[-1]
        ma50 = hist['MA50'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        macd_value = hist.get('MACD', pd.Series([0])).iloc[-1]
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("20-day MA", f"${ma20:.2f}")
            
        with col2:
            st.metric("50-day MA", f"${ma50:.2f}")
            st.metric("RSI", f"{rsi:.1f}")
            
        with col3:
            if 'MACD' in hist.columns:
                st.metric("MACD", f"{macd_value:.3f}")
            
        # Technical signals
        signals = []
        
        # Moving Average signals
        if current_price > ma20:
            signals.append("✅ Price above 20-day MA (Short-term bullish)")
        else:
            signals.append("🔴 Price below 20-day MA (Short-term bearish)")
            
        if current_price > ma50:
            signals.append("✅ Price above 50-day MA (Medium-term bullish)")
        else:
            signals.append("🔴 Price below 50-day MA (Medium-term bearish)")
            
        if ma20 > ma50:
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
        if 'MACD' in hist.columns:
            if macd_value > 0:
                signals.append("✅ MACD above 0 (Bullish)")
            else:
                signals.append("🔴 MACD below 0 (Bearish)")
        
        # Display signals
        st.subheader("Technical Signals")
        for signal in signals:
            st.write(signal)
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def prediction_tab():
    try:
        st.header("Stock Price Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            ticker = st.text_input("Enter Stock Ticker:", "AAPL", key="prediction_ticker").upper()
            prediction_days = st.slider("Prediction Days", 1, 30, 7)
            
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "XGBoost", "LightGBM", "Ensemble", "LLM", "Ollama 3.2"]
            )
            
            if model_type in ["LLM", "Ollama 3.2"]:
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
        
        if st.button("Predict"):
            try:
                with st.spinner("Loading stock data..."):
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1y")
                    
                    if hist.empty:
                        st.error(f"No data found for {ticker}")
                        return
                
                with st.spinner("Calculating technical indicators..."):
                    # Calculate technical indicators
                    hist['MA20'] = hist['Close'].rolling(window=20).mean()
                    hist['MA50'] = hist['Close'].rolling(window=50).mean()
                    
                    # RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    hist['RSI'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
                    hist['MACD'] = exp1 - exp2
                    
                    # Bollinger Bands
                    hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
                    hist['BB_Upper'] = hist['BB_Middle'] + 2 * hist['Close'].rolling(window=20).std()
                    hist['BB_Lower'] = hist['BB_Middle'] - 2 * hist['Close'].rolling(window=20).std()
                    
                    # Drop NaN values
                    hist = hist.dropna()
                
                with st.spinner("Making predictions..."):
                    # Get selected features
                    feature_sets = {
                        "Basic": ["Open", "High", "Low", "Volume"],
                        "Technical": ["Open", "High", "Low", "Volume", "MA20", "MA50", "RSI", "MACD"],
                        "Advanced": ["Open", "High", "Low", "Volume", "MA20", "MA50", "RSI", "MACD", "BB_Upper", "BB_Lower"]
                    }
                    features = feature_sets["Advanced"]  # Use advanced features for better predictions
                    
                    # Verify data preparation
                    data_ok, data_error = test_data_preparation(hist, features, 'Close')
                    if not data_ok:
                        st.error(f"Data preparation error: {data_error}")
                        return
                    
                    # Make prediction
                    result = predict_stock_price(hist, model_type, features, 'Close', prediction_days)
                    
                    if result is None or not isinstance(result, tuple) or len(result) != 3:
                        st.error("""
                        Failed to generate prediction. This could be due to:
                        1. Insufficient historical data
                        2. Invalid feature values
                        3. Model training failure
                        
                        Please try:
                        1. A different stock ticker
                        2. A different prediction model
                        3. Reducing the prediction days
                        """)
                        return
                    
                    # Safely unpack the result
                    predictions, y_test, confidence_bands = result
                    
                    # Additional validation
                    if predictions is None or y_test is None or confidence_bands is None:
                        st.error("Invalid prediction results")
                        return
                        
                    if len(predictions) == 0 or len(y_test) == 0:
                        st.error("Empty prediction results")
                        return
                        
                    if not isinstance(confidence_bands, tuple) or len(confidence_bands) != 2:
                        st.error("Invalid confidence bands")
                        return
                
                with st.spinner("Generating analysis..."):
                    # Generate and display summary
                    summary = generate_prediction_summary(
                        ticker, 
                        hist, 
                        model_type, 
                        prediction_days, 
                        predictions,
                        confidence_bands
                    )
                    
                    if summary:
                        with st.expander("📈 Detailed Prediction Analysis", expanded=True):
                            st.markdown(summary)
                    else:
                        st.warning("Could not generate prediction summary")
                
                with st.spinner("Creating visualization..."):
                    # Display prediction plot
                    fig = go.Figure()
                    
                    # Add actual prices
                    fig.add_trace(go.Scatter(
                        y=y_test,
                        name="Actual Price",
                        line=dict(color='blue')
                    ))
                    
                    # Add predicted prices
                    fig.add_trace(go.Scatter(
                        y=predictions,
                        name="Predicted Price",
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add confidence bands if available
                    if isinstance(confidence_bands, tuple) and len(confidence_bands) == 2:
                        lower_bound, upper_bound = confidence_bands
                        if lower_bound is not None and upper_bound is not None:
                            fig.add_trace(go.Scatter(
                                y=upper_bound,
                                name="Upper Bound",
                                line=dict(color='rgba(200,100,100,0.2)'),
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                y=lower_bound,
                                name="Lower Bound",
                                fill='tonexty',
                                fillcolor='rgba(200,100,100,0.1)',
                                line=dict(color='rgba(200,100,100,0.2)'),
                                showlegend=False
                            ))
                    
                    fig.update_layout(
                        title=f"{ticker} Stock Price Prediction",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"""
                Error during prediction: {str(e)}
                
                This might be due to:
                1. Network connectivity issues
                2. Invalid stock data
                3. Computation error
                
                Please try:
                1. Checking your internet connection
                2. Using a different stock ticker
                3. Selecting a different model
                """)
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
        market_cap = info.get('marketCap', 0) / 1e9  # Convert to billions
        pe_ratio = info.get('forwardPE', 'N/A')
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
- **Market Cap**: ${market_cap:.1f}B
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

def main():
    st.title("Stock Analysis App")
    
    # Add a text input for the stock symbol
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)", "AAPL", key="main_ticker").upper()
    
    if ticker:
        try:
            # Create tabs for different charts
            tab_names = ["Daily Chart", "Technical Analysis", "Historical", "Price Prediction", "Buffett Analysis", "Stock Recommendations"]
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
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
