import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import json
import re
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import percentileofscore
from utils import calculate_rsi, calculate_macd

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

def test_ollama_connection(model_type="Ollama 3.2"):
    """Test connection to Ollama server and model availability"""
    try:
        # Map model types to their Ollama model names
        model_mapping = {
            "Ollama 3.2": "llama2",
            "DeepSeek-R1": "deepseek-r1:latest"
        }
        
        model_name = model_mapping.get(model_type)
        if not model_name:
            return False, f"Unsupported model type: {model_type}"
            
        # Test connection
        response = requests.get('http://localhost:11434/api/tags')
        
        if response.status_code != 200:
            return False, f"Failed to connect to Ollama server: {response.text}"
            
        # Check if model is available
        available_models = response.json().get('models', [])
        if not any(m['name'].startswith(model_name.split(':')[0]) for m in available_models):
            return False, f"Model {model_name} not found. Please run:\n```bash\nollama pull {model_name}\n```"
            
        return True, "Connection successful"
        
    except requests.exceptions.ConnectionError:
        return False, "Failed to connect to Ollama server. Make sure it's running on port 11434."
    except Exception as e:
        return False, f"Unexpected error testing {model_type} connection: {str(e)}"


def get_llm_response(prompt, model_type="Ollama 3.2"):
    """Get response from LLM model"""
    try:
        # Map model types to their Ollama model names
        model_mapping = {
            "Ollama 3.2": "llama2",
            "DeepSeek-R1": "deepseek-r1:latest"
        }
        
        model_name = model_mapping.get(model_type)
        if not model_name:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Check if model is installed
        try:
            tags_response = requests.get('http://localhost:11434/api/tags', timeout=10)
            if tags_response.status_code == 200:
                available_models = tags_response.json().get('models', [])
                model_installed = any(m['name'].startswith(model_name.split(':')[0]) for m in available_models)
                if not model_installed:
                    install_msg = f"Model {model_name} not found. Please install it using:\n```bash\nollama pull {model_name}\n```"
                    st.error(install_msg)
                    logger.error(install_msg)
                    return None
        except Exception as e:
            logger.warning(f"Could not check model availability: {str(e)}")
            
        # Log the request for debugging
        logger.info(f"Sending request to Ollama API with model: {model_name}")
        
        # Show a status message while waiting
        with st.spinner(f'Generating prediction with {model_type}... This may take up to 60 seconds.'):
            # Increased timeout to 60 seconds for larger models
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,  # Lower temperature for more focused predictions
                        'top_p': 0.9,
                        'num_predict': 200
                    }
                },
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                logger.error(f"Ollama API error: {result['error']}")
                st.error(f"Ollama API error: {result['error']}")
                return None
            return result['response']
        else:
            error_msg = f"Error from Ollama API (status {response.status_code}): {response.text}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
            
    except requests.exceptions.Timeout:
        error_msg = f"Request to {model_type} timed out after 60 seconds. Try using a faster model like Llama2."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    except requests.exceptions.ConnectionError:
        error_msg = "Failed to connect to Ollama server. Make sure it's running on port 11434."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error getting LLM response: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

def predict_stock_price(data, prediction_days=7, model_type="Random Forest"):
    """Predict stock prices using various models"""
    try:
        # For LLM models, use a different prediction approach
        if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            return predict_with_llm(data, prediction_days, model_type)
            
        # Traditional ML models
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal']].values)
        
        # Create sequences for training
        X = []
        y = []
        
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Initialize model based on type
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            
            # Make predictions
            test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
            
            # Prepare future prediction data
            last_60_days = scaled_data[-60:]
            future_pred = []
            current_batch = last_60_days.copy()
            
            for _ in range(prediction_days):
                # Prepare current batch for prediction
                current_features = current_batch.reshape(1, -1)
                next_pred = model.predict(current_features)
                
                # Update current batch
                new_row = np.zeros((1, 6))  # 6 features
                new_row[0, 0] = next_pred
                current_batch = np.vstack((current_batch[1:], new_row))
                future_pred.append(float(next_pred))  # Convert to float
                
        elif model_type == "XGBoost":
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            
            # Make predictions
            test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
            
            # Prepare future prediction data
            last_60_days = scaled_data[-60:]
            future_pred = []
            current_batch = last_60_days.copy()
            
            for _ in range(prediction_days):
                current_features = current_batch.reshape(1, -1)
                next_pred = model.predict(current_features)
                
                new_row = np.zeros((1, 6))
                new_row[0, 0] = next_pred
                current_batch = np.vstack((current_batch[1:], new_row))
                future_pred.append(float(next_pred))  # Convert to float
                
        elif model_type == "LightGBM":
            model = lgb.LGBMRegressor(objective='regression', n_estimators=100)
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            
            # Make predictions
            test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
            
            # Prepare future prediction data
            last_60_days = scaled_data[-60:]
            future_pred = []
            current_batch = last_60_days.copy()
            
            for _ in range(prediction_days):
                current_features = current_batch.reshape(1, -1)
                next_pred = model.predict(current_features)
                
                new_row = np.zeros((1, 6))
                new_row[0, 0] = next_pred
                current_batch = np.vstack((current_batch[1:], new_row))
                future_pred.append(float(next_pred))  # Convert to float
        
        # Convert predictions back to original scale
        future_pred = np.array(future_pred).reshape(-1, 1)
        test_pred = test_pred.reshape(-1, 1)
        
        # Create dummy values for other features to use with scaler
        future_dummy = np.zeros((len(future_pred), 6))
        future_dummy[:, 0] = future_pred[:, 0]
        test_dummy = np.zeros((len(test_pred), 6))
        test_dummy[:, 0] = test_pred[:, 0]
        
        # Inverse transform
        future_pred = scaler.inverse_transform(future_dummy)[:, 0]
        test_pred = scaler.inverse_transform(test_dummy)[:, 0]
        
        # Calculate confidence bands (±2% of predicted values)
        confidence_lower = [float(x * 0.98) for x in future_pred]  # Convert to list of floats
        confidence_upper = [float(x * 1.02) for x in future_pred]  # Convert to list of floats
        future_pred = [float(x) for x in future_pred]  # Convert to list of floats
        test_pred = [float(x) for x in test_pred]  # Convert to list of floats
        
        return future_pred, test_pred, (confidence_lower, confidence_upper)
        
    except Exception as e:
        logger.error(f"Error in predict_stock_price: {str(e)}")
        st.error(f"Prediction error: {str(e)}")  # Show error in UI
        return None

def predict_with_llm(data, prediction_days=7, model_type="Ollama 3.2"):
    """Predict stock prices using LLM models"""
    try:
        # Map model types to their Ollama model names
        model_mapping = {
            "Ollama 3.2": "llama2",
            "DeepSeek-R1": "deepseek-r1:latest"
        }
        
        model_name = model_mapping.get(model_type)
        if not model_name:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Get recent price history
        recent_prices = data['Close'].tail(30).tolist()
        current_price = float(data['Close'].iloc[-1])
        
        # Calculate key metrics
        avg_price = sum(recent_prices) / len(recent_prices)
        volatility = np.std(recent_prices) / avg_price * 100
        rsi = float(data['RSI'].iloc[-1])
        macd = float(data['MACD'].iloc[-1])
        signal = float(data['Signal'].iloc[-1])
        ma20 = float(data['MA20'].iloc[-1])
        ma50 = float(data['MA50'].iloc[-1])
        
        # Create prompt for LLM
        prompt = f"""As a financial analyst, predict the next {prediction_days} daily closing prices for a stock with these characteristics:

Current Price: ${current_price:.2f}
30-Day Average: ${avg_price:.2f}
Volatility: {volatility:.1f}%
RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
MACD: {macd:.3f} vs Signal: {signal:.3f} ({'Bullish' if macd > signal else 'Bearish'})
Price vs MA20 (${ma20:.2f}): {'Above' if current_price > ma20 else 'Below'}
Price vs MA50 (${ma50:.2f}): {'Above' if current_price > ma50 else 'Below'}

Recent Daily Prices: {', '.join([f'${x:.2f}' for x in recent_prices[-5:]])}

Instructions:
1. First write "PREDICTIONS:" on a new line
2. Then list exactly {prediction_days} prices, one per line
3. Only write numbers, no dollar signs or text
4. Then write "EXPLANATION:" followed by your analysis

Here's the exact format to follow:

PREDICTIONS:
{current_price:.2f}
{current_price * 1.01:.2f}
{current_price * 1.02:.2f}
{current_price * 1.015:.2f}
{current_price * 1.025:.2f}
{current_price * 1.03:.2f}
{current_price * 1.035:.2f}

EXPLANATION:
Based on the technical indicators...

Remember:
- Write exactly {prediction_days} numbers
- Only numbers in the PREDICTIONS section
- Each prediction on its own line
- No text or symbols in the predictions"""

        # Get LLM response
        response = get_llm_response(prompt, model_type)
        if not response:
            raise Exception("Failed to get LLM response")
            
        # Log full response for debugging
        logger.info(f"Raw LLM Response:\n{response}")
        
        # Parse response
        try:
            # First try exact format
            if "PREDICTIONS:" not in response:
                # Try to find numbers in the response if format is wrong
                logger.warning("Response missing PREDICTIONS section, attempting to extract numbers")
                numbers = []
                for line in response.split('\n'):
                    try:
                        # Clean the line and extract numbers
                        cleaned = ''.join(c for c in line if c.isdigit() or c == '.' or c == ' ')
                        for num in cleaned.split():
                            try:
                                price = float(num)
                                if 0.5 * current_price <= price <= 1.5 * current_price:
                                    numbers.append(price)
                            except:
                                continue
                    except:
                        continue
                
                if len(numbers) >= prediction_days:
                    predictions = numbers[:prediction_days]
                else:
                    # Fallback: Generate predictions using percentage changes
                    base = current_price
                    predictions = []
                    for i in range(prediction_days):
                        # Generate small random changes (-2% to +2%)
                        change = np.random.uniform(-0.02, 0.02)
                        price = base * (1 + change)
                        predictions.append(price)
                        base = price  # Use this as base for next prediction
            else:
                # Parse normal format
                pred_section = response.split("PREDICTIONS:")[1].split("EXPLANATION:")[0].strip()
                
                # Parse predictions into list of floats
                predictions = []
                for line in pred_section.split('\n'):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    try:
                        # Remove any non-numeric characters except decimal point
                        cleaned = ''.join(c for c in line if c.isdigit() or c == '.')
                        if cleaned:
                            price = float(cleaned)
                            if 0.5 * current_price <= price <= 1.5 * current_price:
                                predictions.append(price)
                    except:
                        continue
            
            # Handle prediction count
            if len(predictions) < prediction_days:
                # If we have some predictions but not enough, interpolate
                if len(predictions) > 1:
                    # Linear interpolation between existing predictions
                    step = (predictions[-1] - predictions[0]) / (prediction_days - 1)
                    predictions = [predictions[0] + step * i for i in range(prediction_days)]
                else:
                    # If we only have one or no predictions, use current price with small random changes
                    base = current_price
                    predictions = []
                    for i in range(prediction_days):
                        change = np.random.uniform(-0.02, 0.02)
                        price = base * (1 + change)
                        predictions.append(price)
                        base = price
            
            # Trim if we have too many
            predictions = predictions[:prediction_days]
            
            # Create confidence bands
            confidence_lower = [x * 0.98 for x in predictions]
            confidence_upper = [x * 1.02 for x in predictions]
            test_pred = [current_price]  # Not applicable for LLM
            
            return predictions, test_pred, (confidence_lower, confidence_upper)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}\nResponse: {response}")
            raise Exception(f"Failed to parse LLM response: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in predict_with_llm: {str(e)}")
        st.error(f"LLM prediction error: {str(e)}")
        return None

def get_analyst_writeup(ticker, predictions, model_type="Ollama 3.2"):
    """Get detailed analyst writeup for a stock using LLM"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price and calculate metrics
        current_price = info.get('regularMarketPrice')
        if not current_price:
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            else:
                return None
                
        target_price = predictions[-1] if predictions else None
        if not target_price or not current_price:
            return None
            
        # Calculate key metrics
        price_change = ((target_price - current_price) / current_price) * 100
        recommendation = "Strong Buy" if price_change > 15 else "Buy" if price_change > 5 else "Hold" if price_change > -5 else "Sell"
        
        # Get company info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
        pe_ratio = info.get('forwardPE', 'N/A')
        revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 'N/A'
        profit_margins = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'N/A'
        
        # Get historical performance
        hist_data = stock.history(period="6mo")
        six_month_return = ((current_price - hist_data['Close'].iloc[0]) / hist_data['Close'].iloc[0]) * 100
        
        # Calculate moving averages
        ma20 = hist_data['Close'].tail(20).mean()
        ma50 = hist_data['Close'].tail(50).mean()
        ma_trend = "bullish" if ma20 > ma50 else "bearish"
        
        # Calculate support and resistance
        recent_lows = hist_data['Low'].tail(30)
        recent_highs = hist_data['High'].tail(30)
        support_level = max(recent_lows.nsmallest(3).mean() * 0.98, current_price * 0.85)
        resistance_level = min(recent_highs.nlargest(3).mean() * 1.02, current_price * 1.15)
        
        # Create narrative letter
        letter = f"""
        # ATLAS INVESTMENT LLC
        *Investment Research Division*

        {pd.Timestamp.now().strftime('%B %d, %Y')}

        Dear Valued Investors,

        After conducting an extensive analysis of {company_name} ({ticker}), I want to share my insights and outlook for this {industry} company. As your investment strategist, I believe it's crucial to not just present the numbers, but to explain what they mean for your investment decisions.

        Let me start with our conclusion: We are issuing a **{recommendation.upper()}** rating with a {len(predictions)}-day price target of **${target_price:.2f}**. This represents a potential {price_change:+.1f}% {'upside' if price_change > 0 else 'downside'} from the current price of ${current_price:.2f}. Here's the story behind these numbers.

        First, let's understand where {ticker} stands in the market. With a market capitalization of {market_cap_str}, {company_name} {'is a significant player' if market_cap > 1e9 else 'is an emerging competitor'} in the {sector} sector. {'This size gives them considerable advantages in terms of market access and capital deployment.' if market_cap > 1e9 else 'While smaller, this size allows them to be nimble and adapt quickly to market changes.'} 

        The company's recent performance tells an interesting story. Over the past six months, the stock has {'gained' if six_month_return > 0 else 'declined'} {abs(six_month_return):.1f}%, {'outpacing' if six_month_return > 0 else 'lagging'} its peers. {'This momentum is particularly noteworthy' if six_month_return > 0 else 'This weakness presents an opportunity'} when we consider the broader market context.

        Looking at profitability, {company_name} maintains a {profit_margins:.1f}% profit margin, which is {'impressive' if profit_margins > 15 else 'reasonable' if profit_margins > 8 else 'concerning'}. {'This strong profitability suggests pricing power and operational efficiency' if profit_margins > 15 else 'While not industry-leading, these margins provide stability' if profit_margins > 8 else 'This indicates challenges in maintaining pricing power'}. Revenue growth of {revenue_growth:.1f}% {'demonstrates strong market acceptance' if revenue_growth > 15 else 'shows steady progress' if revenue_growth > 5 else 'suggests market challenges'}.

        The technical picture {'reinforces' if (price_change > 0 and ma_trend == 'bullish') or (price_change < 0 and ma_trend == 'bearish') else 'contrasts with'} our fundamental view. The stock's moving averages show a {ma_trend} trend, with the 20-day average at ${ma20:.2f} and the 50-day at ${ma50:.2f}. {'This upward momentum supports our bullish thesis' if ma_trend == 'bullish' and price_change > 0 else 'This technical weakness may present buying opportunities' if ma_trend == 'bearish' and price_change > 0 else 'The technical picture suggests caution'}.

        Looking forward, several factors shape our {len(predictions)}-day outlook. {'The combination of strong fundamentals and positive technical trends suggests continued upside' if price_change > 0 and ma_trend == 'bullish' else 'Despite technical weakness, fundamental improvements should drive price appreciation' if price_change > 0 else 'Current headwinds may persist, suggesting a cautious approach'}. Our analysis identifies key price levels: support around ${support_level:.2f} and resistance near ${resistance_level:.2f}. These levels provide natural entry and exit points for position management.

        Based on this comprehensive analysis, here's my recommended strategy:

        For investors {'looking to build' if price_change > 0 else 'holding'} positions in {ticker}:
        1. {'Begin accumulating' if price_change > 0 else 'Maintain'} a {'3-5%' if price_change > 0 else '1-2%'} portfolio position
        2. Use ${support_level:.2f} as a key support level for adding to positions
        3. Consider taking profits at ${target_price:.2f}
        4. Maintain a stop-loss at ${support_level * 0.95:.2f} to manage risk

        {'The current setup presents a compelling entry point' if price_change > 0 else 'While we remain cautious, any pullback to support levels could offer opportunities'}. {company_name}'s {'strong' if price_change > 0 else 'evolving'} market position, combined with {'favorable' if price_change > 0 else 'challenging'} technical indicators, suggests {'this is an opportune time to build positions' if price_change > 0 else 'patience is warranted'}.

        I will continue monitoring {ticker}'s progress and provide updates as the situation evolves. As always, please reach out if you would like to discuss this analysis in more detail.

        Best regards,

        **Nour Laaroubi**
        Chief Investment Strategist
        Atlas Investment LLC
        nour.laaroubi@atlasinv.com

        ---
        *This analysis reflects our current view and is subject to change with market conditions.*
        """
        return letter
            
    except Exception as e:
        st.error(f"Error in analyst writeup: {str(e)}")
        return None


def get_buffett_analysis(ticker, financials, info):
    """Generate Warren Buffett style analysis for a stock"""
    try:
        current_price = info.get('currentPrice', 0)
        
        # Calculate key metrics
        metrics = {
            'current_price': current_price,
            'roe': safe_divide(info.get('returnOnEquity', 0), 1) * 100,
            'operating_margin': safe_divide(info.get('operatingMargins', 0), 1) * 100,
            'gross_margin': safe_divide(info.get('grossMargins', 0), 1) * 100,
            'current_ratio': safe_divide(info.get('currentRatio', 0), 1),
            'debt_to_equity': safe_divide(info.get('debtToEquity', 0), 1),
            'fcf_yield': safe_divide(info.get('freeCashflow', 0), info.get('marketCap', 1)) * 100,
            'pe_ratio': safe_divide(info.get('forwardPE', 0), 1),
            'revenue_growth': safe_divide(info.get('revenueGrowth', 0), 1) * 100,
            'margin_of_safety': calculate_margin_of_safety(info)
        }
        
        # Generate analysis text
        analysis = f"""
        Investment Analysis for {ticker}

        Financial Strength:
        - Return on Equity: {metrics['roe']:.1f}%
        - Operating Margin: {metrics['operating_margin']:.1f}%
        - Debt to Equity: {metrics['debt_to_equity']:.2f}

        Growth Metrics:
        - Revenue Growth: {metrics['revenue_growth']:.1f}%
        - Earnings Growth: {safe_divide(info.get('earningsGrowth', 0), 1) * 100:.1f}%

        Valuation:
        - P/E Ratio: {metrics['pe_ratio']:.2f}
        - P/B Ratio: {safe_divide(info.get('priceToBook', 0), 1):.2f}
        - Free Cash Flow Yield: {metrics['fcf_yield']:.1f}%
        """
        
        return metrics, analysis
        
    except Exception as e:
        logger.error(f"Error in get_buffett_analysis: {str(e)}")
        return None, None

def calculate_margin_of_safety(info):
    """Calculate margin of safety based on various valuation methods"""
    try:
        current_price = info.get('currentPrice', 0)
        if not current_price:
            return 0
            
        # Graham Number
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        graham_value = (22.5 * eps * bvps) ** 0.5
        
        # DCF Value
        fcf = info.get('freeCashflow', 0)
        shares = info.get('sharesOutstanding', 1)
        growth_rate = max(info.get('revenueGrowth', 0.05), 0.05)  # Use at least 5% growth
        discount_rate = 0.10  # 10% discount rate
        
        fcf_per_share = safe_divide(fcf, shares)
        dcf_value = calculate_dcf_value(fcf_per_share, growth_rate, discount_rate)
        
        # Owner Earnings (Warren Buffett's preferred metric)
        capex = info.get('capitalExpenditures', 0)
        depreciation = info.get('totalDebt', 0) * 0.1  # Estimate depreciation as 10% of debt
        net_income = info.get('netIncomeToCommon', 0)
        owner_earnings = net_income + depreciation - capex
        owner_earnings_per_share = safe_divide(owner_earnings, shares)
        owner_earnings_value = owner_earnings_per_share * 15  # Apply a 15x multiple
        
        # Calculate average intrinsic value
        intrinsic_values = [v for v in [graham_value, dcf_value, owner_earnings_value] if v > 0]
        if not intrinsic_values:
            return 0
            
        avg_intrinsic_value = sum(intrinsic_values) / len(intrinsic_values)
        margin_of_safety = ((avg_intrinsic_value - current_price) / current_price) * 100
        
        return margin_of_safety
        
    except Exception as e:
        logger.error(f"Error calculating margin of safety: {str(e)}")
        return 0

def calculate_dcf_value(fcf, growth_rate, discount_rate, years=10):
    """Calculate DCF value using perpetuity growth method"""
    try:
        if not fcf:
            return 0
            
        # Calculate present value of future cash flows
        present_value = 0
        terminal_value = fcf * (1 + growth_rate) ** years * (1 + 0.03) / (discount_rate - 0.03)
        
        for year in range(1, years + 1):
            fcf *= (1 + growth_rate)
            present_value += fcf / (1 + discount_rate) ** year
            
        # Add terminal value
        present_value += terminal_value / (1 + discount_rate) ** years
        
        return present_value
        
    except Exception as e:
        logger.error(f"Error in DCF calculation: {str(e)}")
        return 0


def get_stock_data(ticker):
    """Get stock data with technical indicators"""
    try:
        # Configure yfinance with longer timeout and retries
        stock = yf.Ticker(ticker)
        stock._base_url = "https://query2.finance.yahoo.com"
        stock._timeout = 60  # Increase timeout to 60 seconds
        stock._retry = 3     # Number of retries
        stock._pause = 1     # Pause between retries
        
        # Get stock info with error handling
        try:
            info = stock.info
            if not info:
                raise ValueError(f"No data available for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching stock info: {str(e)}")
            raise ValueError(f"Could not fetch data for {ticker}. Error: {str(e)}")
        
        # Get financials with error handling
        try:
            financials = stock.financials
        except Exception as e:
            logger.warning(f"Could not fetch financials: {str(e)}")
            financials = pd.DataFrame()
        
        # Get history with error handling
        try:
            hist = stock.history(period="1y")
            if hist.empty:
                raise ValueError(f"No historical data available for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise ValueError(f"Could not fetch historical data for {ticker}. Error: {str(e)}")
        
        return hist
        
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        raise

def buffett_analysis_tab():
    """Display Warren Buffett style analysis tab"""
    try:
        st.header(" Warren Buffett Analysis")
        
        ticker = st.session_state.get('ticker', '')
        if not ticker:
            st.warning("Please enter a stock ticker in the sidebar.")
            return
            
        with st.spinner("Analyzing through Warren Buffett's lens..."):
            try:
                # Get stock data
                info, financials, hist = get_stock_data(ticker)
                if not info:
                    st.error(f"Could not fetch data for {ticker}. Please verify the ticker symbol.")
                    return
                    
                # Calculate metrics
                metrics = {
                    'current_price': info.get('currentPrice', 0),
                    'roe': safe_divide(info.get('returnOnEquity', 0), 1) * 100,
                    'operating_margin': safe_divide(info.get('operatingMargins', 0), 1) * 100,
                    'gross_margin': safe_divide(info.get('grossMargins', 0), 1) * 100,
                    'debt_to_equity': safe_divide(info.get('debtToEquity', 0), 1),
                    'current_ratio': safe_divide(info.get('currentRatio', 0), 1),
                    'fcf_yield': safe_divide(info.get('freeCashflow', 0), info.get('marketCap', 1)) * 100,
                    'pe_ratio': safe_divide(info.get('forwardPE', 0), 1),
                    'revenue_growth': safe_divide(info.get('revenueGrowth', 0), 1) * 100,
                    'earnings_growth': safe_divide(info.get('earningsGrowth', 0), 1) * 100
                }
                
                # 1. Warren Buffett's Key Metrics Table
                st.subheader(" Warren Buffett's Key Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Return on Equity (ROE)',
                        'Operating Margin',
                        'Gross Margin',
                        'Debt to Equity',
                        'Current Ratio',
                        'Free Cash Flow Yield',
                        'P/E Ratio',
                        'Revenue Growth',
                        'Earnings Growth'
                    ],
                    'Value': [
                        f"{metrics['roe']:.1f}%",
                        f"{metrics['operating_margin']:.1f}%",
                        f"{metrics['gross_margin']:.1f}%",
                        f"{metrics['debt_to_equity']:.2f}",
                        f"{metrics['current_ratio']:.2f}",
                        f"{metrics['fcf_yield']:.1f}%",
                        f"{metrics['pe_ratio']:.2f}",
                        f"{metrics['revenue_growth']:.1f}%",
                        f"{metrics['earnings_growth']:.1f}%"
                    ],
                    'Buffett\'s Criteria': [
                        ' Excellent' if metrics['roe'] > 15 else ' Below Target',
                        ' Strong' if metrics['operating_margin'] > 20 else ' Needs Improvement',
                        ' Good' if metrics['gross_margin'] > 40 else ' Below Target',
                        ' Conservative' if metrics['debt_to_equity'] < 50 else ' High Leverage',
                        ' Strong' if metrics['current_ratio'] > 1.5 else ' Tight Liquidity',
                        ' Attractive' if metrics['fcf_yield'] > 5 else ' Low Yield',
                        ' Fair' if metrics['pe_ratio'] < 20 else ' Expensive',
                        ' Good' if metrics['revenue_growth'] > 10 else ' Slow Growth',
                        ' Strong' if metrics['earnings_growth'] > 10 else ' Weak Growth'
                    ]
                })
                st.table(metrics_df)
                
                # 2. Buy/Sell Price Recommendations
                st.subheader(" Investment Price Targets")
                current_price = metrics['current_price']
                
                targets_df = pd.DataFrame({
                    'Rating': ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'],
                    'Price Range': [
                        f"Below ${current_price * 0.7:.2f}",
                        f"${current_price * 0.7:.2f} - ${current_price * 0.85:.2f}",
                        f"${current_price * 0.85:.2f} - ${current_price * 1.15:.2f}",
                        f"${current_price * 1.15:.2f} - ${current_price * 1.3:.2f}",
                        f"Above ${current_price * 1.3:.2f}"
                    ],
                    'Margin of Safety': [
                        '30%+ discount',
                        '15-30% discount',
                        '±15% of fair value',
                        '15-30% premium',
                        '30%+ premium'
                    ]
                })
                st.table(targets_df)
                
                # 3. Key Takeaways
                st.subheader(" Key Investment Takeaways")
                takeaways = []
                
                # Profitability
                if metrics['roe'] > 15 and metrics['operating_margin'] > 20:
                    takeaways.append(" Strong profitability with high ROE and margins")
                elif metrics['roe'] > 12:
                    takeaways.append(" Decent profitability but room for improvement")
                else:
                    takeaways.append(" Weak profitability metrics")
                
                # Financial Health
                if metrics['debt_to_equity'] < 50 and metrics['current_ratio'] > 1.5:
                    takeaways.append(" Strong balance sheet with conservative debt")
                elif metrics['debt_to_equity'] < 100:
                    takeaways.append(" Moderate financial leverage")
                else:
                    takeaways.append(" High debt levels - potential risk")
                
                # Growth
                if metrics['revenue_growth'] > 10 and metrics['earnings_growth'] > 10:
                    takeaways.append(" Strong growth in both revenue and earnings")
                elif metrics['revenue_growth'] > 5:
                    takeaways.append(" Moderate growth - monitor trends")
                else:
                    takeaways.append(" Weak growth profile")
                
                # Valuation
                if metrics['pe_ratio'] < 15 and metrics['fcf_yield'] > 5:
                    takeaways.append(" Attractively valued with good cash flow yield")
                elif metrics['pe_ratio'] < 20:
                    takeaways.append(" Fair valuation but not a bargain")
                else:
                    takeaways.append(" Relatively expensive valuation")
                
                st.markdown("\n".join(takeaways))
                
                # 4. Warren Buffett's Letter
                st.subheader(" Warren Buffett's Letter to Shareholders")
                st.markdown(f"""
                Dear Fellow Shareholders of {ticker},
                
                I want to share my thoughts on our investment in {info.get('shortName', ticker)}. As you know, 
                we at Berkshire look for businesses with durable competitive advantages, strong management, and 
                attractive financials at reasonable prices.
                
                {info.get('shortName', ticker)} has demonstrated {'strong' if metrics['roe'] > 15 else 'concerning'} 
                profitability with a {metrics['roe']:.1f}% return on equity. The company's operating margin of 
                {metrics['operating_margin']:.1f}% {'reflects solid pricing power' if metrics['operating_margin'] > 20 
                else 'suggests challenges in maintaining pricing power'}.
                
                {'One aspect that particularly impresses me' if metrics['debt_to_equity'] < 50 
                else 'An area of concern'} is the company's financial position. 
                With a debt-to-equity ratio of {metrics['debt_to_equity']:.1f}, the business 
                {'maintains a conservative balance sheet' if metrics['debt_to_equity'] < 50 
                else 'carries more leverage than we typically prefer'}.
                
                The company's free cash flow yield of {metrics['fcf_yield']:.1f}% 
                {'provides a comfortable margin of safety' if metrics['fcf_yield'] > 5 
                else 'is lower than we typically seek'}, and its P/E ratio of {metrics['pe_ratio']:.1f} 
                {'suggests a reasonable valuation' if metrics['pe_ratio'] < 20 
                else 'indicates the market may be too optimistic'}.
                
                {'In conclusion, this business exemplifies the qualities we seek in a long-term investment.' 
                if metrics['roe'] > 15 and metrics['debt_to_equity'] < 50 and metrics['fcf_yield'] > 5 
                else 'While the business has some attractive qualities, it doesn\'t quite meet all our investment criteria at this time.'}
                
                Sincerely,
                Warren Buffett
                """)
                
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")
                logger.error(f"Error in Buffett analysis: {str(e)}")
                
    except Exception as e:
        st.error(f"Error in Buffett analysis tab: {str(e)}")
        logger.error(f"Error in Buffett analysis tab: {str(e)}")


def display_investor_letter(stock_data, predictions, confidence_bands):
    """Display CIO letter with market analysis"""
    try:
        # Get current price and predicted values
        current_price = stock_data['Close'].iloc[-1]
        final_prediction = predictions[-1]
        price_change = ((final_prediction - current_price) / current_price) * 100
        
        # Calculate technical indicators
        rsi = stock_data['RSI'].iloc[-1]
        macd, signal, hist = calculate_macd(stock_data['Close'])
        macd_value = macd.iloc[-1]
        ma20 = stock_data['MA20'].iloc[-1]
        
        # Determine market sentiment
        sentiment = "bullish" if price_change > 0 else "bearish"
        strength = "strong" if abs(price_change) > 5 else "moderate"
        
        with st.expander(" CIO Market Analysis", expanded=False):
            st.markdown(f"""
            ### Letter from the Chief Investment Officer
            
            Dear Valued Investor,
            
            I hope this analysis finds you well. I wanted to share our current market perspective on your selected stock:
            
            #### Market Sentiment
            Our analysis indicates a **{strength} {sentiment}** outlook for this security over the next trading period. 
            We project a potential {price_change:.1f}% move from the current price of ${current_price:.2f}.
            
            #### Technical Analysis
            - RSI ({rsi:.1f}): {"Potentially overbought" if rsi > 70 else "Potentially oversold" if rsi < 30 else "Neutral territory"}
            - MACD: {"Positive momentum" if macd_value > 0 else "Negative momentum"}
            - MA20 (${ma20:.2f}): {"Price above moving average" if current_price > ma20 else "Price below moving average"}
            
            #### Risk Assessment
            Our confidence bands suggest:
            - Upside potential: ${confidence_bands[1][-1]:.2f}
            - Downside risk: ${confidence_bands[0][-1]:.2f}
            
            #### Investment Strategy
            Based on our comprehensive analysis:
            {"- Consider strategic entry points on pullbacks" if sentiment == "bullish" else "- Consider protective stops and position sizing"}
            {"- Look for momentum confirmation" if strength == "strong" else "- Monitor for trend reversal signals"}
            
            As always, please consider your personal risk tolerance and investment goals.
            
            Best regards,
            Chief Investment Officer
            """)
            
    except Exception as e:
        logger.error(f"Error displaying investor letter: {str(e)}")
        st.error("Unable to display investor letter")

def display_fundamental_summary(predictions, confidence_bands):
    """Display fundamental analysis summary"""
    try:
        with st.expander(" Prediction Analysis", expanded=False):
            # Calculate key metrics
            avg_prediction = sum(predictions) / len(predictions)
            max_upside = max(confidence_bands[1])
            max_downside = min(confidence_bands[0])
            volatility = (max_upside - max_downside) / avg_prediction * 100
            
            st.markdown("""
            ### Understanding Your Prediction
            
            #### Key Components
            1. **Base Prediction**
               - Machine learning models analyze historical patterns
               - Technical indicators provide trend confirmation
               - Volume analysis validates price movements
            
            2. **Confidence Bands**
               - Upper band: Optimistic scenario
               - Lower band: Conservative scenario
               - Width indicates market uncertainty
            
            3. **Risk Metrics**
            """)
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Potential Upside",
                    f"${max_upside:.2f}",
                    f"{volatility:.1f}% Volatility"
                )
                
            with col2:
                st.metric(
                    "Potential Downside",
                    f"${max_downside:.2f}",
                    f"{volatility:.1f}% Volatility"
                )
                
            with col3:
                st.metric(
                    "Target Price",
                    f"${avg_prediction:.2f}",
                    "7-day Forecast"
                )
            
            st.markdown("""
            #### Interpretation Guide
            - **Narrow Bands**: Higher prediction confidence
            - **Wide Bands**: More market uncertainty
            - **Trend Direction**: Primary price movement expectation
            - **Volatility**: Market stability indicator
            
            *Note: All predictions are probabilistic and should be used as one of many tools in your investment decision process.*
            """)
            
    except Exception as e:
        logger.error(f"Error displaying fundamental summary: {str(e)}")
        st.error("Unable to display fundamental analysis summary")

def get_prediction_summary(ticker, data, predictions, confidence_bands):
    """Generate a summary of prediction indicators and price recommendations"""
    try:
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions[-1]  # Final predicted price
        
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Unpack confidence bands tuple
        confidence_lower, confidence_upper = confidence_bands
        confidence_range = (confidence_upper[-1] - confidence_lower[-1]) / predicted_price * 100
        
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        signals = []
        if price_change > 5:
            signals.append(1)  # Strong buy
        elif price_change > 2:
            signals.append(0.5)  # Buy
        elif price_change < -5:
            signals.append(-1)  # Strong sell
        elif price_change < -2:
            signals.append(-0.5)  # Sell
        else:
            signals.append(0)  # Hold
            
        if rsi > 70:
            signals.append(-0.5)  # Overbought
        elif rsi < 30:
            signals.append(0.5)  # Oversold
            
        if macd > signal:
            signals.append(0.5)  # Bullish MACD
        else:
            signals.append(-0.5)  # Bearish MACD
            
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.5:
            recommendation = "BUY"
        elif avg_signal < -0.5:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
            
        confidence_score = 100 - min(confidence_range, 100)
        
        if volatility > 40:
            risk_level = "High"
        elif volatility > 20:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        margin = predicted_price * (confidence_range / 100)
        buy_target = predicted_price - margin * 0.5
        sell_target = predicted_price + margin * 0.5
        stop_loss = buy_target - margin * 0.5
        
        technical_analysis = f"""
        The stock is currently {'above' if current_price > ma20 else 'below'} its 20-day moving average and 
        {'above' if current_price > ma50 else 'below'} its 50-day moving average. RSI at {rsi:.1f} indicates 
        the stock is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}. MACD is 
        {'bullish' if macd > signal else 'bearish'} with a {'positive' if macd > 0 else 'negative'} trend.
        """
        
        risk_analysis = f"""
        Volatility is {volatility:.1f}%, indicating {'high' if volatility > 40 else 'moderate' if volatility > 20 else 'low'} risk. 
        The prediction confidence range is {confidence_range:.1f}%, suggesting {'high' if confidence_range > 20 else 'moderate' if confidence_range > 10 else 'low'} uncertainty. 
        Consider position sizing and stop-loss orders accordingly.
        """
        
        investment_strategy = f"""
        Given the {risk_level.lower()} risk profile and {recommendation.lower()} signal:
        - Entry: Consider entries near ${buy_target:.2f}
        - Target: Look to take profits near ${sell_target:.2f}
        - Stop Loss: Protect downside at ${stop_loss:.2f}
        - Position Size: {'Small' if risk_level == 'High' else 'Moderate' if risk_level == 'Medium' else 'Standard'} position recommended
        """
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'volatility': volatility,
            'confidence_score': confidence_score,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'price_target': predicted_price,
            'buy_target': buy_target,
            'sell_target': sell_target,
            'stop_loss': stop_loss,
            'support_level': data['Low'].tail(60).quantile(0.25),
            'resistance_level': data['High'].tail(60).quantile(0.75),
            'technical_analysis': technical_analysis.strip(),
            'risk_analysis': risk_analysis.strip(),
            'investment_strategy': investment_strategy.strip(),
            'technical_indicators': {
                'rsi': rsi,
                'macd': macd,
                'signal': signal,
                'ma20': ma20,
                'ma50': ma50
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_prediction_summary: {str(e)}")
        return None

def prediction_tab(ticker=None):
    """Stock price prediction tab"""
    try:
        st.header(" Stock Price Prediction")
        
        # Initialize session state for active tab if not exists
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 'prediction'
            
        # Get user input
        col1, col2 = st.columns([3, 1])
        with col1:
            if ticker:
                ticker = ticker.upper()
            else:
                ticker = st.text_input("Enter Stock Ticker:", "AAPL", key="prediction_ticker").upper()
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Ollama 3.2", "DeepSeek-R1", "Random Forest", "XGBoost", "LightGBM"],
            help="Choose the model for prediction",
            key="prediction_model",
            index=0
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
            - **Ollama 3.2**: Advanced LLM for market analysis and predictions (Default)
            - **DeepSeek-R1**: Specialized LLM for financial forecasting
            - **Random Forest**: Ensemble method, good for stable predictions
            - **XGBoost**: Gradient boosting, excellent for capturing trends
            - **LightGBM**: Fast gradient boosting, good for large datasets
            """)
        
        if st.button("Generate Prediction", key="generate_prediction"):
            # Set active tab to prediction
            st.session_state.active_tab = 'prediction'
            
            try:
                # Get stock data safely
                stock = yf.Ticker(ticker)
                
                # Get historical data for technical analysis
                data = stock.history(period="2y")  # Get 2 years of data for better prediction
                if data.empty:
                    st.error(f"No data found for {ticker}")
                    return
                    
                # Get current price
                current_price = data['Close'].iloc[-1] if not data.empty else None
                if current_price is None:
                    st.error(f"Could not get current price for {ticker}")
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
                
                # Fill NaN values
                data = data.ffill().bfill()
                
                # Generate predictions
                result = predict_stock_price(data, prediction_days, model_type)
                
                if result is None or any(x is None for x in result):
                    st.error("Failed to generate predictions. Please try again.")
                    return
                    
                predictions, confidence_bands, features = result
                confidence_lower, confidence_upper = confidence_bands
                
                if len(predictions) != prediction_days:
                    st.error("Incorrect number of predictions generated")
                    return
                
                # Create future dates
                future_dates = []
                current_date = data.index[-1]
                
                # Generate exactly prediction_days business days
                while len(future_dates) < len(predictions):
                    current_date = current_date + pd.Timedelta(days=1)
                    if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                        future_dates.append(current_date)
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'Predicted': predictions,
                    'Lower Bound': confidence_lower,
                    'Upper Bound': confidence_upper
                }, index=future_dates)
                
                # Plot predictions
                fig = go.Figure()
                
                # Plot historical data (last 60 days)
                historical_data = data.tail(60)
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Plot predictions
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Predicted'],
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence bands
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Lower Bound'],
                    name='Lower Bound',
                    line=dict(color='gray', dash='dot'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Upper Bound'],
                    name='Upper Bound',
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    line=dict(color='gray', dash='dot'),
                    showlegend=False
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Price Prediction',
                    yaxis_title='Stock Price (USD)',
                    xaxis_title='Date',
                    hovermode='x unified',
                    showlegend=True
                )
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction summary
                st.subheader("Prediction Summary")
                
                # Format the DataFrame for display
                display_df = pd.DataFrame({
                    'Date': pred_df.index,
                    'Predicted Price': [f"${x:.2f}" for x in pred_df['Predicted']],
                    'Lower Bound': [f"${x:.2f}" for x in pred_df['Lower Bound']],
                    'Upper Bound': [f"${x:.2f}" for x in pred_df['Upper Bound']]
                })
                display_df = display_df.set_index('Date')
                st.dataframe(display_df)
                
                # Get and display prediction indicator summary
                st.subheader("Stock Prediction Indicators")
                summary = get_prediction_summary(ticker, data, predictions, confidence_bands)
                
                if summary:
                    # Create three columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Recommendation",
                            summary["recommendation"],
                            f"{summary['confidence_score']:.1f}% Confidence"
                        )
                        
                    with col2:
                        st.metric(
                            "Current Price",
                            f"${summary['current_price']:.2f}",
                            f"{summary['price_change']:+.2f}% Predicted"
                        )
                        
                    with col3:
                        st.metric(
                            "Target Price",
                            f"${summary['predicted_price']:.2f}",
                            "7-day Forecast"
                        )
                    
                    # Create expandable section for detailed analysis
                    with st.expander("View Detailed Price Targets", expanded=True):
                        st.markdown(f"""
                        ### Price Targets
                        - **Buy Target:** ${summary['buy_target']:.2f}
                        - **Sell Target:** ${summary['sell_target']:.2f}
                        - **Stop Loss:** ${summary['stop_loss']:.2f}
                        
                        ### Support & Resistance
                        - **Support Level:** ${summary['support_level']:.2f}
                        - **Resistance Level:** ${summary['resistance_level']:.2f}
                        
                        ### Technical Indicators
                        - **RSI:** {summary['technical_indicators']['rsi']:.1f}
                        - **MACD:** {summary['technical_indicators']['macd']:.2f}
                        - **Signal:** {summary['technical_indicators']['signal']:.2f}
                        - **MA20:** ${summary['technical_indicators']['ma20']:.2f}
                        - **MA50:** ${summary['technical_indicators']['ma50']:.2f}
                        """)
                
                # Generate and display CIO letter
                st.subheader("Chief Investment Officer's Letter")
                with st.spinner("Generating comprehensive market analysis..."):
                    letter = get_cio_letter(ticker, data, predictions, confidence_bands)
                    if letter:
                        st.markdown(letter)
                    else:
                        st.error("Failed to generate CIO letter. Please try again.")
                
            except Exception as e:
                logger.error(f"Error generating prediction: {str(e)}")
                st.error(f"Error generating prediction: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in prediction tab: {str(e)}")
        st.error("Failed to generate predictions. Please try again.")

def calculate_rsi(data, periods=14):
    """Calculate RSI for a given DataFrame"""
    try:
        # Calculate price changes
        close_delta = data['Close'].diff()

        # Create two series: one for gains, one for losses
        gains = close_delta.where(close_delta > 0, 0.0)
        losses = (-close_delta.where(close_delta < 0, 0.0))

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=periods, min_periods=1).mean()
        avg_losses = losses.rolling(window=periods, min_periods=1).mean()

        # Calculate RS and RSI, handling division by zero
        rs = avg_gains / avg_losses.replace(0, float('inf'))
        rsi = 100 - (100 / (1 + rs))

        # Replace infinite values with 100 (fully overbought)
        rsi = rsi.replace([np.inf, -np.inf], 100)
        
        return rsi.fillna(50)  # Fill NaN with neutral RSI value
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(50, index=data.index)  # Return neutral RSI on error

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        # Calculate the EMA
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd = exp1 - exp2
        
        # Calculate signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        hist = macd - signal
        
        return macd, signal, hist
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return pd.Series(0, index=data.index)

def safe_divide(a, b):
    """Safe division function"""
    if b == 0:
        return 0
    return a / b

def plot_predictions(data, predictions, confidence_bands, days):
    """Plot stock price predictions"""
    try:
        # Generate future dates
        last_date = data.index[-1]
        future_dates = [last_date + pd.Timedelta(days=x+1) for x in range(days)]
        
        # Create prediction plot
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=confidence_bands[1] + confidence_bands[0][::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Band'
        ))
        
        # Update layout
        fig.update_layout(
            title="Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            showlegend=True
        )
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")
        st.error("Failed to plot predictions")

def get_cio_letter(ticker, data, predictions, confidence_bands):
    """Generate CIO letter with market analysis"""
    try:
        # Get company info
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
        
        # Calculate metrics
        current_price = data['Close'].iloc[-1]
        final_prediction = predictions[-1]
        price_change = ((final_prediction - current_price) / current_price) * 100
        
        # Calculate returns
        six_month_return = ((data['Close'].iloc[-1] - data['Close'].iloc[-126]) / data['Close'].iloc[-126]) * 100
        
        # Get financial metrics
        profit_margins = info.get('profitMargins', 0) * 100
        revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
        
        # Technical indicators
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Determine market conditions
        market_condition = "bullish" if price_change > 0 else "bearish"
        trend_strength = "strong" if abs(price_change) > 5 else "moderate" if abs(price_change) > 2 else "weak"
        
        # Generate CIO letter
        letter = f"""
        ## Chief Investment Officer's Market Analysis
        
        Dear Valued Investor,
        
        I am writing to provide you with our latest analysis on {company_name} ({ticker}), a leading player in the {industry} industry within the {sector} sector.
        
        ### Market Position and Technical Analysis
        
        Our proprietary models indicate a {market_condition} outlook with {trend_strength} momentum. The stock is currently trading at ${current_price:.2f}, and our analysis suggests a target price of ${final_prediction:.2f}, representing a potential {price_change:+.1f}% movement.
        
        Key technical indicators paint the following picture:
        - RSI at {rsi:.1f} indicates the stock is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}
        - MACD shows {'positive' if macd > signal else 'negative'} momentum
        - Price is trading {'above' if current_price > ma20 else 'below'} the 20-day moving average
        - {'Bullish' if ma20 > ma50 else 'Bearish'} trend confirmed by MA crossover
        
        ### Risk Assessment
        
        The stock exhibits {volatility:.1f}% annualized volatility, which we consider {'high' if volatility > 40 else 'moderate' if volatility > 20 else 'low'}. Our confidence bands suggest a trading range of:
        - Upper Band: ${confidence_bands[1][-1]:.2f}
        - Lower Band: ${confidence_bands[0][-1]:.2f}
        
        ### Investment Strategy
        
        Based on our comprehensive analysis, we recommend a {'STRONG BUY' if price_change > 5 else 'BUY' if price_change > 2 else 'HOLD' if abs(price_change) <= 2 else 'SELL' if price_change < -2 else 'STRONG SELL'} position with the following strategy:
        
        1. Entry Points:
           - Strategic entry near ${confidence_bands[0][-1]:.2f}
           - Add on dips if fundamentals remain strong
        
        2. Exit Strategy:
           - Take profits near ${confidence_bands[1][-1]:.2f}
           - Set stop-loss at ${current_price * 0.95:.2f} (5% below current price)
        
        3. Position Sizing:
           - {'Conservative' if volatility > 30 else 'Moderate' if volatility > 20 else 'Standard'} position size recommended
           - Consider scaling in over multiple entry points
        
        ### Market Catalysts
        
        Key factors influencing our outlook:
        1. Technical Setup: {rsi:.1f} RSI with {'positive' if macd > 0 else 'negative'} MACD momentum
        2. Volatility Profile: {volatility:.1f}% suggests {'high' if volatility > 40 else 'moderate' if volatility > 20 else 'low'} risk
        3. Trend Analysis: {trend_strength.title()} {market_condition} trend in place
        
        We will continue to monitor the situation and provide updates as market conditions evolve.
        
        Best regards,
        Chief Investment Officer
        """
        
        return letter
        
    except Exception as e:
        logger.error(f"Error generating CIO letter: {str(e)}")
        return None

def display_investor_letter(stock_data, predictions, confidence_bands):
    """Display CIO letter with market analysis"""
    try:
        if st.button("Generate CIO Letter", key="gen_letter"):
            with st.spinner("Generating comprehensive market analysis..."):
                letter = get_cio_letter(stock_data['ticker'], stock_data['data'], predictions, confidence_bands)
                if letter:
                    st.markdown(letter)
                else:
                    st.error("Failed to generate CIO letter. Please try again.")
                    
    except Exception as e:
        logger.error(f"Error displaying investor letter: {str(e)}")
        st.error("Failed to display investor letter. Please try again.")

def display_prediction(ticker):
    """Display stock price prediction analysis"""
    try:
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Ollama 3.2", "DeepSeek-R1", "Random Forest", "XGBoost", "LightGBM"],
            help="Choose the model for prediction",
            key="prediction_model_type",
            index=0
        )
        
        # Prediction days
        prediction_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to predict into the future",
            key="prediction_days_slider"
        )
        
        # Model information
        with st.expander("Model Information"):
            st.write("""
            **Available Models:**
            - **Ollama 3.2**: Advanced LLM for market analysis and predictions (Default)
            - **DeepSeek-R1**: Specialized LLM for financial forecasting
            - **Random Forest**: Ensemble method, good for stable predictions
            - **XGBoost**: Gradient boosting, excellent for capturing trends
            - **LightGBM**: Fast gradient boosting, good for large datasets
            """)
            
        if st.button("Generate Prediction", key="generate_prediction_btn"):
            with st.spinner("Generating prediction..."):
                # Get stock data
                stock = yf.Ticker(ticker)
                data = stock.history(period="2y")
                
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
                
                # Fill NaN values
                data = data.ffill().bfill()
                
                # Generate predictions
                result = predict_stock_price(data, prediction_days, model_type)
                
                if result is None:
                    st.error("Failed to generate predictions. Please try again.")
                    return
                
                future_pred, test_pred, confidence = result
                
                # Validate prediction results
                if len(future_pred) != prediction_days:
                    st.error(f"Prediction length mismatch. Expected {prediction_days}, got {len(future_pred)}")
                    return
                    
                if len(confidence[0]) != prediction_days or len(confidence[1]) != prediction_days:
                    st.error("Confidence bands length mismatch")
                    return
                
                # Create future dates
                future_dates = []
                current_date = data.index[-1]
                
                # Generate exactly prediction_days business days
                while len(future_dates) < len(future_pred):
                    current_date = current_date + pd.Timedelta(days=1)
                    if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                        future_dates.append(current_date)
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'Predicted': future_pred,
                    'Lower': confidence[0],
                    'Upper': confidence[1]
                }, index=future_dates[:len(future_pred)])  # Make sure we use the correct number of dates
                
                # Plot predictions
                fig = go.Figure()
                
                # Plot historical data (last 60 days)
                historical_data = data.tail(60)
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Plot predictions
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Predicted'],
                    name='Predicted Price',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence bands
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Lower'],
                    name='Lower Bound',
                    line=dict(color='gray', dash='dot'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Upper'],
                    name='Upper Bound',
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    line=dict(color='gray', dash='dot'),
                    showlegend=False
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Price Prediction',
                    yaxis_title='Stock Price (USD)',
                    xaxis_title='Date',
                    hovermode='x unified',
                    showlegend=True
                )
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction summary
                st.subheader("Prediction Summary")
                
                # Format the DataFrame for display
                display_df = pd.DataFrame({
                    'Date': pred_df.index,
                    'Predicted Price': [f"${x:.2f}" for x in pred_df['Predicted']],
                    'Lower Bound': [f"${x:.2f}" for x in pred_df['Lower']],
                    'Upper Bound': [f"${x:.2f}" for x in pred_df['Upper']]
                })
                display_df = display_df.set_index('Date')
                st.dataframe(display_df)
                
                # Get and display prediction indicator summary
                st.subheader("Stock Prediction Indicators")
                summary = get_prediction_summary(ticker, data, future_pred, confidence)
                
                if summary:
                    # Create three columns for key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Recommendation",
                            summary["recommendation"],
                            f"{summary['confidence_score']:.1f}% Confidence"
                        )
                        
                    with col2:
                        st.metric(
                            "Current Price",
                            f"${summary['current_price']:.2f}",
                            f"{summary['price_change']:+.2f}% Predicted"
                        )
                        
                    with col3:
                        st.metric(
                            "Target Price",
                            f"${summary['predicted_price']:.2f}",
                            "7-day Forecast"
                        )
                    
                    # Create expandable section for detailed analysis
                    with st.expander("View Detailed Price Targets", expanded=True):
                        st.markdown(f"""
                        ### Price Targets
                        - **Buy Target:** ${summary['buy_target']:.2f}
                        - **Sell Target:** ${summary['sell_target']:.2f}
                        - **Stop Loss:** ${summary['stop_loss']:.2f}
                        
                        ### Support & Resistance
                        - **Support Level:** ${summary['support_level']:.2f}
                        - **Resistance Level:** ${summary['resistance_level']:.2f}
                        
                        ### Technical Indicators
                        - **RSI:** {summary['technical_indicators']['rsi']:.1f}
                        - **MACD:** {summary['technical_indicators']['macd']:.2f}
                        - **Signal:** {summary['technical_indicators']['signal']:.2f}
                        - **MA20:** ${summary['technical_indicators']['ma20']:.2f}
                        - **MA50:** ${summary['technical_indicators']['ma50']:.2f}
                        """)
                
                # Display CIO Letter
                st.subheader("Chief Investment Officer's Analysis")
                with st.spinner("Generating CIO analysis..."):
                    letter = get_cio_letter(ticker, data, future_pred, confidence)
                    if letter:
                        # Executive Summary
                        with st.expander("Executive Summary", expanded=True):
                            st.markdown(letter.split("**Company Overview")[0])
                        
                        # Company Overview & Market Position
                        with st.expander("Company Overview & Market Position"):
                            st.markdown("**Company Overview & Market Position**" + letter.split("**Company Overview")[1].split("**Financial Performance")[0])
                        
                        # Financial Performance
                        with st.expander("Financial Performance & Fundamentals"):
                            st.markdown("**Financial Performance & Fundamentals**" + letter.split("**Financial Performance")[1].split("**Technical Analysis")[0])
                        
                        # Technical Analysis
                        with st.expander("Technical Analysis & Market Dynamics"):
                            st.markdown("**Technical Analysis & Market Dynamics**" + letter.split("**Technical Analysis")[1].split("**Risk Assessment")[0])
                        
                        # Risk Assessment
                        with st.expander("Risk Assessment & Portfolio Strategy"):
                            st.markdown("**Risk Assessment**" + letter.split("**Risk Assessment")[1].split("Key risks")[0])
                        
                        # Risk Factors and Conclusion
                        with st.expander("Risk Factors & Conclusion"):
                            st.markdown("Key risks" + letter.split("Key risks")[1])
                    else:
                        st.error("Failed to generate CIO letter. Please try again.")
                    
    except Exception as e:
        st.error(f"Error displaying prediction: {str(e)}")
        logger.error(f"Error in display_prediction: {str(e)}")

def get_cio_letter(ticker, data, predictions, confidence_bands):
    """Generate a detailed CIO letter with market analysis"""
    try:
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        # Calculate key metrics
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Get technical indicators
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Determine market conditions
        market_condition = "bullish" if price_change > 0 else "bearish"
        trend_strength = "strong" if abs(price_change) > 5 else "moderate" if abs(price_change) > 2 else "weak"
        
        # Generate CIO letter
        letter = f"""
        ## Chief Investment Officer's Market Analysis
        
        Dear Valued Investor,
        
        I am writing to provide you with our latest analysis on {company_name} ({ticker}), a leading player in the {industry} industry within the {sector} sector.
        
        ### Market Position and Technical Analysis
        
        Our proprietary models indicate a {market_condition} outlook with {trend_strength} momentum. The stock is currently trading at ${current_price:.2f}, and our analysis suggests a target price of ${predicted_price:.2f}, representing a potential {price_change:+.1f}% movement.
        
        Key technical indicators paint the following picture:
        - RSI at {rsi:.1f} indicates the stock is {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'}
        - MACD shows {'positive' if macd > signal else 'negative'} momentum
        - Price is trading {'above' if current_price > ma20 else 'below'} the 20-day moving average
        - {'Bullish' if ma20 > ma50 else 'Bearish'} trend confirmed by MA crossover
        
        ### Risk Assessment
        
        The stock exhibits {volatility:.1f}% annualized volatility, which we consider {'high' if volatility > 40 else 'moderate' if volatility > 20 else 'low'}. Our confidence bands suggest a trading range of:
        - Upper Band: ${confidence_bands[1][-1]:.2f}
        - Lower Band: ${confidence_bands[0][-1]:.2f}
        
        ### Investment Strategy
        
        Based on our comprehensive analysis, we recommend a {'STRONG BUY' if price_change > 5 else 'BUY' if price_change > 2 else 'HOLD' if abs(price_change) <= 2 else 'SELL' if price_change < -2 else 'STRONG SELL'} position with the following strategy:
        
        1. Entry Points:
           - Strategic entry near ${confidence_bands[0][-1]:.2f}
           - Add on dips if fundamentals remain strong
        
        2. Exit Strategy:
           - Take profits near ${confidence_bands[1][-1]:.2f}
           - Set stop-loss at ${current_price * 0.95:.2f} (5% below current price)
        
        3. Position Sizing:
           - {'Conservative' if volatility > 30 else 'Moderate' if volatility > 20 else 'Standard'} position size recommended
           - Consider scaling in over multiple entry points
        
        ### Market Catalysts
        
        Key factors influencing our outlook:
        1. Technical Setup: {rsi:.1f} RSI with {'positive' if macd > 0 else 'negative'} MACD momentum
        2. Volatility Profile: {volatility:.1f}% suggests {'high' if volatility > 40 else 'moderate' if volatility > 20 else 'low'} risk
        3. Trend Analysis: {trend_strength.title()} {market_condition} trend in place
        
        We will continue to monitor the situation and provide updates as market conditions evolve.
        
        Best regards,
        Chief Investment Officer
        """
        
        return letter
        
    except Exception as e:
        logger.error(f"Error generating CIO letter: {str(e)}")
        return None
