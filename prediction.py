import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb
import requests
import json
import logging
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from scipy.stats import percentileofscore
import logging

from utils import calculate_rsi, calculate_macd


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
        model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-r1:latest"
        
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


def get_llm_response(prompt, model_type="Ollama 3.2"):
    """Get response from LLM model"""
    try:
        model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-r1:latest"
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            logging.error(f"Error from Ollama API: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error getting LLM response: {str(e)}")
        return None


def predict_stock_price(data, prediction_days=7, model_type="Random Forest"):
    """Predict stock prices using various models"""
    try:
        # Ensure we have enough data
        if len(data) < 60:  # Minimum data points needed
            st.error("Insufficient historical data for prediction")
            return None
            
        # Prepare features
        data = data.copy()
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data)
        data['MACD'] = calculate_macd(data['Close'])[0]
        
        # Fill NaN values
        data = data.ffill().bfill()
        
        # Choose and train model
        if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
            try:
                # Get current price and technical indicators
                current_price = float(data['Close'].iloc[-1])
                
                # Format recent price history
                recent_prices = data['Close'].iloc[-5:].tolist()
                price_history = ", ".join([f"{p:.2f}" for p in recent_prices])
                
                # Create a strict prompt that forces array output
                prompt = f"""You are a stock price prediction model. Your task is to predict {prediction_days} future stock prices.

IMPORTANT: You must respond with ONLY a JSON array of {prediction_days} numbers representing daily prices. No other text.

Current price: {current_price:.2f}
Recent prices: {price_history}

Example response format:
[{current_price+0.5}, {current_price+1.0}, {current_price+1.5}]

YOUR RESPONSE (ONLY numbers in array format):"""

                # Set up API call
                model_name = "llama3.2" if model_type == "Ollama 3.2" else "deepseek-r1:latest"
                
                # Make API call
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"LLM API returned status code {response.status_code}")
                
                # Parse response
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Find JSON array in response
                array_pattern = r'\[[\d\s,.]+\]'
                matches = re.findall(array_pattern, response_text)
                
                if not matches:
                    raise ValueError("No valid predictions found")
                
                # Process first valid array found
                predictions = None
                for potential_json in matches:
                    try:
                        clean_json = re.sub(r'[^\[\]\d.,]', '', potential_json)
                        pred_array = json.loads(clean_json)
                        if isinstance(pred_array, list):
                            predictions = [float(p) for p in pred_array]
                            if len(predictions) >= prediction_days:
                                predictions = predictions[:prediction_days]
                                break
                    except:
                        continue
                
                if predictions is None or len(predictions) != prediction_days:
                    raise ValueError(f"Failed to get {prediction_days} valid predictions")
                
                # Calculate confidence bands (±2%)
                confidence_lower = [p * 0.98 for p in predictions]
                confidence_upper = [p * 1.02 for p in predictions]
                
                return predictions, None, (confidence_lower, confidence_upper)
                
            except Exception as e:
                logging.error(f"LLM prediction error: {str(e)}")
                # Fall back to Random Forest
                model_type = "Random Forest"
        
        if model_type in ["Random Forest", "XGBoost", "LightGBM"]:
            # Prepare features for ML models
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
            X = data[features].values
            y = data['Close'].values
            
            # Scale the data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train on all data except last sequence
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            else:  # LightGBM
                model = lgb.LGBMRegressor(objective='regression', random_state=42)
            
            # Fit model on all data
            model.fit(X_scaled, y)
            
            # Generate predictions
            future_pred = []
            last_data = X_scaled[-1:]  # Start with last known data point
            
            for _ in range(prediction_days):
                # Predict next value
                pred = float(model.predict(last_data)[0])
                future_pred.append(pred)
                
                # Update last_data for next prediction
                new_row = last_data.copy()
                new_row[0, features.index('Close')] = pred
                last_data = new_row
            
            # Calculate confidence bands (±2%)
            confidence_lower = [p * 0.98 for p in future_pred]
            confidence_upper = [p * 1.02 for p in future_pred]
            
            return future_pred, None, (confidence_lower, confidence_upper)
            
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        st.error(f"Error in prediction: {str(e)}")
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
        
        return info, financials, hist
        
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


def prediction_tab(ticker, data):
    """Stock price prediction tab"""
    try:
        st.header(" Stock Price Prediction")
        
        if not ticker or data is None or data.empty:
            st.error("Please enter a valid ticker symbol")
            return
        
        # Model selection with Ollama 3.2 as default
        model_type = st.selectbox(
            "Select Model",
            ["Ollama 3.2", "DeepSeek-R1", "Random Forest", "XGBoost", "LightGBM"],
            index=0,  # Set Ollama 3.2 as default
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
            if data is None or data.empty:
                st.error(f"No data found for {ticker}")
                return
                
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Generate predictions
            result = predict_stock_price(data, prediction_days, model_type)
            
            if result is None:
                st.error("Failed to generate predictions. Please try again.")
                return
            
            # Unpack predictions
            future_pred, test_pred, confidence = result
            confidence_lower, confidence_upper = confidence
            
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
                'Lower Bound': confidence_lower,
                'Upper Bound': confidence_upper
            }, index=future_dates)
            
            # Analysis sections container
            with st.container():
                # 1. Price Predictions Chart
                with st.expander("Price Predictions Chart", expanded=True):
                    # Create figure
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=data.index[-30:],
                        y=data['Close'][-30:],
                        name="Historical",
                        line=dict(color='blue')
                    ))
                    
                    # Add predictions
                    fig.add_trace(go.Scatter(
                        x=pred_df.index,
                        y=pred_df['Predicted'],
                        name="Predicted",
                        line=dict(color='red')
                    ))
                    
                    # Add confidence bands
                    fig.add_trace(go.Scatter(
                        x=pred_df.index,
                        y=pred_df['Upper Bound'],
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.2)'),
                        name="Upper Bound"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=pred_df.index,
                        y=pred_df['Lower Bound'],
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.2)'),
                        name="Lower Bound"
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
                
                # 2. Detailed Predictions Table
                with st.expander("Detailed Price Predictions", expanded=False):
                    st.dataframe(pred_df.style.format("{:.2f}"))
                
                # 3. AI Analysis
                if model_type in ["Ollama 3.2", "DeepSeek-R1"]:
                    with st.expander("AI Technical Analysis", expanded=False):
                        with st.spinner("Generating AI analysis..."):
                            analysis = get_llm_response(f"""
                            Analyze the following market data for {ticker}:
                            - Current Price: ${current_price:.2f}
                            - Predicted Price: ${future_pred[-1]:.2f} ({((future_pred[-1] - current_price) / current_price) * 100:.2f}%)
                            - RSI: {calculate_rsi(data)[-1]:.1f}
                            - MACD: {calculate_macd(data['Close'])[0][-1]:.2f}
                            - Volume Trend: {(data['Volume'].tail(5).mean() / data['Volume'].tail(20).mean()):.2f}x average

                            Provide a detailed market analysis in markdown format.
                            Focus on key drivers, risks, and potential catalysts.
                            Keep the analysis concise but informative.
                            """, model_type)
                            if analysis:
                                st.markdown(analysis)
                            else:
                                st.error("Failed to generate AI analysis. Please try again.")
                
                # 4. Analyst Letter
                with st.expander("Analyst Investment Letter", expanded=True):
                    with st.spinner("Generating analyst letter..."):
                        letter = get_analyst_writeup(ticker, future_pred, model_type)
                        if letter:
                            st.markdown(letter)
                        else:
                            st.error("Failed to generate analyst letter. Please try again.")
            
    except Exception as e:
        st.error(f"Error in prediction tab: {str(e)}")
        logging.error(f"Error in prediction tab: {str(e)}")
