import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import logging
import re
from utils import format_large_number, calculate_rsi, calculate_macd, calculate_bollinger_bands
import traceback
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def get_market_data():
    """Get real-time market data for analysis"""
    try:
        logger.info("Starting market data collection...")
        # List of popular stocks to analyze
        stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'PFE', 'CSCO', 'ADBE'
        ]
        
        market_data = {}
        for symbol in stocks:
            try:
                logger.info(f"Fetching data for {symbol}...")
                stock = yf.Ticker(symbol)
                info = stock.info
                if not info:
                    logger.error(f"No info available for {symbol}")
                    continue
                
                hist = stock.history(period="1y")
                if hist.empty:
                    logger.error(f"No historical data available for {symbol}")
                    continue
                
                current_price = info.get('currentPrice')
                if current_price is None:
                    current_price = hist['Close'].iloc[-1]
                    logger.info(f"Using historical close price for {symbol}: {current_price}")
                
                previous_close = info.get('previousClose')
                if previous_close is None:
                    previous_close = hist['Close'].iloc[-2]
                    logger.info(f"Using historical previous close for {symbol}: {previous_close}")
                    
                market_data[symbol] = {
                    'Symbol': symbol,
                    'Company': info.get('longName', symbol),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Price': current_price,
                    'Previous Close': previous_close,
                    'Change %': ((current_price - previous_close) / previous_close) * 100,
                    'Volume': info.get('volume', hist['Volume'].iloc[-1]),
                    'Market Cap': info.get('marketCap', 0),
                    'P/E Ratio': info.get('trailingPE', 0),
                    'Dividend Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    'Beta': info.get('beta', 0),
                    'Revenue Growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                    'Profit Margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                    '52W High': info.get('fiftyTwoWeekHigh', 0),
                    '52W Low': info.get('fiftyTwoWeekLow', 0)
                }
                
                # Calculate technical indicators
                hist['RSI'] = calculate_rsi(hist)
                macd_data = calculate_macd(hist)
                bb_data = calculate_bollinger_bands(hist)
                
                market_data[symbol].update({
                    'RSI': hist['RSI'].iloc[-1],
                    'MACD': macd_data['MACD'].iloc[-1],
                    'Signal': macd_data['Signal'].iloc[-1],
                    'BB_Upper': bb_data['BB_Upper'].iloc[-1],
                    'BB_Middle': bb_data['BB_Middle'].iloc[-1],
                    'BB_Lower': bb_data['BB_Lower'].iloc[-1]
                })
                
                logger.info(f"Successfully processed data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}\n{traceback.format_exc()}")
                continue
        
        if not market_data:
            logger.error("No market data collected")
            return None
            
        logger.info(f"Successfully collected data for {len(market_data)} stocks")
        return market_data
        
    except Exception as e:
        logger.error(f"Error in get_market_data: {str(e)}\n{traceback.format_exc()}")
        return None

def analyze_stocks(market_data):
    """Analyze stocks using real-time data and return recommendations"""
    try:
        logger.info("Starting stock analysis...")
        if not market_data:
            logger.error("No market data provided for analysis")
            return None
            
        # Convert market data to list of dictionaries
        stocks_list = []
        for symbol, data in market_data.items():
            try:
                # Calculate technical score
                technical_score = calculate_technical_score(
                    data['RSI'],
                    data['MACD'],
                    data['Signal'],
                    data['Price'],
                    data['BB_Middle'],
                    data['Change %'],
                    data['Volume'],
                    data['Beta']
                )
                
                # Calculate fundamental score
                fundamental_score = calculate_fundamental_score(
                    data['P/E Ratio'],
                    0,  # P/B ratio not available
                    data['Dividend Yield'],
                    data['Revenue Growth'],
                    0,  # Earnings growth not available
                    data['Profit Margin'],
                    0  # Debt to equity not available
                )
                
                # Calculate total score
                total_score = (technical_score + fundamental_score) / 2
                
                # Add scores to data
                data['Technical Score'] = technical_score
                data['Fundamental Score'] = fundamental_score
                data['Score'] = total_score
                
                stocks_list.append(data)
                logger.info(f"Analyzed {symbol}: Score={total_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        if not stocks_list:
            logger.error("No stocks analyzed successfully")
            return None
            
        # Sort by total score
        stocks_list.sort(key=lambda x: x['Score'], reverse=True)
        logger.info(f"Successfully analyzed {len(stocks_list)} stocks")
        return stocks_list
        
    except Exception as e:
        logger.error(f"Error in analyze_stocks: {str(e)}\n{traceback.format_exc()}")
        return None

def process_stock(symbol):
    """Process individual stock data"""
    try:
        logger.info(f"Processing data for {symbol}...")
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info:
            logger.error(f"No info available for {symbol}")
            return None
        
        hist = stock.history(period="1y")
        if hist.empty:
            logger.error(f"No historical data available for {symbol}")
            return None
        
        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        previous_close = info.get('previousClose', hist['Close'].iloc[-2])
        
        # Calculate growth metrics
        revenue_growth = info.get('revenueGrowth', 0)
        profit_margin = info.get('profitMargins', 0)
        earnings_growth = info.get('earningsGrowth', 0)
        
        # Calculate momentum score
        price_change_1m = ((current_price - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]) * 100
        price_change_3m = ((current_price - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60]) * 100
        volume_ratio = hist['Volume'].iloc[-1] / hist['Volume'].mean()
        
        momentum_score = (
            price_change_1m * 0.4 +  # Recent momentum
            price_change_3m * 0.3 +  # Medium-term momentum
            volume_ratio * 0.3       # Volume strength
        )
        
        return {
            'Symbol': symbol,
            'Company': info.get('longName', symbol),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Price': current_price,
            'Previous Close': previous_close,
            'Change %': ((current_price - previous_close) / previous_close) * 100,
            'Volume': info.get('volume', hist['Volume'].iloc[-1]),
            'Market Cap': info.get('marketCap', 0),
            'P/E Ratio': info.get('trailingPE', 0),
            'Revenue Growth': revenue_growth * 100 if revenue_growth else 0,
            'Profit Margin': profit_margin * 100 if profit_margin else 0,
            'Earnings Growth': earnings_growth * 100 if earnings_growth else 0,
            'Momentum Score': momentum_score,
            'RSI': calculate_rsi(hist).iloc[-1],
            'MACD': calculate_macd(hist)['MACD'].iloc[-1],
            'Signal': calculate_macd(hist)['Signal'].iloc[-1],
            'BB_Data': calculate_bollinger_bands(hist)
        }
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_technical_score(rsi, macd, signal, price, bb_middle, price_change, volume, beta):
    """Calculate technical analysis score"""
    try:
        score = 0
        logger.info(f"Calculating technical score: RSI={rsi}, MACD={macd}, Signal={signal}, Price={price}, BB_Middle={bb_middle}, Price_Change={price_change}, Volume={volume}, Beta={beta}")
        
        # RSI (0-30 oversold, 70-100 overbought)
        if rsi is not None:
            if 40 <= rsi <= 60:  # Ideal range
                score += 30
            elif 30 <= rsi <= 70:  # Good range
                score += 20
            else:  # Extreme range
                score += 10
        
        # MACD
        if macd is not None and signal is not None:
            if macd > signal:  # Bullish
                score += 20
        
        # Price vs Moving Average
        if price is not None and bb_middle is not None:
            if price > bb_middle:  # Above MA
                score += 15
        
        # Momentum
        if price_change is not None:
            if price_change > 0:
                score += 15
        
        # Volume (compared to average)
        if volume is not None and volume > 0:
            score += 10
        
        # Beta (risk adjustment)
        if beta is not None:
            if 0.5 <= beta <= 1.5:  # Moderate risk
                score += 10
            elif beta < 0.5:  # Low risk
                score += 5
        
        logger.info(f"Technical score calculated: {score}")
        return min(100, max(0, score))
        
    except Exception as e:
        logger.error(f"Error calculating technical score: {str(e)}\n{traceback.format_exc()}")
        return 0

def calculate_fundamental_score(pe_ratio, pb_ratio, dividend_yield, revenue_growth, earnings_growth, profit_margins, debt_to_equity):
    """Calculate fundamental analysis score"""
    try:
        score = 0
        logger.info(f"Calculating fundamental score: PE={pe_ratio}, PB={pb_ratio}, Div={dividend_yield}, Rev={revenue_growth}, Earn={earnings_growth}, Margin={profit_margins}, D/E={debt_to_equity}")
        
        # P/E Ratio
        if pe_ratio is not None and pe_ratio > 0:
            if pe_ratio <= 20:
                score += 20
            elif pe_ratio <= 30:
                score += 15
            elif pe_ratio <= 40:
                score += 10
        
        # P/B Ratio
        if pb_ratio is not None and pb_ratio > 0:
            if pb_ratio <= 3:
                score += 10
            elif pb_ratio <= 5:
                score += 5
        
        # Dividend Yield
        if dividend_yield is not None and dividend_yield > 0:
            if dividend_yield > 4:
                score += 15
            elif dividend_yield >= 2:
                score += 10
            else:
                score += 5
        
        # Revenue Growth
        if revenue_growth is not None:
            if revenue_growth > 20:
                score += 15
            elif revenue_growth >= 10:
                score += 10
            elif revenue_growth > 0:
                score += 5
        
        # Earnings Growth
        if earnings_growth is not None:
            if earnings_growth > 20:
                score += 15
            elif earnings_growth >= 10:
                score += 10
            elif earnings_growth > 0:
                score += 5
        
        # Profit Margins
        if profit_margins is not None:
            if profit_margins > 20:
                score += 15
            elif profit_margins >= 10:
                score += 10
            elif profit_margins > 0:
                score += 5
        
        # Debt to Equity
        if debt_to_equity is not None and debt_to_equity >= 0:
            if debt_to_equity <= 1:
                score += 10
            elif debt_to_equity <= 2:
                score += 5
        
        logger.info(f"Fundamental score calculated: {score}")
        return min(100, max(0, score))
        
    except Exception as e:
        logger.error(f"Error calculating fundamental score: {str(e)}\n{traceback.format_exc()}")
        return 0

def get_ai_analysis(symbol, model_type="llama3.2"):
    """Get AI analysis for a stock"""
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
        
        if not info or hist.empty:
            logger.error(f"No data available for {symbol}")
            return None
            
        # Prepare data for analysis
        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        previous_close = info.get('previousClose', hist['Close'].iloc[-2])
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        # Calculate technical indicators
        rsi = calculate_rsi(hist).iloc[-1]
        macd_data = calculate_macd(hist)
        macd = macd_data['MACD'].iloc[-1]
        signal = macd_data['Signal'].iloc[-1]
        
        # Create prompt with comprehensive data
        prompt = f"""You are a financial analyst. Analyze {symbol} ({info.get('longName', 'Unknown Company')}) and provide a JSON response with your analysis.

Key Metrics:
- Current Price: ${current_price:.2f}
- Price Change: {change_percent:+.2f}%
- Market Cap: ${info.get('marketCap', 0)/1e9:.1f}B
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 0)*100:.1f}%
- Profit Margin: {info.get('profitMargins', 0)*100:.1f}%
- RSI: {rsi:.1f}
- MACD: {macd:.2f}
- Signal: {signal:.2f}

Technical State:
- RSI indicates: {"Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"}
- MACD Signal: {"Bullish" if macd > signal else "Bearish"}
- 52W Range: ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}

Business Summary: {info.get('longBusinessSummary', 'N/A')}

Respond with a JSON object in this exact format:
{{
    "summary": "2-3 sentences about current state",
    "strengths": [
        "strength 1",
        "strength 2",
        "strength 3"
    ],
    "risks": [
        "risk 1",
        "risk 2",
        "risk 3"
    ],
    "recommendation": "BUY/HOLD/SELL with target price"
}}"""
        
        logger.info(f"Sending prompt to {model_type} for {symbol}")
        
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                  json={
                                      'model': 'llama3.2',
                                      'prompt': prompt,
                                      'stream': False
                                  },
                                  timeout=30)
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            if 'response' not in result:
                logger.error(f"Invalid response from Ollama for {symbol}: {result}")
                return None
                
            # Parse JSON response
            try:
                # Find JSON content between curly braces
                response_text = result['response']
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    # Clean up the string to ensure valid JSON
                    json_str = json_str.replace('\n', ' ').replace('\r', '')
                    analysis = json.loads(json_str)
                    logger.info(f"Successfully got AI analysis for {symbol}")
                    return analysis
                else:
                    logger.error(f"No JSON found in response for {symbol}")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama response for {symbol}: {str(e)}\nResponse: {result['response']}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama for {symbol}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting AI analysis for {symbol}: {str(e)}\n{traceback.format_exc()}")
        return None

def plot_stock_analysis(symbol):
    """Create a technical analysis plot for a stock"""
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return None
            
        # Calculate indicators
        df['RSI'] = calculate_rsi(df)
        macd_data = calculate_macd(df)
        bb_data = calculate_bollinger_bands(df)
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=(f'{symbol} Price', 'MACD', 'RSI'),
                           row_heights=[0.5, 0.25, 0.25])

        # Add candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='OHLC'),
                     row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, 
                                y=bb_data['BB_Upper'],
                                name='Upper BB',
                                line=dict(color='gray', dash='dash')),
                     row=1, col=1)
                     
        fig.add_trace(go.Scatter(x=df.index,
                                y=bb_data['BB_Middle'],
                                name='Middle BB',
                                line=dict(color='gray')),
                     row=1, col=1)
                     
        fig.add_trace(go.Scatter(x=df.index,
                                y=bb_data['BB_Lower'],
                                name='Lower BB',
                                line=dict(color='gray', dash='dash')),
                     row=1, col=1)

        # Add MACD
        fig.add_trace(go.Scatter(x=df.index,
                                y=macd_data['MACD'],
                                name='MACD',
                                line=dict(color='blue')),
                     row=2, col=1)
                     
        fig.add_trace(go.Scatter(x=df.index,
                                y=macd_data['Signal'],
                                name='Signal',
                                line=dict(color='orange')),
                     row=2, col=1)

        # Add RSI
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['RSI'],
                                name='RSI',
                                line=dict(color='purple')),
                     row=3, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            title=f"{symbol} Technical Analysis"
        )

        return fig
        
    except Exception as e:
        logger.error(f"Error creating plot for {symbol}: {str(e)}\n{traceback.format_exc()}")
        return None

def test_stock_analysis(symbol="AAPL"):
    """Test stock analysis functionality"""
    try:
        logger.info(f"Testing stock analysis with {symbol}")
        
        # Test getting stock data
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
        
        if info is None or hist.empty:
            logger.error(f"Failed to get data for {symbol}")
            return False, f"Failed to get data for {symbol}"
            
        logger.info(f"Successfully got data for {symbol}")
        
        # Test calculating indicators
        try:
            rsi = calculate_rsi(hist)
            macd_data = calculate_macd(hist)
            bb_data = calculate_bollinger_bands(hist)
            
            if rsi is None or macd_data is None or bb_data is None:
                logger.error("Failed to calculate technical indicators")
                return False, "Failed to calculate technical indicators"
                
            logger.info("Successfully calculated technical indicators")
            
            # Test calculating scores
            technical_score = calculate_technical_score(
                rsi.iloc[-1],
                macd_data['MACD'].iloc[-1],
                macd_data['Signal'].iloc[-1],
                hist['Close'].iloc[-1],
                bb_data['BB_Middle'].iloc[-1],
                ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100,
                hist['Volume'].iloc[-1],
                info.get('beta', 1)
            )
            
            fundamental_score = calculate_fundamental_score(
                info.get('trailingPE', 0),
                info.get('priceToBook', 0),
                info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                0,  # earnings growth not available
                info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                info.get('debtToEquity', 0)
            )
            
            if technical_score is None or fundamental_score is None:
                logger.error("Failed to calculate scores")
                return False, "Failed to calculate scores"
                
            logger.info(f"Successfully calculated scores: Technical={technical_score}, Fundamental={fundamental_score}")
            return True, "Test passed successfully"
            
        except Exception as e:
            logger.error(f"Error in calculations: {str(e)}")
            return False, f"Error in calculations: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False, f"Error in test: {str(e)}"

def get_recommendation_reason(data):
    """Generate a recommendation reason based on metrics"""
    reasons = []
    
    # Technical Analysis
    if data['RSI'] < 30:
        reasons.append("Oversold (RSI < 30)")
    elif data['RSI'] > 70:
        reasons.append("Overbought (RSI > 70)")
        
    if data['MACD'] > data['Signal']:
        reasons.append("Bullish MACD crossover")
    elif data['MACD'] < data['Signal']:
        reasons.append("Bearish MACD crossover")
        
    # Fundamental Analysis
    if data['Revenue Growth'] > 10:
        reasons.append(f"Strong revenue growth ({data['Revenue Growth']:.1f}%)")
    if data['Profit Margin'] > 15:
        reasons.append(f"High profit margin ({data['Profit Margin']:.1f}%)")
        
    # Price Action
    if data['Change %'] > 0:
        reasons.append(f"Upward momentum (+{data['Change %']:.1f}%)")
    else:
        reasons.append(f"Price pullback ({data['Change %']:.1f}%)")
        
    return "; ".join(reasons[:2])  # Return top 2 reasons

def stock_recommendations_tab():
    """Display stock recommendations"""
    try:
        st.header("📈 Stock Recommendations")
        
        # Initialize session state for recommendations if not exists
        if 'recommendations_df' not in st.session_state:
            st.session_state.recommendations_df = None
            
        try:
            logger.info("Starting market recommendations...")
            if st.session_state.recommendations_df is None:
                with st.spinner("Analyzing market data..."):
                    # Get market data
                    market_data = get_market_data()
                    if market_data is None:
                        st.error("Failed to fetch market data. Please try again.")
                        logger.error("Market data is None")
                        return
                    
                    logger.info(f"Got market data for {len(market_data)} stocks")
                    
                    # Analyze stocks
                    recommendations = analyze_stocks(market_data)
                    if recommendations is None:
                        st.error("Failed to analyze stocks. Please try again.")
                        logger.error("Stock analysis returned None")
                        return
                    
                    # Get top 5 stocks
                    st.session_state.recommendations_df = pd.DataFrame(recommendations).head(5)
            
            df = st.session_state.recommendations_df
            
            # Create two tabs
            rec_tab, analysis_tab = st.tabs(["Top 5 Recommendations", "Detailed Analysis"])
            
            with rec_tab:
                st.subheader("Top 5 Stock Recommendations")
                
                # Format numeric columns
                if not df.empty:
                    logger.info("Formatting recommendation data...")
                    
                    # Create display dataframe
                    display_data = []
                    for idx, row in df.iterrows():
                        display_data.append({
                            'Rank': idx + 1,
                            'Symbol': row['Symbol'],
                            'Company': row['Company'],
                            'Price': f"${row['Price']:.2f}",
                            'Change %': f"{row['Change %']:+.2f}%",
                            'Market Cap': format_market_cap(row['Market Cap']),
                            'Score': f"{row['Score']:.2f}",
                            'Reason': get_recommendation_reason(row)
                        })
                    
                    display_df = pd.DataFrame(display_data)
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        column_config={
                            "Rank": st.column_config.NumberColumn("Rank", width="small"),
                            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                            "Company": st.column_config.TextColumn("Company", width="medium"),
                            "Price": st.column_config.TextColumn("Price", width="small"),
                            "Change %": st.column_config.TextColumn("Change %", width="small"),
                            "Market Cap": st.column_config.TextColumn("Market Cap", width="small"),
                            "Score": st.column_config.TextColumn("Score", width="small"),
                            "Reason": st.column_config.TextColumn("Why Selected", width="large")
                        }
                    )
                    
                    logger.info("Successfully displayed recommendations")
                else:
                    st.warning("No recommendations available")
                    logger.warning("Recommendations DataFrame is empty")
            
            with analysis_tab:
                st.subheader("Detailed Stock Analysis")
                
                # Create a stock selector
                if not df.empty:
                    selected_stock = st.selectbox(
                        "Select Stock for Analysis",
                        df['Symbol'].tolist(),
                        key="analysis_stock_selector"
                    )
                    
                    if selected_stock:
                        # Display stock chart
                        st.subheader(f"{selected_stock} Technical Analysis")
                        fig = plot_stock_analysis(selected_stock)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### Technical Indicators")
                            st.metric("RSI", f"{df[df['Symbol'] == selected_stock]['RSI'].iloc[0]:.1f}")
                            st.metric("MACD", f"{df[df['Symbol'] == selected_stock]['MACD'].iloc[0]:.2f}")
                            st.metric("Signal", f"{df[df['Symbol'] == selected_stock]['Signal'].iloc[0]:.2f}")
                        
                        with col2:
                            st.markdown("### Price Metrics")
                            st.metric("Current Price", f"${df[df['Symbol'] == selected_stock]['Price'].iloc[0]:.2f}")
                            st.metric("Change", f"{df[df['Symbol'] == selected_stock]['Change %'].iloc[0]:+.2f}%")
                            st.metric("Volume", f"{df[df['Symbol'] == selected_stock]['Volume'].iloc[0]:,.0f}")
                        
                        with col3:
                            st.markdown("### Scores")
                            st.metric("Total Score", f"{df[df['Symbol'] == selected_stock]['Score'].iloc[0]:.2f}")
                            st.metric("Technical Score", f"{df[df['Symbol'] == selected_stock]['Technical Score'].iloc[0]:.2f}")
                            st.metric("Fundamental Score", f"{df[df['Symbol'] == selected_stock]['Fundamental Score'].iloc[0]:.2f}")
                        
                        # Get AI analysis
                        st.subheader("AI Analysis")
                        with st.spinner(f"Getting AI analysis for {selected_stock}..."):
                            analysis = get_ai_analysis(selected_stock, "llama3.2")  # Always use llama3.2
                            if analysis:
                                st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")
                                st.markdown("**Strengths:**")
                                for strength in analysis.get('strengths', []):
                                    st.markdown(f"- {strength}")
                                st.markdown("**Risks:**")
                                for risk in analysis.get('risks', []):
                                    st.markdown(f"- {risk}")
                                st.markdown(f"**Recommendation:** {analysis.get('recommendation', 'N/A')}")
                            else:
                                st.warning(f"Could not get AI analysis for {selected_stock}")
                else:
                    st.warning("No stocks available for analysis")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in stock recommendations tab: {str(e)}\n{traceback.format_exc()}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in stock recommendations tab: {str(e)}\n{traceback.format_exc()}")

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    try:
        logger.info("Calculating Bollinger Bands...")
        # Calculate middle band (20-day SMA)
        middle_band = data['Close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        std = data['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        # Create DataFrame with the bands
        bands = pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': middle_band,
            'BB_Lower': lower_band
        })
        
        logger.info("Successfully calculated Bollinger Bands")
        return bands
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}\n{traceback.format_exc()}")
        return None

def format_market_cap(value):
    """Format market cap in B/M format"""
    try:
        if value >= 1e9:
            return f"${value/1e9:.1f}B"
        elif value >= 1e6:
            return f"${value/1e6:.1f}M"
        else:
            return f"${value:,.0f}"
    except:
        return "N/A"
