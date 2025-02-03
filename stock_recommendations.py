import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import logging
from utils import format_large_number, calculate_rsi, calculate_macd, calculate_bollinger_bands

logger = logging.getLogger(__name__)

def get_top_stocks():
    """Get list of top stocks to analyze"""
    # List of major stocks to analyze
    stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSM', 'AVGO',
        'ASML', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC',
        'TXN', 'IBM', 'NOW', 'SAP', 'SHOP', 'SQ', 'PYPL', 'V', 'MA',
        'JPM', 'BAC', 'WFC', 'GS', 'MS'
    ]
    return stocks

def analyze_stock(ticker):
    """Analyze a single stock and return its metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data
        hist = stock.history(period='1y')
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Calculate technical indicators
        rsi = calculate_rsi(hist['Close'])[-1]
        macd, signal, _ = calculate_macd(hist['Close'])
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]
        upper, middle, lower = calculate_bollinger_bands(hist['Close'])
        
        # Get fundamental metrics
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        return {
            'Ticker': ticker,
            'Company': info.get('longName', ticker),
            'Price': current_price,
            'Change %': price_change,
            'Market Cap': market_cap,
            'P/E Ratio': pe_ratio,
            'P/B Ratio': pb_ratio,
            'RSI': rsi,
            'MACD': macd_value,
            'Signal': signal_value,
            'Dividend Yield': dividend_yield,
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown'),
            'Description': info.get('longBusinessSummary', '')
        }
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        return None

def get_stock_recommendations(model="ollama"):
    """Get stock recommendations using specified model"""
    try:
        stocks = get_top_stocks()
        analyzed_stocks = []
        
        # Analyze each stock
        for ticker in stocks:
            metrics = analyze_stock(ticker)
            if metrics:
                analyzed_stocks.append(metrics)
        
        if not analyzed_stocks:
            return []
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(analyzed_stocks)
        
        # Calculate composite score
        df['Technical Score'] = (
            (df['RSI'].between(30, 70).astype(int) * 0.3) +
            ((df['MACD'] > df['Signal']).astype(int) * 0.4) +
            (df['Change %'].gt(0).astype(int) * 0.3)
        ) * 100
        
        df['Fundamental Score'] = (
            (df['P/E Ratio'].between(0, 30).astype(int) * 0.4) +
            (df['P/B Ratio'].between(0, 5).astype(int) * 0.3) +
            (df['Dividend Yield'].gt(0).astype(int) * 0.3)
        ) * 100
        
        df['Total Score'] = df['Technical Score'] * 0.5 + df['Fundamental Score'] * 0.5
        
        # Sort by total score
        df = df.sort_values('Total Score', ascending=False)
        
        # Get top 5 recommendations
        top_5 = df.head()
        
        recommendations = []
        for _, stock in top_5.iterrows():
            recommendations.append({
                'Ticker': stock['Ticker'],
                'Company': stock['Company'],
                'Price': stock['Price'],
                'Change %': stock['Change %'],
                'Market Cap': stock['Market Cap'],
                'Score': stock['Total Score'],
                'Technical Score': stock['Technical Score'],
                'Fundamental Score': stock['Fundamental Score'],
                'Sector': stock['Sector'],
                'Industry': stock['Industry'],
                'Description': stock['Description']
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return []

def plot_stock_analysis(ticker):
    """Create detailed analysis chart for a stock"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        
        if hist.empty:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Calculate and add technical indicators
        upper, middle, lower = calculate_bollinger_bands(hist['Close'])
        fig.add_trace(go.Scatter(x=hist.index, y=upper, name='Upper BB', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=hist.index, y=middle, name='Middle BB', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=hist.index, y=lower, name='Lower BB', line=dict(dash='dash')))
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Analysis',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            xaxis_title='Date',
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting analysis for {ticker}: {str(e)}")
        return None

def stock_recommendations_tab():
    """Display stock recommendations tab"""
    st.header("🎯 Stock Recommendations")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📈 Top Recommendations", "🔍 Detailed Analysis"])
    
    with tab1:
        # Model selection
        col1, col2 = st.columns([2, 1])
        with col1:
            model = st.selectbox(
                "Select Analysis Model",
                ["Traditional", "Ollama 3.2"],
                help="Traditional uses technical and fundamental analysis. Ollama 3.2 uses AI for more nuanced analysis."
            )
        
        with col2:
            refresh = st.button("🔄 Refresh Analysis")
        
        if refresh or 'recommendations' not in st.session_state:
            with st.spinner("Analyzing stocks..."):
                if model == "Traditional":
                    recommendations = get_stock_recommendations()
                else:
                    recommendations = get_stock_recommendations_ollama()
                st.session_state.recommendations = recommendations
        
        if st.session_state.recommendations:
            # Display recommendations in cards
            for rec in st.session_state.recommendations:
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.subheader(f"${rec['Ticker']}")
                        price_change = rec['Change %']
                        st.metric(
                            "Price",
                            f"${rec['Price']:.2f}",
                            f"{price_change:+.2f}%" if price_change else None,
                            delta_color="normal"
                        )
                        
                    with col2:
                        st.subheader(rec['Company'])
                        st.caption(f"{rec['Sector']} | {rec['Industry']}")
                        st.progress(rec['Score']/100, text=f"Overall Score: {rec['Score']:.0f}/100")
                        tech_col, fund_col = st.columns(2)
                        with tech_col:
                            st.progress(rec['Technical Score']/100, text=f"Technical: {rec['Technical Score']:.0f}/100")
                        with fund_col:
                            st.progress(rec['Fundamental Score']/100, text=f"Fundamental: {rec['Fundamental Score']:.0f}/100")
                    
                    with col3:
                        st.metric("Market Cap", format_large_number(rec['Market Cap']))
                        if st.button("📊 Analyze", key=f"analyze_{rec['Ticker']}"):
                            st.session_state.current_ticker = rec['Ticker']
                            st.experimental_rerun()
                    
                    # Expandable description
                    with st.expander("📝 Company Description", expanded=False):
                        st.write(rec['Description'])
                    
                st.markdown("---")
        else:
            st.warning("No recommendations available. Try refreshing the analysis.")
    
    with tab2:
        # Detailed analysis of a specific stock
        ticker = st.text_input("Enter Stock Ticker:", value=st.session_state.get('current_ticker', ''))
        
        if ticker:
            col1, col2 = st.columns(2)
            
            with col1:
                timeframe = st.selectbox(
                    "Select Timeframe",
                    ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
                    index=3
                )
            
            with col2:
                indicators = st.multiselect(
                    "Select Technical Indicators",
                    ["Bollinger Bands", "RSI", "MACD", "Moving Averages"],
                    default=["Bollinger Bands", "RSI"]
                )
            
            # Plot stock analysis
            fig = plot_stock_analysis(ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Stock metrics
                metrics = analyze_stock(ticker)
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RSI", f"{metrics['RSI']:.1f}")
                    with col2:
                        st.metric("MACD", f"{metrics['MACD']:.2f}")
                    with col3:
                        st.metric("P/E Ratio", f"{metrics['P/E Ratio']:.1f}")
                    with col4:
                        st.metric("Dividend Yield", f"{metrics['Dividend Yield']:.1f}%")
                    
                    # Analyst writeup
                    if model == "Ollama 3.2":
                        with st.expander("📝 AI Analyst Report", expanded=True):
                            writeup = get_analyst_writeup(ticker)
                            if writeup:
                                st.markdown(writeup)
            else:
                st.error(f"Could not fetch data for {ticker}")

def test_ollama_connection(model_type):
    """Test connection to Ollama server and model availability"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            return False, "Could not connect to Ollama server"
            
        models = [tag['name'] for tag in response.json()['models']]
        model_name = 'llama2' if model_type == "Ollama 3.2" else "deepseek-r1:latest"
        
        if not any(m.startswith(model_name.split(':')[0]) for m in models):
            install_msg = f"Model {model_name} not found. Please install it using:\n```bash\nollama pull {model_name}\n```"
            st.error(install_msg)
            return False, install_msg
            
        return True, "Connection successful"
        
    except requests.exceptions.RequestException:
        return False, "Could not connect to Ollama server"

def get_stock_recommendations_ollama(model_type="Ollama 3.2"):
    """Get top 5 stock recommendations using LLM analysis"""
    try:
        # Test model availability
        is_available, message = test_ollama_connection(model_type)
        if not is_available:
            st.error(message)
            return None
            
        # Map model types to their Ollama model names
        model_mapping = {
            "Ollama 3.2": "llama2",
            "DeepSeek-R1": "deepseek-r1:latest"
        }
        
        model_name = model_mapping.get(model_type)
        if not model_name:
            st.error(f"Unsupported model type: {model_type}")
            return None
            
        # Create prompt for stock recommendations
        prompt = """You are a senior portfolio manager. Your task is to recommend exactly 5 stocks to analyze based on current market conditions.

IMPORTANT: Your response must be valid JSON. Do not include any text before or after the JSON.

Required format:
{
    "recommendations": [
        {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "sector": "Technology",
            "points": [
                "Strong iPhone sales momentum",
                "Growing services revenue",
                "Robust balance sheet"
            ],
            "target_range": {
                "low": 180,
                "high": 210
            },
            "risk": "Low"
        }
    ]
}

Rules:
1. Include exactly 5 stocks
2. Use only real stock tickers
3. Keep points concise (3-4 bullet points)
4. Target prices should be numbers (no $ signs)
5. Risk must be "Low", "Medium", or "High"
6. Response must be valid JSON"""
        
        # Get LLM response
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'num_predict': 1000
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    st.error(f"Ollama API error: {result['error']}")
                    return None
                    
                # Parse JSON response
                try:
                    # Clean the response text
                    response_text = result['response']
                    # Find the JSON part (between first { and last })
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start == -1 or json_end == 0:
                        st.error("Could not find JSON in response")
                        return None
                    json_str = response_text[json_start:json_end]
                    
                    # Parse JSON
                    recommendations = json.loads(json_str)
                    recs = recommendations.get('recommendations', [])
                    
                    # Validate recommendations
                    if not recs or len(recs) == 0:
                        st.error("No recommendations found in response")
                        return None
                        
                    # Validate each recommendation
                    valid_recs = []
                    for rec in recs:
                        try:
                            # Ensure required fields exist
                            required_fields = ['ticker', 'company', 'sector', 'points', 'target_range', 'risk']
                            if not all(field in rec for field in required_fields):
                                continue
                                
                            # Ensure target_range has low and high
                            if not isinstance(rec['target_range'], dict) or \
                               'low' not in rec['target_range'] or \
                               'high' not in rec['target_range']:
                                continue
                                
                            # Convert target prices to float
                            rec['target_range']['low'] = float(rec['target_range']['low'])
                            rec['target_range']['high'] = float(rec['target_range']['high'])
                            
                            # Ensure points is a list
                            if not isinstance(rec['points'], list):
                                rec['points'] = [rec['points']]
                                
                            valid_recs.append(rec)
                        except:
                            continue
                            
                    if not valid_recs:
                        st.error("No valid recommendations found")
                        return None
                        
                    return valid_recs[:5]  # Return at most 5 recommendations
                    
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse LLM response as JSON: {str(e)}")
                    return None
                except Exception as e:
                    st.error(f"Error processing recommendations: {str(e)}")
                    return None
            else:
                st.error(f"Error from Ollama API: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error(f"Request to {model_type} timed out after 60 seconds")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to Ollama server")
            return None
        except Exception as e:
            st.error(f"Error getting LLM response: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error getting stock recommendations: {str(e)}")
        return None

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
        
        # Create prompt
        prompt = f"""You are a senior equity analyst. Write a detailed analysis for {ticker} ({info.get('longName', '')}).
        
        Include:
        1. Business Overview
        2. Competitive Position
        3. Growth Drivers
        4. Risk Factors
        5. Valuation Analysis
        6. Investment Recommendation
        
        Keep the analysis professional, data-driven, and concise.
        Format using markdown with clear section headers.
        """
        
        # Make API call to Ollama
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama2' if model_type == "Ollama 3.2" else "deepseek-r1:latest",
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                st.error("Error getting analyst writeup")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to LLM service: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error getting analyst writeup: {str(e)}")
        return None

def recommendations_tab():
    """Stock recommendations tab with detailed analysis"""
    try:
        st.header("Top Stock Picks")
        
        # Add model selection
        model_type = st.selectbox(
            "Select Model",
            ["Ollama 3.2", "DeepSeek-R1"],
            index=0,
            help="Choose a model for stock recommendations"
        )
        
        # Get recommendations using selected model
        with st.spinner("Generating stock recommendations..."):
            recommendations = get_stock_recommendations_ollama(model_type)
            
            if recommendations:
                # Create tabs for different views
                summary_tab, detail_tab = st.tabs(["Summary View", "Detailed Analysis"])
                
                with summary_tab:
                    # Create a summary table
                    summary_data = []
                    for rec in recommendations:
                        summary_data.append({
                            "Ticker": rec['ticker'],
                            "Company": rec['company'],
                            "Sector": rec['sector'],
                            "Risk": rec['risk'],
                            "Target Range": f"${rec['target_range']['low']} - ${rec['target_range']['high']}"
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
                        with st.expander(f"#{i}: {rec['ticker']} - {rec['company']}", expanded=i==1):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("Investment Summary")
                                st.write(f"**Sector:** {rec['sector']}")
                                st.write(f"**Risk Level:** {rec['risk']}")
                                st.write("**Key Points:**")
                                for point in rec['points']:
                                    st.write(f"• {point}")
                                
                            with col2:
                                try:
                                    # Get real-time stock data
                                    stock = yf.Ticker(rec['ticker'])
                                    history = stock.history(period="1d")
                                    if not history.empty:
                                        current_price = history['Close'].iloc[-1]
                                        
                                        # Calculate potential returns
                                        low_return = (rec['target_range']['low'] / current_price - 1) * 100
                                        high_return = (rec['target_range']['high'] / current_price - 1) * 100
                                        
                                        st.metric("Current Price", f"${current_price:.2f}")
                                        st.metric("Target Range", 
                                                f"${rec['target_range']['low']:.2f} - ${rec['target_range']['high']:.2f}",
                                                f"{low_return:.1f}% to {high_return:.1f}%")
                                    else:
                                        st.warning(f"Could not fetch current price for {rec['ticker']}")
                                        st.metric("Target Range", 
                                                f"${rec['target_range']['low']:.2f} - ${rec['target_range']['high']:.2f}")
                                except Exception as e:
                                    st.warning(f"Error getting price data for {rec['ticker']}: {str(e)}")
                                    st.metric("Target Range", 
                                            f"${rec['target_range']['low']:.2f} - ${rec['target_range']['high']:.2f}")
                            
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
                                
    except Exception as e:
        st.error(f"Error in recommendations tab: {str(e)}")

def plot_stock_history(ticker, period='1y'):
    """Plot historical stock data with multiple timeframes and technical indicators"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            st.warning(f"No historical data available for {ticker}")
            return None
            
        # Calculate technical indicators
        try:
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate RSI
            rsi_data = calculate_rsi(data['Close'])
            data['RSI'] = rsi_data
            
            # Calculate MACD
            macd_data = calculate_macd(data['Close'])
            data['MACD'] = macd_data['MACD']
            data['Signal'] = macd_data['Signal']
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC"
            ))
            
            # Add moving averages
            if not data['SMA20'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    name="20 SMA",
                    line=dict(color='orange')
                ))
            
            if not data['SMA50'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    name="50 SMA",
                    line=dict(color='blue')
                ))
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Price History ({period})',
                yaxis_title='Price (USD)',
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Error calculating indicators for {ticker}: {str(e)}")
            return None
            
    except Exception as e:
        st.warning(f"Error getting historical data for {ticker}: {str(e)}")
        return None
