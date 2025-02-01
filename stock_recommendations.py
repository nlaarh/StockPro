import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import json
import requests
from utils import calculate_rsi, calculate_macd

def test_ollama_connection(model_type):
    """Test connection to Ollama server and model availability"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            return False, "Could not connect to Ollama server"
            
        models = [tag['name'] for tag in response.json()['models']]
        model_name = 'llama3.2:latest' if model_type == "Ollama 3.2" else "deepseek-coder:latest"
        
        if model_name not in models:
            return False, f"Model {model_name} not available"
            
        return True, "Connection successful"
        
    except requests.exceptions.RequestException:
        return False, "Could not connect to Ollama server"

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
                    'model': 'llama3.2:latest' if model_type == "Ollama 3.2" else "deepseek-coder:latest",
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
                    'model': 'llama3.2:latest' if model_type == "Ollama 3.2" else "deepseek-coder:latest",
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
                                    
    except Exception as e:
        st.error(f"Error in recommendations tab: {str(e)}")

def plot_stock_history(ticker, period='1y'):
    """Plot historical stock data with multiple timeframes and technical indicators"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            st.error(f"No historical data available for {ticker}")
            return None
            
        # Calculate technical indicators
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
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA20'],
            name="20 SMA",
            line=dict(color='orange')
        ))
        
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
        st.error(f"Error creating chart for {ticker}: {str(e)}")
        return None
