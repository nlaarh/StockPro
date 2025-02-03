import streamlit as st
import requests
import json
from typing import List, Optional
import yfinance as yf
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_web_search_results(query: str) -> str:
    """Get recent web search results about a stock"""
    try:
        # Using DuckDuckGo API (no key required)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(search_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Extract relevant information
            if 'Abstract' in data and data['Abstract']:
                results.append(f"Summary: {data['Abstract']}")
            
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:3]:  # Get first 3 related topics
                    if 'Text' in topic:
                        results.append(f"- {topic['Text']}")
            
            return "\n".join(results) if results else "No recent web information found."
        return "Unable to fetch web search results."
    except Exception as e:
        return f"Error fetching web data: {str(e)}"

def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            return sorted(models)
        return ["llama2", "codellama", "mistral"]  # Fallback default models
    except:
        return ["llama2", "codellama", "mistral"]  # Fallback default models

def get_realtime_stock_data(symbol: str) -> dict:
    """Get real-time stock data from Alpha Vantage"""
    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return {"error": "Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in .env file"}

        # Get real-time quote
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        quote_response = requests.get(quote_url)
        quote_data = quote_response.json()

        # Get intraday data
        intraday_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"
        intraday_response = requests.get(intraday_url)
        intraday_data = intraday_response.json()

        # Get company overview
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
        overview_response = requests.get(overview_url)
        overview_data = overview_response.json()

        # Extract relevant information
        real_time_data = {
            "quote": quote_data.get("Global Quote", {}),
            "intraday": list(intraday_data.get("Time Series (5min)", {}).items())[:12],  # Last hour of data
            "overview": overview_data
        }

        return real_time_data
    except Exception as e:
        return {"error": f"Error fetching real-time data: {str(e)}"}

def format_realtime_data(data: dict) -> str:
    """Format real-time stock data for LLM context"""
    if "error" in data:
        return data["error"]

    try:
        quote = data["quote"]
        overview = data["overview"]
        intraday = data["intraday"]

        # Format current quote data
        current_data = f"""
Real-Time Stock Data (as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):
Current Price: ${quote.get('05. price', 'N/A')}
Change: {quote.get('09. change', 'N/A')} ({quote.get('10. change percent', 'N/A')})
Volume: {quote.get('06. volume', 'N/A')}
Previous Close: ${quote.get('08. previous close', 'N/A')}

Company Overview:
Market Cap: ${overview.get('MarketCapitalization', 'N/A')}
PE Ratio: {overview.get('PERatio', 'N/A')}
52-Week High: ${overview.get('52WeekHigh', 'N/A')}
52-Week Low: ${overview.get('52WeekLow', 'N/A')}
Dividend Yield: {overview.get('DividendYield', 'N/A')}
Profit Margin: {overview.get('ProfitMargin', 'N/A')}

Recent Price Movement (Last Hour):"""

        # Add intraday price movement
        for timestamp, values in intraday:
            current_data += f"\n{timestamp}: ${values.get('4. close', 'N/A')}"

        return current_data
    except Exception as e:
        return f"Error formatting real-time data: {str(e)}"

def get_llm_response(model: str, prompt: str, stock_context: Optional[str] = None, web_data: Optional[str] = None, realtime_data: Optional[str] = None) -> str:
    """Get response from selected LLM model"""
    try:
        # Build comprehensive context
        context_parts = []
        
        if realtime_data:
            context_parts.append(f"Real-Time Market Data:\n{realtime_data}")
            
        if stock_context:
            context_parts.append(f"Stock Information:\n{stock_context}")
        
        if web_data:
            context_parts.append(f"Recent Web Data:\n{web_data}")
            
        context_parts.append(f"User Question: {prompt}")
        
        # Create final prompt
        full_prompt = "\n\n".join([
            "You are a financial analyst AI assistant. Provide a detailed analysis based on the following information:",
            *context_parts,
            "Please provide a comprehensive analysis that includes real-time market data, technical analysis, and recent market sentiment."
        ])

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Failed to get response from model. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def should_use_realtime_data(question: str) -> tuple[bool, str]:
    """Determine if real-time data is needed based on the question type"""
    
    # Keywords indicating real-time data need
    realtime_keywords = [
        'current', 'now', 'today', 'moment', 'price', 'trading',
        'buy', 'sell', 'enter', 'exit', 'volatile', 'movement',
        'volume', 'trend', 'intraday', 'momentum', 'session'
    ]
    
    # Keywords indicating historical/fundamental analysis
    historical_keywords = [
        'history', 'historical', 'past', 'longterm', 'fundamental',
        'company', 'business', 'management', 'strategy', 'sector',
        'industry', 'competition', 'moat', 'risk'
    ]
    
    question = question.lower()
    
    # Count keyword matches
    realtime_score = sum(1 for word in realtime_keywords if word in question)
    historical_score = sum(1 for word in historical_keywords if word in question)
    
    # Determine if real-time data is needed
    needs_realtime = realtime_score > historical_score
    
    # Provide reason for the decision
    if needs_realtime:
        reason = "Question involves current market conditions or trading decisions"
    else:
        reason = "Question focuses on historical or fundamental analysis"
    
    return needs_realtime, reason

def stock_chat_tab():
    """Stock Chat tab for interacting with LLMs"""
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .chat-container {
        margin-top: 2rem;
    }
    .model-selector {
        margin-bottom: 1rem;
    }
    .real-time-data {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for API key
    if not os.getenv('ALPHA_VANTAGE_API_KEY'):
        st.warning("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in .env file for real-time data.")
    
    # Get available models
    models = get_available_models()
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select LLM Model",
            models,
            key="llm_model_selector",
            help="Choose the LLM model to chat with about stocks"
        )
    
    # Initialize session state for real-time data
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = None
    
    # Get stock context if ticker is available
    stock_context = None
    web_data = None
    if 'ticker' in st.session_state and st.session_state['ticker']:
        try:
            # Get basic stock info from yfinance
            stock = yf.Ticker(st.session_state['ticker'])
            info = stock.info
            stock_context = f"""
            Company: {info.get('longName', st.session_state['ticker'])}
            Sector: {info.get('sector', 'N/A')}
            Industry: {info.get('industry', 'N/A')}
            Description: {info.get('longBusinessSummary', 'N/A')}
            """
            
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            st.write("Debug: Exception traceback:", traceback.format_exc())
    
    # User input
    user_question = st.text_area(
        "Enter your question about the stock",
        height=100,
        placeholder="Example: What are the key risks and opportunities for this stock based on current market conditions?",
        help="Ask any question about the stock and get AI-powered insights with real-time data"
    )
    
    # Determine if real-time data is needed
    if user_question:
        needs_realtime, reason = should_use_realtime_data(user_question)
    else:
        needs_realtime, reason = False, ""
    
    # Data source toggles with smart defaults
    col1, col2 = st.columns(2)
    with col1:
        include_web_search = st.checkbox(
            "Include latest web information",
            value=True,
            help="Include recent web search results in the analysis"
        )
    with col2:
        include_realtime = st.checkbox(
            "Include real-time market data",
            value=needs_realtime,
            help=f"Include real-time stock data in the analysis\n{reason if reason else ''}"
        )
        
        # Show recommendation if real-time determination was made
        if user_question and reason:
            st.caption(f"💡 {reason}")
    
    # Submit button
    if st.button("Get Analysis", type="primary"):
        if user_question:
            with st.spinner(f"Getting analysis from {selected_model}..."):
                # Get fresh web data if requested
                if include_web_search:
                    web_data = get_web_search_results(f"{user_question} {st.session_state.get('ticker', '')} stock")
                
                # Get fresh real-time data if needed
                realtime_data = None
                if include_realtime and st.session_state.get('ticker'):
                    rt_data = get_realtime_stock_data(st.session_state['ticker'])
                    if "error" not in rt_data:
                        realtime_data = format_realtime_data(rt_data)
                        st.session_state.realtime_data = realtime_data
                        
                        # Add explanation of data usage
                        if needs_realtime:
                            st.info("Using real-time market data for current market analysis")
                
                response = get_llm_response(
                    selected_model,
                    user_question,
                    stock_context,
                    web_data if include_web_search else None,
                    realtime_data if include_realtime else None
                )
                
                # Display response in a nice format
                st.markdown("### AI Analysis")
                st.markdown(response)
                
                # Show data sources
                with st.expander("View Data Sources"):
                    if include_realtime and realtime_data:
                        st.markdown("### Real-Time Market Data")
                        st.markdown(realtime_data)
                        if needs_realtime:
                            st.caption("Real-time data was used due to the nature of your question")
                    
                    if include_web_search and web_data:
                        st.markdown("### Web Search Results")
                        st.markdown(web_data)
                
                # Add disclaimer
                st.markdown("""
                ---
                *Disclaimer: This analysis is generated using artificial intelligence and should not be considered as financial advice. 
                Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*
                """)
        else:
            st.warning("Please enter a question to get analysis.")
