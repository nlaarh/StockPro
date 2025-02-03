import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

def get_options_data(ticker: str) -> Tuple[yf.Ticker, List[str]]:
    """Get options data for a given ticker"""
    try:
        # Create new Ticker object
        stock = yf.Ticker(ticker)
        
        # Get options expiration dates
        try:
            expiration_dates = stock.options
            if not expiration_dates:
                st.warning(f"No options data available for {ticker}")
                return None, []
            return stock, expiration_dates
        except Exception as e:
            st.warning(f"Error getting options data: {str(e)}")
            return None, []
            
    except Exception as e:
        st.error(f"Error creating Ticker object: {str(e)}")
        return None, []

def calculate_implied_volatility(row: pd.Series) -> float:
    """Calculate implied volatility"""
    try:
        return row['impliedVolatility'] * 100
    except:
        return np.nan

def get_cio_letter(stock: yf.Ticker, expiry_date: str, stock_price: float, 
                   iv_percentile: float, earnings_date: Optional[datetime],
                   options_chain: pd.DataFrame) -> str:
    """Generate a CIO letter with options trade recommendations"""
    
    # Get stock info and current date
    try:
        info = stock.info
        # Get the ticker symbol from the stock object's ticker attribute
        ticker_symbol = getattr(stock, 'ticker', None)
        if ticker_symbol is None:
            # Try to get it from the info dictionary
            ticker_symbol = info.get('symbol', 'Unknown')
            
        company_name = info.get('longName', ticker_symbol)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
    except:
        ticker_symbol = getattr(stock, 'ticker', 'Unknown')
        company_name = ticker_symbol
        sector = 'N/A'
        industry = 'N/A'
    
    current_date = datetime.now().strftime('%B %d, %Y')
    expiry = datetime.strptime(expiry_date, '%Y-%m-%d').strftime('%B %d, %Y')
    
    # Format earnings date if available
    earnings_str = ""
    if earnings_date:
        days_to_earnings = (earnings_date - datetime.now()).days
        if days_to_earnings > 0:
            earnings_str = f"\nEarnings are scheduled in {days_to_earnings} days on {earnings_date.strftime('%B %d, %Y')}."
        elif days_to_earnings == 0:
            earnings_str = "\nEarnings are scheduled for today."
    
    # Calculate days to expiration
    days_to_expiry = (datetime.strptime(expiry_date, '%Y-%m-%d') - datetime.now()).days
    
    # Analyze options data
    calls = options_chain[options_chain['contractType'] == 'call'] if 'contractType' in options_chain.columns else options_chain.head(len(options_chain)//2)
    puts = options_chain[options_chain['contractType'] == 'put'] if 'contractType' in options_chain.columns else options_chain.tail(len(options_chain)//2)
    
    # Find ATM options
    atm_strike = min(options_chain['strike'], key=lambda x: abs(x - stock_price))
    atm_call = calls[calls['strike'] == atm_strike].iloc[0] if not calls[calls['strike'] == atm_strike].empty else None
    atm_put = puts[puts['strike'] == atm_strike].iloc[0] if not puts[puts['strike'] == atm_strike].empty else None
    
    # Determine market sentiment
    sentiment = ""
    if iv_percentile > 75:
        sentiment = "highly volatile"
    elif iv_percentile > 50:
        sentiment = "moderately volatile"
    elif iv_percentile > 25:
        sentiment = "relatively stable"
    else:
        sentiment = "very stable"
    
    # Generate strategy recommendations
    strategies = []
    
    # High IV strategies
    if iv_percentile > 60:
        strategies.append({
            'name': "Iron Condor",
            'description': "Sell OTM call and put credit spreads to take advantage of high implied volatility",
            'risk_level': "Medium"
        })
        strategies.append({
            'name': "Credit Spread",
            'description': f"Sell OTM {'put' if stock_price > atm_strike else 'call'} spread to benefit from IV crush",
            'risk_level': "Medium"
        })
    
    # Low IV strategies
    else:
        strategies.append({
            'name': "Long Call/Put",
            'description': f"Buy {'calls' if stock_price < atm_strike else 'puts'} to benefit from potential directional move",
            'risk_level': "High"
        })
        strategies.append({
            'name': "Calendar Spread",
            'description': "Buy longer-term options and sell shorter-term options to benefit from time decay",
            'risk_level': "Medium"
        })
    
    # Generate the letter
    letter = f"""
    # Options Analysis for {company_name} ({ticker_symbol})
    
    **Date:** {current_date}
    
    Dear Valued Investor,
    
    I hope this letter finds you well. Today, I am writing to provide you with a comprehensive analysis of the options market for {company_name} ({sector} - {industry}).
    
    ## Current Market Context
    - Stock Price: ${stock_price:.2f}
    - Days to Expiration: {days_to_expiry}
    - IV Percentile: {iv_percentile:.1f}%{earnings_str}
    
    The options market is currently indicating a {sentiment} environment for {company_name}. 
    
    ## Volatility Analysis
    With an IV percentile of {iv_percentile:.1f}%, the market is pricing in {'significant' if iv_percentile > 50 else 'moderate'} movement in the stock. 
    {'This elevated volatility presents opportunities for premium selling strategies.' if iv_percentile > 60 else 'The lower volatility environment may favor directional strategies.'}
    
    ## Recommended Options Strategies
    
    Based on the current market conditions, here are my recommended strategies:
    """
    
    # Add strategy details
    for i, strategy in enumerate(strategies, 1):
        letter += f"""
    {i}. **{strategy['name']}**
       - {strategy['description']}
       - Risk Level: {strategy['risk_level']}
        """
    
    # Add risk management guidelines
    letter += """
    ## Risk Management Guidelines
    1. Position Size: Limit each options position to 1-3% of your portfolio
    2. Stop Loss: Set stop losses at 50% of maximum risk for defined-risk trades
    3. Profit Targets: Take profits at 50-75% of maximum potential profit
    4. Time Management: Close positions when 21 days or less to expiration
    
    Please note that these recommendations are based on current market conditions and should be adjusted as conditions change. Always conduct your own due diligence and consider your risk tolerance before entering any position.
    
    Best regards,
    Your Options Strategy Team
    """
    
    return letter

def get_current_price(stock: yf.Ticker) -> float:
    """Get current stock price using multiple fallback methods"""
    try:
        # Try getting from history first (most reliable)
        try:
            hist = stock.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
            
        # Try getting from info
        try:
            info = stock.info
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                return float(info['regularMarketPrice'])
            if 'currentPrice' in info and info['currentPrice']:
                return float(info['currentPrice'])
        except:
            pass
            
        # Try fast info as last resort
        try:
            fast_info = stock.fast_info
            if hasattr(fast_info, 'last_price') and fast_info.last_price:
                return float(fast_info.last_price)
        except:
            pass
            
        raise ValueError("Could not get current price from any source")
        
    except Exception as e:
        st.error(f"Error getting current price: {str(e)}")
        return None

def calculate_option_greeks(row: pd.Series, stock_price: float, risk_free_rate: float = 0.05) -> pd.Series:
    """Calculate option Greeks if they're not provided"""
    try:
        # Convert time to expiry to years
        days_to_expiry = (pd.to_datetime(row['expiration']) - pd.Timestamp.now()).days / 365.0
        
        # Use Black-Scholes formula to calculate Greeks
        S = stock_price  # Current stock price
        K = row['strike']  # Strike price
        r = risk_free_rate  # Risk-free rate
        sigma = row['impliedVolatility']  # Implied volatility
        t = days_to_expiry  # Time to expiration in years
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + sigma**2/2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        
        # Calculate Greeks
        from scipy.stats import norm
        N = norm.cdf
        n = norm.pdf
        
        # Delta
        if row['contractType'] == 'call':
            delta = N(d1)
        else:
            delta = N(d1) - 1
            
        # Gamma (same for calls and puts)
        gamma = n(d1)/(S*sigma*np.sqrt(t))
        
        # Theta
        theta = (-S*sigma*n(d1))/(2*np.sqrt(t)) - r*K*np.exp(-r*t)*N(d2 if row['contractType'] == 'call' else -d2)
        
        # Vega (same for calls and puts)
        vega = S*np.sqrt(t)*n(d1)
        
        return pd.Series({
            'delta': delta,
            'gamma': gamma,
            'theta': theta/365,  # Convert to daily theta
            'vega': vega/100  # Convert to 1% change in IV
        })
    except:
        return pd.Series({
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        })

def highlight_atm(x, stock_price):
    """Highlight ATM and near-ATM options"""
    if x.name == 'strike':
        return ['background-color: #e6ffe6' if abs(stock_price - v) < 5 else '' for v in x]
    return ['' for _ in x]

def options_analysis_tab():
    """Options analysis tab with CIO letter and options chain"""
    try:
        if 'ticker' not in st.session_state:
            st.warning("Please enter a stock symbol in the main input field")
            return
            
        # Get and validate ticker
        ticker = st.session_state['ticker']
        if not ticker:
            st.warning("Please enter a valid stock symbol")
            return
        
        # Get options data
        stock, expiration_dates = get_options_data(ticker)
        
        if not stock or not expiration_dates:
            st.error(f"No options data available for {ticker}")
            return
            
        # Get current stock price
        stock_price = get_current_price(stock)
        if stock_price is None:
            st.error(f"Could not get current price for {ticker}")
            return
            
        # Get earnings date if available
        try:
            earnings_date = pd.to_datetime(stock.calendar.iloc[0,0])
        except:
            earnings_date = None
        
        # Expiry date selection
        st.markdown("## 🔍 Options Chain")
        expiry_date = st.selectbox(
            "Select Expiration Date",
            expiration_dates,
            format_func=lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%B %d, %Y')
        )
        
        # Get options chain
        options = stock.option_chain(expiry_date)
        
        # Calculate IV percentile
        try:
            # Get historical volatility data
            hist_vol = stock.history(period='1y')['Close'].pct_change().std() * np.sqrt(252) * 100
            current_iv = options.calls['impliedVolatility'].mean() * 100
            iv_percentile = np.percentile([hist_vol, current_iv], 75)
        except:
            iv_percentile = 50  # Default to neutral if calculation fails
        
        # Create combined options chain
        calls = options.calls.copy()
        puts = options.puts.copy()
        
        # Add contract type and expiration
        calls['contractType'] = 'call'
        puts['contractType'] = 'put'
        calls['expiration'] = expiry_date
        puts['expiration'] = expiry_date
        
        # Calculate Greeks if they're missing
        required_columns = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'volume', 'openInterest']
        greek_columns = ['delta', 'gamma', 'theta', 'vega']
        
        # Ensure required columns exist
        for col in required_columns:
            if col not in calls.columns:
                calls[col] = 0.0
            if col not in puts.columns:
                puts[col] = 0.0
        
        # Calculate Greeks if any are missing
        if not all(col in calls.columns for col in greek_columns):
            greek_data = calls.apply(lambda row: calculate_option_greeks(row, stock_price), axis=1)
            for col in greek_columns:
                calls[col] = greek_data[col]
                
        if not all(col in puts.columns for col in greek_columns):
            greek_data = puts.apply(lambda row: calculate_option_greeks(row, stock_price), axis=1)
            for col in greek_columns:
                puts[col] = greek_data[col]
        
        # Combine chains
        chain = pd.concat([calls, puts])
        
        # Format the options chains
        format_dict = {
            'lastPrice': '${:.2f}',
            'strike': '${:.2f}',
            'bid': '${:.2f}',
            'ask': '${:.2f}',
            'impliedVolatility': '{:.1%}',
            'delta': '{:.3f}',
            'gamma': '{:.3f}',
            'theta': '{:.3f}',
            'vega': '{:.3f}',
            'volume': '{:,.0f}',
            'openInterest': '{:,.0f}'
        }
        
        # Select columns to display
        display_columns = [
            'strike', 'lastPrice', 'bid', 'ask', 
            'impliedVolatility', 'volume', 'openInterest',
            'delta', 'gamma', 'theta', 'vega'
        ]
        
        # Display options chain
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📞 Calls")
            # Sort calls by strike price
            calls_display = calls[display_columns].sort_values('strike')
            # Format and display
            st.dataframe(
                calls_display.style
                .format(format_dict)
                .apply(lambda x: highlight_atm(x, stock_price)),
                height=400
            )
        
        with col2:
            st.markdown("### 📞 Puts")
            # Sort puts by strike price
            puts_display = puts[display_columns].sort_values('strike')
            # Format and display
            st.dataframe(
                puts_display.style
                .format(format_dict)
                .apply(lambda x: highlight_atm(x, stock_price)),
                height=400
            )
        
        # Add summary metrics
        st.markdown("### 📊 Options Summary")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Total Call Volume", f"{calls['volume'].sum():,.0f}")
        with metrics_col2:
            st.metric("Total Put Volume", f"{puts['volume'].sum():,.0f}")
        with metrics_col3:
            cp_ratio = calls['volume'].sum() / puts['volume'].sum() if puts['volume'].sum() > 0 else float('inf')
            st.metric("Call/Put Ratio", f"{cp_ratio:.2f}")
        with metrics_col4:
            st.metric("Avg IV", f"{current_iv:.1f}%")
            
        # Generate and display CIO letter
        st.markdown("## 📈 Options Analysis and Recommendations")
        letter = get_cio_letter(stock, expiry_date, stock_price, iv_percentile, earnings_date, chain)
        st.markdown(letter)
                
    except Exception as e:
        st.error(f"Error in options analysis: {str(e)}")
        return

def get_options_chain(ticker):
    """Fetch options chain data for a given ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get options expiration dates
        try:
            expirations = stock.options
        except:
            # Retry with force=True to bypass cache
            stock = yf.Ticker(ticker, session=None)
            expirations = stock.options
        
        if not expirations:
            st.warning("""
            No options data available. This could be due to:
            1. Market hours (options data may be delayed outside market hours)
            2. Data provider limitations
            3. No options trading for this stock
            
            Try again in a few minutes or during market hours.
            """)
            return None, None, None
            
        # Get nearest expiration date
        expiration = expirations[0]
        
        # Get options chain with retry
        try:
            opt = stock.option_chain(expiration)
        except:
            # Retry with force=True
            stock = yf.Ticker(ticker, session=None)
            opt = stock.option_chain(expiration)
            
        if not hasattr(opt, 'calls') or not hasattr(opt, 'puts'):
            st.warning(f"Invalid options data structure for {ticker}")
            return None, None, None
        
        # Add expiration date to both DataFrames
        calls = opt.calls.copy()
        puts = opt.puts.copy()
        calls['expiration'] = expiration
        puts['expiration'] = expiration
        
        # Ensure required columns exist
        required_columns = ['strike', 'impliedVolatility', 'inTheMoney']
        for col in required_columns:
            if col not in calls.columns or col not in puts.columns:
                st.warning(f"Missing required column: {col}")
                return None, None, None
        
        return calls, puts, expiration
        
    except Exception as e:
        st.error(f"Error fetching options chain: {str(e)}")
        return None, None, None

def calculate_max_pain(calls, puts):
    """Calculate the max pain point"""
    try:
        # Get unique strike prices
        strikes = sorted(set(calls['strike'].unique()) | set(puts['strike'].unique()))
        
        # Calculate total value of options at each strike
        max_pain = 0
        min_value = float('inf')
        
        for strike in strikes:
            total_value = 0
            
            # Add call values
            call_options = calls[calls['strike'] <= strike]
            total_value += (strike - call_options['strike']) * call_options['volume'].fillna(0)
            
            # Add put values
            put_options = puts[puts['strike'] >= strike]
            total_value += (put_options['strike'] - strike) * put_options['volume'].fillna(0)
            
            # Update max pain if this strike has lower total value
            if total_value.sum() < min_value:
                min_value = total_value.sum()
                max_pain = strike
                
        return max_pain
        
    except Exception as e:
        st.error(f"Error calculating max pain: {str(e)}")
        return None

def analyze_options_sentiment(iv, cp_ratio, current_price, max_pain):
    """Analyze overall options market sentiment"""
    try:
        sentiment = ""
        
        # Analyze IV
        if iv > 0.4:
            sentiment += "High implied volatility suggests market uncertainty. "
        elif iv < 0.2:
            sentiment += "Low implied volatility suggests market complacency. "
        else:
            sentiment += "Moderate implied volatility suggests balanced market expectations. "
        
        # Analyze call/put ratio
        if cp_ratio > 1.5:
            sentiment += "High call/put ratio indicates bullish sentiment. "
        elif cp_ratio < 0.5:
            sentiment += "Low call/put ratio indicates bearish sentiment. "
        else:
            sentiment += "Balanced call/put ratio suggests neutral sentiment. "
        
        # Analyze max pain
        if max_pain > current_price * 1.05:
            sentiment += "Max pain above current price suggests potential upward pressure. "
        elif max_pain < current_price * 0.95:
            sentiment += "Max pain below current price suggests potential downward pressure. "
        else:
            sentiment += "Max pain near current price suggests price stability. "
            
        return sentiment
        
    except Exception as e:
        st.error(f"Error analyzing options sentiment: {str(e)}")
        return None

def get_implied_volatility(ticker):
    """Calculate average implied volatility from options chain"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get nearest expiration options
        expiry = stock.options[0]
        chain = stock.option_chain(expiry)
        
        # Calculate average IV
        all_iv = pd.concat([
            chain.calls['impliedVolatility'],
            chain.puts['impliedVolatility']
        ])
        
        return np.mean(all_iv) * 100
        
    except Exception as e:
        st.error(f"Error calculating IV: {str(e)}")
        return None

def get_call_put_ratio(ticker):
    """Calculate call/put ratio based on volume"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get nearest expiration options
        expiry = stock.options[0]
        chain = stock.option_chain(expiry)
        
        # Calculate volume ratio
        call_volume = chain.calls['volume'].sum()
        put_volume = chain.puts['volume'].sum()
        
        return call_volume / put_volume if put_volume > 0 else 0
        
    except Exception as e:
        st.error(f"Error calculating call/put ratio: {str(e)}")
        return None

def display_analysis(ticker):
    """Display options analysis"""
    try:
        st.subheader(f"Options Analysis for {ticker}")
        
        # Get stock data and options chain
        calls, puts, expiration = get_options_chain(ticker)
        if calls is None or puts is None:
            return
            
        # Get current price and other metrics
        current_price = get_current_price(ticker)
        if current_price is None:
            st.error(f"Could not get current price for {ticker}")
            return
            
        # Create metrics columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}"
            )
            
        with col2:
            iv = get_implied_volatility(ticker)
            st.metric(
                "Implied Volatility",
                f"{iv:.1f}%" if iv else "N/A"
            )
            
        with col3:
            cp_ratio = get_call_put_ratio(ticker)
            st.metric(
                "Call/Put Ratio",
                f"{cp_ratio:.2f}" if cp_ratio else "N/A"
            )
            
        # Display options chain
        st.subheader("Options Chain")
        
        # Format options chain for display
        def format_chain(chain, option_type):
            if chain is not None and not chain.empty:
                # Select and rename columns
                display_cols = {
                    'strike': 'Strike',
                    'lastPrice': 'Last',
                    'bid': 'Bid',
                    'ask': 'Ask',
                    'volume': 'Volume',
                    'openInterest': 'Open Int',
                    'impliedVolatility': 'IV'
                }
                
                chain_display = chain[display_cols.keys()].copy()
                chain_display.columns = display_cols.values()
                
                # Format numeric columns
                chain_display['Strike'] = chain_display['Strike'].map('${:,.2f}'.format)
                chain_display['Last'] = chain_display['Last'].map('${:,.2f}'.format)
                chain_display['Bid'] = chain_display['Bid'].map('${:,.2f}'.format)
                chain_display['Ask'] = chain_display['Ask'].map('${:,.2f}'.format)
                chain_display['IV'] = chain_display['IV'].map('{:.1%}'.format)
                
                return chain_display
            return pd.DataFrame()
            
        # Create tabs for calls and puts
        call_tab, put_tab = st.tabs(["Calls", "Puts"])
        
        with call_tab:
            st.dataframe(
                format_chain(calls, 'call'),
                use_container_width=True,
                hide_index=True
            )
            
        with put_tab:
            st.dataframe(
                format_chain(puts, 'put'),
                use_container_width=True,
                hide_index=True
            )
            
        # Calculate and display max pain
        max_pain = calculate_max_pain(calls, puts)
        if max_pain:
            st.metric("Max Pain", f"${max_pain:.2f}")
            
        # Display options sentiment
        sentiment = analyze_options_sentiment(get_implied_volatility(ticker), get_call_put_ratio(ticker), current_price, max_pain)
        if sentiment:
            st.markdown(f"### Market Sentiment\n{sentiment}")
            
    except Exception as e:
        st.error(f"Error displaying options analysis: {str(e)}")

def calculate_max_pain(calls, puts):
    """Calculate the max pain point"""
    try:
        # Get unique strike prices
        strikes = sorted(set(calls['strike'].unique()) | set(puts['strike'].unique()))
        
        # Calculate total loss at each strike price
        max_pain = 0
        min_loss = float('inf')
        
        for strike in strikes:
            # Calculate loss for calls
            call_loss = sum(
                max(0, strike - call_strike) * call_volume
                for call_strike, call_volume in zip(calls['strike'], calls['volume'])
                if call_strike <= strike
            )
            
            # Calculate loss for puts
            put_loss = sum(
                max(0, put_strike - strike) * put_volume
                for put_strike, put_volume in zip(puts['strike'], puts['volume'])
                if put_strike >= strike
            )
            
            total_loss = call_loss + put_loss
            
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain = strike
                
        return max_pain
        
    except Exception as e:
        st.error(f"Error calculating max pain: {str(e)}")
        return None

def analyze_options_sentiment(iv, cp_ratio, current_price, max_pain):
    """Analyze overall options market sentiment"""
    try:
        sentiment = ""
        
        # Analyze IV
        if iv > 50:
            sentiment += "High implied volatility suggests significant price movement is expected. "
        else:
            sentiment += "Low implied volatility suggests relatively stable price movement. "
            
        # Analyze call/put ratio
        if cp_ratio > 1.5:
            sentiment += "High call/put ratio indicates bullish sentiment. "
        elif cp_ratio < 0.5:
            sentiment += "Low call/put ratio indicates bearish sentiment. "
        else:
            sentiment += "Balanced call/put ratio suggests neutral sentiment. "
            
        # Analyze max pain
        if max_pain:
            diff = ((max_pain - current_price) / current_price) * 100
            if abs(diff) > 5:
                sentiment += f"Max pain point is significantly {'above' if diff > 0 else 'below'} current price, "
                sentiment += "suggesting potential price movement towards ${:.2f}. ".format(max_pain)
            else:
                sentiment += "Current price is near the max pain point. "
                
        return sentiment
        
    except Exception as e:
        st.error(f"Error analyzing options sentiment: {str(e)}")
        return ""
