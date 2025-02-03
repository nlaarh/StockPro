import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import percentileofscore
from datetime import datetime, timedelta
from utils import calculate_rsi, calculate_macd


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


def get_options_strategy(ticker, price, rsi, volatility):
    """Determine the best options strategy based on technical indicators"""
    try:
        # Convert rsi to float if it's a Series
        if isinstance(rsi, pd.Series):
            rsi = float(rsi.iloc[-1])
        
        strategy = {
            'name': None,
            'description': None,
            'risk_level': None
        }
        
        # Determine market conditions
        if rsi > 70:  # Overbought
            if volatility > 0.3:  # High volatility
                strategy['name'] = "Put Credit Spread"
                strategy['description'] = "Sell OTM put spread to benefit from high IV and potential mean reversion"
                strategy['risk_level'] = "Medium"
            else:  # Low volatility
                strategy['name'] = "Covered Call"
                strategy['description'] = "Write covered calls against long stock position"
                strategy['risk_level'] = "Low"
        elif rsi < 30:  # Oversold
            if volatility > 0.3:  # High volatility
                strategy['name'] = "Call Credit Spread"
                strategy['description'] = "Sell OTM call spread to benefit from high IV and potential mean reversion"
                strategy['risk_level'] = "Medium"
            else:  # Low volatility
                strategy['name'] = "Long Call"
                strategy['description'] = "Buy calls to benefit from potential upside"
                strategy['risk_level'] = "High"
        else:  # Neutral
            if volatility > 0.3:
                strategy['name'] = "Iron Condor"
                strategy['description'] = "Sell OTM call and put spreads to benefit from high IV"
                strategy['risk_level'] = "Medium"
            else:
                strategy['name'] = "Calendar Spread"
                strategy['description'] = "Sell front-month option and buy back-month option"
                strategy['risk_level'] = "Medium"
                
        return strategy
        
    except Exception as e:
        st.error(f"Error determining options strategy: {str(e)}")
        return None


def get_options_analyst_letter(ticker, current_price, rsi, volatility, strategy, calls, puts, expiration):
    """Generate analyst perspective for options strategy"""
    try:
        # Get nearest ATM options (handle empty DataFrames)
        if calls.empty or puts.empty:
            st.warning("No options data available for analysis")
            return None
            
        # Find ATM options safely
        try:
            atm_call_idx = abs(calls['strike'] - current_price).idxmin()
            atm_put_idx = abs(puts['strike'] - current_price).idxmin()
            atm_call = calls.loc[atm_call_idx]
            atm_put = puts.loc[atm_put_idx]
        except Exception as e:
            st.warning("Could not find ATM options")
            return None
        
        # Calculate key metrics safely
        try:
            days_to_exp = (pd.to_datetime(expiration) - pd.Timestamp.now()).days
            if 'impliedVolatility' in calls.columns:
                iv_rank = percentileofscore(calls['impliedVolatility'].dropna(), calls['impliedVolatility'].mean())
            else:
                iv_rank = 50  # Default to neutral if no IV data
        except Exception as e:
            st.warning("Error calculating options metrics")
            return None
        
        # Determine market sentiment
        if rsi > 70:
            sentiment = "overbought"
            bias = "bearish"
        elif rsi < 30:
            sentiment = "oversold"
            bias = "bullish"
        else:
            sentiment = "neutral"
            bias = "neutral"
            
        # Generate strategy-specific recommendations
        if strategy['name'] == "Put Credit Spread":
            strikes = f"Sell {current_price * 0.95:.2f} Put, Buy {current_price * 0.90:.2f} Put"
            explanation = f"""
            Given the overbought conditions (RSI: {rsi:.1f}) and high volatility ({volatility:.1%}), 
            we recommend a Put Credit Spread strategy. This involves:
            
            1. Selling a put option at the {current_price * 0.95:.2f} strike
            2. Buying a put option at the {current_price * 0.90:.2f} strike
            3. Both options expiring in {days_to_exp} days
            
            This strategy allows you to profit from:
            - High implied volatility ({iv_rank:.0f}th percentile)
            - Potential mean reversion in the stock price
            - Time decay (theta)
            
            Maximum profit is the credit received, while maximum loss is limited to the difference between strikes.
            """
            
        elif strategy['name'] == "Covered Call":
            strikes = f"Sell {current_price * 1.05:.2f} Call"
            explanation = f"""
            With the stock showing overbought signals (RSI: {rsi:.1f}) but low volatility ({volatility:.1%}), 
            a Covered Call strategy is optimal. This involves:
            
            1. Holding (or buying) 100 shares of {ticker}
            2. Selling a call option at the {current_price * 1.05:.2f} strike
            3. Expiring in {days_to_exp} days
            
            This strategy allows you to:
            - Generate additional income from your stock position
            - Provide some downside protection
            - Benefit from time decay
            
            Your upside is capped at the strike price, but you keep the premium regardless.
            """
            
        elif strategy['name'] == "Iron Condor":
            strikes = f"Sell {current_price * 1.05:.2f}/{current_price * 1.10:.2f} Call Spread, {current_price * 0.95:.2f}/{current_price * 0.90:.2f} Put Spread"
            explanation = f"""
            With neutral market conditions (RSI: {rsi:.1f}) and high volatility ({volatility:.1%}), 
            an Iron Condor strategy offers the best risk/reward. This involves:
            
            1. Selling a call spread:
               - Sell {current_price * 1.05:.2f} call
               - Buy {current_price * 1.10:.2f} call
            2. Selling a put spread:
               - Sell {current_price * 0.95:.2f} put
               - Buy {current_price * 0.90:.2f} put
            3. All options expiring in {days_to_exp} days
            
            This strategy profits from:
            - High implied volatility ({iv_rank:.0f}th percentile)
            - Time decay
            - Stock trading sideways
            
            Maximum profit is the net credit received, with defined risk on both sides.
            """
            
        else:
            strikes = f"ATM Call: {atm_call['strike']:.2f}, ATM Put: {atm_put['strike']:.2f}"
            explanation = f"""
            Given the current market conditions:
            - RSI: {rsi:.1f} ({sentiment})
            - Volatility: {volatility:.1%}
            - Days to expiration: {days_to_exp}
            
            We recommend a {strategy['name']} strategy. This approach is suitable because:
            - Market bias is {bias}
            - Implied volatility is in the {iv_rank:.0f}th percentile
            - {strategy['description']}
            
            Consider options near the current price ({current_price:.2f}) for optimal risk/reward.
            Risk level is {strategy['risk_level'].lower()}.
            """
            
        letter = f"""
        # Options Strategy Analysis for {ticker}
        
        Dear Valued Investor,
        
        I am writing to provide you with a detailed analysis of the current options market conditions for {ticker} and our recommended strategy based on the prevailing market environment.
        
        ## Market Analysis
        
        Current market conditions indicate:
        - Stock Price: ${current_price:.2f}
        - RSI: {rsi:.1f} ({sentiment})
        - Volatility: {volatility:.1%}
        - IV Rank: {iv_rank:.0f}th percentile
        - Days to Expiration: {days_to_exp}
        
        ## Strategy Recommendation: {strategy['name']}
        
        Based on our quantitative and qualitative analysis, we recommend implementing a {strategy['name']} strategy with the following specifications:
        
        **{strikes}**
        
        {explanation}
        
        ## Risk Management Framework
        
        1. **Position Sizing:**
           - Initial Position: 2-3% of portfolio value
           - Maximum Position: 5% of portfolio value
           - Scale-in Approach: Consider multiple entries
        
        2. **Risk Parameters:**
           - Stop Loss: Exit at 50% of maximum loss
           - Profit Target: Take profits at 50-75% of maximum gain
           - Time Stop: Close or roll position at 21 DTE
        
        3. **Volatility Considerations:**
           - Current IV Percentile: {iv_rank:.0f}%
           - IV Outlook: {'Elevated - consider premium selling strategies' if iv_rank > 60 else 'Low - consider premium buying strategies' if iv_rank < 30 else 'Moderate - neutral strategies preferred'}
           - Expected Move: ${current_price * volatility / np.sqrt(252):.2f} (1-day, 1 SD)
        
        ## Execution Guidelines
        
        1. Entry Timing:
           - {'Wait for price consolidation' if volatility > 0.3 else 'Consider immediate entry' if rsi < 40 else 'Scale in gradually'}
           - Monitor volume for confirmation
           - Check bid-ask spreads for liquidity
        
        2. Position Management:
           - Regular delta adjustments if needed
           - Monitor gamma exposure near expiration
           - Consider rolling at 21 DTE
        
        3. Hedging Considerations:
           - {'Consider protective puts' if strategy['risk_level'] == 'High' else 'No additional hedging needed'}
           - Maintain balanced portfolio exposure
           - Monitor correlation with existing positions
        
        ## Additional Considerations
        
        - Market Environment: {'Volatile' if volatility > 0.3 else 'Stable'} with {'strong' if abs(rsi - 50) > 20 else 'moderate' if abs(rsi - 50) > 10 else 'neutral'} momentum
        - Earnings Impact: Monitor upcoming announcements
        - Sector Analysis: Consider correlation with sector movements
        
        This strategy aligns with our current market outlook and offers a {'conservative' if strategy['risk_level'] == 'Low' else 'balanced' if strategy['risk_level'] == 'Medium' else 'aggressive'} approach to options trading in the present environment.
        
        Best regards,
        
        Nour Learoubi
        Chief Investment Officer
        StockPro Analytics
        
        *Note: This analysis is valid as of {pd.Timestamp.now().strftime('%Y-%m-%d')} and should be reviewed regularly as market conditions change.*
        """
        
        return letter
        
    except Exception as e:
        st.error(f"Error generating options analyst letter: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
        sentiment = analyze_options_sentiment(iv, cp_ratio, current_price, max_pain)
        if sentiment:
            st.markdown(f"### Market Sentiment\n{sentiment}")
            
    except Exception as e:
        st.error(f"Error displaying options analysis: {str(e)}")


def calculate_max_pain(calls, puts):
    """Calculate the max pain point"""
    try:
        strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        
        # Calculate total value for each strike
        pain = []
        for strike in strikes:
            # Calculate call pain
            call_pain = sum(
                max(0, strike - k) * v 
                for k, v in zip(calls['strike'], calls['openInterest'])
                if k <= strike
            )
            
            # Calculate put pain
            put_pain = sum(
                max(0, k - strike) * v 
                for k, v in zip(puts['strike'], puts['openInterest'])
                if k >= strike
            )
            
            pain.append(call_pain + put_pain)
            
        # Return strike price with minimum pain
        return strikes[pain.index(min(pain))]
    except:
        return 0

def analyze_options_sentiment(iv, cp_ratio, current_price, max_pain):
    """Analyze overall options market sentiment"""
    # Score different factors
    scores = []
    
    # Implied volatility
    if iv > 50:
        scores.append("High implied volatility suggests significant uncertainty")
    elif iv > 30:
        scores.append("Moderate implied volatility indicates normal market conditions")
    else:
        scores.append("Low implied volatility suggests stable price expectations")
    
    # Call/put ratio
    if cp_ratio > 1.5:
        scores.append("Strong bullish sentiment from call/put ratio")
    elif cp_ratio < 0.5:
        scores.append("Strong bearish sentiment from call/put ratio")
    else:
        scores.append("Neutral sentiment from call/put ratio")
    
    # Max pain
    pain_diff = ((current_price / max_pain - 1) * 100) if max_pain > 0 else 0
    if abs(pain_diff) < 2:
        scores.append("Price near max pain point suggests potential consolidation")
    elif pain_diff > 0:
        scores.append(f"Price {pain_diff:.1f}% above max pain may face resistance")
    else:
        scores.append(f"Price {-pain_diff:.1f}% below max pain may find support")
    
    return "\n".join(scores)

def get_current_price(ticker):
    """Get current stock price"""
    try:
        stock = yf.Ticker(ticker)
        # Try different price fields in order of preference
        price = (
            stock.info.get('currentPrice') or 
            stock.info.get('regularMarketPrice') or
            stock.info.get('previousClose') or
            stock.history(period='1d')['Close'].iloc[-1]
        )
        return price
    except Exception as e:
        st.error(f"Error getting price: {str(e)}")
        return None

def get_implied_volatility(ticker):
    """Calculate average implied volatility from options chain"""
    try:
        stock = yf.Ticker(ticker)
        # Get options expiration dates
        expirations = stock.options
        
        if not expirations:
            return 0
            
        # Get nearest expiration date
        nearest_exp = expirations[0]
        
        # Get options chain for nearest expiration
        calls = stock.option_chain(nearest_exp).calls
        puts = stock.option_chain(nearest_exp).puts
        
        # Calculate average implied volatility
        call_iv = calls['impliedVolatility'].mean() * 100
        put_iv = puts['impliedVolatility'].mean() * 100
        
        return (call_iv + put_iv) / 2
    except:
        return 0

def get_call_put_ratio(ticker):
    """Calculate call/put ratio based on volume"""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        
        if not expirations:
            return 1.0
            
        # Get nearest expiration date
        nearest_exp = expirations[0]
        
        # Get options chain
        calls = stock.option_chain(nearest_exp).calls
        puts = stock.option_chain(nearest_exp).puts
        
        # Calculate total volume
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        
        if put_volume == 0:
            return 1.0
            
        return call_volume / put_volume
    except:
        return 1.0

def options_analysis_tab():
    """Display options analysis and trading strategies"""
    st.subheader("Options Analysis")
    
    # Get ticker from session state
    ticker = st.session_state.ticker
    
    if ticker:
        display_analysis(ticker)
    else:
        st.warning("Please enter a stock symbol to analyze options.")
