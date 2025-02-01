import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import percentileofscore
from datetime import datetime

from utils import calculate_rsi, calculate_macd


def get_options_chain(ticker):
    """Fetch options chain data for a given ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get options expiration dates
        expirations = stock.options
        
        if not expirations:
            st.warning(f"No options data available for {ticker}")
            return None, None, None
            
        # Get nearest expiration date
        expiration = expirations[0]
        
        # Get options chain
        opt = stock.option_chain(expiration)
        if not hasattr(opt, 'calls') or not hasattr(opt, 'puts'):
            st.warning(f"Invalid options data for {ticker}")
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


def highlight_options(df, current_price, is_calls=True):
    """Highlight optimal entry/exit points in options chain"""
    # Implementation of options highlighting
    pass


def get_best_options(calls, puts, current_price, strategy, data=None):
    """Get the best options to trade based on strategy and market conditions"""
    # Implementation of best options selection
    pass


def analyze_market_conditions(data, current_price, volatility, rsi):
    """Analyze market conditions for options strategy selection"""
    # Implementation of market condition analysis
    pass


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


def options_analysis_tab(ticker):
    """Display options analysis and trading strategies"""
    try:
        st.subheader(f"Options Analysis for {ticker}")
        
        # Add investment essay
        with st.expander("📚 Understanding Options Investment", expanded=True):
            st.markdown("""
            # A Guide to Options Investment
            
            Options trading represents a sophisticated approach to investing that can offer both enhanced returns and risk management capabilities. This comprehensive guide will help you understand the fundamentals and advanced concepts of options investing.
            
            ## What Are Options?
            Options are financial derivatives that give buyers the right, but not the obligation, to buy (calls) or sell (puts) an underlying asset at a predetermined price within a specific time frame. They serve multiple purposes in a modern investment portfolio:
            
            1. **Income Generation**
               - Writing covered calls for regular premium income
               - Selling puts to acquire stocks at desired prices
               - Creating structured income strategies
            
            2. **Risk Management**
               - Hedging portfolio positions against market downturns
               - Protecting unrealized gains in long positions
               - Reducing portfolio volatility
            
            3. **Leverage and Capital Efficiency**
               - Controlling more shares with less capital
               - Amplifying returns on directional moves
               - Implementing sophisticated spread strategies
            
            ## Key Components of Options
            
            ### 1. Strike Price
            - The predetermined price at which the option can be exercised
            - Determines if an option is In-The-Money (ITM) or Out-of-The-Money (OTM)
            - Critical for strategy selection and risk/reward profiles
            
            ### 2. Expiration Date
            - The deadline for exercising the option
            - Affects time decay (theta) and option premium
            - Shorter dates = faster time decay but lower cost
            
            ### 3. Implied Volatility (IV)
            - Market's forecast of likely movement
            - Higher IV = more expensive options
            - Key for selecting optimal entry points
            
            ## Advanced Concepts
            
            ### 1. The Greeks
            - **Delta**: Directional risk exposure
            - **Theta**: Time decay impact
            - **Vega**: Volatility sensitivity
            - **Gamma**: Rate of delta change
            
            ### 2. Volatility Analysis
            - Historical vs. Implied Volatility
            - Volatility skew and term structure
            - Mean reversion opportunities
            
            ## Risk Management Principles
            
            1. **Position Sizing**
               - Never risk more than 2-3% on any single trade
               - Consider portfolio correlation
               - Account for maximum possible loss
            
            2. **Strategy Selection**
               - Match strategies to market outlook
               - Consider volatility environment
               - Account for liquidity constraints
            
            3. **Exit Planning**
               - Predefined profit targets
               - Stop-loss levels
               - Time-based exits
            
            ## Best Practices
            
            1. **Start Small**
               - Begin with simple strategies
               - Paper trade to gain experience
               - Gradually increase complexity
            
            2. **Continuous Learning**
               - Study market behavior
               - Analyze past trades
               - Stay informed about market conditions
            
            3. **Risk First**
               - Focus on risk management
               - Use defined-risk strategies
               - Always know your maximum loss
            
            ## Common Pitfalls to Avoid
            
            1. **Overleverage**
               - Don't use excessive position sizes
               - Consider worst-case scenarios
               - Maintain adequate cash reserves
            
            2. **Ignoring Volatility**
               - Pay attention to IV levels
               - Consider volatility mean reversion
               - Adjust strategies based on IV environment
            
            3. **Poor Position Management**
               - Don't let small losses become large
               - Take profits when available
               - Adjust positions when necessary
            
            ## Conclusion
            
            Options trading requires a disciplined approach combining technical analysis, risk management, and strategic thinking. Success comes from:
            
            - Understanding fundamental concepts
            - Implementing proper risk management
            - Maintaining emotional discipline
            - Continuous education and improvement
            
            Remember that options trading involves substantial risk and is not suitable for all investors. Always consider your risk tolerance and investment objectives before implementing any strategy.
            """)
        
        # Add strategy explanation section
        with st.expander("ℹ️ Understanding Options Strategies"):
            st.markdown("""
            ### Options Trading Strategies Guide
            
            #### Key Concepts:
            - **Call Options**: Right to buy stock at strike price
            - **Put Options**: Right to sell stock at strike price
            - **In-The-Money (ITM)**: Options with immediate exercise value (shown in 🟢)
            - **Out-of-The-Money (OTM)**: Options with no immediate exercise value
            
            #### Common Strategies:
            1. **Covered Call**
               - Low risk, income generation
               - Sell calls against owned stock
            
            2. **Put Credit Spread**
               - Medium risk, defined risk/reward
               - Profit from high volatility
            
            3. **Iron Condor**
               - Medium risk, market neutral
               - Profit from time decay
            
            4. **Long Call/Put**
               - High risk, directional bet
               - Maximum leverage
            """)
        
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            st.error(f"No historical data available for {ticker}")
            return
            
        current_price = data['Close'].iloc[-1]
        
        # Calculate technical indicators
        rsi = calculate_rsi(data)[-1]
        volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        
        # Get options chain
        calls, puts, expiration = get_options_chain(ticker)
        
        if calls is None or puts is None or expiration is None:
            st.warning("Unable to analyze options. Please try a different stock.")
            return
            
        # Get options strategy
        strategy = get_options_strategy(ticker, current_price, rsi, volatility)
        
        if strategy:
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                
            with col2:
                st.metric("RSI", f"{rsi:.1f}", 
                    delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                    delta_color="inverse")
                
            with col3:
                st.metric("Volatility", f"{volatility:.1%}",
                    delta="High" if volatility > 0.3 else "Low",
                    delta_color="off")
            
            # Display strategy recommendation with better formatting
            st.markdown("### 📈 Strategy Recommendation")
            rec_col1, rec_col2 = st.columns([1, 2])
            
            with rec_col1:
                st.info(f"**Strategy:** {strategy['name']}")
                st.warning(f"**Risk Level:** {strategy['risk_level']}")
                
            with rec_col2:
                st.write(f"**Description:** {strategy['description']}")
            
            # Display options chains with legend
            st.markdown("### 📊 Options Chain Analysis")
            st.markdown("""
            <small>🟢 In-The-Money (ITM) options are highlighted in green. These options have intrinsic value.</small>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["📈 Calls", "📉 Puts"])
            
            with tab1:
                st.write("#### Call Options")
                st.dataframe(calls.style.apply(
                    lambda x: ['background-color: #90EE90' if x['inTheMoney'] else '' for i in x],
                    axis=1
                ))
                
            with tab2:
                st.write("#### Put Options")
                st.dataframe(puts.style.apply(
                    lambda x: ['background-color: #90EE90' if x['inTheMoney'] else '' for i in x],
                    axis=1
                ))
            
            # Display risk analysis with better formatting
            st.markdown("### ⚠️ Risk Analysis")
            
            # Calculate days to nearest expiration
            exp_date = pd.to_datetime(expiration)
            days_to_exp = (exp_date - pd.Timestamp.now()).days
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.metric("Days to Expiration", days_to_exp,
                    delta="Short-term" if days_to_exp < 30 else "Long-term",
                    delta_color="off")
                st.metric("Implied Volatility Rank", 
                    f"{percentileofscore(data['Close'].pct_change().rolling(30).std(), volatility):.0f}%")
                
            with risk_col2:
                st.metric("Break-even Price", f"${current_price + calls['impliedVolatility'].mean():.2f}")
                st.metric("Max Loss", "Limited" if strategy['risk_level'] != "High" else "Unlimited",
                    delta=strategy['risk_level'],
                    delta_color="inverse")
            
            # Display analyst letter with better formatting
            st.markdown("### 📝 Options Strategy Analysis")
            with st.expander("Click to read detailed options strategy analysis"):
                letter = get_options_analyst_letter(ticker, current_price, rsi, volatility, strategy, calls, puts, expiration)
                if letter:
                    st.markdown(letter)
                
    except Exception as e:
        st.error(f"Error in options analysis tab: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
