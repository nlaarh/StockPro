"""Warren Buffett style analysis for StockPro"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import traceback
from utils import format_large_number
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_price_targets(analysis, current_price):
    """Calculate buy, sell, and hold price targets based on current price"""
    try:
        # Simple percentage-based targets from current price
        targets = {
            'Buy': current_price * 0.85,        # 15% below current price
            'Hold_Low': current_price * 0.95,   # 5% below current price
            'Hold_High': current_price * 1.05,  # 5% above current price
            'Sell': current_price * 1.15,       # 15% above current price
            'Stop_Loss': current_price * 0.85   # 15% stop loss
        }
        
        return targets
    except Exception as e:
        st.error(f"Error calculating price targets: {str(e)}")
        return None

def get_buffett_metrics_interpretation(metric_name, value):
    """Get interpretation of Warren Buffett metrics"""
    interpretations = {
        'ROE': {
            'ideal': 15,
            'ranges': [
                (20, 'Excellent - Company generates strong returns on shareholder equity'),
                (15, 'Good - Solid returns on equity, meeting Buffett\'s criteria'),
                (10, 'Fair - Moderate returns, may need improvement'),
                (0, 'Poor - Below Buffett\'s minimum requirements')
            ]
        },
        'Debt/Equity': {
            'ideal': 0.5,
            'ranges': [
                (0.3, 'Excellent - Very conservative debt level'),
                (0.5, 'Good - Manageable debt level'),
                (1.0, 'Fair - Higher debt, but still acceptable'),
                (float('inf'), 'Poor - High debt level, increased risk')
            ]
        },
        'Operating Margin': {
            'ideal': 20,
            'ranges': [
                (20, 'Excellent - Strong pricing power and efficiency'),
                (15, 'Good - Healthy operating efficiency'),
                (10, 'Fair - Moderate operating efficiency'),
                (0, 'Poor - Weak operating efficiency')
            ]
        },
        'P/E Ratio': {
            'ideal': 15,
            'ranges': [
                (10, 'Excellent - Potentially undervalued'),
                (15, 'Good - Reasonably priced'),
                (20, 'Fair - Somewhat expensive'),
                (float('inf'), 'Poor - Expensive by Buffett standards')
            ]
        },
        'Profit Margin': {
            'ideal': 20,
            'ranges': [
                (20, 'Excellent - Strong profitability'),
                (15, 'Good - Healthy profit margins'),
                (10, 'Fair - Moderate profitability'),
                (0, 'Poor - Weak profitability')
            ]
        }
    }
    
    if metric_name not in interpretations:
        return None, None
    
    metric_info = interpretations[metric_name]
    ideal = metric_info['ideal']
    
    for threshold, interpretation in metric_info['ranges']:
        if value <= threshold:
            return ideal, interpretation
            
    return ideal, metric_info['ranges'][-1][1]

def plot_buffett_metrics_comparison(metrics):
    """Create a comparison chart of actual vs ideal metrics"""
    try:
        # Prepare data for plotting
        categories = list(metrics.keys())
        actual_values = [metrics[cat]['actual'] for cat in categories]
        ideal_values = [metrics[cat]['ideal'] for cat in categories]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=categories, y=actual_values,
                  marker_color='rgb(55, 83, 109)'),
            go.Bar(name='Ideal', x=categories, y=ideal_values,
                  marker_color='rgb(26, 118, 255)')
        ])
        
        # Update layout
        fig.update_layout(
            title='Buffett Metrics: Actual vs Ideal',
            xaxis_title='Metrics',
            yaxis_title='Value (%)',
            barmode='group',
            height=400
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating metrics comparison chart: {str(e)}")
        return None

def generate_buffett_letter(info, current_price, targets):
    """Generate Warren Buffett style analysis letter"""
    try:
        company_name = info.get('longName', info.get('shortName', 'the company'))
        sector = info.get('sector', 'its sector')
        industry = info.get('industry', 'its industry')
        
        # Get key metrics with proper formatting
        market_cap = format_large_number(info.get('marketCap', 0))
        revenue = format_large_number(info.get('totalRevenue', 0))
        profit_margin = info.get('profitMargins', 0) * 100
        roe = info.get('returnOnEquity', 0) * 100
        operating_margin = info.get('operatingMargins', 0) * 100
        pe_ratio = info.get('trailingPE', 0)
        debt_equity = info.get('debtToEquity', 0)
        dividend_yield = info.get('dividendYield', 0) * 100
        
        # Determine company's moat strength
        moat_strength = (
            "wide and enduring" if operating_margin > 20 and roe > 15 else
            "moderate but defensible" if operating_margin > 15 and roe > 10 else
            "narrow or uncertain"
        )
        
        # Determine management effectiveness
        management_effectiveness = (
            "exceptional" if roe > 20 and profit_margin > 20 else
            "competent" if roe > 15 and profit_margin > 15 else
            "adequate" if roe > 10 and profit_margin > 10 else
            "concerning"
        )
        
        # Determine financial position
        financial_position = (
            "fortress-like" if debt_equity < 0.3 and operating_margin > 20 else
            "strong" if debt_equity < 0.5 and operating_margin > 15 else
            "adequate" if debt_equity < 1.0 and operating_margin > 10 else
            "concerning"
        )
        
        letter = f"""
Dear Fellow Shareholders,

As I sit down to write this letter about {company_name}, I'm reminded of a lesson I learned from my mentor, Ben Graham, 
many years ago. He taught me that investing isn't about finding what's popular or trendy – it's about finding businesses 
that create real value and purchasing them at sensible prices. Today, I want to share my thoughts on one such business 
that has caught our attention.

The Business and Its Moat
------------------------
{company_name}, operating in the {industry} segment of the {sector} sector, presents an interesting case study in 
competitive dynamics. With {revenue} in annual revenue and a market capitalization of {market_cap}, it has certainly 
achieved meaningful scale. But as you know, at Berkshire, we're far more interested in the quality of the business 
than its size.

What particularly interests me about {company_name} is its {moat_strength} economic moat. The company's 
{operating_margin:.1f}% operating margin {"is truly exceptional and suggests significant pricing power" if operating_margin > 20 
else "indicates a healthy competitive position" if operating_margin > 15 
else "suggests some competitive challenges"}. In my experience, sustained high margins like {"these" if operating_margin > 20 
else "this"} often indicate the presence of {"strong" if operating_margin > 20 else "some"} competitive advantages.

Management and Capital Allocation
-------------------------------
Charlie and I have always emphasized the importance of honest and capable management. At {company_name}, the 
leadership team has demonstrated {management_effectiveness} capabilities in capital allocation, as evidenced by 
their {roe:.1f}% return on equity. {
"This is the kind of performance that makes us sit up and take notice." if roe > 20 else
"While not extraordinary, this represents solid stewardship of shareholder capital." if roe > 15 else
"This leaves some room for improvement." if roe > 10 else
"This is an area that requires significant improvement."
}

The company's financial position is {financial_position}, with a debt-to-equity ratio of {debt_equity:.2f}. {
"This conservative financial structure gives us great comfort." if debt_equity < 0.3 else
"This represents a prudent balance between leverage and financial flexibility." if debt_equity < 0.5 else
"While manageable, we would prefer to see less leverage." if debt_equity < 1.0 else
"This level of leverage leaves the company vulnerable to economic downturns."
}

Valuation and Our Approach
-------------------------
At Berkshire, we believe that time is the friend of the wonderful business and the enemy of the mediocre. 
With {company_name} trading at ${current_price:.2f} per share, representing {pe_ratio:.1f} times earnings, 
the market is {"undervaluing" if pe_ratio < 15 else "fairly valuing" if pe_ratio < 20 else "generously valuing"} 
this business.

Our analysis suggests several key price points for consideration:
• We would be enthusiastic buyers below ${targets['Buy']:.2f}
• The range of ${targets['Hold_Low']:.2f} to ${targets['Hold_High']:.2f} represents fair value
• Above ${targets['Sell']:.2f}, we would carefully review our position

{company_name} {f'provides a {dividend_yield:.1f}% dividend yield, which ' if dividend_yield > 0 else ''}{
f'represents a meaningful return of capital to shareholders' if dividend_yield > 3 else
f'provides a modest but growing stream of income' if dividend_yield > 0 else
'currently retains all earnings for reinvestment in the business'}.

Our Verdict
----------
{
"I am particularly excited about this opportunity. The combination of a wide moat, excellent management, and attractive valuation is rare. " 
if moat_strength == "wide and enduring" and management_effectiveness == "exceptional" and pe_ratio < 15 else

"While the business demonstrates quality characteristics, patience may be rewarded by waiting for a more attractive entry point. " 
if moat_strength in ["wide and enduring", "moderate but defensible"] and management_effectiveness in ["exceptional", "competent"] and pe_ratio >= 20 else

"The current situation requires careful monitoring. While there are positive attributes, we need to see improvements in key areas before considering a significant investment."
}

Remember what I've always said: it's far better to buy a wonderful company at a fair price than a fair company 
at a wonderful price. {company_name} appears to be {
"a wonderful company at an attractive price" if moat_strength == "wide and enduring" and pe_ratio < 15 else
"a quality business at a full price" if moat_strength in ["wide and enduring", "moderate but defensible"] and pe_ratio >= 20 else
"a company with potential that needs to prove itself further"
}.

As always, our favorite holding period is forever – but this assumes the fundamentals remain strong and the price 
paid is reasonable. We will continue to monitor {company_name}'s progress with great interest.

Sincerely,
Warren Buffett

P.S. Remember, as Charlie would say, "All I want to know is where I'm going to die, so I'll never go there." 
In investing, this means avoiding businesses with fundamental problems while embracing those with enduring strengths.
"""
        return letter
    except Exception as e:
        logger.error(f"Error generating Buffett letter: {str(e)}")
        return None

def get_price_recommendation(current_price, targets, metrics):
    """Generate price recommendation based on Buffett's principles"""
    try:
        # Calculate overall quality score
        roe_score = min(100, max(0, metrics['ROE']['actual'] * 5))  # ROE weight: 5x
        margin_score = min(100, max(0, metrics['Operating Margin']['actual'] * 4))  # Operating Margin weight: 4x
        debt_score = min(100, max(0, (1 - metrics['Debt/Equity']['actual']) * 100))  # Lower debt is better
        pe_score = min(100, max(0, (20 - metrics['P/E Ratio']['actual']) * 5))  # Lower P/E is better
        
        quality_score = (roe_score * 0.4 + margin_score * 0.3 + debt_score * 0.2 + pe_score * 0.1)
        
        # Determine recommendation
        if current_price <= targets['Buy']:
            action = "Strong Buy" if quality_score >= 70 else "Buy"
            color = "green"
        elif current_price <= targets['Hold_Low']:
            action = "Accumulate"
            color = "lightgreen"
        elif current_price <= targets['Hold_High']:
            action = "Hold"
            color = "orange"
        elif current_price <= targets['Sell']:
            action = "Reduce"
            color = "lightred"
        else:
            action = "Sell"
            color = "red"
            
        # Generate detailed recommendation
        if action in ["Strong Buy", "Buy"]:
            detail = f"""
            **Price Analysis**: Currently trading at ${current_price:.2f}, which is {'significantly ' if action == 'Strong Buy' else ''}below our estimated fair value.
            
            **Quality Score**: {quality_score:.0f}/100
            - Return on Equity is {metrics['ROE']['actual']:.1f}% ({roe_score:.0f}/100)
            - Operating Margin is {metrics['Operating Margin']['actual']:.1f}% ({margin_score:.0f}/100)
            - Debt/Equity Ratio is {metrics['Debt/Equity']['actual']:.2f} ({debt_score:.0f}/100)
            - P/E Ratio is {metrics['P/E Ratio']['actual']:.1f} ({pe_score:.0f}/100)
            
            **Price Targets**:
            - Current Price: ${current_price:.2f}
            - Buy Target: ${targets['Buy']:.2f}
            - Fair Value Range: ${targets['Hold_Low']:.2f} - ${targets['Hold_High']:.2f}
            - Sell Target: ${targets['Sell']:.2f}
            
            **Margin of Safety**: {((targets['Hold_High'] - current_price) / targets['Hold_High'] * 100):.1f}%
            """
        elif action == "Hold":
            detail = f"""
            **Price Analysis**: Currently trading at ${current_price:.2f}, which is within our estimated fair value range.
            
            **Quality Score**: {quality_score:.0f}/100
            - Return on Equity is {metrics['ROE']['actual']:.1f}% ({roe_score:.0f}/100)
            - Operating Margin is {metrics['Operating Margin']['actual']:.1f}% ({margin_score:.0f}/100)
            - Debt/Equity Ratio is {metrics['Debt/Equity']['actual']:.2f} ({debt_score:.0f}/100)
            - P/E Ratio is {metrics['P/E Ratio']['actual']:.1f} ({pe_score:.0f}/100)
            
            **Price Targets**:
            - Buy Target: ${targets['Buy']:.2f}
            - Current Price: ${current_price:.2f}
            - Fair Value Range: ${targets['Hold_Low']:.2f} - ${targets['Hold_High']:.2f}
            - Sell Target: ${targets['Sell']:.2f}
            """
        else:
            detail = f"""
            **Price Analysis**: Currently trading at ${current_price:.2f}, which is {'significantly ' if action == 'Sell' else ''}above our estimated fair value.
            
            **Quality Score**: {quality_score:.0f}/100
            - Return on Equity is {metrics['ROE']['actual']:.1f}% ({roe_score:.0f}/100)
            - Operating Margin is {metrics['Operating Margin']['actual']:.1f}% ({margin_score:.0f}/100)
            - Debt/Equity Ratio is {metrics['Debt/Equity']['actual']:.2f} ({debt_score:.0f}/100)
            - P/E Ratio is {metrics['P/E Ratio']['actual']:.1f} ({pe_score:.0f}/100)
            
            **Price Targets**:
            - Buy Target: ${targets['Buy']:.2f}
            - Fair Value Range: ${targets['Hold_Low']:.2f} - ${targets['Hold_High']:.2f}
            - Current Price: ${current_price:.2f}
            - Sell Target: ${targets['Sell']:.2f}
            """
            
        return {
            'action': action,
            'color': color,
            'detail': detail,
            'quality_score': quality_score
        }
    except Exception as e:
        logger.error(f"Error generating price recommendation: {str(e)}")
        return None

def buffett_analysis_tab():
    """Display Warren Buffett style analysis"""
    try:
        st.header("Warren Buffett Analysis")
        
        # Get current ticker from session state
        ticker = st.session_state.get('ticker', 'AAPL')
        
        # Get stock data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            current_price = info.get('currentPrice')
            if current_price is None:
                current_price = info.get('regularMarketPrice')
            if current_price is None:
                st.error(f"Could not get current price for {ticker}")
                return
            
            # Calculate simple price targets
            targets = calculate_price_targets(info, current_price)
            
            # Get metrics for recommendation
            metrics = {
                'ROE': {
                    'actual': info.get('returnOnEquity', 0) * 100,
                    'ideal': 15
                },
                'Debt/Equity': {
                    'actual': info.get('debtToEquity', 0),
                    'ideal': 0.5
                },
                'Operating Margin': {
                    'actual': info.get('operatingMargins', 0) * 100,
                    'ideal': 20
                },
                'P/E Ratio': {
                    'actual': info.get('trailingPE', 0),
                    'ideal': 15
                },
                'Profit Margin': {
                    'actual': info.get('profitMargins', 0) * 100,
                    'ideal': 20
                }
            }
            
            # Get price recommendation
            recommendation = get_price_recommendation(current_price, targets, metrics)
            if recommendation:
                st.subheader("Warren's Recommendation")
                st.markdown(f"### :dart: {recommendation['action']}")
                st.markdown(recommendation['detail'])
            
            # Display price targets
            st.subheader("Price Targets")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Buy Below", f"${targets['Buy']:.2f}")
                st.metric("Stop Loss", f"${targets['Stop_Loss']:.2f}")
            
            with col2:
                hold_range = f"${targets['Hold_Low']:.2f} - ${targets['Hold_High']:.2f}"
                st.metric("Hold Range", hold_range)
                st.metric("Sell Above", f"${targets['Sell']:.2f}")
            
            with col3:
                margin_of_safety = ((current_price - targets['Buy']) / current_price * 100)
                st.metric("Margin of Safety", f"{margin_of_safety:.1f}%")
                st.metric("Quality Score", f"{recommendation['quality_score']:.0f}/100")
            
            # Buffett Metrics Table
            st.subheader("Warren Buffett's Key Metrics")
            metrics = {
                'ROE': {
                    'actual': info.get('returnOnEquity', 0) * 100,
                    'ideal': 15
                },
                'Debt/Equity': {
                    'actual': info.get('debtToEquity', 0),
                    'ideal': 0.5
                },
                'Operating Margin': {
                    'actual': info.get('operatingMargins', 0) * 100,
                    'ideal': 20
                },
                'P/E Ratio': {
                    'actual': info.get('trailingPE', 0),
                    'ideal': 15
                },
                'Profit Margin': {
                    'actual': info.get('profitMargins', 0) * 100,
                    'ideal': 20
                }
            }
            
            # Create metrics table
            table_data = []
            for metric_name, values in metrics.items():
                actual = values['actual']
                ideal, interpretation = get_buffett_metrics_interpretation(metric_name, actual)
                table_data.append({
                    'Metric': metric_name,
                    'Actual': f"{actual:.1f}",
                    'Ideal': f"{ideal:.1f}",
                    'Interpretation': interpretation
                })
            
            df = pd.DataFrame(table_data)
            st.table(df)
            
            # Plot metrics comparison
            fig = plot_buffett_metrics_comparison(metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Company Metrics Section
            with st.expander("Company Metrics", expanded=False):
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.subheader("Financial Metrics")
                    st.metric("Market Cap", format_large_number(info.get('marketCap', 0)))
                    st.metric("Revenue", format_large_number(info.get('totalRevenue', 0)))
                    st.metric("Net Income", format_large_number(info.get('netIncome', 0)))
                    st.metric("Free Cash Flow", format_large_number(info.get('freeCashflow', 0)))
                
                with metrics_col2:
                    st.subheader("Per Share Metrics")
                    st.metric("EPS (TTM)", f"${info.get('trailingEps', 0):.2f}")
                    st.metric("Book Value/Share", f"${info.get('bookValue', 0):.2f}")
                    st.metric("Dividend/Share", f"${info.get('dividendRate', 0):.2f}")
                    st.metric("52 Week Range", f"${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}")
            
            # Key Ratios Section
            with st.expander("Key Ratios", expanded=False):
                ratios_col1, ratios_col2 = st.columns(2)
                
                with ratios_col1:
                    st.subheader("Profitability")
                    st.metric("Return on Equity", f"{info.get('returnOnEquity', 0) * 100:.1f}%")
                    st.metric("Operating Margin", f"{info.get('operatingMargins', 0) * 100:.1f}%")
                    st.metric("Profit Margin", f"{info.get('profitMargins', 0) * 100:.1f}%")
                
                with ratios_col2:
                    st.subheader("Valuation")
                    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}")
                    st.metric("Price/Book", f"{info.get('priceToBook', 0):.2f}")
                    st.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}")
            
            # Warren's Letter Section
            with st.expander("Warren's Letter to Shareholders", expanded=False):
                letter = generate_buffett_letter(info, current_price, targets)
                if letter:
                    st.markdown(letter)
                else:
                    st.error("Could not generate Warren's analysis letter")
        
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error fetching data for {ticker}. Please try again.")
            return
            
    except Exception as e:
        logger.error(f"Error in Buffett analysis tab: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again.")
