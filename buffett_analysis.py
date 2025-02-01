import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

def safe_divide(a, b, default=0):
    """Safely divide two numbers"""
    try:
        return float(a) / float(b) if float(b) != 0 else default
    except (TypeError, ValueError):
        return default

def plot_price_targets(current_price):
    """Create a visual representation of Buffett's price targets"""
    price_ranges = {
        'Strong Buy': [0, current_price * 0.7],
        'Buy': [current_price * 0.7, current_price * 0.85],
        'Hold': [current_price * 0.85, current_price * 1.15],
        'Sell': [current_price * 1.15, current_price * 1.3],
        'Strong Sell': [current_price * 1.3, current_price * 1.5]
    }
    
    colors = {
        'Strong Buy': 'darkgreen',
        'Buy': 'green',
        'Hold': 'yellow',
        'Sell': 'orange',
        'Strong Sell': 'red'
    }
    
    fig = go.Figure()
    
    for rating, (min_price, max_price) in price_ranges.items():
        fig.add_trace(go.Bar(
            name=rating,
            x=[rating],
            y=[max_price - min_price],
            base=[min_price],
            marker_color=colors[rating],
            text=f"${min_price:.2f} - ${max_price:.2f}",
            textposition='auto',
            hovertemplate=f"{rating}<br>Range: ${min_price:.2f} - ${max_price:.2f}<extra></extra>"
        ))
    
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Current Price: ${current_price:.2f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Warren Buffett's Investment Price Targets",
        yaxis_title="Stock Price ($)",
        showlegend=False,
        height=400,
        template="plotly_dark",
        bargap=0.3
    )
    
    return fig

def get_buffett_metrics(ticker):
    """Calculate Warren Buffett style metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
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
        
        return metrics, info
        
    except Exception as e:
        logger.error(f"Error getting Buffett metrics: {str(e)}")
        return None, None

def get_buffett_writeup(ticker, metrics):
    """Generate a Warren Buffett style investment writeup"""
    try:
        writeup = []
        
        # Profitability Analysis
        if metrics['roe'] > 15 and metrics['operating_margin'] > 20:
            writeup.append(" Strong profitability metrics indicate a durable competitive advantage")
        elif metrics['roe'] > 12:
            writeup.append(" Decent profitability but room for improvement")
        else:
            writeup.append(" Weak profitability raises concerns about competitive position")
            
        # Financial Health
        if metrics['debt_to_equity'] < 50 and metrics['current_ratio'] > 1.5:
            writeup.append(" Conservative balance sheet with manageable debt levels")
        elif metrics['debt_to_equity'] < 100:
            writeup.append(" Moderate financial leverage requires monitoring")
        else:
            writeup.append(" High debt levels increase financial risk")
            
        # Growth Assessment
        if metrics['revenue_growth'] > 10 and metrics['earnings_growth'] > 10:
            writeup.append(" Strong and sustainable growth in both revenue and earnings")
        elif metrics['revenue_growth'] > 5:
            writeup.append(" Moderate growth - need to assess sustainability")
        else:
            writeup.append(" Limited growth potential")
            
        # Valuation
        if metrics['pe_ratio'] < 15 and metrics['fcf_yield'] > 5:
            writeup.append(" Attractive valuation with good cash flow generation")
        elif metrics['pe_ratio'] < 20:
            writeup.append(" Fair valuation but limited margin of safety")
        else:
            writeup.append(" Rich valuation leaves little room for error")
        
        return "\n\n".join(writeup)
        
    except Exception as e:
        logger.error(f"Error generating Buffett writeup: {str(e)}")
        return None

def get_buffett_letter(ticker, metrics, info):
    """Generate a Warren Buffett style shareholder letter"""
    try:
        # Determine company's strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if metrics['roe'] > 15:
            strengths.append(f"strong return on equity of {metrics['roe']:.1f}%")
        else:
            weaknesses.append(f"concerning return on equity of {metrics['roe']:.1f}%")
            
        if metrics['operating_margin'] > 20:
            strengths.append(f"impressive operating margin of {metrics['operating_margin']:.1f}%")
        else:
            weaknesses.append(f"operating margin of {metrics['operating_margin']:.1f}% needs improvement")
            
        if metrics['debt_to_equity'] < 50:
            strengths.append(f"conservative debt-to-equity ratio of {metrics['debt_to_equity']:.1f}")
        else:
            weaknesses.append(f"high debt-to-equity ratio of {metrics['debt_to_equity']:.1f}")
            
        if metrics['fcf_yield'] > 5:
            strengths.append(f"attractive free cash flow yield of {metrics['fcf_yield']:.1f}%")
        else:
            weaknesses.append(f"low free cash flow yield of {metrics['fcf_yield']:.1f}%")
            
        # Forward-looking assessment
        growth_outlook = "promising" if metrics['revenue_growth'] > 10 and metrics['earnings_growth'] > 10 else "challenging"
        valuation_status = "attractive" if metrics['pe_ratio'] < 15 else "rich"
        
        letter = f"""
        Dear Fellow Shareholders of {ticker},

        I want to share my thoughts on our investment in {info.get('shortName', ticker)}. As you know, 
        we at Berkshire look for businesses with durable competitive advantages, strong management, and 
        attractive financials at reasonable prices.

        BUSINESS ANALYSIS:
        {info.get('shortName', ticker)} has demonstrated {', '.join(strengths) if strengths else 'few notable strengths'}. 
        However, we must also acknowledge {', '.join(weaknesses) if weaknesses else 'no significant weaknesses'}.

        COMPETITIVE POSITION:
        The company's operating metrics suggest {'a strong' if len(strengths) > len(weaknesses) else 'a concerning'} competitive position 
        in its market. {'The combination of high returns on equity and strong margins indicates pricing power and efficient operations.' 
        if metrics['roe'] > 15 and metrics['operating_margin'] > 20 
        else 'There is room for improvement in operational efficiency and market positioning.'}

        FINANCIAL HEALTH:
        {'The conservative balance sheet provides flexibility for future opportunities.' 
        if metrics['debt_to_equity'] < 50 
        else 'The current debt levels require careful monitoring and management attention.'}
        
        VALUATION AND FUTURE PROSPECTS:
        Looking ahead, we see {growth_outlook} growth prospects, with revenue growing at {metrics['revenue_growth']:.1f}% 
        and earnings at {metrics['earnings_growth']:.1f}%. The current valuation appears {valuation_status}, 
        trading at {metrics['pe_ratio']:.1f}x earnings with a {metrics['fcf_yield']:.1f}% free cash flow yield.

        INVESTMENT OUTLOOK:
        {'We believe the company is well-positioned to generate sustainable returns for shareholders.' 
        if len(strengths) > len(weaknesses) 
        else 'While the business has potential, several areas need improvement before it meets our strict investment criteria.'}
        
        {'The current market price provides an attractive entry point for long-term investors.' 
        if metrics['pe_ratio'] < 15 and metrics['fcf_yield'] > 5 
        else 'We would become more interested at a more conservative valuation level.'}

        Sincerely,
        Warren Buffett
        """
        
        return letter
        
    except Exception as e:
        logger.error(f"Error generating Buffett letter: {str(e)}")
        return None

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
                # Get metrics and info
                metrics, info = get_buffett_metrics(ticker)
                if not metrics or not info:
                    st.error("Could not calculate Buffett metrics. Please try again later.")
                    return
                
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
                    'Buffett\'s Assessment': [
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
                
                # 2. Buy/Sell Price Recommendations with Visual Chart
                st.subheader(" Investment Price Targets")
                
                # Display the price targets chart
                fig = plot_price_targets(metrics['current_price'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed price targets table
                targets_df = pd.DataFrame({
                    'Rating': ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'],
                    'Price Range': [
                        f"Below ${metrics['current_price'] * 0.7:.2f}",
                        f"${metrics['current_price'] * 0.7:.2f} - ${metrics['current_price'] * 0.85:.2f}",
                        f"${metrics['current_price'] * 0.85:.2f} - ${metrics['current_price'] * 1.15:.2f}",
                        f"${metrics['current_price'] * 1.15:.2f} - ${metrics['current_price'] * 1.3:.2f}",
                        f"Above ${metrics['current_price'] * 1.3:.2f}"
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
                
                # 3. Key Investment Takeaways
                st.subheader(" Key Investment Takeaways")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Strengths**")
                    if metrics['roe'] > 15:
                        st.markdown(" Strong return on equity")
                    if metrics['operating_margin'] > 20:
                        st.markdown(" Excellent operating margins")
                    if metrics['debt_to_equity'] < 50:
                        st.markdown(" Conservative debt levels")
                    if metrics['fcf_yield'] > 5:
                        st.markdown(" Attractive free cash flow yield")
                
                with col2:
                    st.markdown("**Areas of Concern**")
                    if metrics['roe'] <= 15:
                        st.markdown(" Below-target ROE")
                    if metrics['operating_margin'] <= 20:
                        st.markdown(" Margin improvement needed")
                    if metrics['debt_to_equity'] >= 50:
                        st.markdown(" High leverage")
                    if metrics['fcf_yield'] <= 5:
                        st.markdown(" Low cash flow yield")
                
                # 4. Warren Buffett's Letter to Shareholders (Expandable)
                st.subheader(" Warren Buffett's Letter to Shareholders")
                with st.expander("Click to read Warren Buffett's detailed analysis"):
                    letter = get_buffett_letter(ticker, metrics, info)
                    if letter:
                        st.markdown(letter)
                
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")
                logger.error(f"Error in Buffett analysis: {str(e)}")
                
    except Exception as e:
        st.error(f"Error in Buffett analysis tab: {str(e)}")
        logger.error(f"Error in Buffett analysis tab: {str(e)}")
