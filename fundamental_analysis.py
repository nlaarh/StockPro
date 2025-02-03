import streamlit as st
import yfinance as yf
import pandas as pd
from utils import (
    format_large_number, calculate_growth_rate,
    calculate_intrinsic_value, safe_divide
)

def fundamental_analysis_tab():
    """Display fundamental analysis for a stock"""
    if not st.session_state.get('ticker'):
        st.warning("Please enter a stock symbol above")
        return
        
    try:
        ticker = yf.Ticker(st.session_state['ticker'])
        info = ticker.info
        
        st.subheader("Fundamental Analysis")
        
        # Financial Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pe_ratio = info.get('forwardPE', 0)
            st.metric("Forward P/E", f"{pe_ratio:.2f}")
            
            pb_ratio = info.get('priceToBook', 0)
            st.metric("Price/Book", f"{pb_ratio:.2f}")
            
            ps_ratio = info.get('priceToSalesTrailing12Months', 0)
            st.metric("Price/Sales", f"{ps_ratio:.2f}")
            
        with col2:
            profit_margin = info.get('profitMargins', 0) * 100
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            roe = info.get('returnOnEquity', 0) * 100
            st.metric("ROE", f"{roe:.1f}%")
            
            roa = info.get('returnOnAssets', 0) * 100
            st.metric("ROA", f"{roa:.1f}%")
            
        with col3:
            dividend_yield = info.get('dividendYield', 0) * 100
            st.metric("Dividend Yield", f"{dividend_yield:.2f}%")
            
            payout_ratio = info.get('payoutRatio', 0) * 100
            st.metric("Payout Ratio", f"{payout_ratio:.1f}%")
            
            beta = info.get('beta', 0)
            st.metric("Beta", f"{beta:.2f}")
            
        # Growth Metrics
        st.subheader("Growth Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_growth = info.get('revenueGrowth', 0) * 100
            st.metric("Revenue Growth", f"{revenue_growth:.1f}%")
            
            earnings_growth = info.get('earningsGrowth', 0) * 100
            st.metric("Earnings Growth", f"{earnings_growth:.1f}%")
            
        with col2:
            eps_growth = info.get('earningsQuarterlyGrowth', 0) * 100
            st.metric("EPS Growth", f"{eps_growth:.1f}%")
            
            free_cash_flow = info.get('freeCashflow', 0)
            st.metric("Free Cash Flow", format_large_number(free_cash_flow))
            
        # Valuation Analysis
        st.subheader("Valuation Analysis")
        
        # Calculate free cash flow per share
        fcf_per_share = safe_divide(
            info.get('freeCashflow', 0),
            info.get('sharesOutstanding', 1)
        )
        
        intrinsic_value = calculate_intrinsic_value(
            fcf=fcf_per_share,
            growth_rate=earnings_growth/100,
            discount_rate=0.10,  # 10% discount rate
            terminal_growth=0.03,  # 3% terminal growth
            years=5
        )
        
        current_price = info.get('currentPrice', 0)
        margin_of_safety = safe_divide(
            (intrinsic_value - current_price),
            intrinsic_value
        ) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
            
        with col2:
            st.metric("Intrinsic Value", f"${intrinsic_value:.2f}")
            
        with col3:
            st.metric("Margin of Safety", f"{margin_of_safety:.1f}%")
            
    except Exception as e:
        st.error(f"Error analyzing fundamentals: {str(e)}")
