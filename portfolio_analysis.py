import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from utils import format_large_number

def portfolio_analysis_tab():
    """Display portfolio analysis tools"""
    if not st.session_state.get('ticker'):
        st.warning("Please enter a stock symbol above")
        return
        
    try:
        ticker = yf.Ticker(st.session_state['ticker'])
        info = ticker.info
        
        st.subheader("Portfolio Analysis")
        
        # Portfolio Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            beta = info.get('beta', 0)
            st.metric("Beta", f"{beta:.2f}")
            
            volatility = info.get('52WeekChange', 0) * 100
            st.metric("52-Week Volatility", f"{volatility:.1f}%")
            
        with col2:
            alpha = info.get('threeYearAverageReturn', 0) * 100
            st.metric("3-Year Alpha", f"{alpha:.1f}%")
            
            sharpe = info.get('fiveYearAverageReturn', 0)
            st.metric("5-Year Sharpe", f"{sharpe:.2f}")
            
        with col3:
            volume = info.get('averageVolume', 0)
            st.metric("Avg Volume", format_large_number(volume))
            
            market_cap = info.get('marketCap', 0)
            st.metric("Market Cap", format_large_number(market_cap))
            
        # Risk Analysis
        st.subheader("Risk Analysis")
        
        risk_score = min(100, max(0, (
            (beta * 20) +  # Beta weight
            (abs(volatility) * 0.5) +  # Volatility weight
            (100 - alpha)  # Alpha weight (inverse)
        ) / 3))  # Average of components
        
        st.progress(risk_score/100)
        st.write(f"Risk Score: {risk_score:.1f}/100")
        
        # Risk Factors
        factors = []
        
        if beta > 1.5:
            factors.append("High market sensitivity")
        elif beta < 0.5:
            factors.append("Low market correlation")
            
        if volatility > 30:
            factors.append("High price volatility")
        elif volatility < 10:
            factors.append("Low price volatility")
            
        if alpha < 0:
            factors.append("Underperforming benchmark")
        elif alpha > 20:
            factors.append("Strong outperformance")
            
        if market_cap < 2e9:  # $2B
            factors.append("Small-cap stock")
        elif market_cap > 200e9:  # $200B
            factors.append("Mega-cap stock")
            
        if volume < 500000:
            factors.append("Low trading liquidity")
            
        st.write("### Risk Factors")
        for factor in factors:
            st.write(f"- {factor}")
            
        # Portfolio Fit
        st.subheader("Portfolio Fit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Diversification Impact")
            if beta < 0.5:
                st.write("Good hedge against market downturns")
            elif beta > 1.5:
                st.write("Amplifies market movements")
            else:
                st.write("Moderate market correlation")
                
        with col2:
            st.write("### Position Sizing")
            if risk_score > 75:
                st.write("Consider small position size (<2%)")
            elif risk_score > 50:
                st.write("Moderate position size (2-5%)")
            else:
                st.write("Can handle larger position (>5%)")
                
    except Exception as e:
        st.error(f"Error analyzing portfolio metrics: {str(e)}")
