"""Technical analysis module for StockPro"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import calculate_rsi, calculate_macd, calculate_bollinger_bands

def technical_analysis_tab():
    """Technical analysis tab showing various technical indicators"""
    try:
        st.header("📊 Technical Analysis")
        
        # Get ticker from session state
        ticker = st.session_state.get('current_ticker', '')
        if not ticker:
            st.warning("Please enter a stock ticker above.")
            return
            
        # Fetch stock data
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
                
            # Calculate indicators
            rsi = calculate_rsi(data)
            macd_data = calculate_macd(data)
            bb_data = calculate_bollinger_bands(data)
            
            # Create tabs for different indicators
            tab1, tab2, tab3 = st.tabs(["Price & Volume", "RSI & MACD", "Bollinger Bands"])
            
            with tab1:
                st.subheader("Price & Volume Analysis")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                # Volume chart
                fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    yaxis='y2'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} Price and Volume",
                    yaxis_title="Price ($)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right"
                    ),
                    xaxis_title="Date",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                st.subheader("RSI & MACD")
                
                # RSI Plot
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=rsi,
                    name='RSI'
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                
                fig_rsi.update_layout(
                    title=f"{ticker} RSI (14)",
                    yaxis_title="RSI",
                    xaxis_title="Date",
                    height=300
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD Plot
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=macd_data['MACD'],
                    name='MACD'
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=macd_data['Signal'],
                    name='Signal'
                ))
                fig_macd.add_trace(go.Bar(
                    x=data.index,
                    y=macd_data['Histogram'],
                    name='Histogram'
                ))
                
                fig_macd.update_layout(
                    title=f"{ticker} MACD",
                    yaxis_title="MACD",
                    xaxis_title="Date",
                    height=300
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
                
            with tab3:
                st.subheader("Bollinger Bands")
                
                fig_bb = go.Figure()
                
                # Add price
                fig_bb.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name='Price',
                    line=dict(color='blue')
                ))
                
                # Add Bollinger Bands
                fig_bb.add_trace(go.Scatter(
                    x=data.index,
                    y=bb_data['Upper'],
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=data.index,
                    y=bb_data['Lower'],
                    name='Lower Band',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))
                
                fig_bb.update_layout(
                    title=f"{ticker} Bollinger Bands",
                    yaxis_title="Price ($)",
                    xaxis_title="Date",
                    height=600
                )
                
                st.plotly_chart(fig_bb, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

def plot_daily_candlestick(ticker, data=None):
    """Plot daily candlestick chart"""
    try:
        if data is None:
            # Fetch data if not provided
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
        
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        fig.update_layout(
            title=f"{ticker} Daily Price",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error plotting candlestick: {str(e)}")
        return None

def plot_stock_history(ticker, period='1y'):
    """Plot historical stock data with multiple timeframes"""
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.error(f"No data available for {ticker}")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=ticker
        ))
        
        # Add volume bars
        colors = ['red' if row['Open'] - row['Close'] >= 0 
                 else 'green' for index, row in hist.iterrows()]
        
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color=colors,
            yaxis="y2"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Historical Data",
            yaxis_title="Price",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right"
            ),
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error plotting stock history: {str(e)}")
        return None
