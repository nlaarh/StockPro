import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import calculate_rsi, calculate_macd

def technical_analysis_tab(ticker, data):
    """Technical analysis tab with customizable indicators"""
    try:
        st.subheader(f"Technical Analysis for {ticker}")
        
        # Add indicator selection
        indicators = st.multiselect(
            "Select Technical Indicators",
            ["Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume"],
            default=["Moving Averages", "RSI"]
        )
        
        # Create figure with multiple subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          row_heights=[0.6, 0.2, 0.2],
                          subplot_titles=('Price', 'RSI', 'MACD'))
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Add selected indicators
        if "Moving Averages" in indicators:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA20'],
                name='20-day SMA',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA50'],
                name='50-day SMA',
                line=dict(color='blue', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA200'],
                name='200-day SMA',
                line=dict(color='purple', width=1)
            ), row=1, col=1)
        
        if "RSI" in indicators:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple', width=1)
            ), row=2, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        if "MACD" in indicators:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='blue', width=1)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Signal'],
                name='Signal',
                line=dict(color='orange', width=1)
            ), row=3, col=1)
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Hist'],
                name='MACD Histogram',
                marker_color='gray'
            ), row=3, col=1)
        
        if "Volume" in indicators:
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='lightgray'
            ), row=3, col=1)
            
        if "Bollinger Bands" in indicators:
            # Calculate Bollinger Bands
            period = 20
            std_dev = 2
            
            data['BB_middle'] = data['Close'].rolling(window=period).mean()
            data['BB_upper'] = data['BB_middle'] + (data['Close'].rolling(window=period).std() * std_dev)
            data['BB_lower'] = data['BB_middle'] - (data['Close'].rolling(window=period).std() * std_dev)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                name='Upper BB',
                line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_middle'],
                name='Middle BB',
                line=dict(color='gray', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                name='Lower BB',
                line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            yaxis_title='Price (USD)',
            yaxis2_title='RSI',
            yaxis3_title='MACD/Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD/Volume", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display current indicator values
        st.subheader("Current Technical Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("20-day SMA", f"${data['SMA20'].iloc[-1]:.2f}")
            st.metric("50-day SMA", f"${data['SMA50'].iloc[-1]:.2f}")
            
        with col2:
            st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
            st.metric("200-day SMA", f"${data['SMA200'].iloc[-1]:.2f}")
            
        with col3:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.3f}")
            st.metric("Signal", f"{data['Signal'].iloc[-1]:.3f}")
        
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

def plot_daily_candlestick(data):
    """Plot daily candlestick chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                       row_width=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='OHLC'),
                 row=1, col=1)

    fig.add_trace(go.Bar(x=data.index,
                        y=data['Volume'],
                        name='Volume'),
                 row=2, col=1)

    fig.update_layout(
        title='Daily Trading Activity',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

def plot_stock_history(data, indicators=None):
    """Plot stock history with selected technical indicators"""
    if indicators is None:
        indicators = ['SMA20', 'SMA50', 'RSI', 'MACD']
        
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.6, 0.2, 0.2],
                       subplot_titles=('Price', 'RSI', 'MACD'))

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ), row=1, col=1)

    # Add moving averages
    if 'SMA20' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA20'],
            name='20 SMA',
            line=dict(color='orange')
        ), row=1, col=1)

    if 'SMA50' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA50'],
            name='50 SMA',
            line=dict(color='blue')
        ), row=1, col=1)

    # Add RSI
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Add MACD
    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Signal'],
            name='Signal',
            line=dict(color='orange')
        ), row=3, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['MACD_Hist'],
            name='MACD Histogram',
            marker_color='gray'
        ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Technical Analysis',
        yaxis_title='Price (USD)',
        yaxis2_title='RSI',
        yaxis3_title='MACD',
        xaxis_rangeslider_visible=False,
        height=800
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig
