import numpy as np
import pandas as pd
import streamlit as st


def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    try:
        # Convert data to Series if it's not already
        if isinstance(data, pd.DataFrame):
            close_prices = data['Close']
        else:
            close_prices = pd.Series(data)
            
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains (up) and losses (down)
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        avg_losses = losses.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        
        # Calculate relative strength
        rs = avg_gains / avg_losses
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        if isinstance(data, pd.DataFrame):
            return pd.Series(index=data.index)
        return pd.Series()


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        if isinstance(data, pd.DataFrame):
            close_prices = data['Close']
        else:
            close_prices = pd.Series(data)
            
        # Calculate EMAs
        exp1 = close_prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd = exp1 - exp2
        
        # Calculate signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        hist = macd - signal
        
        return macd.iloc[-signal_period:], signal.iloc[-signal_period:], hist.iloc[-signal_period:]
        
    except Exception as e:
        st.error(f"Error calculating MACD: {str(e)}")
        if isinstance(data, pd.DataFrame):
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
        return pd.Series(), pd.Series(), pd.Series()


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': macd - signal_line
        }, index=prices.index)
        
    except Exception as e:
        st.error(f"Error calculating MACD: {str(e)}")
        return pd.DataFrame({
            'MACD': pd.Series(index=prices.index),
            'Signal': pd.Series(index=prices.index),
            'Histogram': pd.Series(index=prices.index)
        })
