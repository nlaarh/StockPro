"""Utility functions for StockPro"""
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    try:
        # Check if data is a DataFrame or Series
        if isinstance(data, pd.DataFrame):
            close_data = data['Close']
        else:
            close_data = data
            
        # Calculate RSI
        close_delta = close_data.diff()
        
        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        
        # Calculate the EWMA
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        
        return rsi
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=data.index)

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        # Check if data is a DataFrame or Series
        if isinstance(data, pd.DataFrame):
            close_data = data['Close']
        else:
            close_data = data
            
        # Calculate the MACD line
        exp1 = close_data.ewm(span=fast_period, adjust=False).mean()
        exp2 = close_data.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        
        # Calculate the signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate the histogram
        hist = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        })
    except Exception as e:
        st.error(f"Error calculating MACD: {str(e)}")
        return pd.DataFrame(index=data.index)

def calculate_macd_new(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        # Calculate the fast and slow EMAs
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd = exp1 - exp2
        
        # Calculate the signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate the histogram
        hist = macd - signal
        
        return macd, signal, hist
    except:
        return pd.Series(), pd.Series(), pd.Series()

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    try:
        # Check if data is a DataFrame or Series
        if isinstance(data, pd.DataFrame):
            close_data = data['Close']
        else:
            close_data = data
            
        # Calculate middle band (simple moving average)
        middle_band = close_data.rolling(window=window).mean()
        
        # Calculate standard deviation
        std = close_data.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return pd.DataFrame({
            'Middle': middle_band,
            'Upper': upper_band,
            'Lower': lower_band
        })
    except Exception as e:
        st.error(f"Error calculating Bollinger Bands: {str(e)}")
        return pd.DataFrame(index=data.index)

def safe_divide(a, b, default=0):
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if b == 0:
            return default
        return a / b
    except:
        return default

def format_large_number(num):
    """Format large numbers into K, M, B format"""
    try:
        num = float(num)
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.1f}"
    except:
        return "N/A"

def format_percentage(value):
    """Format a number as a percentage"""
    try:
        return f"{value:.1f}%" if value is not None else "N/A"
    except:
        return "N/A"

def calculate_delta(S, K, T, r, sigma, is_call=True):
    """Calculate option Delta"""
    try:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if is_call:
            return norm.cdf(d1)
        return norm.cdf(d1) - 1
    except:
        return 0

def calculate_gamma(S, K, T, r, sigma):
    """Calculate option Gamma"""
    try:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except:
        return 0

def calculate_theta(S, K, T, r, sigma, is_call=True):
    """Calculate option Theta"""
    try:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        return theta / 365  # Convert to daily theta
    except:
        return 0

def calculate_vega(S, K, T, r, sigma):
    """Calculate option Vega"""
    try:
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1) / 100  # Divide by 100 for percentage move
    except:
        return 0

def calculate_growth_rate(data, periods=5):
    """Calculate compound annual growth rate"""
    try:
        if len(data) < periods:
            return 0
        
        start_value = data.iloc[-periods]
        end_value = data.iloc[0]
        
        if start_value <= 0:
            return 0
        
        # Calculate CAGR
        cagr = (end_value / start_value) ** (1/periods) - 1
        return cagr * 100
    except:
        return 0

def calculate_intrinsic_value(fcf, growth_rate, discount_rate=0.10, terminal_growth=0.03, years=10):
    """Calculate intrinsic value using DCF model"""
    try:
        if fcf <= 0:
            return 0
            
        value = 0
        
        # Calculate present value of future cash flows
        for year in range(1, years + 1):
            future_fcf = fcf * (1 + growth_rate/100) ** year
            value += future_fcf / (1 + discount_rate) ** year
        
        # Add terminal value
        terminal_fcf = fcf * (1 + growth_rate/100) ** years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        value += terminal_value / (1 + discount_rate) ** years
        
        # Add a safety check for unreasonable values
        if value <= 0 or value > fcf * 100:  # Cap at 100x FCF
            return fcf * 15  # Use simple 15x FCF multiple as fallback
            
        return value
        
    except Exception as e:
        logger.warning(f"Error calculating intrinsic value: {str(e)}")
        return 0
