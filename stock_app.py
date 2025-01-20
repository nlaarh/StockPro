import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time
import traceback
from llm_predictions import LLMPredictor

# Function to predict stock prices
def predict_stock_price(ticker, prediction_days=30, model_type='linear', llm_model=None):
    """Predict stock prices using various models including LLMs"""
    try:
        # Get stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, None, "No data available for this stock"
            
        print(f"\nData shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Prepare data - use only Close prices
        df = data['Close'].copy()
        df = df.sort_index()  # Ensure data is sorted by date
        df = df.fillna(method='ffill').fillna(method='bfill')  # Handle any missing values
        
        # Check if we have enough data
        min_required_points = prediction_days + 60  # Need more points for training
        if len(df) < min_required_points:
            return None, None, f"Insufficient data points. Need at least {min_required_points} points, got {len(df)} points."
            
        print(f"Number of data points: {len(df)}")
        
        # Convert to numpy array for easier processing
        prices = df.values
        
        # Create sequences of previous prices to predict next price
        X = []
        y = []
        sequence_length = min(30, len(prices) // 4)  # Use shorter sequences for shorter datasets
        
        for i in range(len(prices) - sequence_length):
            X.append(prices[i:(i + sequence_length)])
            y.append(prices[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        
        if len(X) < min_required_points:
            return None, None, f"Not enough data points after sequence creation. Need at least {min_required_points}, got {len(X)}."
        
        # Split the data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, sequence_length))
        X_test_scaled = scaler.transform(X_test.reshape(-1, sequence_length))
        
        # Reshape back to 3D
        X_train_scaled = X_train_scaled.reshape(-1, sequence_length, 1)
        X_test_scaled = X_test_scaled.reshape(-1, sequence_length, 1)
        
        # Choose and train model
        if model_type == 'linear':
            model = LinearRegression()
            # Reshape for linear regression
            X_train_2d = X_train_scaled.reshape(-1, sequence_length)
            X_test_2d = X_test_scaled.reshape(-1, sequence_length)
            model.fit(X_train_2d, y_train)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            # Reshape for random forest
            X_train_2d = X_train_scaled.reshape(-1, sequence_length)
            X_test_2d = X_test_scaled.reshape(-1, sequence_length)
            model.fit(X_train_2d, y_train)
        else:  # SVR
            model = SVR(kernel='rbf', C=1000.0, gamma=0.1)
            # Reshape for SVR
            X_train_2d = X_train_scaled.reshape(-1, sequence_length)
            X_test_2d = X_test_scaled.reshape(-1, sequence_length)
            model.fit(X_train_2d, y_train)
        
        # Make future predictions
        last_sequence = prices[-sequence_length:]
        predictions = []
        
        for _ in range(prediction_days):
            # Scale the last sequence
            last_sequence_scaled = scaler.transform(last_sequence.reshape(1, -1))
            
            # Reshape based on model type
            if model_type in ['linear', 'random_forest', 'svr']:
                next_pred = model.predict(last_sequence_scaled)[0]
            
            predictions.append(next_pred)
            
            # Update the sequence
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred
        
        # Create future dates for prediction
        future_dates = pd.date_range(start=df.index[-1], periods=prediction_days + 1)[1:]
        
        # Create visualization
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.values,
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Add prediction
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{ticker} Stock Price Prediction',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            showlegend=True,
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig, float(np.mean(predictions)), "Prediction based on historical price patterns"
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return None, None, f"Error in prediction: {str(e)}"

# Function to calculate Buffett metrics
def calculate_buffett_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price and target prices
        current_price = info.get('currentPrice', 0)
        target_mean_price = info.get('targetMeanPrice', 0)
        target_low_price = info.get('targetLowPrice', 0)
        target_high_price = info.get('targetHighPrice', 0)
        
        # Calculate valuation metrics
        pe_ratio = info.get('forwardPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        # Calculate different valuation methods
        dcf_value = current_price * 1.1  # Simplified DCF
        graham_value = (info.get('bookValue', 0) * pe_ratio) ** 0.5  # Graham's Square Root Formula
        pe_based_value = info.get('forwardEps', 0) * 15  # Conservative PE-based valuation
        
        # Basic metrics
        metrics = {
            'Entry Price Analysis': {
                'Current Price': current_price,
                'Target Price': target_mean_price,
                'Entry Points': {
                    'Strong Buy Below': target_low_price,
                    'Buy Below': target_mean_price * 0.9 if target_mean_price else current_price * 0.9,
                    'Hold Above': target_high_price
                },
                'Valuation Methods': {
                    'DCF Value': round(dcf_value, 2) if dcf_value else 0,
                    'Graham Value': round(graham_value, 2) if graham_value else 0,
                    'PE-Based Value': round(pe_based_value, 2) if pe_based_value else 0,
                    'Analyst Target': round(target_mean_price, 2) if target_mean_price else 0
                }
            },
            'Business Understanding': {
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Business Model': info.get('longBusinessSummary', 'N/A')
            },
            'Competitive Advantage': {
                'Market Cap': info.get('marketCap', 0),
                'Market Position': info.get('industry', 'N/A') + ' Industry',
                'Brand Value': 'Strong' if info.get('marketCap', 0) > 10e9 else 'Medium'
            },
            'Management Quality': {
                'Insider Ownership': str(info.get('heldPercentInsiders', 0) * 100) + '%',
                'Institutional Ownership': str(info.get('heldPercentInstitutions', 0) * 100) + '%',
                'ROE': str(round(info.get('returnOnEquity', 0) * 100, 2)) + '%' if info.get('returnOnEquity') else 'N/A'
            },
            'Financial Health': {
                'Debt to Equity': round(info.get('debtToEquity', 0), 2),
                'Current Ratio': round(info.get('currentRatio', 0), 2),
                'Profit Margin': str(round(info.get('profitMargins', 0) * 100, 2)) + '%' if info.get('profitMargins') else 'N/A'
            },
            'Value Metrics': {
                'P/E Ratio': round(pe_ratio, 2) if pe_ratio else 'N/A',
                'P/B Ratio': round(pb_ratio, 2) if pb_ratio else 'N/A',
                'Dividend Yield': str(round(dividend_yield, 2)) + '%' if dividend_yield else 'N/A'
            }
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating Buffett metrics: {str(e)}")
        return None

def plot_daily_candlestick(ticker):
    """Plot daily candlestick chart"""
    stock = yf.Ticker(ticker)
    df = stock.history(period='1y')
    
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Stock Price (USD)',
        xaxis_title='Date'
    )
    
    return fig

def plot_technical_analysis(data, ticker, indicators):
    """Plot technical analysis indicators"""
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='OHLC'))
    
    # Add Moving Averages
    if 'Moving Averages' in indicators:
        ma_types = data.attrs.get('ma_types', ["MA20", "MA50"])
        for ma_type in ma_types:
            period = int(ma_type[2:])  # Extract number from MA20, MA50, etc.
            data[ma_type] = data['Close'].rolling(window=period).mean()
            fig.add_trace(go.Scatter(x=data.index, y=data[ma_type], name=ma_type))
    
    # Add Bollinger Bands
    if 'Bollinger Bands' in indicators:
        period = data.attrs.get('bb_period', 20)
        std_dev = data.attrs.get('bb_std', 2)
        
        data['MA'] = data['Close'].rolling(window=period).mean()
        data['STD'] = data['Close'].rolling(window=period).std()
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MA'] + (data['STD'] * std_dev), 
                               name=f'Upper Band ({std_dev}σ)', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA'], 
                               name=f'Middle Band (SMA{period})', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA'] - (data['STD'] * std_dev), 
                               name=f'Lower Band ({std_dev}σ)', line=dict(dash='dash')))
    
    # Add MACD
    if 'MACD' in indicators:
        fast_period = data.attrs.get('macd_fast', 12)
        slow_period = data.attrs.get('macd_slow', 26)
        signal_period = data.attrs.get('macd_signal', 9)
        
        data['EMA_fast'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Create a secondary y-axis for MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                               name=f'MACD ({fast_period},{slow_period})', yaxis='y2'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], 
                               name=f'Signal ({signal_period})', yaxis='y2'))
    
    # Add RSI
    if 'RSI' in indicators:
        period = data.attrs.get('rsi_period', 14)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Create a secondary y-axis for RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                               name=f'RSI ({period})', yaxis='y2'))
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yaxis='y2')
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yaxis='y2')
    
    # Add Stochastic Oscillator
    if 'Stochastic' in indicators:
        k_period = data.attrs.get('stoch_k', 14)
        d_period = data.attrs.get('stoch_d', 3)
        
        data['L14'] = data['Low'].rolling(window=k_period).min()
        data['H14'] = data['High'].rolling(window=k_period).max()
        data['%K'] = (data['Close'] - data['L14']) / (data['H14'] - data['L14']) * 100
        data['%D'] = data['%K'].rolling(window=d_period).mean()
        
        # Create a secondary y-axis for Stochastic
        fig.add_trace(go.Scatter(x=data.index, y=data['%K'], 
                               name=f'%K ({k_period})', yaxis='y2'))
        fig.add_trace(go.Scatter(x=data.index, y=data['%D'], 
                               name=f'%D ({d_period})', yaxis='y2'))
        
        # Add Stochastic reference lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, yaxis='y2')
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, yaxis='y2')
    
    # Update layout based on indicator type
    if any(ind in indicators for ind in ['MACD', 'RSI', 'Stochastic']):
        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            yaxis=dict(title='Price', domain=[0.3, 1.0]),
            yaxis2=dict(title='Indicator', domain=[0, 0.25]),
            xaxis_title='Date',
            height=800  # Make the chart taller to accommodate both plots
        )
    else:
        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            yaxis_title='Price',
            xaxis_title='Date',
            height=600
        )
    
    return fig

def plot_stock_history(ticker, period='1y'):
    """Plot historical stock data"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price'))
    
    fig.update_layout(
        title=f'{ticker} Historical Price Data',
        yaxis_title='Stock Price (USD)',
        xaxis_title='Date'
    )
    
    return fig

def get_buffett_recommendation(metrics):
    """Generate investment recommendation based on Warren Buffett's principles"""
    if not metrics:
        return "Unable to generate recommendation due to insufficient data."
    
    points = 0
    reasons = []
    
    # 1. Competitive Advantage
    try:
        gross_margin = float(metrics['Competitive Advantage']['Gross Margin (%)'].strip('%'))
        if gross_margin > 40:
            points += 2
            reasons.append("Strong gross margin indicates competitive advantage")
        
        roe = float(metrics['Competitive Advantage']['Return on Equity (%)'].strip('%'))
        if roe > 15:
            points += 2
            reasons.append("High return on equity shows efficient capital use")
    except:
        pass
    
    # 2. Management Quality
    try:
        insider_ownership = float(metrics['Management Quality']['Insider Ownership (%)'].strip('%'))
        if insider_ownership > 5:
            points += 1
            reasons.append("Significant insider ownership aligns management interests")
        
        debt_to_equity = float(metrics['Management Quality']['Debt to Equity'])
        if debt_to_equity < 0.5:
            points += 1
            reasons.append("Conservative debt management")
    except:
        pass
    
    # 3. Financial Health
    try:
        if 'Error' not in metrics['Financial Health']:
            fcf = metrics['Financial Health']['Free Cash Flow']
            if float(fcf.strip('$').strip('B')) > 0:
                points += 2
                reasons.append("Positive free cash flow generation")
    except:
        pass
    
    # 4. Value Metrics
    try:
        margin_of_safety = float(metrics['Value Metrics']['Margin of Safety (%)'].strip('%'))
        if margin_of_safety > 20:
            points += 3
            reasons.append(f"Large margin of safety ({margin_of_safety:.1f}%)")
        
        forward_pe = float(metrics['Value Metrics']['Forward P/E'])
        if forward_pe < 20:
            points += 1
            reasons.append("Reasonable forward P/E ratio")
        
        peg = float(metrics['Value Metrics']['PEG Ratio'])
        if peg < 1.5:
            points += 1
            reasons.append("Attractive PEG ratio indicates fair price for growth")
    except:
        pass
    
    # Generate recommendation
    if points >= 10:
        recommendation = "Strong Buy"
        color = "success"
    elif points >= 7:
        recommendation = "Buy"
        color = "success"
    elif points >= 5:
        recommendation = "Hold"
        color = "warning"
    else:
        recommendation = "Not Recommended"
        color = "error"
    
    return recommendation, color, reasons

def main():
    st.title("Stock Analysis App")
    
    # Add a text input for the stock symbol
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)", "AAPL").upper()
    
    if ticker:
        try:
            # Create tabs for different charts
            chart_tabs = st.tabs(["Daily Chart", "Technical Analysis", "Historical", "Price Prediction", "Buffett Analysis"])
            
            # Daily Chart Tab
            with chart_tabs[0]:
                st.plotly_chart(plot_daily_candlestick(ticker), use_container_width=True)
            
            # Technical Analysis Tab
            with chart_tabs[1]:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Technical Analysis Method Selector
                    analysis_method = st.selectbox(
                        "Select Analysis Method",
                        ["Moving Averages", "Bollinger Bands", "MACD", "RSI", "Stochastic"]
                    )
                    
                    # Parameters for each method
                    if analysis_method == "Moving Averages":
                        ma_types = st.multiselect(
                            "Select MA Types",
                            ["MA20", "MA50", "MA200"],
                            default=["MA20", "MA50"]
                        )
                    
                    elif analysis_method == "Bollinger Bands":
                        bb_period = st.slider("Period", 5, 50, 20)
                        bb_std = st.slider("Standard Deviation", 1, 4, 2)
                    
                    elif analysis_method == "MACD":
                        macd_fast = st.slider("Fast Period", 5, 20, 12)
                        macd_slow = st.slider("Slow Period", 15, 40, 26)
                        macd_signal = st.slider("Signal Period", 5, 15, 9)
                    
                    elif analysis_method == "RSI":
                        rsi_period = st.slider("RSI Period", 5, 30, 14)
                    
                    elif analysis_method == "Stochastic":
                        stoch_k = st.slider("%K Period", 5, 30, 14)
                        stoch_d = st.slider("%D Period", 2, 10, 3)
                
                with col2:
                    # Add button to apply analysis
                    if st.button("Apply Analysis"):
                        stock = yf.Ticker(ticker)
                        hist_data = stock.history(period='1y')
                        
                        # Create a set with only the selected method
                        selected_indicators = {analysis_method}
                        
                        # Add any additional parameters to the data object
                        if analysis_method == "Moving Averages":
                            hist_data.attrs['ma_types'] = ma_types
                        elif analysis_method == "Bollinger Bands":
                            hist_data.attrs['bb_period'] = bb_period
                            hist_data.attrs['bb_std'] = bb_std
                        elif analysis_method == "MACD":
                            hist_data.attrs['macd_fast'] = macd_fast
                            hist_data.attrs['macd_slow'] = macd_slow
                            hist_data.attrs['macd_signal'] = macd_signal
                        elif analysis_method == "RSI":
                            hist_data.attrs['rsi_period'] = rsi_period
                        elif analysis_method == "Stochastic":
                            hist_data.attrs['stoch_k'] = stoch_k
                            hist_data.attrs['stoch_d'] = stoch_d
                        
                        st.plotly_chart(plot_technical_analysis(hist_data, ticker, selected_indicators), use_container_width=True)
                    
                    # Add explanation for the selected method
                    st.markdown("### Analysis Method Explanation")
                    if analysis_method == "Moving Averages":
                        st.info("""
                        Moving averages help identify trends by smoothing out price data. 
                        - MA20: Short-term trend
                        - MA50: Medium-term trend
                        - MA200: Long-term trend
                        """)
                    elif analysis_method == "Bollinger Bands":
                        st.info("""
                        Bollinger Bands show volatility and potential overbought/oversold conditions.
                        - Upper Band: Mean + (STD × 2)
                        - Middle Band: Simple Moving Average
                        - Lower Band: Mean - (STD × 2)
                        """)
                    elif analysis_method == "MACD":
                        st.info("""
                        MACD (Moving Average Convergence Divergence) shows momentum and trend direction.
                        - MACD Line: Difference between fast and slow EMAs
                        - Signal Line: EMA of MACD
                        - Histogram: MACD - Signal
                        """)
                    elif analysis_method == "RSI":
                        st.info("""
                        RSI (Relative Strength Index) indicates overbought/oversold conditions.
                        - Above 70: Potentially overbought
                        - Below 30: Potentially oversold
                        - 50: Neutral
                        """)
                    elif analysis_method == "Stochastic":
                        st.info("""
                        Stochastic Oscillator shows momentum and trend reversal points.
                        - %K: Fast stochastic indicator
                        - %D: Slow stochastic indicator
                        - Above 80: Overbought
                        - Below 20: Oversold
                        """)
            
            # Historical Chart Tab
            with chart_tabs[2]:
                period = st.select_slider("Select Time Period", 
                                        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                                        value='1y')
                st.plotly_chart(plot_stock_history(ticker, period), use_container_width=True)
            
            # Price Prediction Tab
            with chart_tabs[3]:
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    prediction_days = st.slider("Select number of days to predict", 7, 60, 30)
                
                with col2:
                    model_type = st.selectbox(
                        "Select Prediction Model",
                        options=['linear', 'random_forest', 'svr']
                    )
                
                with col3:
                    predict_button = st.button("Generate Prediction")
                
                if predict_button:
                    with st.spinner("Generating prediction..."):
                        pred_fig, pred_value, explanation = predict_stock_price(
                            ticker, 
                            prediction_days, 
                            model_type
                        )
                        
                        if pred_fig is not None and pred_value is not None:
                            st.plotly_chart(pred_fig, use_container_width=True)
                            st.info(f"Predicted price in {prediction_days} days: ${pred_value:.2f}")
                        else:
                            st.error(f"Failed to generate prediction: {explanation}")
            
            # Warren Buffett Analysis Tab
            with chart_tabs[4]:
                st.subheader("Warren Buffett Investment Analysis")
                st.write("Analyzing stock based on Warren Buffett's investment principles")
                
                # Calculate Buffett metrics
                metrics = calculate_buffett_metrics(ticker)
                
                if metrics and metrics.get('Entry Price Analysis'):
                    entry_analysis = metrics['Entry Price Analysis']
                    
                    # Display Entry Price Analysis first
                    st.markdown("### 💰 Entry Price Analysis")
                    
                    # Create a price comparison table
                    price_data = {
                        "Current Price": [entry_analysis['Current Price']],
                        "Target Price": [entry_analysis['Target Price']],
                        "Strong Buy Below": [entry_analysis['Entry Points']['Strong Buy Below']],
                        "Buy Below": [entry_analysis['Entry Points']['Buy Below']],
                        "Hold Above": [entry_analysis['Entry Points']['Hold Above']]
                    }
                    df_prices = pd.DataFrame(price_data)
                    st.dataframe(df_prices.style.highlight_min(axis=1, color='lightgreen')
                                                .highlight_max(axis=1, color='lightcoral'),
                               use_container_width=True)
                    
                    # Show detailed valuation methods
                    st.markdown("#### 📊 Valuation Methods")
                    methods_data = pd.DataFrame({
                        "Method": list(entry_analysis['Valuation Methods'].keys()),
                        "Target Price": list(entry_analysis['Valuation Methods'].values())
                    })
                    st.dataframe(methods_data, use_container_width=True)
                    
                    # Add explanation
                    st.info("""
                    **Entry Price Guidelines:**
                    - **Strong Buy:** Excellent entry point with maximum margin of safety
                    - **Buy:** Good entry point with reasonable margin of safety
                    - **Hold:** Fair value, consider holding if already owned
                    - **Above Hold:** Consider taking profits or reducing position
                    """)
                    
                    # Display recommendation
                    recommendation, color, reasons = get_buffett_recommendation(metrics)
                    st.markdown(f"### Overall Recommendation: :{color}[{recommendation}]")
                    
                    # Display reasons
                    st.markdown("### Key Findings:")
                    for reason in reasons:
                        st.write(f"- {reason}")
                    
                    # Display detailed metrics in expandable sections
                    st.markdown("### 📈 Detailed Analysis")
                    
                    # 1. Business Understanding
                    with st.expander("🏢 Business Understanding"):
                        st.write(metrics['Business Understanding'])
                        if 'Business Model' in metrics['Business Understanding']:
                            st.markdown("### Business Model")
                            st.write(metrics['Business Understanding']['Business Model'])
                    
                    # 2. Competitive Advantage (Moat)
                    with st.expander("🏰 Competitive Advantage (Moat)"):
                        for key, value in metrics['Competitive Advantage'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # 3. Management Quality
                    with st.expander("👥 Management Quality"):
                        for key, value in metrics['Management Quality'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # 4. Financial Health
                    with st.expander("💰 Financial Health"):
                        for key, value in metrics['Financial Health'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # 5. Value Metrics
                    with st.expander("📊 Value Metrics"):
                        for key, value in metrics['Value Metrics'].items():
                            st.write(f"**{key}:** {value}")
                    
                    # Additional Notes
                    st.markdown("### 📝 Investment Notes")
                    st.info("""
                    Warren Buffett's Investment Principles:
                    1. Understand the business
                    2. Look for companies with strong competitive advantages
                    3. Focus on management quality and integrity
                    4. Ensure strong financial health
                    5. Buy at a reasonable price with a margin of safety
                    """)
                else:
                    st.error("Unable to perform Buffett analysis. Insufficient data available.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
