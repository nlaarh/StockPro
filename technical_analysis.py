"""Technical analysis module for StockPro"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import calculate_rsi, calculate_macd, calculate_bollinger_bands

def get_available_indicators() -> dict:
    """Get available technical indicators with their parameters"""
    return {
        "Moving Averages": {
            "periods": [20, 50, 200],
            "description": "Simple Moving Averages (SMA) for trend identification",
            "overlay": True  # Overlays on main price chart
        },
        "RSI": {
            "periods": [14],
            "description": "Relative Strength Index for overbought/oversold conditions",
            "overlay": False  # Separate panel
        },
        "MACD": {
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "description": "Moving Average Convergence Divergence for trend momentum",
            "overlay": False
        },
        "Bollinger Bands": {
            "period": 20,
            "std_dev": 2,
            "description": "Volatility bands based on standard deviation",
            "overlay": True
        },
        "Volume": {
            "description": "Trading volume with moving average",
            "overlay": False
        },
        "Stochastic": {
            "k_period": 14,
            "d_period": 3,
            "description": "Stochastic oscillator for momentum",
            "overlay": False
        },
        "ATR": {
            "period": 14,
            "description": "Average True Range for volatility",
            "overlay": False
        }
    }

def calculate_indicator(data: pd.DataFrame, indicator: str, params: dict) -> pd.DataFrame:
    """Calculate technical indicator values"""
    df = data.copy()
    
    if indicator == "Moving Averages":
        for period in params["periods"]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    elif indicator == "RSI":
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params["periods"][0]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params["periods"][0]).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    elif indicator == "MACD":
        exp1 = df['Close'].ewm(span=params["fast"], adjust=False).mean()
        exp2 = df['Close'].ewm(span=params["slow"], adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=params["signal"], adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    elif indicator == "Bollinger Bands":
        sma = df['Close'].rolling(window=params["period"]).mean()
        std = df['Close'].rolling(window=params["period"]).std()
        df['BB_Upper'] = sma + (std * params["std_dev"])
        df['BB_Middle'] = sma
        df['BB_Lower'] = sma - (std * params["std_dev"])
    
    elif indicator == "Stochastic":
        low_min = df['Low'].rolling(window=params["k_period"]).min()
        high_max = df['High'].rolling(window=params["k_period"]).max()
        df['K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['D'] = df['K'].rolling(window=params["d_period"]).mean()
    
    elif indicator == "ATR":
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=params["period"]).mean()
    
    return df

def get_ai_interpretation(data: pd.DataFrame, selected_indicators: list, indicator_params: dict) -> str:
    """Get AI interpretation of technical indicators with clear action items"""
    analysis = []
    signals = {"bullish": 0, "bearish": 0, "neutral": 0}
    
    # Price Trend Analysis
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-20]
    price_change = ((current_price - prev_price) / prev_price) * 100
    trend = "upward" if price_change > 0 else "downward"
    
    analysis.append(f"**Current Price:** ${current_price:.2f}")
    analysis.append(f"**20-Day Price Change:** {price_change:.2f}% ({trend} trend)")
    
    # Analyze each indicator
    indicator_signals = []
    for indicator in selected_indicators:
        if indicator == "Moving Averages":
            ma_signals = []
            for period in indicator_params[indicator]["periods"]:
                sma = data[f'SMA_{period}'].iloc[-1]
                if current_price > sma:
                    ma_signals.append(f"Price above {period}-day SMA (${sma:.2f})")
                    signals["bullish"] += 1
                else:
                    ma_signals.append(f"Price below {period}-day SMA (${sma:.2f})")
                    signals["bearish"] += 1
            if ma_signals:
                analysis.append(f"**Moving Averages:**\n" + "\n".join(f"- {s}" for s in ma_signals))
        
        elif indicator == "RSI":
            rsi = data['RSI'].iloc[-1]
            rsi_text = f"RSI at {rsi:.2f}: "
            if rsi > 70:
                rsi_text += "Overbought territory"
                signals["bearish"] += 2
                indicator_signals.append(("RSI", "Strong Sell", "Overbought conditions suggest taking profits"))
            elif rsi < 30:
                rsi_text += "Oversold territory"
                signals["bullish"] += 2
                indicator_signals.append(("RSI", "Strong Buy", "Oversold conditions suggest potential reversal"))
            else:
                rsi_text += "Neutral territory"
                signals["neutral"] += 1
                indicator_signals.append(("RSI", "Hold", "Neutral momentum"))
            analysis.append(f"**RSI Analysis:**\n- {rsi_text}")
        
        elif indicator == "MACD":
            macd = data['MACD'].iloc[-1]
            signal = data['Signal'].iloc[-1]
            hist = data['MACD_Hist'].iloc[-1]
            prev_hist = data['MACD_Hist'].iloc[-2]
            
            macd_text = []
            if hist > 0 and hist > prev_hist:
                macd_text.append("Increasing bullish momentum")
                signals["bullish"] += 2
                indicator_signals.append(("MACD", "Buy", "Strong bullish momentum"))
            elif hist < 0 and hist < prev_hist:
                macd_text.append("Increasing bearish momentum")
                signals["bearish"] += 2
                indicator_signals.append(("MACD", "Sell", "Strong bearish momentum"))
            else:
                macd_text.append("Mixed momentum signals")
                signals["neutral"] += 1
                indicator_signals.append(("MACD", "Hold", "Mixed signals"))
            
            if macd > signal:
                macd_text.append("MACD above signal line")
                signals["bullish"] += 1
            else:
                macd_text.append("MACD below signal line")
                signals["bearish"] += 1
                
            analysis.append(f"**MACD Analysis:**\n" + "\n".join(f"- {s}" for s in macd_text))
        
        elif indicator == "Bollinger Bands":
            upper = data['BB_Upper'].iloc[-1]
            lower = data['BB_Lower'].iloc[-1]
            
            bb_text = []
            if current_price > upper:
                bb_text.append(f"Price (${current_price:.2f}) above upper band (${upper:.2f})")
                signals["bearish"] += 2
                indicator_signals.append(("Bollinger Bands", "Sell", "Price above upper band suggests overbought"))
            elif current_price < lower:
                bb_text.append(f"Price (${current_price:.2f}) below lower band (${lower:.2f})")
                signals["bullish"] += 2
                indicator_signals.append(("Bollinger Bands", "Buy", "Price below lower band suggests oversold"))
            else:
                bb_text.append(f"Price (${current_price:.2f}) within bands")
                signals["neutral"] += 1
                indicator_signals.append(("Bollinger Bands", "Hold", "Price within normal range"))
                
            analysis.append(f"**Bollinger Bands Analysis:**\n" + "\n".join(f"- {s}" for s in bb_text))
    
    # Calculate overall sentiment
    total_signals = sum(signals.values())
    if total_signals > 0:
        bullish_pct = (signals["bullish"] / total_signals) * 100
        bearish_pct = (signals["bearish"] / total_signals) * 100
        
        if bullish_pct > 60:
            action = "Strong Buy"
            confidence = "High" if bullish_pct > 80 else "Moderate"
            emoji = "🟢"
        elif bearish_pct > 60:
            action = "Strong Sell"
            confidence = "High" if bearish_pct > 80 else "Moderate"
            emoji = "🔴"
        elif bullish_pct > bearish_pct:
            action = "Buy"
            confidence = "Moderate"
            emoji = "🟡"
        elif bearish_pct > bullish_pct:
            action = "Sell"
            confidence = "Moderate"
            emoji = "🟡"
        else:
            action = "Hold"
            confidence = "Neutral"
            emoji = "⚪"
    
    # Combine all analysis
    full_analysis = "### Technical Analysis Summary\n\n"
    full_analysis += "\n\n".join(analysis)
    
    # Add Action Items section
    full_analysis += "\n\n### 🎯 Recommended Actions\n\n"
    full_analysis += f"**Overall Recommendation:** {emoji} {action} (Confidence: {confidence})\n\n"
    
    if indicator_signals:
        full_analysis += "**Indicator-Specific Actions:**\n"
        for ind, act, reason in indicator_signals:
            full_analysis += f"- {ind}: {act} - {reason}\n"
    
    # Add specific trade suggestions based on the action
    full_analysis += "\n**Trade Suggestions:**\n"
    if action in ["Strong Buy", "Buy"]:
        stop_loss = current_price * 0.95  # 5% below current price
        target = current_price * 1.1  # 10% above current price
        full_analysis += f"- Entry Point: Current price (${current_price:.2f})\n"
        full_analysis += f"- Stop Loss: ${stop_loss:.2f} (-5%)\n"
        full_analysis += f"- Target Price: ${target:.2f} (+10%)\n"
        full_analysis += "- Consider scaling in to reduce risk\n"
    elif action in ["Strong Sell", "Sell"]:
        full_analysis += f"- Consider taking profits at current price (${current_price:.2f})\n"
        full_analysis += "- Set trailing stop orders to protect gains\n"
        full_analysis += "- Consider scaling out of position\n"
    else:  # Hold
        full_analysis += "- Monitor current positions\n"
        full_analysis += "- Consider setting stop losses to protect gains\n"
        full_analysis += "- Wait for stronger signals before new positions\n"
    
    full_analysis += "\n*Note: This analysis is generated by AI based on technical indicators. Always conduct your own research and consider fundamental factors before making investment decisions.*"
    
    return full_analysis

def technical_analysis_tab():
    """Technical analysis tab showing various technical indicators"""
    if 'ticker' not in st.session_state:
        st.warning("Please enter a stock symbol in the main input field")
        return
        
    st.header("📊 Technical Analysis")
    
    # Get available indicators
    indicators = get_available_indicators()
    
    # Create two columns for the controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select for indicators
        selected_indicators = st.multiselect(
            "Select Technical Indicators to Display",
            options=list(indicators.keys()),
            default=["Moving Averages", "RSI"],
            help="Choose technical indicators to display on the chart"
        )
    
    with col2:
        # Time period selection
        timeframe = st.selectbox(
            "Select Timeframe",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Choose the time period for analysis"
        )
    
    # Show indicator descriptions
    if selected_indicators:
        with st.expander("📈 Indicator Descriptions", expanded=False):
            for indicator in selected_indicators:
                st.markdown(f"**{indicator}**: {indicators[indicator]['description']}")
    
    # Parameters for selected indicators in a cleaner format
    if selected_indicators:
        st.markdown("### 🔧 Indicator Parameters")
        param_cols = st.columns(len(selected_indicators))
        
        indicator_params = {}
        for idx, indicator in enumerate(selected_indicators):
            with param_cols[idx]:
                st.markdown(f"**{indicator}**")
                
                if indicator == "Moving Averages":
                    indicator_params[indicator] = {
                        "periods": st.multiselect(
                            "MA Periods",
                            options=[20, 50, 100, 200],
                            default=[20, 50],
                            key=f"ma_periods_{idx}"
                        )
                    }
                elif indicator == "Bollinger Bands":
                    indicator_params[indicator] = {
                        "period": st.slider("BB Period", 5, 50, 20, key=f"bb_period_{idx}"),
                        "std_dev": st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1, key=f"bb_std_{idx}")
                    }
                elif indicator == "MACD":
                    indicator_params[indicator] = {
                        "fast": st.slider("Fast Period", 5, 20, 12, key=f"macd_fast_{idx}"),
                        "slow": st.slider("Slow Period", 15, 40, 26, key=f"macd_slow_{idx}"),
                        "signal": st.slider("Signal Period", 5, 15, 9, key=f"macd_signal_{idx}")
                    }
                elif indicator == "RSI":
                    indicator_params[indicator] = {
                        "periods": [st.slider("RSI Period", 5, 30, 14, key=f"rsi_period_{idx}")]
                    }
                elif indicator == "Stochastic":
                    indicator_params[indicator] = {
                        "k_period": st.slider("K Period", 5, 30, 14, key=f"stoch_k_{idx}"),
                        "d_period": st.slider("D Period", 2, 10, 3, key=f"stoch_d_{idx}")
                    }
                elif indicator == "ATR":
                    indicator_params[indicator] = {
                        "period": st.slider("ATR Period", 5, 30, 14, key=f"atr_period_{idx}")
                    }
    
    # Get stock data
    try:
        if selected_indicators:
            with st.spinner(f"Fetching data for {st.session_state['ticker']}..."):
                stock = yf.Ticker(st.session_state['ticker'])
                data = stock.history(period=timeframe)
                
                if data.empty:
                    st.error(f"No data available for {st.session_state['ticker']}")
                    return
                
                # Calculate indicators
                for indicator in selected_indicators:
                    data = calculate_indicator(data, indicator, indicator_params[indicator])
                
                # Create subplots based on selected indicators
                non_overlay_count = sum(1 for ind in selected_indicators if not indicators[ind]["overlay"])
                fig = make_subplots(
                    rows=1 + non_overlay_count,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.5] + [0.5/non_overlay_count]*non_overlay_count if non_overlay_count > 0 else [1]
                )
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add indicators
                current_row = 2
                for indicator in selected_indicators:
                    if indicator == "Moving Averages":
                        for period in indicator_params[indicator]["periods"]:
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=data[f'SMA_{period}'],
                                    name=f'SMA {period}',
                                    line=dict(width=1)
                                ),
                                row=1, col=1
                            )
                    
                    elif indicator == "RSI":
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['RSI'],
                                name='RSI'
                            ),
                            row=current_row, col=1
                        )
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                        current_row += 1
                    
                    elif indicator == "MACD":
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['MACD'], name='MACD'),
                            row=current_row, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['Signal'], name='Signal'),
                            row=current_row, col=1
                        )
                        fig.add_trace(
                            go.Bar(x=data.index, y=data['MACD_Hist'], name='MACD Histogram'),
                            row=current_row, col=1
                        )
                        current_row += 1
                    
                    elif indicator == "Bollinger Bands":
                        for band in ['Upper', 'Middle', 'Lower']:
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=data[f'BB_{band}'],
                                    name=f'BB {band}',
                                    line=dict(dash='dash' if band != 'Middle' else 'solid')
                                ),
                                row=1, col=1
                            )
                    
                    elif indicator == "Volume":
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['Volume'],
                                name='Volume'
                            ),
                            row=current_row, col=1
                        )
                        current_row += 1
                    
                    elif indicator == "Stochastic":
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['K'], name='%K'),
                            row=current_row, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['D'], name='%D'),
                            row=current_row, col=1
                        )
                        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
                        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
                        current_row += 1
                    
                    elif indicator == "ATR":
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['ATR'], name='ATR'),
                            row=current_row, col=1
                        )
                        current_row += 1
                
                # Update layout
                fig.update_layout(
                    title=f"{st.session_state['ticker']} - Technical Analysis ({timeframe})",
                    xaxis_title="Date",
                    height=200 + 200*len(selected_indicators),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Get and display AI interpretation
                interpretation = get_ai_interpretation(data, selected_indicators, indicator_params)
                st.markdown(interpretation)
        else:
            st.info("👆 Please select at least one technical indicator above to begin analysis")
        
    except Exception as e:
        st.error(f"Error generating technical analysis: {str(e)}")
        st.write("Debug: Exception traceback:", traceback.format_exc())

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
