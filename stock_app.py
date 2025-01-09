import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from textblob import TextBlob
import nltk

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

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
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='MA200'))
    
    # Add Bollinger Bands
    if 'Bollinger Bands' in indicators:
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['20dSTD'] = data['Close'].rolling(window=20).std()
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'] + (data['20dSTD'] * 2), name='Upper Band'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'] - (data['20dSTD'] * 2), name='Lower Band'))
    
    # Add MACD
    if 'MACD' in indicators:
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Signal'))
    
    # Add RSI
    if 'RSI' in indicators:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
    
    # Add Stochastic Oscillator
    if 'Stochastic' in indicators:
        data['L14'] = data['Low'].rolling(window=14).min()
        data['H14'] = data['High'].rolling(window=14).max()
        data['%K'] = (data['Close'] - data['L14']) / (data['H14'] - data['L14']) * 100
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        fig.add_trace(go.Scatter(x=data.index, y=data['%K'], name='%K'))
        fig.add_trace(go.Scatter(x=data.index, y=data['%D'], name='%D'))
    
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title='Price',
        xaxis_title='Date'
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

def predict_stock_price(ticker, prediction_days=30, model_type='linear'):
    """Predict stock prices using various models"""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y')
        
        # Prepare data for models
        data = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Prepare training data
        X = []
        y = []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape data for models
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Select and train model
        if model_type == 'linear':
            model = LinearRegression()
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        elif model_type == 'svr':
            model = SVR(kernel='rbf')
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        model.fit(X_train, y_train)
        
        # Make predictions
        last_60_days = scaled_data[-60:]
        X_pred = []
        
        for i in range(prediction_days):
            X_pred = last_60_days[-60:].reshape(1, -1)
            pred = model.predict(X_pred)
            last_60_days = np.append(last_60_days, pred)
        
        # Scale back predictions
        predictions = scaler.inverse_transform(last_60_days[-prediction_days:].reshape(-1, 1))
        
        # Create dates for prediction
        last_date = df.index[-1]
        prediction_dates = pd.date_range(start=last_date, periods=prediction_days+1)[1:]
        
        # Create plot
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                               name='Actual Price',
                               line=dict(color='blue')))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(),
                               name='Predicted Price',
                               line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title=f'{ticker} Stock Price Prediction',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            showlegend=True
        )
        
        return fig, np.mean(predictions), None
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None

def calculate_buffett_metrics(ticker_symbol):
    """Calculate Warren Buffett's key investment metrics"""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        # 1. Business Understanding
        business_metrics = {
            'Company': info.get('longName', ticker_symbol),
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown'),
            'Business Model': info.get('longBusinessSummary', 'No description available')
        }
        
        # 2. Competitive Advantage (Moat)
        try:
            gross_margin = (info.get('grossMargins', 0) * 100)
            operating_margin = (info.get('operatingMargins', 0) * 100)
            net_margin = (info.get('profitMargins', 0) * 100)
            
            moat_metrics = {
                'Gross Margin (%)': f"{gross_margin:.2f}%",
                'Operating Margin (%)': f"{operating_margin:.2f}%",
                'Net Margin (%)': f"{net_margin:.2f}%",
                'Return on Equity (%)': f"{(info.get('returnOnEquity', 0) * 100):.2f}%",
                'Market Share': info.get('marketPosition', 'Not available')
            }
        except:
            moat_metrics = {'Error': 'Unable to calculate moat metrics'}
        
        # 3. Management Quality
        try:
            management_metrics = {
                'Insider Ownership (%)': f"{info.get('heldPercentInsiders', 0) * 100:.2f}%",
                'Institutional Ownership (%)': f"{info.get('heldPercentInstitutions', 0) * 100:.2f}%",
                'Debt to Equity': f"{info.get('debtToEquity', 0):.2f}",
                'Current Ratio': info.get('currentRatio', 0)
            }
        except:
            management_metrics = {'Error': 'Unable to calculate management metrics'}
        
        # 4. Financial Health
        try:
            # Get total cash and total debt from info
            total_cash = info.get('totalCash', 0)
            total_debt = info.get('totalDebt', 0)
            free_cash_flow = info.get('freeCashflow', 0)
            
            # Calculate ratios
            debt_to_cash = total_debt / total_cash if total_cash > 0 else float('inf')
            
            # Format values
            financial_health = {
                'Cash Position': f"${total_cash/1e9:.2f}B" if total_cash >= 1e9 else f"${total_cash/1e6:.2f}M",
                'Total Debt': f"${total_debt/1e9:.2f}B" if total_debt >= 1e9 else f"${total_debt/1e6:.2f}M",
                'Free Cash Flow': f"${free_cash_flow/1e9:.2f}B" if free_cash_flow >= 1e9 else f"${free_cash_flow/1e6:.2f}M",
                'Debt to Cash Ratio': f"{debt_to_cash:.2f}",
                'Current Ratio': f"{info.get('currentRatio', 0):.2f}",
                'Quick Ratio': f"{info.get('quickRatio', 0):.2f}"
            }
        except Exception as e:
            st.error(f"Financial Health Error: {str(e)}")
            financial_health = {'Error': 'Unable to calculate financial health metrics'}
        
        # 5. Value Metrics
        try:
            current_price = info.get('currentPrice', 0)
            book_value = info.get('bookValue', 0)
            forward_pe = info.get('forwardPE', 0)
            peg_ratio = info.get('pegRatio', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            intrinsic_value = calculate_intrinsic_value(stock)
            margin_of_safety = ((intrinsic_value - current_price) / intrinsic_value * 100) if intrinsic_value > 0 else 0
            
            value_metrics = {
                'Current Price': f"${current_price:.2f}",
                'Book Value': f"${book_value:.2f}",
                'Forward P/E': f"{forward_pe:.2f}",
                'PEG Ratio': f"{peg_ratio:.2f}",
                'Dividend Yield (%)': f"{dividend_yield:.2f}%",
                'Estimated Intrinsic Value': f"${intrinsic_value:.2f}",
                'Margin of Safety (%)': f"{margin_of_safety:.2f}%"
            }
        except:
            value_metrics = {'Error': 'Unable to calculate value metrics'}
        
        # Add entry price analysis
        entry_price_analysis = calculate_entry_price(stock, info, {
            'Value Metrics': value_metrics
        })
        
        return {
            'Business Understanding': business_metrics,
            'Competitive Advantage': moat_metrics,
            'Management Quality': management_metrics,
            'Financial Health': financial_health,
            'Value Metrics': value_metrics,
            'Entry Price Analysis': entry_price_analysis
        }
        
    except Exception as e:
        st.error(f"Error in Buffett analysis: {str(e)}")
        return None

def calculate_intrinsic_value(stock):
    """Calculate intrinsic value using Discounted Cash Flow (DCF) method"""
    try:
        # Get historical free cash flows
        cashflow = stock.cashflow
        if cashflow.empty:
            return 0
            
        fcf = cashflow.loc['Free Cash Flow']
        
        # Calculate average FCF growth rate
        fcf_growth_rates = fcf.pct_change()
        avg_fcf_growth = fcf_growth_rates.mean()
        
        # Use conservative growth rate (min of historical or 15%)
        growth_rate = min(avg_fcf_growth, 0.15)
        
        # Latest FCF as starting point
        latest_fcf = fcf.iloc[0]
        
        # Project future cash flows (10 years)
        future_fcf = []
        for i in range(1, 11):
            future_fcf.append(latest_fcf * (1 + growth_rate) ** i)
        
        # Terminal value (using conservative growth rate of 3%)
        terminal_growth_rate = 0.03
        terminal_value = future_fcf[-1] * (1 + terminal_growth_rate) / (0.09 - terminal_growth_rate)
        
        # Discount rate (using 9% as default)
        discount_rate = 0.09
        
        # Calculate present value of future cash flows
        present_value = 0
        for i, fcf in enumerate(future_fcf):
            present_value += fcf / (1 + discount_rate) ** (i + 1)
        
        # Add present value of terminal value
        present_value += terminal_value / (1 + discount_rate) ** 10
        
        # Get shares outstanding
        shares_outstanding = stock.info.get('sharesOutstanding', 0)
        
        if shares_outstanding > 0:
            intrinsic_value = present_value / shares_outstanding
            return intrinsic_value
        return 0
        
    except:
        return 0

def calculate_entry_price(stock, info, metrics):
    """Calculate the optimal entry price based on Warren Buffett's principles"""
    try:
        current_price = info.get('currentPrice', 0)
        
        # 1. Based on Intrinsic Value (DCF)
        intrinsic_value = float(metrics['Value Metrics']['Estimated Intrinsic Value'].replace('$', ''))
        dcf_target = intrinsic_value * 0.8  # 20% margin of safety
        
        # 2. Based on P/E Ratio
        earnings_per_share = info.get('trailingEps', 0)
        conservative_pe = min(15, info.get('forwardPE', 15))  # Use lower of 15 or forward P/E
        pe_target = earnings_per_share * conservative_pe
        
        # 3. Based on Book Value
        book_value = info.get('bookValue', 0)
        price_to_book = info.get('priceToBook', 0)
        if price_to_book > 0:
            conservative_ptb = min(3, price_to_book)  # Use lower of 3 or current P/B
            book_target = book_value * conservative_ptb
        else:
            book_target = book_value * 1.5  # Default to 1.5x book value
        
        # 4. Based on Free Cash Flow Yield
        market_cap = info.get('marketCap', 0)
        free_cash_flow = info.get('freeCashflow', 0)
        if market_cap > 0 and free_cash_flow > 0:
            current_fcf_yield = (free_cash_flow / market_cap) * 100
            target_fcf_yield = 7  # Target 7% FCF yield (conservative)
            fcf_target = (free_cash_flow * 100) / target_fcf_yield
            fcf_target = fcf_target / info.get('sharesOutstanding', 1)  # Per share
        else:
            fcf_target = 0
        
        # Calculate weighted average target price
        valid_targets = [price for price in [dcf_target, pe_target, book_target, fcf_target] if price > 0]
        if valid_targets:
            average_target = sum(valid_targets) / len(valid_targets)
            
            # Add more weight to DCF and P/E if they're available
            weighted_target = average_target
            if dcf_target > 0 and pe_target > 0:
                weighted_target = (dcf_target * 0.4 + pe_target * 0.3 + 
                                 book_target * 0.15 + fcf_target * 0.15)
        else:
            weighted_target = current_price
        
        # Calculate entry points
        strong_buy = weighted_target * 0.7   # 30% below target
        buy = weighted_target * 0.8          # 20% below target
        hold = weighted_target * 0.9         # 10% below target
        
        return {
            'Current Price': f"${current_price:.2f}",
            'Target Price': f"${weighted_target:.2f}",
            'Entry Points': {
                'Strong Buy Below': f"${strong_buy:.2f}",
                'Buy Below': f"${buy:.2f}",
                'Hold Above': f"${hold:.2f}"
            },
            'Valuation Methods': {
                'DCF Target': f"${dcf_target:.2f}" if dcf_target > 0 else "N/A",
                'P/E Target': f"${pe_target:.2f}" if pe_target > 0 else "N/A",
                'Book Value Target': f"${book_target:.2f}" if book_target > 0 else "N/A",
                'FCF Target': f"${fcf_target:.2f}" if fcf_target > 0 else "N/A"
            }
        }
    except Exception as e:
        st.error(f"Error calculating entry price: {str(e)}")
        return None

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
                # Technical Analysis Selector
                available_indicators = [
                    'Moving Averages',
                    'Bollinger Bands',
                    'MACD',
                    'RSI',
                    'Stochastic'
                ]
                
                selected_indicators = st.multiselect(
                    "Select Technical Indicators",
                    available_indicators,
                    default=['Moving Averages', 'MACD', 'RSI']
                )
                
                if selected_indicators:
                    stock = yf.Ticker(ticker)
                    hist_data = stock.history(period='1y')
                    st.plotly_chart(plot_technical_analysis(hist_data, ticker, selected_indicators), use_container_width=True)
            
            # Historical Chart Tab
            with chart_tabs[2]:
                period = st.select_slider("Select Time Period", 
                                        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                                        value='1y')
                st.plotly_chart(plot_stock_history(ticker, period), use_container_width=True)
            
            # Price Prediction Tab
            with chart_tabs[3]:
                col1, col2 = st.columns([2, 1])
                with col1:
                    prediction_days = st.slider("Select number of days to predict", 7, 60, 30)
                with col2:
                    model_type = st.selectbox(
                        "Select Prediction Model",
                        ['linear', 'random_forest', 'svr'],
                        format_func=lambda x: {
                            'linear': 'Linear Regression',
                            'random_forest': 'Random Forest',
                            'svr': 'Support Vector Regression'
                        }[x]
                    )
                
                if st.button("Generate Prediction"):
                    with st.spinner("Generating prediction..."):
                        pred_fig, pred_value, _ = predict_stock_price(ticker, prediction_days, model_type)
                        
                        if pred_fig is not None:
                            st.plotly_chart(pred_fig, use_container_width=True)
                            st.info(f"Average predicted price for next {prediction_days} days: ${pred_value:.2f}")
            
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
