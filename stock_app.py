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
        # Get stock data with retry mechanism
        max_retries = 3
        retry_count = 0
        data = None
        
        print(f"\nStarting prediction for {ticker} with {model_type} model")
        print(f"Input parameters:")
        print(f"- ticker: {ticker} ({type(ticker)})")
        print(f"- prediction_days: {prediction_days} ({type(prediction_days)})")
        print(f"- model_type: {model_type} ({type(model_type)})")
        
        while retry_count < max_retries:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                print(f"\nDownloaded data:")
                print(f"- Shape: {data.shape}")
                print(f"- Columns: {data.columns}")
                print(f"- Types:\n{data.dtypes}")
                print(f"- First few rows:\n{data.head()}")
                
                if len(data) > 0:
                    break
            except Exception as e:
                print(f"Error fetching data: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)  # Wait 1 second before retrying
        
        if data is None or len(data) == 0:
            return None, None, f"Failed to fetch historical data for {ticker} after {max_retries} attempts"
        
        # Convert to float type and handle missing values
        try:
            print("Converting Close column to float type and handling missing values")
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce').fillna(method='ffill')
            print("Close column after conversion:")
            print(f"- Type: {type(data['Close'])}")
            print(f"- Shape: {data['Close'].shape}")
            print(f"- First few values:\n{data['Close'].head()}")
            
            # Check if the series is empty or contains only NaNs
            if data['Close'].empty:
                raise ValueError("Close series is empty after conversion")
            if data['Close'].isna().all():
                raise ValueError("All values in Close series are NaN after conversion")
        except Exception as e:
            print(f"Error processing Close column: {str(e)}")
            print(traceback.format_exc(limit=5))  # Format traceback with 5 frames
            return None, None, f"Error processing Close column: {str(e)}"
        
        if model_type in ['linear', 'random_forest', 'svr']:
            # Traditional ML prediction logic
            df = pd.DataFrame(data['Close'])  # Create a new DataFrame with just Close prices
            df['Prediction'] = df['Close'].shift(-prediction_days)
            
            X = np.array(df.drop(['Prediction'], axis=1))[:-prediction_days]
            y = np.array(df['Prediction'])[:-prediction_days]
            
            # Split data
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0,1))
            x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1))
            x_test_scaled = scaler.transform(x_test.reshape(-1, 1))
            
            # Choose and train model
            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100)
            else:  # SVR
                model = SVR(kernel='rbf', C=1000.0, gamma=0.1)
            
            model.fit(x_train_scaled, y_train)
            
            # Prepare future data for prediction
            last_days = np.array(df.drop(['Prediction'], axis=1))[-prediction_days:]
            last_days_scaled = scaler.transform(last_days)
            
            # Make prediction
            prediction = model.predict(last_days_scaled)
            
            # Create visualization
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'].values,
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            # Add prediction
            future_dates = pd.date_range(start=data.index[-1], periods=prediction_days+1)[1:]
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=prediction,
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
            
            return fig, float(np.mean(prediction)), "Traditional ML model prediction"
            
        else:
            # LLM-based prediction
            llm_predictor = LLMPredictor()
            
            if 'llama' in model_type or 'mistral' in model_type or 'codellama' in model_type:
                try:
                    # Pass the Close column directly
                    print("\nPreparing data for LLM prediction:")
                    close_series = data['Close'].copy()
                    print(f"- Type: {type(close_series)}")
                    print(f"- Shape: {close_series.shape}")
                    print(f"- First few values:\n{close_series.head()}")
                    
                    # Check if the series is empty or contains only NaNs
                    if close_series.empty:
                        raise ValueError("Close series is empty before prediction")
                    if close_series.isna().all():
                        raise ValueError("All values in Close series are NaN before prediction")

                    predicted_price, reasoning = llm_predictor.predict_ollama(ticker, prediction_days, model_type, close_series)
                    if predicted_price is None:
                        return None, None, reasoning  # Return the error message from predict_ollama
                except Exception as e:
                    print(f"Error in LLM prediction: {str(e)}")
                    traceback.print_exc()
                    return None, None, f"Error with Ollama prediction: {str(e)}"
            elif 'nvidia' in model_type:
                try:
                    predicted_price, reasoning = llm_predictor.predict_nvidia(ticker, prediction_days)
                    if predicted_price is None:
                        return None, None, reasoning
                except Exception as e:
                    return None, None, f"Error with Nvidia prediction: {str(e)}"
            else:
                return None, None, f"Unsupported model type: {model_type}"
            
            # Create visualization only if we have a valid prediction
            if predicted_price is not None:
                # Create visualization
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'].values,
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Add prediction point
                future_date = data.index[-1] + pd.Timedelta(days=prediction_days)
                current_price = float(data['Close'].iloc[-1])
                
                fig.add_trace(go.Scatter(
                    x=[data.index[-1], future_date],
                    y=[current_price, float(predicted_price)],
                    name='LLM Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{ticker} Stock Price Prediction (LLM)',
                    yaxis_title='Stock Price (USD)',
                    xaxis_title='Date',
                    showlegend=True,
                    template='plotly_white',
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                return fig, predicted_price, reasoning
            
            return None, None, "Failed to generate valid prediction"
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return None, None, f"Error in prediction: {str(e)}"

# Function to calculate Buffett metrics
def calculate_buffett_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic metrics
        metrics = {
            'Entry Price Analysis': {
                'Current Price': info.get('currentPrice', 0),
                'Target Price': info.get('targetMeanPrice', 0),
                'Entry Points': {
                    'Strong Buy Below': info.get('targetLowPrice', 0),
                    'Buy Below': info.get('targetMeanPrice', 0) * 0.9,
                    'Hold Above': info.get('targetHighPrice', 0)
                }
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
    # Set page config at the very beginning
    st.set_page_config(
        page_title="Stock Analysis App",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
