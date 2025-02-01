import streamlit as st
import pandas as pd
import requests
import ssl
import certifi


def get_market_movers():
    """Get top gainers, losers, and most active stocks with SSL handling"""
    try:
        # Create session with SSL verification and headers
        session = requests.Session()
        session.verify = certifi.where()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Function to safely get data
        def get_data(url):
            response = session.get(url)
            response.raise_for_status()
            df = pd.read_html(response.text)[0]
            
            # Select and rename required columns
            df = df[['Symbol', 'Name', 'Price (Intraday)', 'Change %', 'Volume', 'Market Cap']]
            df = df.rename(columns={
                'Change %': '% Change',
                'Price (Intraday)': 'Price'
            })
            return df
            
        # Helper functions to clean data
        def clean_percentage(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val).replace('%', '').replace('+', '').replace(',', '')
            try:
                return float(val)
            except:
                return 0.0
                
        def clean_price(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val).replace('$', '').replace('+', '').replace(',', '')
            try:
                return float(val)
            except:
                return 0.0
                
        def clean_volume(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val = str(val)
            try:
                val = val.replace(',', '')
                if 'K' in val:
                    return float(val.replace('K', '')) * 1000
                elif 'M' in val:
                    return float(val.replace('M', '')) * 1000000
                elif 'B' in val:
                    return float(val.replace('B', '')) * 1000000000
                else:
                    return float(val)
            except:
                return 0.0
        
        # Get market data from Yahoo Finance with updated URLs
        gainers = get_data('https://finance.yahoo.com/screener/predefined/day_gainers')
        losers = get_data('https://finance.yahoo.com/screener/predefined/day_losers')
        active = get_data('https://finance.yahoo.com/screener/predefined/most_actives')
        
        # Clean up data
        for df in [gainers, losers, active]:
            df['% Change'] = df['% Change'].apply(clean_percentage)
            df['Price'] = df['Price'].apply(clean_price)
            df['Volume'] = df['Volume'].apply(clean_volume)
            
        return gainers.head(), losers.head(), active.head()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching market data: {str(e)}")
        return None, None, None
    except ValueError as e:
        st.error(f"Error processing market data: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None, None, None


def display_movers_table(df, category):
    """Display market movers table with formatting"""
    try:
        # Format the dataframe
        df['Price'] = df['Price'].apply(lambda x: f"${x:.2f}")
        df['% Change'] = df['% Change'].apply(lambda x: f"{x:+.2f}%")
        df['Volume'] = df['Volume'].apply(lambda x: f"{x:,.0f}")
        df['Market Cap'] = df['Market Cap'].apply(lambda x: f"${x:,.0f}")
        
        # Display the table with custom formatting
        st.dataframe(
            df.style.set_properties(**{
                'background-color': '#f0f2f6',
                'color': 'black',
                'border-color': 'white'
            })
        )
    except Exception as e:
        st.error(f"Error displaying {category} table: {str(e)}")


def market_movers_tab():
    """Market movers tab with enhanced UI and analysis"""
    try:
        st.subheader("Market Movers")
        
        with st.spinner("Fetching market data..."):
            # Get market movers data
            gainers, losers, active = get_market_movers()
            
            if gainers is not None and losers is not None and active is not None:
                # Create three columns for different categories
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("### 📈 Top Gainers")
                    display_movers_table(gainers, "gainers")
                    
                with col2:
                    st.write("### 📉 Top Losers")
                    display_movers_table(losers, "losers")
                    
                with col3:
                    st.write("### 📊 Most Active")
                    display_movers_table(active, "most active")
                    
                # Add market summary
                st.write("### Market Summary")
                avg_gain = gainers['% Change'].mean()
                avg_loss = losers['% Change'].mean()
                total_volume = active['Volume'].sum()
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Average Gain", f"{avg_gain:+.2f}%")
                    
                with summary_col2:
                    st.metric("Average Loss", f"{avg_loss:+.2f}%")
                    
                with summary_col3:
                    st.metric("Total Volume", f"{total_volume:,.0f}")
                    
                # Add refresh button
                if st.button("🔄 Refresh Data"):
                    st.experimental_rerun()
                    
    except Exception as e:
        st.error(f"Error in market movers tab: {str(e)}")
