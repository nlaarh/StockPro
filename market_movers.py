import streamlit as st
import pandas as pd
import requests
import ssl
import certifi
import logging
import io
import traceback

logger = logging.getLogger(__name__)

def format_number(value):
    """Format numbers for display"""
    try:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, str):
            # Try to convert percentage strings
            if '%' in value:
                value = float(value.strip('%')) / 100
            else:
                value = float(value)
        
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif isinstance(value, float):
            if abs(value) < 0.01:
                return f"${value:.4f}"
            else:
                return f"${value:.2f}"
        else:
            return str(value)
    except Exception as e:
        logger.error(f"Error formatting number {value}: {str(e)}")
        return str(value)

def format_percentage(value):
    """Format percentage values"""
    try:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, str):
            # Remove % if present and convert to float
            value = float(value.strip('%').strip('+').strip('-'))
        return f"{value:+.2f}%"
    except Exception as e:
        logger.error(f"Error formatting percentage {value}: {str(e)}")
        return "N/A"

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
            df = pd.read_html(io.StringIO(response.text))[0]
            
            # Print column names for debugging
            logger.info(f"Columns in dataframe: {df.columns.tolist()}")
            
            # Map old column names to new ones
            column_map = {
                'Symbol': 'Symbol',
                'Name': 'Name',
                'Price': 'Price',
                'Price (Intraday)': 'Price',
                'Change': 'Change',
                'Change %': '% Change',
                '% Change': '% Change',
                'Volume': 'Volume',
                'Avg Vol (3 month)': 'Volume',
                'Market Cap': 'Market Cap'
            }
            
            # Select required columns, handling different possible names
            required_columns = []
            for new_col, old_cols in {
                'Symbol': ['Symbol'],
                'Name': ['Name'],
                'Price': ['Price', 'Price (Intraday)'],
                '% Change': ['% Change', 'Change %'],
                'Volume': ['Volume', 'Avg Vol (3 month)'],
                'Market Cap': ['Market Cap']
            }.items():
                # Find first matching column
                found = False
                for old_col in old_cols:
                    if old_col in df.columns:
                        df[new_col] = df[old_col]
                        found = True
                        break
                if not found:
                    df[new_col] = 'N/A'
                required_columns.append(new_col)
            
            df = df[required_columns]
            
            # Convert numeric columns
            df['Price'] = pd.to_numeric(df['Price'].str.replace('$', '').str.replace(',', ''), errors='coerce')
            df['% Change'] = pd.to_numeric(df['% Change'].str.replace('%', '').str.replace('+', '').str.replace(',', ''), errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
            df['Market Cap'] = pd.to_numeric(df['Market Cap'].str.replace('$', '').str.replace(',', '').str.replace('T', 'e12').str.replace('B', 'e9').str.replace('M', 'e6').str.replace('K', 'e3'), errors='coerce')
            
            return df
            
        # Get market data from Yahoo Finance with updated URLs
        gainers = get_data('https://finance.yahoo.com/gainers')
        losers = get_data('https://finance.yahoo.com/losers')
        active = get_data('https://finance.yahoo.com/most-active')
        
        return gainers.head(), losers.head(), active.head()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching market data: {str(e)}")
        return None, None, None
    except ValueError as e:
        logger.error(f"Error processing market data: {str(e)}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None, None, None

def format_market_cap(value):
    """Format market cap values"""
    try:
        if pd.isna(value):
            return "N/A"
        
        value = float(value)
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value/1e3:.2f}K"
    except:
        return "N/A"

def display_movers_table(df, category):
    """Display market movers table with formatting"""
    try:
        if df is None or df.empty:
            st.error(f"No data available for {category}")
            return
            
        # Create a copy for display
        display_df = df.copy()
        
        # Format numeric columns
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
        display_df['% Change'] = display_df['% Change'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
        display_df['Market Cap'] = display_df['Market Cap'].apply(format_market_cap)
        
        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "% Change": st.column_config.TextColumn("% Change", width="small"),
                "Volume": st.column_config.TextColumn("Volume", width="small"),
                "Market Cap": st.column_config.TextColumn("Market Cap", width="small")
            }
        )
    except Exception as e:
        logger.error(f"Error displaying {category} table: {str(e)}")
        st.error(f"Error displaying {category} table: {str(e)}")

def market_movers_tab():
    """Display market movers data"""
    try:
        st.header("Market Movers")
        
        # Get market movers data
        gainers, losers, active = get_market_movers()
        
        # Create tabs for different categories
        gainers_tab, losers_tab, active_tab = st.tabs(["Top Gainers", "Top Losers", "Most Active"])
        
        # Display top gainers
        with gainers_tab:
            st.markdown("### 📈 Top Gainers")
            if gainers is not None:
                display_movers_table(gainers, "Top Gainers")
            else:
                st.error("Unable to fetch gainers")
        
        # Display top losers
        with losers_tab:
            st.markdown("### 📉 Top Losers")
            if losers is not None:
                display_movers_table(losers, "Top Losers")
            else:
                st.error("Unable to fetch losers")
        
        # Display most active
        with active_tab:
            st.markdown("### 📊 Most Active")
            if active is not None:
                display_movers_table(active, "Most Active")
            else:
                st.error("Unable to fetch active")
            
    except Exception as e:
        logger.error(f"Error in market movers tab: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error loading market movers data: {str(e)}")
