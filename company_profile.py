import streamlit as st
import pandas as pd
import yfinance as yf


def company_profile_tab():
    """Display company profile and key metrics"""
    try:
        st.header("📊 Company Profile")
        
        # Get ticker from session state
        ticker = st.session_state.get('current_ticker', '')
        if not ticker:
            st.warning("Please enter a stock ticker above.")
            return
            
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                st.error(f"No data found for {ticker}")
                return
                
            # Company Overview
            st.subheader("Company Overview")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Company name and description
                st.markdown(f"### {info.get('longName', ticker)}")
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
                st.markdown(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
                st.markdown("### Business Summary")
                st.write(info.get('longBusinessSummary', 'No description available.'))
            
            with col2:
                # Key Stats
                st.markdown("### Key Statistics")
                market_cap = info.get('marketCap', 0)
                st.metric("Market Cap", f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) and market_cap > 0 else "N/A")
                
                pe_ratio = info.get('trailingPE')
                st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A")
                
                beta = info.get('beta')
                st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A")
                
                high_52w = info.get('fiftyTwoWeekHigh')
                st.metric("52W High", f"${high_52w:.2f}" if isinstance(high_52w, (int, float)) else "N/A")
                
                low_52w = info.get('fiftyTwoWeekLow')
                st.metric("52W Low", f"${low_52w:.2f}" if isinstance(low_52w, (int, float)) else "N/A")
            
            # Financial Metrics
            st.subheader("Financial Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Valuation")
                forward_pe = info.get('forwardPE')
                st.metric("Forward P/E", f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else "N/A")
                
                peg_ratio = info.get('pegRatio')
                st.metric("PEG Ratio", f"{peg_ratio:.2f}" if isinstance(peg_ratio, (int, float)) else "N/A")
                
                pb_ratio = info.get('priceToBook')
                st.metric("Price/Book", f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else "N/A")
                
                ps_ratio = info.get('priceToSalesTrailing12Months')
                st.metric("Price/Sales", f"{ps_ratio:.2f}" if isinstance(ps_ratio, (int, float)) else "N/A")
            
            with col2:
                st.markdown("### Growth & Margins")
                rev_growth = info.get('revenueGrowth')
                st.metric("Revenue Growth", f"{rev_growth*100:.1f}%" if isinstance(rev_growth, (int, float)) else "N/A")
                
                gross_margin = info.get('grossMargins')
                st.metric("Gross Margin", f"{gross_margin*100:.1f}%" if isinstance(gross_margin, (int, float)) else "N/A")
                
                op_margin = info.get('operatingMargins')
                st.metric("Operating Margin", f"{op_margin*100:.1f}%" if isinstance(op_margin, (int, float)) else "N/A")
                
                profit_margin = info.get('profitMargins')
                st.metric("Profit Margin", f"{profit_margin*100:.1f}%" if isinstance(profit_margin, (int, float)) else "N/A")
            
            with col3:
                st.markdown("### Dividends & Returns")
                div_yield = info.get('dividendYield')
                st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if isinstance(div_yield, (int, float)) else "N/A")
                
                roe = info.get('returnOnEquity')
                st.metric("ROE", f"{roe*100:.1f}%" if isinstance(roe, (int, float)) else "N/A")
                
                roa = info.get('returnOnAssets')
                st.metric("ROA", f"{roa*100:.1f}%" if isinstance(roa, (int, float)) else "N/A")
                
                payout_ratio = info.get('payoutRatio')
                st.metric("Payout Ratio", f"{payout_ratio*100:.1f}%" if isinstance(payout_ratio, (int, float)) else "N/A")
            
            # Financial Reports Links
            st.subheader("Financial Reports & Resources")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### SEC Filings")
                sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={info.get('SEC_CIK', '')}&owner=exclude"
                st.markdown(f"[View SEC Filings]({sec_url})")
                
                # Yahoo Finance links
                base_url = f"https://finance.yahoo.com/quote/{ticker}"
                st.markdown(f"[Income Statement]({base_url}/financials)")
                st.markdown(f"[Balance Sheet]({base_url}/balance-sheet)")
                st.markdown(f"[Cash Flow]({base_url}/cash-flow)")
            
            with col2:
                st.markdown("### Investor Resources")
                st.markdown(f"[Earnings Calls Transcripts](https://seekingalpha.com/symbol/{ticker}/earnings)")
                st.markdown(f"[Latest News](https://finance.yahoo.com/quote/{ticker}/news)")
                ir_website = info.get('website', '').rstrip('/') + '/investor-relations'
                st.markdown(f"[Investor Relations]({ir_website})")
            
            # Ownership and Insiders
            st.subheader("Ownership Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Institutional Ownership")
                try:
                    inst_holders = stock.institutional_holders
                    if isinstance(inst_holders, pd.DataFrame) and not inst_holders.empty:
                        st.dataframe(inst_holders.head())
                    else:
                        st.write("No institutional ownership data available")
                except Exception as e:
                    st.write("No institutional ownership data available")
            
            with col2:
                st.markdown("### Major Holders")
                try:
                    major_holders = stock.major_holders
                    if isinstance(major_holders, pd.DataFrame) and not major_holders.empty:
                        st.dataframe(major_holders)
                    else:
                        st.write("No major holders data available")
                except Exception as e:
                    st.write("No major holders data available")
        
        except Exception as e:
            st.error(f"Error loading company profile: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
