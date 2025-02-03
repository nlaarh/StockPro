"""Warren Buffett style analysis for StockPro"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import calculate_rsi, calculate_macd, safe_divide, format_large_number, format_percentage

def get_buffett_analysis(ticker, financials, info):
    """Generate Warren Buffett style analysis for a stock"""
    try:
        # Safely get values with defaults
        def safe_get(data, key, default=0):
            try:
                value = data.get(key, default)
                return float(value) if value is not None else default
            except:
                return default
        
        # Get key metrics
        market_cap = safe_get(info, 'marketCap', 0)
        revenue = safe_get(info, 'totalRevenue', 0)
        net_income = safe_get(info, 'netIncome', 0)
        total_assets = safe_get(info, 'totalAssets', 0)
        total_debt = safe_get(info, 'totalDebt', 0)
        free_cash_flow = safe_get(info, 'freeCashflow', 0)
        shares_outstanding = safe_get(info, 'sharesOutstanding', 0)
        current_price = safe_get(info, 'currentPrice', 0)
        
        # Calculate key ratios
        pe_ratio = safe_divide(current_price * shares_outstanding, net_income)
        price_to_book = safe_divide(market_cap, total_assets - total_debt)
        roe = safe_divide(net_income, total_assets - total_debt) * 100
        operating_margin = safe_get(info, 'operatingMargins', 0) * 100
        net_margin = safe_divide(net_income, revenue) * 100
        debt_to_equity = safe_divide(total_debt, total_assets - total_debt)
        
        # Store metrics
        metrics = {
            'market_cap': market_cap,
            'revenue': revenue,
            'net_income': net_income,
            'total_assets': total_assets,
            'total_debt': total_debt,
            'free_cash_flow': free_cash_flow,
            'shares_outstanding': shares_outstanding,
            'current_price': current_price,
            'pe_ratio': pe_ratio,
            'price_to_book': price_to_book,
            'roe': roe,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'debt_to_equity': debt_to_equity
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error in Buffett analysis: {str(e)}")
        return {}

def get_buffett_metrics_description():
    """Get Warren Buffett's key metrics and their interpretations"""
    return {
        'Return on Equity (ROE)': {
            'description': 'Measures how efficiently a company uses shareholders\' equity to generate profits',
            'interpretation': {
                'Excellent': '> 20%',
                'Good': '15-20%',
                'Average': '10-15%',
                'Poor': '< 10%'
            },
            'buffett_preference': 'Buffett prefers companies with ROE > 15% consistently over 10 years'
        },
        'Debt to Equity': {
            'description': 'Measures financial leverage and risk',
            'interpretation': {
                'Excellent': '< 0.3',
                'Good': '0.3-0.5',
                'Caution': '0.5-1.0',
                'High Risk': '> 1.0'
            },
            'buffett_preference': 'Buffett prefers companies with low debt, typically D/E < 0.5'
        },
        'Operating Margin': {
            'description': 'Indicates pricing power and operational efficiency',
            'interpretation': {
                'Excellent': '> 20%',
                'Good': '15-20%',
                'Average': '10-15%',
                'Poor': '< 10%'
            },
            'buffett_preference': 'Buffett looks for high margins indicating competitive advantages'
        },
        'Net Margin': {
            'description': 'Shows how much profit is generated from revenue',
            'interpretation': {
                'Excellent': '> 15%',
                'Good': '10-15%',
                'Average': '5-10%',
                'Poor': '< 5%'
            },
            'buffett_preference': 'Higher margins indicate better business models'
        },
        'Price to Earnings (P/E)': {
            'description': 'Valuation metric comparing price to earnings',
            'interpretation': {
                'Potentially Undervalued': '< 15',
                'Fair Value': '15-20',
                'Premium': '20-25',
                'Expensive': '> 25'
            },
            'buffett_preference': 'Buffett looks for reasonable P/E ratios relative to growth'
        },
        'Price to Book (P/B)': {
            'description': 'Compares market price to book value',
            'interpretation': {
                'Potentially Undervalued': '< 1.5',
                'Fair Value': '1.5-2.5',
                'Premium': '2.5-3.5',
                'Expensive': '> 3.5'
            },
            'buffett_preference': 'Buffett typically seeks companies with P/B < 2.5'
        },
        'Free Cash Flow Yield': {
            'description': 'Free cash flow relative to market cap',
            'interpretation': {
                'Excellent': '> 8%',
                'Good': '5-8%',
                'Average': '3-5%',
                'Poor': '< 3%'
            },
            'buffett_preference': 'Higher FCF yield indicates better value'
        }
    }

def get_buffett_recommendation(analysis):
    """Get Buffett-style investment recommendation"""
    try:
        # Quality Score (0-100)
        quality_score = (
            min(100, (analysis['roe'] / 20) * 30) +  # ROE (max 30 points)
            min(40, (analysis['operating_margin'] / 20) * 40) +  # Operating Margin (max 40 points)
            min(30, max(0, (1 - analysis['debt_to_equity']) * 30))  # Low Debt (max 30 points)
        )
        
        # Valuation Score (0-100)
        valuation_score = (
            min(50, max(0, (25 - analysis['pe_ratio']) * 2)) +  # PE Ratio (max 50 points)
            min(50, max(0, (3 - analysis['price_to_book']) * 25))  # PB Ratio (max 50 points)
        )
        
        # Overall Score
        overall_score = (quality_score * 0.6) + (valuation_score * 0.4)  # Quality weighted higher
        
        recommendation = {
            'quality_score': quality_score,
            'valuation_score': valuation_score,
            'overall_score': overall_score,
            'action': 'STRONG BUY' if overall_score >= 80 else
                     'BUY' if overall_score >= 70 else
                     'HOLD' if overall_score >= 50 else
                     'SELL',
            'reasons': []
        }
        
        # Add detailed reasons
        if analysis['roe'] > 15:
            recommendation['reasons'].append("✅ High ROE indicates strong competitive advantage")
        else:
            recommendation['reasons'].append("❌ Low ROE suggests weak competitive position")
            
        if analysis['operating_margin'] > 20:
            recommendation['reasons'].append("✅ High operating margin shows pricing power")
        else:
            recommendation['reasons'].append("❌ Low operating margin indicates competitive pressures")
            
        if analysis['debt_to_equity'] < 0.5:
            recommendation['reasons'].append("✅ Conservative debt level")
        else:
            recommendation['reasons'].append("❌ High debt level increases risk")
            
        if analysis['pe_ratio'] < 15:
            recommendation['reasons'].append("✅ Attractive valuation (P/E)")
        else:
            recommendation['reasons'].append("❌ Premium valuation (P/E)")
            
        return recommendation
        
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        return None

def plot_buffett_comparison(analysis):
    """Create a simple bar chart comparing actual vs Buffett's preferred values"""
    try:
        # Define metrics and their preferred values
        metrics = {
            'Return on Equity': {
                'actual': analysis['roe'],
                'preferred': 15,
                'format': '%'
            },
            'Operating Margin': {
                'actual': analysis['operating_margin'],
                'preferred': 20,
                'format': '%'
            },
            'Debt to Equity': {
                'actual': analysis['debt_to_equity'],
                'preferred': 0.5,
                'format': 'x'
            },
            'P/E Ratio': {
                'actual': analysis['pe_ratio'],
                'preferred': 15,
                'format': 'x'
            },
            'Price to Book': {
                'actual': analysis['price_to_book'],
                'preferred': 2.5,
                'format': 'x'
            }
        }
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for actual values
        fig.add_trace(go.Bar(
            name='Company',
            x=list(metrics.keys()),
            y=[m['actual'] for m in metrics.values()],
            marker_color='blue',
            text=[f"{v['actual']:.1f}{v['format']}" for v in metrics.values()],
            textposition='auto',
        ))
        
        # Add bars for preferred values
        fig.add_trace(go.Bar(
            name='Buffett Preferred',
            x=list(metrics.keys()),
            y=[m['preferred'] for m in metrics.values()],
            marker_color='green',
            text=[f"{v['preferred']}{v['format']}" for v in metrics.values()],
            textposition='auto',
        ))
        
        # Update layout
        fig.update_layout(
            title="Company Metrics vs. Buffett's Preferences",
            barmode='group',
            height=400,
            showlegend=True,
            yaxis_title="Value",
            xaxis_title="Metric"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

def calculate_price_targets(analysis, current_price):
    """Calculate buy, sell, and hold price targets based on Buffett's principles"""
    try:
        # Calculate intrinsic value using various methods
        pe_based_value = current_price * (15 / max(analysis['pe_ratio'], 1))  # Normalize to PE of 15
        pb_based_value = current_price * (2 / max(analysis['price_to_book'], 0.1))  # Normalize to PB of 2
        
        # Weight the values based on company quality
        quality_score = (
            (analysis['roe'] / 15) * 0.4 +  # ROE weight
            (analysis['operating_margin'] / 20) * 0.3 +  # Operating margin weight
            (min(1, 0.5 / max(analysis['debt_to_equity'], 0.1))) * 0.3  # Debt weight
        )
        
        # Calculate target prices
        intrinsic_value = (pe_based_value * 0.6 + pb_based_value * 0.4) * quality_score
        
        targets = {
            'Buy': max(current_price * 0.7, intrinsic_value * 0.8),  # Conservative buy price
            'Hold_Low': intrinsic_value * 0.9,
            'Hold_High': intrinsic_value * 1.1,
            'Sell': min(current_price * 1.3, intrinsic_value * 1.2),  # Conservative sell price
            'Stop_Loss': current_price * 0.85  # 15% stop loss
        }
        
        return targets
    except Exception as e:
        st.error(f"Error calculating price targets: {str(e)}")
        return None

def generate_buffett_letter(analysis, company_name):
    """Generate a Warren Buffett style shareholder letter"""
    try:
        # Format metrics
        roe = format_percentage(analysis['roe'])
        op_margin = format_percentage(analysis['operating_margin'])
        net_margin = format_percentage(analysis['net_margin'])
        pe = f"{analysis['pe_ratio']:.1f}"
        debt_equity = f"{analysis['debt_to_equity']:.2f}"
        
        letter = f"""
        Dear Fellow Shareholders,

        I am writing to share my thoughts on {company_name} as a potential investment opportunity. As you know, 
        we at Berkshire have always focused on businesses with strong fundamentals, competitive advantages, and 
        fair valuations.

        Business Performance
        -------------------
        {company_name} has demonstrated a return on equity of {roe}, which {
        'indicates strong business fundamentals' if analysis['roe'] > 15 else 
        'suggests room for improvement in capital efficiency'}. The company's operating margin of {op_margin} {
        'reflects significant pricing power and efficiency' if analysis['operating_margin'] > 20 else
        'indicates a competitive but challenging market environment'}.

        Financial Position
        -----------------
        With a debt-to-equity ratio of {debt_equity}, the company {
        'maintains a conservative financial position' if analysis['debt_to_equity'] < 0.5 else
        'carries a significant debt load that warrants attention'}. The net profit margin of {net_margin} {
        'demonstrates strong cost management' if analysis['net_margin'] > 15 else
        'suggests opportunities for improved profitability'}.

        Valuation
        ---------
        At the current price-to-earnings ratio of {pe}, the stock {
        'appears attractively valued' if analysis['pe_ratio'] < 15 else
        'trades at a premium to our preferred entry points' if analysis['pe_ratio'] > 20 else
        'is reasonably valued'}. {
        'This valuation, combined with the company\'s strong fundamentals, makes it an attractive investment candidate.' 
        if analysis['pe_ratio'] < 20 and analysis['roe'] > 15 else
        'We continue to monitor the company for more attractive entry points.'}

        Economic Moat
        ------------
        {company_name}'s competitive position {
        'appears strong, with sustainable advantages' if analysis['operating_margin'] > 20 and analysis['roe'] > 15 else
        'shows some defensive characteristics but faces competitive pressures' if analysis['operating_margin'] > 15 else
        'may be vulnerable to competitive pressures'}. {
        'The combination of high returns on equity and strong margins suggests a durable competitive advantage.' 
        if analysis['roe'] > 15 and analysis['operating_margin'] > 20 else
        'We will continue to monitor the company\'s ability to maintain and strengthen its market position.'}

        Outlook
        -------
        {
        'Given the strong fundamentals and reasonable valuation, we view this as an attractive opportunity for long-term investors.' 
        if analysis['pe_ratio'] < 20 and analysis['roe'] > 15 and analysis['operating_margin'] > 15 else
        'While the business has attractive qualities, we await a more compelling valuation before considering a significant position.' 
        if analysis['roe'] > 15 else
        'The current combination of business performance and valuation suggests better opportunities may exist elsewhere.'
        }

        As always, we maintain our discipline in seeking businesses with strong fundamentals at fair prices.

        Sincerely,
        Warren Buffett
        """
        return letter
    except Exception as e:
        return f"Error generating letter: {str(e)}"

def buffett_analysis_tab():
    """Display Warren Buffett style analysis"""
    try:
        st.header("🎩 Buffett Analysis")
        
        # Get ticker from session state
        ticker = st.session_state.get('current_ticker', '')
        if not ticker:
            st.warning("Please enter a stock ticker above.")
            return
            
        # Fetch stock data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials
            company_name = info.get('longName', ticker)
            
            if not info:
                st.error(f"Could not fetch data for {ticker}")
                return
                
            # Get analysis
            analysis = get_buffett_analysis(ticker, financials, info)
            
            # Get recommendation
            recommendation = get_buffett_recommendation(analysis)
            
            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Company Size")
                st.metric("Market Cap", format_large_number(analysis['market_cap']))
                st.metric("Revenue", format_large_number(analysis['revenue']))
                st.metric("Net Income", format_large_number(analysis['net_income']))
            
            with col2:
                st.subheader("Profitability")
                st.metric("Operating Margin", format_percentage(analysis['operating_margin']))
                st.metric("Net Margin", format_percentage(analysis['net_margin']))
                st.metric("Return on Equity", format_percentage(analysis['roe']))
            
            with col3:
                st.subheader("Valuation")
                st.metric("P/E Ratio", f"{analysis['pe_ratio']:.2f}")
                st.metric("Price/Book", f"{analysis['price_to_book']:.2f}")
                st.metric("Debt/Equity", f"{analysis['debt_to_equity']:.2f}")
            
            # Recommendation Section
            st.markdown("---")
            rec_col1, rec_col2 = st.columns([1, 2])
            
            with rec_col1:
                st.subheader("Buffett Score")
                st.metric("Quality", f"{recommendation['quality_score']:.0f}/100")
                st.metric("Valuation", f"{recommendation['valuation_score']:.0f}/100")
                st.metric("Overall", f"{recommendation['overall_score']:.0f}/100")
                
                # Color-coded recommendation
                rec_color = {
                    'STRONG BUY': 'success',
                    'BUY': 'info',
                    'HOLD': 'warning',
                    'SELL': 'error'
                }
                getattr(st, rec_color[recommendation['action']].lower())(
                    f"### {recommendation['action']}"
                )
            
            with rec_col2:
                st.subheader("Analysis")
                for reason in recommendation['reasons']:
                    st.write(reason)
            
            # Expandable sections
            with st.expander("📊 Metrics Comparison", expanded=False):
                fig = plot_buffett_comparison(analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("💰 Price Targets", expanded=False):
                targets = calculate_price_targets(analysis, analysis['current_price'])
                if targets:
                    price_cols = st.columns(5)
                    with price_cols[0]:
                        st.metric("Buy Below", f"${targets['Buy']:.2f}", 
                                 f"{((targets['Buy'] - analysis['current_price']) / analysis['current_price'] * 100):.1f}%")
                    with price_cols[1]:
                        st.metric("Hold Range", f"${targets['Hold_Low']:.2f} - ${targets['Hold_High']:.2f}")
                    with price_cols[2]:
                        st.metric("Sell Above", f"${targets['Sell']:.2f}",
                                 f"{((targets['Sell'] - analysis['current_price']) / analysis['current_price'] * 100):.1f}%")
                    with price_cols[3]:
                        st.metric("Stop Loss", f"${targets['Stop_Loss']:.2f}",
                                 f"{((targets['Stop_Loss'] - analysis['current_price']) / analysis['current_price'] * 100):.1f}%")
                    with price_cols[4]:
                        st.metric("Current", f"${analysis['current_price']:.2f}")
            
            with st.expander("🎯 Buffett's Key Metrics Guide", expanded=False):
                metrics = get_buffett_metrics_description()
                metrics_data = []
                for metric, details in metrics.items():
                    metrics_data.append({
                        'Metric': metric,
                        'Description': details['description'],
                        'Interpretation': ' | '.join([f"{k}: {v}" for k, v in details['interpretation'].items()]),
                        'Buffett\'s Preference': details['buffett_preference']
                    })
                
                st.dataframe(
                    pd.DataFrame(metrics_data),
                    use_container_width=True,
                    hide_index=True
                )
            
            with st.expander("📜 Warren's Letter to Shareholders", expanded=False):
                letter = generate_buffett_letter(analysis, company_name)
                st.markdown(letter)
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
