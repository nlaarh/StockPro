import yfinance as yf
import pandas as pd
import numpy as np

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def test_stock_metrics(symbol="NVDA"):
    """Test all stock metrics for a given symbol"""
    print_section(f"Testing metrics for {symbol}")
    
    stock = yf.Ticker(symbol)
    
    # Test Fast Info
    print_section("Fast Info")
    fast_info = stock.fast_info
    for attr in dir(fast_info):
        if not attr.startswith('_'):
            try:
                value = getattr(fast_info, attr)
                print(f"{attr}: {value}")
            except Exception as e:
                print(f"Error getting {attr}: {str(e)}")
    
    # Test Earnings Forecasts
    print_section("Earnings Forecasts")
    try:
        forecasts = stock.earnings_forecasts
        print("\nForecasts DataFrame:")
        print(forecasts)
    except Exception as e:
        print(f"Error getting earnings forecasts: {str(e)}")
    
    # Test Recommendations
    print_section("Recommendations")
    try:
        recommendations = stock.recommendations
        print("\nRecommendations DataFrame:")
        print(recommendations)
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
    
    # Test Financials
    print_section("Financials")
    try:
        financials = stock.financials
        print("\nFinancials DataFrame:")
        print(financials)
    except Exception as e:
        print(f"Error getting financials: {str(e)}")
    
    # Test Income Statement
    print_section("Income Statement")
    try:
        income_stmt = stock.income_stmt
        print("\nColumns:")
        print(income_stmt.columns.tolist())
        print("\nIndex:")
        print(income_stmt.index.tolist())
        print("\nLatest Quarter Data:")
        print(income_stmt.iloc[:, 0])
    except Exception as e:
        print(f"Error getting income statement: {str(e)}")
    
    # Test Balance Sheet
    print_section("Balance Sheet")
    try:
        balance = stock.balance_sheet
        print("\nColumns:")
        print(balance.columns.tolist())
        print("\nIndex:")
        print(balance.index.tolist())
        print("\nLatest Quarter Data:")
        print(balance.iloc[:, 0])
    except Exception as e:
        print(f"Error getting balance sheet: {str(e)}")
    
    # Test Cash Flow
    print_section("Cash Flow")
    try:
        cashflow = stock.cashflow
        print("\nColumns:")
        print(cashflow.columns.tolist())
        print("\nIndex:")
        print(cashflow.index.tolist())
        print("\nLatest Quarter Data:")
        print(cashflow.iloc[:, 0])
    except Exception as e:
        print(f"Error getting cash flow: {str(e)}")
    
    # Calculate Key Metrics
    print_section("Calculated Metrics")
    try:
        # Market Cap
        market_cap = fast_info.market_cap if hasattr(fast_info, 'market_cap') else None
        print(f"Market Cap: {market_cap:,} ({market_cap/1e9:.2f}B)" if market_cap else "Market Cap: N/A")
        
        # Current Price
        current_price = fast_info.last_price if hasattr(fast_info, 'last_price') else None
        print(f"Current Price: {current_price}" if current_price else "Current Price: N/A")
        
        # Forward P/E
        try:
            forecasts = stock.earnings_forecasts
            if not forecasts.empty:
                forward_eps = forecasts['Earnings Estimate'].iloc[0]
                forward_pe = current_price / forward_eps if forward_eps > 0 else None
                print(f"Forward P/E (from forecasts): {forward_pe:.2f}" if forward_pe else "Forward P/E: N/A")
        except Exception as e:
            print(f"Error calculating Forward P/E: {str(e)}")
        
        # Price/Book
        try:
            if not balance.empty and current_price:
                equity = balance.loc['Stockholders Equity'].iloc[0]
                shares = fast_info.shares if hasattr(fast_info, 'shares') else None
                if equity and shares:
                    book_value_per_share = equity / shares
                    pb_ratio = current_price / book_value_per_share
                    print(f"Price/Book: {pb_ratio:.2f}")
                else:
                    print("Price/Book: Missing data")
        except Exception as e:
            print(f"Error calculating Price/Book: {str(e)}")
        
        # Price/Sales
        try:
            if not income_stmt.empty and market_cap:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                ps_ratio = market_cap / revenue if revenue > 0 else None
                print(f"Price/Sales: {ps_ratio:.2f}" if ps_ratio else "Price/Sales: N/A")
        except Exception as e:
            print(f"Error calculating Price/Sales: {str(e)}")
        
        # EV/EBITDA
        try:
            if not income_stmt.empty and not balance.empty and market_cap:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                total_debt = balance.loc['Total Debt'].iloc[0]
                cash = balance.loc['Cash And Cash Equivalents'].iloc[0]
                enterprise_value = market_cap + total_debt - cash
                ev_ebitda = enterprise_value / ebitda if ebitda > 0 else None
                print(f"EV/EBITDA: {ev_ebitda:.2f}" if ev_ebitda else "EV/EBITDA: N/A")
        except Exception as e:
            print(f"Error calculating EV/EBITDA: {str(e)}")
        
        # FCF Yield
        try:
            if not cashflow.empty and market_cap:
                fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                fcf_yield = fcf / market_cap
                print(f"FCF Yield: {fcf_yield:.2%}")
        except Exception as e:
            print(f"Error calculating FCF Yield: {str(e)}")
                
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def test_company_info(symbol="NVDA"):
    """Test all available company info"""
    print_section(f"Testing company info for {symbol}")
    
    stock = yf.Ticker(symbol)
    
    # Test Info
    print_section("Stock Info")
    try:
        info = stock.info
        print("\nInfo Dict:")
        for key, value in info.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error getting info: {str(e)}")
    
    # Test Fast Info
    print_section("Fast Info")
    try:
        fast_info = stock.fast_info
        print("\nFast Info:")
        for attr in dir(fast_info):
            if not attr.startswith('_'):
                try:
                    value = getattr(fast_info, attr)
                    print(f"{attr}: {value}")
                except Exception as e:
                    print(f"Error getting {attr}: {str(e)}")
    except Exception as e:
        print(f"Error getting fast info: {str(e)}")
    
    # Test Institutional Holders
    print_section("Institutional Holders")
    try:
        holders = stock.institutional_holders
        print("\nInstitutional Holders:")
        print(holders)
    except Exception as e:
        print(f"Error getting institutional holders: {str(e)}")
    
    # Test Major Holders
    print_section("Major Holders")
    try:
        major = stock.major_holders
        print("\nMajor Holders:")
        print(major)
    except Exception as e:
        print(f"Error getting major holders: {str(e)}")

if __name__ == "__main__":
    test_stock_metrics()
    test_company_info()
