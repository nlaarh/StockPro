import yfinance as yf
import pandas as pd
import numpy as np

def print_debug(msg):
    print(f"DEBUG: {msg}")

def test_valuation_metrics():
    """Test valuation metrics calculation"""
    # Test with a known stock
    ticker = "NVDA"
    stock = yf.Ticker(ticker)
    
    print(f"\nTesting valuation metrics for {ticker}")
    print("=" * 50)
    
    # Get info dictionary
    info = stock.info
    print("\nInfo dictionary keys:")
    print("-" * 30)
    for key in sorted(info.keys()):
        print(f"{key}: {info[key]}")
    
    # Test market cap
    print("\nMarket Cap:")
    print("-" * 30)
    market_cap = info.get('marketCap')
    print(f"Market Cap: {market_cap:,} ({market_cap/1e9:.2f}B)")
    
    # Test current price
    print("\nCurrent Price:")
    print("-" * 30)
    current_price = info.get('currentPrice', info.get('regularMarketPrice'))
    print(f"Current Price: {current_price}")
    
    # Test P/E ratios
    print("\nP/E Ratios:")
    print("-" * 30)
    print(f"Forward P/E (direct): {info.get('forwardPE')}")
    print(f"Trailing P/E (direct): {info.get('trailingPE')}")
    
    if current_price and 'forwardEps' in info:
        calculated_forward_pe = current_price / info['forwardEps']
        print(f"Calculated Forward P/E: {calculated_forward_pe}")
    
    # Test Price/Book
    print("\nPrice/Book:")
    print("-" * 30)
    print(f"Price/Book (direct): {info.get('priceToBook')}")
    if current_price and 'bookValue' in info and info['bookValue'] > 0:
        calculated_pb = current_price / info['bookValue']
        print(f"Calculated P/B: {calculated_pb}")
    
    # Test Price/Sales
    print("\nPrice/Sales:")
    print("-" * 30)
    print(f"P/S (direct): {info.get('priceToSalesTrailing12Months')}")
    
    # Test PEG Ratio
    print("\nPEG Ratio:")
    print("-" * 30)
    print(f"PEG (direct): {info.get('pegRatio')}")
    
    # Test EV/EBITDA
    print("\nEV/EBITDA:")
    print("-" * 30)
    print(f"EV/EBITDA (direct): {info.get('enterpriseToEbitda')}")
    
    # Test FCF Yield
    print("\nFCF Yield:")
    print("-" * 30)
    try:
        cashflow = stock.cashflow
        if not cashflow.empty:
            print("\nCash Flow Statement Columns:")
            print(cashflow.columns.tolist())
            print("\nCash Flow Statement Index:")
            print(cashflow.index.tolist())
            
            # Try to get operating cash flow and capex
            if 'Operating Cash Flow' in cashflow.index:
                operating_cashflow = cashflow.loc['Operating Cash Flow'].iloc[0]
                print(f"Operating Cash Flow: {operating_cashflow:,}")
            
            if 'Capital Expenditure' in cashflow.index:
                capex = abs(cashflow.loc['Capital Expenditure'].iloc[0])
                print(f"Capital Expenditure: {capex:,}")
                
            if 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                fcf = operating_cashflow - capex
                if market_cap:
                    fcf_yield = fcf / market_cap
                    print(f"Calculated FCF Yield: {fcf_yield:.2%}")
    except Exception as e:
        print(f"Error calculating FCF Yield: {str(e)}")

if __name__ == "__main__":
    test_valuation_metrics()
