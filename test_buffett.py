import yfinance as yf
import pandas as pd

def test_stock_data(ticker_symbol):
    print(f"\nTesting data for {ticker_symbol}")
    print("-" * 50)
    
    stock = yf.Ticker(ticker_symbol)
    
    # Test different ways to get balance sheet
    print("\nTesting Balance Sheet:")
    try:
        bs1 = stock.balance_sheet
        print("stock.balance_sheet shape:", bs1.shape if not bs1.empty else "Empty")
        if not bs1.empty:
            print("Columns:", bs1.columns.tolist())
            print("Index:", bs1.index.tolist())
    except Exception as e:
        print("Error with balance_sheet:", str(e))
        
    try:
        bs2 = stock.quarterly_balance_sheet
        print("\nstock.quarterly_balance_sheet shape:", bs2.shape if not bs2.empty else "Empty")
        if not bs2.empty:
            print("Columns:", bs2.columns.tolist())
            print("Index:", bs2.index.tolist())
    except Exception as e:
        print("Error with quarterly_balance_sheet:", str(e))
    
    # Test different ways to get income statement
    print("\nTesting Income Statement:")
    try:
        is1 = stock.income_stmt
        print("stock.income_stmt shape:", is1.shape if not is1.empty else "Empty")
        if not is1.empty:
            print("Columns:", is1.columns.tolist())
            print("Index:", is1.index.tolist())
    except Exception as e:
        print("Error with income_stmt:", str(e))
        
    try:
        is2 = stock.quarterly_financials
        print("\nstock.quarterly_financials shape:", is2.shape if not is2.empty else "Empty")
        if not is2.empty:
            print("Columns:", is2.columns.tolist())
            print("Index:", is2.index.tolist())
    except Exception as e:
        print("Error with quarterly_financials:", str(e))
    
    # Test different ways to get cash flow
    print("\nTesting Cash Flow:")
    try:
        cf1 = stock.cashflow
        print("stock.cashflow shape:", cf1.shape if not cf1.empty else "Empty")
        if not cf1.empty:
            print("Columns:", cf1.columns.tolist())
            print("Index:", cf1.index.tolist())
    except Exception as e:
        print("Error with cashflow:", str(e))
        
    try:
        cf2 = stock.quarterly_cashflow
        print("\nstock.quarterly_cashflow shape:", cf2.shape if not cf2.empty else "Empty")
        if not cf2.empty:
            print("Columns:", cf2.columns.tolist())
            print("Index:", cf2.index.tolist())
    except Exception as e:
        print("Error with quarterly_cashflow:", str(e))

if __name__ == "__main__":
    test_stock_data("AAPL")
