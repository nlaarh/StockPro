import yfinance as yf
import pandas as pd

# Test direct data fetch
ticker = "AAPL"
stock = yf.Ticker(ticker)
hist = stock.history(period="1y")

print(f"Fetching data for {ticker}")
print(f"Data shape: {hist.shape}")
print("\nFirst few rows:")
print(hist.head())
