import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test direct download
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Downloading data for {ticker}")
print(f"Start date: {start_date}")
print(f"End date: {end_date}")

data = yf.download(ticker, start=start_date, end=end_date, progress=False)
print(f"\nData shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())
