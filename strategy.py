import yfinance as yf
from datetime import datetime

data = yf.download('AAPL', start='2024-07-01', end='2025-07-01')
data.to_csv('data/aapl_2024_2025.csv')