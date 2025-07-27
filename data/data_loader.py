import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

class DataLoader:
    """Utility class để tải và lưu dữ liệu tài chính"""
    
    def __init__(self, data_dir='data/saved/'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_stock_data(self, symbol, period='2y', interval='1d', save=True):
        """
        Tải dữ liệu cổ phiếu từ Yahoo Finance
        
        Args:
            symbol: Mã cổ phiếu (VD: 'AAPL', 'SPY')
            period: Thời gian ('1y', '2y', '5y', 'max')
            interval: Khoảng thời gian ('1d', '1h', '5m')
            save: Có lưu file CSV không
        """
        try:
            print(f"Đang tải dữ liệu {symbol}...")
            
            # Tải dữ liệu
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"Không có dữ liệu cho {symbol}")
            
            # Làm sạch dữ liệu
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Lưu file nếu cần
            if save:
                filename = f"{self.data_dir}{symbol}_{period}_{interval}.csv"
                data.to_csv(filename)
                print(f"Đã lưu dữ liệu vào {filename}")
            
            print(f"Tải thành công {len(data)} dòng dữ liệu")
            return data
            
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol):
        """Lấy thông tin cơ bản về cổ phiếu"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
        except:
            return None

# Test ngay
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.download_stock_data('AAPL', period='1y')
    print(data.head())
    info = loader.get_stock_info('AAPL')
    print(f"Thông tin Apple: {info}")
