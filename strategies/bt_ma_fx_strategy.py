import backtrader as bt
import pandas as pd
from datetime import datetime
import yfinance as yf
import os

class ForexLoader:
    """
    Lớp để tải và nạp dữ liệu Forex từ file CSV cho Backtrader.
    """

    def __init__(self, data_dir='data/saved/forex/'):
        """Khởi tạo loader và tạo thư mục lưu trữ nếu cần."""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_forex_data(self, symbol, start_date, end_date):
        """
        Tải dữ liệu Forex bằng yfinance và lưu vào file CSV.
        """
        yf_symbol = f"{symbol.replace('_', '')}=X"
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        
        print(f"Đang tải dữ liệu Forex cho {symbol} từ yfinance...")
        data = yf.download(yf_symbol, start=start_date, end=end_date)
        
        if data.empty:
            print(f"Không tải được dữ liệu cho {symbol}.")
            return None
        
        data.to_csv(filepath)
        print(f"Đã lưu dữ liệu vào: {filepath}")
        return filepath

    def get_data_from_csv(self, symbol):
        """
        Đọc dữ liệu từ file CSV và tạo data feed cho Backtrader.
        """
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(filepath):
            print(f"File {filepath} không tồn tại. Vui lòng chạy download_forex_data() trước.")
            return None
            
        print(f"Đang đọc dữ liệu từ {filepath}...")
        
        dataframe = pd.read_csv(
            filepath,
            index_col=0,
            parse_dates=True,
            date_format='%Y-%m-%d'  # THÊM DÒNG NÀY ĐỂ LOẠI BỎ CẢNH BÁO
        )
        
        # --- SỬA LỖI Ở ĐÂY: Chuyển đổi các cột sang dạng số ---
        # Đổi tên cột thành chữ thường trước
        dataframe.columns = [col.lower() for col in dataframe.columns]
        
        # Ép kiểu các cột cần thiết sang dạng số
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in dataframe.columns:
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        
        # Xóa các dòng có thể bị lỗi sau khi ép kiểu
        dataframe.dropna(inplace=True)
        # ----------------------------------------------------
        
        # Tạo data feed từ Pandas DataFrame
        return bt.feeds.PandasData(dataname=dataframe)

# Ví dụ cách sử dụng
if __name__ == '__main__':
    forex_loader = ForexLoader()
    
    start = datetime(2023, 1, 1)
    end = datetime(2024, 1, 1)
    
    forex_loader.download_forex_data('EUR_USD', start, end)
    
    eurusd_data = forex_loader.get_data_from_csv('EUR_USD')
    
    if eurusd_data is not None:
        cerebro = bt.Cerebro()
        cerebro.adddata(eurusd_data, name="EURUSD")
        
        print("\nĐã thêm dữ liệu EURUSD vào Cerebro thành công.")
