import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class ForexLoader:
    """
    Lớp để tải dữ liệu Forex và Cổ phiếu từ Yahoo Finance.
    """
    def fetch(self, pair="EURUSD=X", tf="1h", period="100d"):
        """
        Tải dữ liệu và trả về dưới dạng Pandas DataFrame.
        Sử dụng start/end date để tăng độ ổn định.
        
        Args:
            pair (str): Cặp tiền tệ hoặc mã cổ phiếu.
            tf (str): Khung thời gian ('1h', '1d').
            period (str): Khoảng thời gian ('100d', '1y').
        """
        print(f"Đang tải dữ liệu cho {pair} khung thời gian {tf}...")
        try:
            # --- PHẦN SỬA LỖI ---
            # Chuyển đổi period (ví dụ: '100d') thành ngày bắt đầu cụ thể
            # Điều này giúp yfinance hoạt động ổn định hơn
            
            # Lấy số ngày từ chuỗi period (ví dụ: '100d' -> 100)
            days_to_subtract = int(period[:-1])
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_to_subtract)
            
            # Tải dữ liệu bằng yfinance với start và end date
            data = yf.download(
                pair, 
                start=start_date, 
                end=end_date, 
                interval=tf, 
                progress=False
            )
            # --------------------
            
            if data.empty:
                print(f"Không có dữ liệu cho {pair}.")
                return pd.DataFrame()
            
            # Đổi tên cột để nhất quán
            data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }, inplace=True)
            
            print("Tải dữ liệu thành công.")
            return data

        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return pd.DataFrame()

# Ví dụ cách sử dụng
if __name__ == '__main__':
    loader = ForexLoader()
    print("--- Tải dữ liệu Forex ---")
    forex_df = loader.fetch("EURUSD=X", tf="1d", period="30d")
    if not forex_df.empty:
        print(forex_df.tail())
