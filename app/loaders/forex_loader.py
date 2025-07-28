import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class ForexLoader:
    """
    Lớp để tải dữ liệu Forex và Cổ phiếu từ Yahoo Finance.
    """
    def fetch(self, pair="EURUSD=X", tf="1h", period="100d"):
        """
        Tải dữ liệu và trả về dưới dạng Pandas DataFrame đã được chuẩn hóa.
        
        Args:
            pair (str): Cặp tiền tệ hoặc mã cổ phiếu.
            tf (str): Khung thời gian ('1h', '1d', v.v.).
            period (str): Khoảng thời gian ('100d', '1y', '6mo', v.v.).
        """
        print(f"Đang tải dữ liệu cho {pair}, khung thời gian {tf}, khoảng thời gian {period}...")
        try:
            # --- PHẦN SỬA LỖI ---
            # Sử dụng trực tiếp tham số 'period' của yfinance.
            # Thư viện này có thể tự xử lý các định dạng như '100d', '1y', 'max'.
            # Điều này đơn giản và đáng tin cậy hơn việc tự tính toán ngày bắt đầu/kết thúc.
            data = yf.download(
                tickers=pair, 
                period=period, 
                interval=tf, 
                progress=False
            )
            
            if data.empty:
                print(f"Không có dữ liệu cho {pair} với các tham số đã cho.")
                return pd.DataFrame()
            
            # --- SỬA LỖI QUAN TRỌNG ---
            # Chuẩn hóa tên cột để xử lý cả trường hợp tên cột là tuple (ví dụ: ('Open', ''))
            # hoặc chuỗi thường (ví dụ: 'Open'). Điều này tránh lỗi 'KeyError' và 'AttributeError'.
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

            # Kiểm tra xem các cột cần thiết có tồn tại không
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                 print(f"Dữ liệu tải về thiếu các cột cần thiết. Các cột hiện có: {list(data.columns)}")
                 return pd.DataFrame()

            print("Tải dữ liệu thành công.")
            return data

        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return pd.DataFrame()

# Ví dụ cách sử dụng
if __name__ == '__main__':
    loader = ForexLoader()
    
    print("\n--- Tải dữ liệu Forex (30 ngày) ---")
    forex_df = loader.fetch("EURUSD=X", tf="1d", period="30d")
    if not forex_df.empty:
        print(forex_df.tail())

    print("\n--- Tải dữ liệu Cổ phiếu (1 năm) ---")
    stock_df = loader.fetch("AAPL", tf="1d", period="1y")
    if not stock_df.empty:
        print(stock_df.tail())
