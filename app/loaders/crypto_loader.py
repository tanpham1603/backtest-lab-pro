import ccxt
import pandas as pd
import time

class CryptoLoader:
    """
    Lớp để tải dữ liệu crypto từ các sàn giao dịch sử dụng CCXT.
    """
    def __init__(self, exchange_id="binance"):
        """Khởi tạo với một sàn giao dịch cụ thể."""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({'enableRateLimit': True})
            self.exchange.load_markets()
        except Exception as e:
            print(f"Lỗi khi kết nối đến sàn {exchange_id}: {e}")
            self.exchange = None

    def fetch(self, symbol="BTC/USDT", tf="1h", limit=500):
        """
        Tải dữ liệu OHLC và trả về dưới dạng Pandas DataFrame.
        """
        if not self.exchange:
            return pd.DataFrame()

        print(f"Đang tải dữ liệu cho {symbol} khung thời gian {tf}...")
        try:
            raw_data = self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            
            if not raw_data:
                print(f"Không có dữ liệu cho {symbol}.")
                return pd.DataFrame()

            df = pd.DataFrame(raw_data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("Date", inplace=True)
            df.drop(columns="timestamp", inplace=True)
            
            print("Tải dữ liệu Crypto thành công.")
            return df

        except Exception as e:
            print(f"Lỗi khi tải dữ liệu OHLC: {e}")
            return pd.DataFrame()
