import backtrader as bt
from datetime import datetime
import sys
import os

# Thêm đường dẫn gốc của dự án vào sys.path để có thể import các module khác
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- PHẦN SỬA LỖI QUAN TRỌNG ---
# Import các công cụ tải dữ liệu từ các file riêng biệt
from data.forex_loader import ForexLoader
# from data.crypto_loader import CryptoLoader # Bỏ comment nếu bạn muốn dùng crypto
# ---------------------------------

# --- PHẦN 1: ĐỊNH NGHĨA CHIẾN LƯỢC ĐA TÀI SẢN ---
class MultiAssetStrategy(bt.Strategy):
    """
    Một chiến lược ví dụ giao dịch trên nhiều loại tài sản (Forex và Crypto)
    với các quy tắc phí và ký quỹ riêng biệt.
    """
    params = dict(
        period=20
    )

    def __init__(self):
        """Khởi tạo chỉ báo và thiết lập rủi ro cho từng tài sản."""
        
        self.sma = {}
        
        # Lặp qua tất cả các data feed đã được thêm vào cerebro
        for d in self.datas:
            data_name = d._name
            
            # Áp dụng các quy tắc riêng cho từng loại tài sản
            if "USD" in data_name and len(data_name) == 6: # Mẹo để nhận diện Forex
                print(f"Thiết lập thông số cho Forex: {data_name}")
                self.broker.setcommission(
                    name=data_name,
                    mult=100000,
                    commission=0.00002,
                    margin=3333
                )
            elif "USDT" in data_name: # Mẹo để nhận diện Crypto
                print(f"Thiết lập thông số cho Crypto: {data_name}")
                self.broker.setcommission(
                    name=data_name,
                    commission=0.0004,
                    mult=1,
                    margin=None
                )
            
            # Tạo chỉ báo SMA cho từng data feed
            self.sma[d] = bt.ind.SMA(d, period=self.p.period)

    def next(self):
        """Logic chạy cho mỗi nến trên từng data feed."""
        for d in self.datas:
            if not self.getposition(d).size:
                if d.close[0] > self.sma[d][0]:
                    self.buy(data=d, size=1)
            elif d.close[0] < self.sma[d][0]:
                self.close(data=d)

# --- PHẦN 2: HÀM ĐỂ CHẠY BACKTEST ---
def run_multi_asset_backtest():
    """Hàm chính để thiết lập và chạy backtest đa tài sản"""
    
    cerebro = bt.Cerebro()
    
    # --- Thêm dữ liệu Forex bằng cách sử dụng ForexLoader ---
    print("--- Đang nạp dữ liệu Forex ---")
    forex_loader = ForexLoader()
    eurusd_data = forex_loader.get_data_from_csv('EUR_USD')
    
    if eurusd_data is not None:
        cerebro.adddata(eurusd_data, name="EURUSD")
        print("✅ Đã thêm dữ liệu EURUSD.")
    else:
        print("⚠️ Không tìm thấy dữ liệu Forex.")
        
    # Kiểm tra xem có dữ liệu nào được thêm không
    if not cerebro.datas:
        print("\n❌ Lỗi: Không có dữ liệu nào được nạp vào backtester. Dừng chương trình.")
        return

    # Thêm chiến lược
    cerebro.addstrategy(MultiAssetStrategy)
    
    # Thiết lập vốn ban đầu
    cerebro.broker.setcash(100000.0)
    
    # Chạy backtest
    print(f'\nBắt đầu backtest với vốn ban đầu: ${cerebro.broker.getvalue():,.2f}')
    cerebro.run()
    print(f'Kết thúc backtest. Vốn cuối kỳ: ${cerebro.broker.getvalue():,.2f}')
    
    # Vẽ biểu đồ
    print("\nĐang vẽ biểu đồ kết quả...")
    try:
        cerebro.plot(style='candlestick')
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ. Lỗi: {e}")

# --- PHẦN 3: CHẠY CODE ---
if __name__ == '__main__':
    run_multi_asset_backtest()
