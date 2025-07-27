import vectorbt as vbt
import pandas as pd
from datetime import datetime, timedelta

def run_ma_cross_comparison():
    """
    Hàm để chạy và so sánh hiệu suất của chiến lược MA Crossover
    trên nhiều cặp FX và Crypto.
    """
    print("Bắt đầu backtest chiến lược MA Crossover với VectorBT...")

    # 1. Tải dữ liệu cho nhiều cặp tiền
    # Lưu ý: VectorBT dùng yfinance, nên ký hiệu FX là 'EURUSD=X'
    # và Crypto là 'BTC-USD'
    symbols = ["EURUSD=X", "GBPUSD=X", "BTC-USD", "ETH-USD"]
    
    # --- SỬA LỖI Ở ĐÂY: Lùi ngày bắt đầu để nằm trong giới hạn 730 ngày của yfinance ---
    # Lấy dữ liệu 1h chỉ có sẵn trong 730 ngày gần nhất
    start_date = datetime.now() - timedelta(days=729)
    
    # Tải dữ liệu giá đóng cửa khung thời gian 1 giờ
    price = vbt.YFData.download(
        symbols,
        start=start_date, 
        interval='1h',
        missing_index='drop'
    ).get('Close')

    # Chuyển đổi kiểu dữ liệu của giá thành float để tương thích với Numba
    price = price.astype(float)
    # --------------------

    print(f"Đã tải thành công dữ liệu cho: {price.columns.tolist()}")

    # 2. Tính toán các chỉ báo
    fast_ma = vbt.MA.run(price, window=20)
    slow_ma = vbt.MA.run(price, window=50)

    # 3. Tạo tín hiệu giao dịch
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    # 4. Chạy backtest
    portfolio = vbt.Portfolio.from_signals(
        price, 
        entries, 
        exits,
        init_cash=10000, # Vốn ban đầu cho mỗi cặp tiền
        fees=0.001,      # Phí giao dịch 0.1%
        freq='1H'        # Tần suất dữ liệu là 1 giờ
    )

    # 5. In kết quả thống kê
    print("\n--- KẾT QUẢ BACKTEST ---")
    
    # --- SỬA LỖI Ở ĐÂY: Dùng .sum() để kiểm tra tổng số giao dịch ---
    if portfolio.trades.count().sum() > 0:
        print(portfolio.stats())
    else:
        print("Không có giao dịch nào được thực hiện trong khoảng thời gian này.")
    # -----------------------------------------------
    
    # (Tùy chọn) Vẽ biểu đồ
    # portfolio.plot().show()

# Chạy hàm chính khi thực thi file này
if __name__ == "__main__":
    run_ma_cross_comparison()
