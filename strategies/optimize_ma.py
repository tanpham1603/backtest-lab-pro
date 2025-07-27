import vectorbt as vbt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from numba import njit

# --- PHẦN 1: ĐỊNH NGHĨA LOGIC CHIẾN LƯỢC BẰNG NUMBA ---
# Sử dụng Numba để tăng tốc độ tính toán

# Import trực tiếp các hàm Numba cần thiết.
try:
    from vectorbt.generic.nb import rolling_mean_nb, crossed_above_nb
except ImportError:
    print("Lỗi nghiêm trọng: Không thể import các hàm Numba cần thiết. Vui lòng kiểm tra lại phiên bản vectorbt.")
    exit()

@njit
def ma_cross_nb(close, fast_window, slow_window):
    """
    Hàm tính toán tín hiệu giao cắt MA, được tối ưu hóa bởi Numba.
    """
    # Tính toán hai đường MA
    fast_ma = rolling_mean_nb(close, window=fast_window, minp=1)
    slow_ma = rolling_mean_nb(close, window=slow_window, minp=1)
    
    # Tín hiệu Mua: MA nhanh cắt lên trên MA chậm
    entries = crossed_above_nb(fast_ma, slow_ma)
    
    # Tín hiệu Bán: MA chậm cắt lên trên MA nhanh (tương đương MA nhanh cắt xuống dưới)
    exits = crossed_above_nb(slow_ma, fast_ma)
    
    return entries, exits

# --- PHẦN 2: TẠO INDICATOR TÙY CHỈNH TỪ LOGIC TRÊN ---
# IndicatorFactory giúp đóng gói logic và quản lý tham số
MACrossover = vbt.IndicatorFactory(
    class_name='MACrossover',
    input_names=['close'],
    param_names=['fast_window', 'slow_window'],
    output_names=['entries', 'exits']
).from_apply_func(ma_cross_nb)


# --- PHẦN 3: HÀM TỐI ƯU HÓA ---
def optimize_ma_cross_strategy(
    symbols=["EURUSD=X", "GBPUSD=X", "BTC-USD", "ETH-USD"],
    fast_windows=np.arange(10, 40, 5),
    slow_windows=np.arange(50, 100, 10),
    start_date=None,
    interval='1h'
):
    """
    Hàm để TỐI ƯU HÓA chiến lược MA Crossover trên nhiều cặp FX & Crypto.
    """
    print("="*50)
    print(f"Bắt đầu TỐI ƯU HÓA chiến lược MA Crossover")
    print("="*50)

    # 1. Tải dữ liệu
    if start_date is None:
        start_date = datetime.now() - timedelta(days=729)

    try:
        price = vbt.YFData.download(
            symbols,
            start=start_date,
            interval=interval,
            missing_index='drop'
        ).get('Close')
        
        if price.empty:
            print("\nLỗi: Không tải được dữ liệu.")
            return

        price = price.astype(float)
        print(f"Đã tải thành công dữ liệu cho: {price.columns.tolist()}")

    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình tải dữ liệu: {e}")
        return

    # 2. Chạy indicator tùy chỉnh với tất cả các cặp tham số
    indicator = MACrossover.run(
        price,
        fast_window=fast_windows,
        slow_window=slow_windows,
        param_product=True # Tự động tạo ra mọi sự kết hợp
    )
    
    entries = indicator.entries
    exits = indicator.exits

    # 3. Chạy backtest
    portfolio = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=10000,
        fees=0.001,
        freq=interval
    )

    # 4. Tìm và in ra kết quả tốt nhất
    print("\n--- KẾT QUẢ TỐI ƯU HÓA ---")
    
    # Lấy ra Sharpe Ratio của tất cả các lần chạy
    sharpe_ratios = portfolio.sharpe_ratio()

    if sharpe_ratios.empty:
        print("Không có kết quả nào để phân tích.")
        return
        
    # --- SỬA LỖI Ở ĐÂY: Truy cập level bằng vị trí (0 và 1) thay vì tên ---
    # Lọc ra các kết quả hợp lệ (loại bỏ các cặp MA nhanh >= MA chậm)
    valid_sharpe = sharpe_ratios[sharpe_ratios.index.get_level_values(0) < sharpe_ratios.index.get_level_values(1)]

    if valid_sharpe.empty:
        print("Không có kết quả hợp lệ nào sau khi lọc.")
        return
    # ---------------------------------------------------------

    # Tìm ra cặp tham số có Sharpe Ratio cao nhất từ các kết quả đã lọc
    best_params = valid_sharpe.idxmax()
    best_sharpe = valid_sharpe.max()
    
    print(f"\nKết quả tốt nhất dựa trên Sharpe Ratio:")
    print(f"  - Cặp tham số tối ưu: MA(fast={best_params[0]}, slow={best_params[1]}) cho mã {best_params[2]}")
    print(f"  - Sharpe Ratio cao nhất: {best_sharpe:.2f}")

    # In ra các chỉ số khác của cặp tham số tốt nhất
    best_stats = portfolio[best_params].stats()
    print("\nThống kê chi tiết cho cặp tham số tốt nhất:")
    print(best_stats[['Total Return [%]', 'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades']])


# Chạy hàm chính khi thực thi file này
if __name__ == "__main__":
    optimize_ma_cross_strategy()
