import streamlit as st
import vectorbt as vbt
import pandas as pd
import numpy as np
import sys
import os

# Thêm đường dẫn để import các module từ thư mục app
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from app.loaders.crypto_loader import CryptoLoader
    from app.loaders.forex_loader import ForexLoader
except ImportError:
    st.error("Lỗi import: Không tìm thấy các file loader. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
st.title("🧪 Backtest Chiến lược MA-Cross")

# --- Sidebar để người dùng tùy chỉnh ---
st.sidebar.header("🎛️ Cấu hình Backtest")

asset = st.sidebar.selectbox("Loại tài sản:", ["Crypto", "Forex", "Stocks"])

if asset == "Crypto":
    symbol = st.sidebar.text_input("Cặp giao dịch:", "BTC/USDT")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=1)
elif asset == "Forex":
    symbol = st.sidebar.text_input("Cặp giao dịch:", "EURUSD=X")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0)
else: # Stocks
    symbol = st.sidebar.text_input("Mã cổ phiếu:", "AAPL")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1d"], index=0)

fast_ma = st.sidebar.slider("MA Nhanh", 5, 50, 20)
slow_ma = st.sidebar.slider("MA Chậm", 10, 200, 50)

# --- Hàm tải dữ liệu với cache (ĐÃ CẬP NHẬT) ---
@st.cache_data(ttl=600) # Cache kết quả trong 10 phút
def load_price_data(asset_type, sym, timeframe):
    """Tải về dữ liệu giá cho backtest một cách an toàn."""
    try:
        if asset_type == "Crypto":
            data = CryptoLoader().fetch(sym, timeframe, 1000)
        else: # Forex và Stocks dùng chung yfinance
            data = ForexLoader().fetch(sym, timeframe, "730d")

        # --- KIỂM TRA AN TOÀN ---
        if data is None or data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
            return None
        
        if 'Close' not in data.columns:
            st.error(f"Dữ liệu trả về cho {sym} không chứa cột 'Close'.")
            return None
        
        return data["Close"] # Chỉ trả về cột 'Close' sau khi đã kiểm tra
        # ----------------------
    except Exception as e:
        st.error(f"Lỗi hệ thống khi tải dữ liệu: {e}")
        return None

# --- Hàm trợ giúp để lấy giá trị số từ kết quả của vectorbt ---
def get_scalar(value):
    """Trích xuất một giá trị số từ một scalar hoặc một Series."""
    if isinstance(value, pd.Series):
        if not value.empty:
            return value.iloc[0]
        return np.nan # Trả về NaN nếu Series rỗng
    return value # Trả về chính nó nếu đã là scalar

# --- Chạy backtest khi người dùng nhấn nút ---
if st.sidebar.button("🚀 Chạy Backtest", type="primary"):
    
    # Đảm bảo MA nhanh luôn nhỏ hơn MA chậm
    if fast_ma >= slow_ma:
        st.error("Lỗi: MA Nhanh phải nhỏ hơn MA Chậm.")
    else:
        with st.spinner(f"Đang chạy backtest cho {symbol}..."):
            # Tải dữ liệu
            price = load_price_data(asset, symbol, tf)
            
            if price is not None and not price.empty:
                # Tính toán tín hiệu
                fast_ma_series = price.rolling(fast_ma).mean()
                slow_ma_series = price.rolling(slow_ma).mean()
                
                entries = fast_ma_series > slow_ma_series
                exits = fast_ma_series < slow_ma_series
                
                # Chạy backtest với vectorbt
                pf = vbt.Portfolio.from_signals(
                    price, 
                    entries, 
                    exits, 
                    fees=0.001, # Phí 0.1%
                    freq=tf # Tần suất dữ liệu
                )
                
                # Hiển thị kết quả
                st.subheader("Kết quả Backtest")
                
                col1, col2, col3 = st.columns(3)
                
                # --- SỬA LỖI: Dùng hàm get_scalar để đảm bảo giá trị là số ---
                total_return = get_scalar(pf.total_return())
                sharpe_ratio = get_scalar(pf.sharpe_ratio())
                win_rate = get_scalar(pf.trades.win_rate())

                col1.metric("Tổng lợi nhuận (Total Return)", f"{total_return:.2%}" if not np.isnan(total_return) else "N/A")
                col2.metric("Tỷ lệ Sharpe (Sharpe Ratio)", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")
                col3.metric("Tỷ lệ thắng (Win Rate)", f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A")
                # ----------------------------------------------------
                
                st.plotly_chart(pf.plot(), use_container_width=True)
                
                st.subheader("Thống kê chi tiết")
                st.dataframe(pf.stats())
            else:
                st.warning("Không có dữ liệu để chạy backtest. Quá trình đã dừng lại.")
else:
    st.info("👈 Vui lòng cấu hình các tham số và nhấn 'Chạy Backtest' ở thanh bên trái.")
