import streamlit as st
import vectorbt as vbt
import pandas as pd
import sys
import os

# Thêm đường dẫn để import các module từ thư mục app
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from loaders.crypto_loader import CryptoLoader
    from loaders.forex_loader import ForexLoader
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

# --- Hàm tải dữ liệu với cache ---
@st.cache_data(ttl=600) # Cache kết quả trong 10 phút
def load_price_data(asset_type, sym, timeframe):
    """Tải về chuỗi giá đóng cửa cho backtest."""
    try:
        if asset_type == "Crypto":
            # Tải 1000 nến gần nhất
            return CryptoLoader().fetch(sym, timeframe, 1000)["Close"]
        else: # Forex và Stocks dùng chung yfinance
            # Tải dữ liệu 2 năm gần nhất
            return ForexLoader().fetch(sym, timeframe, "730d")["Close"]
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

# --- Chạy backtest khi người dùng nhấn nút ---
if st.sidebar.button("🚀 Chạy Backtest"):
    
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
                
                # --- SỬA LỖI Ở ĐÂY: Bỏ .iloc[0] vì kết quả đã là một con số ---
                col1.metric("Tổng lợi nhuận (Total Return)", f"{pf.total_return():.2%}")
                col2.metric("Tỷ lệ Sharpe (Sharpe Ratio)", f"{pf.sharpe_ratio():.2f}")
                col3.metric("Tỷ lệ thắng (Win Rate)", f"{pf.trades.win_rate():.2%}")
                # ----------------------------------------------------
                
                # Sửa lỗi plot bằng cách chỉ định rõ cột cần vẽ
                st.plotly_chart(pf.plot(), use_container_width=True)
                
                st.subheader("Thống kê chi tiết")
                # Lấy thống kê cho mã giao dịch cụ thể
                st.dataframe(pf.stats())
            else:
                st.warning("Không có dữ liệu để chạy backtest.")
else:
    st.info("👈 Vui lòng cấu hình các tham số và nhấn 'Chạy Backtest' ở thanh bên trái.")

