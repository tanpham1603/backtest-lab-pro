import streamlit as st
import pandas as pd
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
st.set_page_config(page_title="Data Center", page_icon="🗃️", layout="wide")
st.title("🗃️ Data Center")
st.markdown("### Tải và xem dữ liệu tài chính thô từ nhiều nguồn khác nhau.")

# --- Sidebar để người dùng tùy chỉnh ---
st.sidebar.header("Cấu hình Dữ liệu")
asset_class = st.sidebar.radio("Loại tài sản:", ["Crypto", "Forex", "Stocks"])

if asset_class == "Crypto":
    symbol = st.sidebar.text_input("Cặp giao dịch:", "BTC/USDT")
    tf = st.sidebar.selectbox("Khung thời gian:", ["15m", "1h", "4h", "1d"], index=1)
    limit = st.sidebar.slider("Số nến:", 200, 2000, 500)
elif asset_class == "Forex":
    symbol = st.sidebar.text_input("Cặp giao dịch:", "EURUSD=X")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0)
    limit = st.sidebar.slider("Số nến:", 200, 1000, 500)
else: # Stocks
    symbol = st.sidebar.text_input("Mã cổ phiếu:", "AAPL")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1d", "1h"], index=0)
    limit = st.sidebar.slider("Số nến:", 200, 1000, 500)

# --- Hàm tải dữ liệu với cache (ĐÃ CẬP NHẬT) ---
@st.cache_data(ttl=300) # Cache kết quả trong 5 phút
def load_data(asset, sym, timeframe, data_limit):
    """Tải dữ liệu dựa trên lựa chọn của người dùng một cách an toàn."""
    with st.spinner(f"Đang tải dữ liệu cho {sym}..."):
        try:
            if asset == "Crypto":
                data = CryptoLoader().fetch(sym, timeframe, data_limit)
            else: # Forex và Stocks đều dùng ForexLoader (yfinance)
                if timeframe == '1d':
                    period = f"{data_limit}d"
                else:
                    # yfinance giới hạn data < 1d trong 730 ngày
                    period = f"{min(data_limit // 8, 729)}d"
                data = ForexLoader().fetch(sym, timeframe, period)

            # --- KIỂM TRA AN TOÀN ---
            if data is None or data.empty:
                st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
                return None
            
            return data
            # ----------------------
        except Exception as e:
            st.error(f"Lỗi hệ thống khi tải dữ liệu cho {sym}: {e}")
            return None

# --- Nút để tải và hiển thị dữ liệu (ĐÃ CẬP NHẬT) ---
if st.sidebar.button("Tải Dữ liệu", type="primary"):
    df = load_data(asset_class, symbol, tf, limit)
    
    if df is not None and not df.empty:
        # Thêm kiểm tra cột 'Close' trước khi sử dụng
        if 'Close' not in df.columns:
            st.error(f"Dữ liệu cho {symbol} không có cột 'Close'.")
        else:
            st.success(f"Đã tải thành công {len(df)} dòng dữ liệu cho {symbol}.")
            
            st.subheader("Biểu đồ đường (Line Chart)")
            st.line_chart(df['Close'])
            
            st.subheader("Dữ liệu thô (50 dòng cuối)")
            st.dataframe(df.tail(50))
    else:
        # Thông báo lỗi chi tiết hơn đã được hiển thị bên trong hàm load_data
        st.info("Quá trình tải dữ liệu đã kết thúc. Nếu có lỗi, thông báo sẽ hiển thị ở trên.")
else:
    st.info("👈 Vui lòng cấu hình và nhấn 'Tải Dữ liệu' ở thanh bên trái.")
