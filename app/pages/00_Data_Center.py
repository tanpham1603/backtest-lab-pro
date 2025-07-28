import streamlit as st
import pandas as pd
import sys
import os
import ccxt 
import yfinance as yf

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

# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=300)
def load_data(asset, sym, timeframe, data_limit):
    """Tải dữ liệu một cách an toàn và chi tiết."""
    with st.spinner(f"Đang tải dữ liệu cho {sym}..."):
        try:
            if asset == "Crypto":
                exchange = ccxt.kucoin()
                ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=data_limit)
                data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
            else: # Forex và Stocks
                period = "5y"
                if timeframe not in ['1d', '1wk', '1mo']:
                    period = "730d"
                data = yf.download(sym, period=period, interval=timeframe, progress=False)
                data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

            if data is None or data.empty:
                st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
                return None
            
            return data
        except Exception as e:
            st.error(f"Lỗi hệ thống khi tải dữ liệu cho {sym}: {e}")
            return None

# --- Nút để tải và hiển thị dữ liệu ---
if st.sidebar.button("Tải Dữ liệu", type="primary"):
    df = load_data(asset_class, symbol, tf, limit)
    
    if df is not None and not df.empty:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dữ liệu cho {symbol} không có đủ các cột cần thiết. Các cột hiện có: {list(df.columns)}")
        else:
            st.success(f"Đã tải thành công {len(df)} dòng dữ liệu cho {symbol}.")
            st.subheader("Biểu đồ đường (Line Chart)")
            st.line_chart(df['Close'])
            st.subheader("Dữ liệu thô (50 dòng cuối)")
            st.dataframe(df.tail(50))
    else:
        st.info("Quá trình tải dữ liệu đã kết thúc. Nếu có lỗi, thông báo sẽ hiển thị ở trên.")
else:
    st.info("👈 Vui lòng cấu hình và nhấn 'Tải Dữ liệu' ở thanh bên trái.")
