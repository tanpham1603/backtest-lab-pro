import streamlit as st
import pandas as pd
import sys
import os
import ccxt 
import yfinance as yf
import plotly.graph_objects as go

# --- Cấu hình trang ---
st.set_page_config(page_title="Data Center", page_icon="🗃️", layout="wide")

# --- Tùy chỉnh CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("⚙️ Cấu hình Dữ liệu")
    asset_class = st.radio("Loại tài sản:", ["Crypto", "Forex", "Stocks"])

    if asset_class == "Crypto":
        symbol = st.text_input("Cặp giao dịch:", "BTC/USDT")
        tf = st.selectbox("Khung thời gian:", ["15m", "1h", "4h", "1d"], index=1)
        limit = st.slider("Số nến:", 200, 2000, 500)
    elif asset_class == "Forex":
        symbol = st.text_input("Cặp giao dịch:", "EURUSD=X")
        tf = st.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0)
        limit = st.slider("Số nến:", 200, 1000, 500)
    else: # Stocks
        symbol = st.text_input("Mã cổ phiếu:", "AAPL")
        tf = st.selectbox("Khung thời gian:", ["1d", "1h"], index=0)
        limit = st.slider("Số nến:", 200, 1000, 500)

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
                data = yf.download(sym, period=period, interval=timeframe, progress=False, auto_adjust=True)
                data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

            if data is None or data.empty:
                st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
                return None
            
            return data
        except Exception as e:
            st.error(f"Lỗi hệ thống khi tải dữ liệu cho {sym}: {e}")
            return None

# --- Giao diện chính ---
st.title("🗃️ Data Center")
st.markdown("### Tải và xem dữ liệu tài chính thô từ nhiều nguồn khác nhau.")

if st.sidebar.button("Tải Dữ liệu", type="primary"):
    df = load_data(asset_class, symbol, tf, limit)
    
    if df is not None and not df.empty:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dữ liệu cho {symbol} không có đủ các cột cần thiết. Các cột hiện có: {list(df.columns)}")
        else:
            st.success(f"Đã tải thành công {len(df)} dòng dữ liệu cho {symbol}.")
            
            # Hiển thị các chỉ số chính bằng st.metric
            st.subheader("Tổng quan Dữ liệu Mới nhất")
            latest_data = df.iloc[-1]
            change = latest_data['Close'] - df.iloc[-2]['Close']
            change_pct = (change / df.iloc[-2]['Close']) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Giá Đóng cửa", f"${latest_data['Close']:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
            col2.metric("Giá Cao nhất", f"${latest_data['High']:.2f}")
            col3.metric("Giá Thấp nhất", f"${latest_data['Low']:.2f}")
            col4.metric("Khối lượng", f"{latest_data['Volume']:,.0f}")

            # Sử dụng Plotly để vẽ biểu đồ nến tương tác
            st.subheader("Biểu đồ Nến (Candlestick Chart)")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol
            )])
            
            fig.update_layout(
                title=f'Biểu đồ giá cho {symbol}',
                yaxis_title='Giá',
                template='plotly_dark',
                height=500,
                xaxis_rangeslider_visible=False # Tắt thanh trượt mặc định để gọn hơn
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Ẩn dữ liệu thô trong expander
            with st.expander("🔬 Xem Dữ liệu thô"):
                st.dataframe(df.tail(100))
    else:
        st.info("Quá trình tải dữ liệu đã kết thúc. Nếu có lỗi, thông báo sẽ hiển thị ở trên.")
else:
    st.info(" Vui lòng cấu hình và nhấn 'Tải Dữ liệu' ở thanh bên trái.")
