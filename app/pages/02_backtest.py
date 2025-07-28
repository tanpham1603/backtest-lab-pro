import streamlit as st
import vectorbt as vbt
import pandas as pd
import numpy as np
import sys
import os
import ccxt
import yfinance as yf

# --- Cấu hình trang ---
st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")

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
    st.header("🎛️ Cấu hình Backtest")
    asset = st.selectbox("Loại tài sản:", ["Crypto", "Forex", "Stocks"])

    if asset == "Crypto":
        symbol = st.text_input("Cặp giao dịch:", "BTC/USDT")
        tf = st.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=1)
    else: # Stocks and Forex
        default_symbol = "AAPL" if asset == "Stocks" else "EURUSD=X"
        symbol = st.text_input("Mã cổ phiếu/Forex:", default_symbol)
        tf = st.selectbox("Khung thời gian:", ["1d"], index=0)

    fast_ma = st.slider("MA Nhanh", 5, 50, 20)
    slow_ma = st.slider("MA Chậm", 10, 200, 50)

# --- Hàm tải dữ liệu ---
@st.cache_data(ttl=600)
def load_price_data(asset_type, sym, timeframe):
    """Tải về dữ liệu giá cho backtest một cách an toàn."""
    try:
        if asset_type == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=1000)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            period = "5y"
            if timeframe not in ['1d', '1wk', '1mo']:
                period = "730d"
            data = yf.download(sym, period=period, interval=timeframe, progress=False, auto_adjust=True)
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

        if data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}.")
            return None
        return data["Close"]
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None

# --- Giao diện chính ---
st.title("🧪 Backtest Chiến lược MA-Cross")
st.markdown("### Phân tích hiệu suất chiến lược giao cắt đường trung bình động trên dữ liệu lịch sử.")

if st.sidebar.button("🚀 Chạy Backtest", type="primary"):
    if fast_ma >= slow_ma:
        st.error("Lỗi: MA Nhanh phải nhỏ hơn MA Chậm.")
    else:
        with st.spinner("⏳ Đang tải dữ liệu và chạy backtest..."):
            price = load_price_data(asset, symbol, tf)
            
            if price is not None and not price.empty:
                fast_ma_series = price.rolling(fast_ma).mean()
                slow_ma_series = price.rolling(slow_ma).mean()
                entries = fast_ma_series > slow_ma_series
                exits = fast_ma_series < slow_ma_series
                
                pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.001, freq=tf)
                
                st.header("📊 Kết quả Backtest")
                
                # Hiển thị các chỉ số chính
                stats = pf.stats()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return [%]", f"{stats['Total Return [%]']:.2f}")
                col2.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
                col3.metric("Win Rate [%]", f"{stats['Win Rate [%]']:.2f}")
                col4.metric("Max Drawdown [%]", f"{stats['Max Drawdown [%]']:.2f}")
                
                # Biểu đồ
                st.subheader("📈 Biểu đồ Lợi nhuận Lũy kế")
                fig = pf.cumulative_returns().vbt.plot()
                st.plotly_chart(fig, use_container_width=True)
                
                # Bảng thống kê chi tiết
                with st.expander("🔬 Xem thống kê chi tiết"):
                    st.dataframe(stats)
            else:
                st.warning("Không có dữ liệu để chạy backtest.")
else:
    st.info("👈 Vui lòng cấu hình các tham số và nhấn 'Chạy Backtest' ở thanh bên trái.")
