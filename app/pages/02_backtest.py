import streamlit as st
import vectorbt as vbt
import pandas as pd
import numpy as np
import sys
import os
import ccxt # Thêm ccxt
import yfinance as yf # Thêm yfinance

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

# --- Hàm tải dữ liệu với cache (ĐÃ CẬP NHẬT HOÀN CHỈNH) ---
@st.cache_data(ttl=600) # Cache kết quả trong 10 phút
def load_price_data(asset_type, sym, timeframe):
    """Tải về dữ liệu giá cho backtest một cách an toàn."""
    try:
        if asset_type == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=1000)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks dùng yfinance
            if timeframe not in ['1d', '1wk', '1mo']:
                period = "730d" # yfinance giới hạn dữ liệu intraday
            else:
                period = "5y" # Tải 5 năm cho dữ liệu ngày
            data = yf.download(sym, period=period, interval=timeframe, progress=False)
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

        # --- KIỂM TRA AN TOÀN ---
        if data is None or data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
            return None
        
        if 'Close' not in data.columns:
            st.error(f"Dữ liệu trả về cho {sym} không chứa cột 'Close'.")
            return None
        
        return data["Close"]
        # ----------------------
    except ccxt.BadSymbol as e:
        st.error(f"Lỗi từ CCXT: Mã giao dịch '{sym}' không hợp lệ hoặc không được hỗ trợ trên Kucoin. Lỗi: {e}")
        return None
    except ccxt.NetworkError as e:
        st.error(f"Lỗi mạng CCXT: Không thể kết nối đến sàn giao dịch. Vui lòng thử lại. Lỗi: {e}")
        return None
    except Exception as e:
        st.error(f"Lỗi hệ thống khi tải dữ liệu: {e}")
        return None

# --- Hàm trợ giúp để lấy giá trị số ---
def get_scalar(value):
    """Trích xuất một giá trị số từ một scalar hoặc một Series."""
    if isinstance(value, pd.Series):
        if not value.empty:
            return value.iloc[0]
        return np.nan
    return value

# --- Chạy backtest khi người dùng nhấn nút ---
if st.sidebar.button("🚀 Chạy Backtest", type="primary"):
    
    if fast_ma >= slow_ma:
        st.error("Lỗi: MA Nhanh phải nhỏ hơn MA Chậm.")
    else:
        with st.spinner(f"Đang chạy backtest cho {symbol}..."):
            price = load_price_data(asset, symbol, tf)
            
            if price is not None and not price.empty:
                fast_ma_series = price.rolling(fast_ma).mean()
                slow_ma_series = price.rolling(slow_ma).mean()
                
                entries = fast_ma_series > slow_ma_series
                exits = fast_ma_series < slow_ma_series
                
                pf = vbt.Portfolio.from_signals(
                    price, entries, exits, fees=0.001, freq=tf
                )
                
                st.subheader("Kết quả Backtest")
                
                col1, col2, col3 = st.columns(3)
                
                total_return = get_scalar(pf.total_return())
                sharpe_ratio = get_scalar(pf.sharpe_ratio())
                win_rate = get_scalar(pf.trades.win_rate())

                col1.metric("Tổng lợi nhuận (Total Return)", f"{total_return:.2%}" if not np.isnan(total_return) else "N/A")
                col2.metric("Tỷ lệ Sharpe (Sharpe Ratio)", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")
                col3.metric("Tỷ lệ thắng (Win Rate)", f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A")
                
                st.subheader("Biểu đồ Lợi nhuận Lũy kế")
                fig = pf.cumulative_returns().vbt.plot()
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Thống kê chi tiết")
                st.dataframe(pf.stats())
            else:
                st.warning("Không có dữ liệu để chạy backtest. Quá trình đã dừng lại.")
else:
    st.info("👈 Vui lòng cấu hình các tham số và nhấn 'Chạy Backtest' ở thanh bên trái.")
