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

    # Khung thời gian phổ biến
    crypto_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    yfinance_timeframes = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']


    if asset_class == "Crypto":
        symbol = st.text_input("Cặp giao dịch:", "BTC/USDT")
        tf = st.selectbox("Khung thời gian:", crypto_timeframes, index=4) 
        limit = st.slider("Số nến:", 200, 2000, 500)
    
    elif asset_class == "Forex":
        symbol = st.text_input("Cặp giao dịch:", "EURUSD=X")
        tf = st.selectbox("Khung thời gian:", yfinance_timeframes, index=4) 
        limit = st.slider("Số nến:", 200, 1000, 500)

    else: # Stocks
        symbol = st.text_input("Mã cổ phiếu:", "AAPL")
        tf = st.selectbox("Khung thời gian:", yfinance_timeframes, index=5)
        limit = st.slider("Số nến:", 200, 1000, 500)


# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=300)
def load_data(asset, sym, timeframe, data_limit):
    """Tải dữ liệu một cách an toàn và chi tiết."""
    with st.spinner(f"Đang tải dữ liệu cho {sym}..."):
        try:
            if asset == "Crypto":
                exchange = ccxt.kucoin() 
                if not exchange.has['fetchOHLCV']:
                    st.error(f"Sàn {exchange.id} không hỗ trợ tải dữ liệu OHLCV.")
                    return None
                ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=data_limit)
                data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
            
            else: # Forex và Stocks
                if timeframe in ['1m', '2m', '5m', '15m', '30m']:
                    period = "7d"
                elif timeframe == '1h':
                    period = '730d'
                else:
                    period = "5y"

                data = yf.download(sym, period=period, interval=timeframe, progress=False, auto_adjust=True)

                # SỬA ĐỔI: Logic xử lý tên cột phức tạp từ yfinance
                # Chuyển đổi các cột dạng ('Close', 'EURUSD=X') thành 'Close'
                new_columns = []
                for col in data.columns:
                    if isinstance(col, tuple):
                        new_columns.append(col[0].capitalize())
                    else:
                        new_columns.append(str(col).capitalize())
                data.columns = new_columns


            if data is None or data.empty:
                st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
                return None
            
            return data.tail(data_limit)
        
        except Exception as e:
            st.error(f"Lỗi hệ thống khi tải dữ liệu cho {sym}: {e}")
            return None


# --- Giao diện chính ---
st.title("🗃️ Data Center")
st.markdown("### Tải và xem dữ liệu tài chính thô từ nhiều nguồn khác nhau.")

if st.sidebar.button("Tải Dữ liệu", type="primary"):
    df = load_data(asset_class, symbol, tf, limit)
    
    if df is not None and not df.empty and len(df) > 1:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dữ liệu cho {symbol} không có đủ các cột cần thiết. Các cột hiện có: {list(df.columns)}")
        else:
            st.success(f"Đã tải thành công {len(df)} dòng dữ liệu cho {symbol}.")
            
            st.subheader("Tổng quan Dữ liệu")
            
            latest_data = df.iloc[-1]
            previous_data = df.iloc[-2]
            change = latest_data['Close'] - previous_data['Close']
            change_pct = (change / previous_data['Close']) * 100
            
            period_high = df['High'].max()
            period_low = df['Low'].min()
            period_avg_volume = df['Volume'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Giá Đóng cửa", f"${latest_data['Close']:,.4f}", f"{change:,.4f} ({change_pct:.2f}%)")
            col2.metric("Giá Cao nhất (Chu kỳ)", f"${period_high:,.4f}")
            col3.metric("Giá Thấp nhất (Chu kỳ)", f"${period_low:,.4f}")
            col4.metric("Khối lượng (TB)", f"{period_avg_volume:,.0f}")

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
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🔬 Xem Dữ liệu thô"):
                st.dataframe(df)
    
    elif df is not None and len(df) <= 1:
        st.warning("Không đủ dữ liệu để hiển thị (cần ít nhất 2 dòng).")
        
    else:
        st.info("Quá trình tải dữ liệu đã kết thúc. Nếu có lỗi, thông báo sẽ hiển thị ở trên.")
else:
    st.info(" Vui lòng cấu hình và nhấn 'Tải Dữ liệu' ở thanh bên trái.")