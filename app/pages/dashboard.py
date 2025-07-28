import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ccxt
import yfinance as yf
import pandas_ta as ta

# --- Cấu hình trang ---
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

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
    st.header("⚙️ Tùy chỉnh Dashboard")
    asset_class = st.selectbox("Loại tài sản:", ["Crypto", "Forex", "Stocks"], key="dashboard_asset")

    if asset_class == "Crypto":
        symbol = st.text_input("Cặp giao dịch:", "BTC/USDT", key="dashboard_crypto_symbol")
        tf = st.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0, key="dashboard_crypto_tf")
        limit = st.slider("Số nến:", 200, 1000, 500, key="dashboard_crypto_limit")
    else: # Forex và Stocks
        symbol = st.text_input("Mã giao dịch:", "EURUSD=X" if asset_class == "Forex" else "AAPL", key="dashboard_stock_symbol")
        tf = st.selectbox("Khung thời gian:", ["1d", "1wk"], index=0, key="dashboard_stock_tf")
        limit = 500 # Giới hạn mặc định cho yfinance để tránh lỗi

    st.sidebar.subheader("📈 Chỉ báo kỹ thuật")
    show_ma = st.sidebar.checkbox("Moving Averages (20, 50)", value=True)
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)")
    show_rsi = st.sidebar.checkbox("RSI (14)")


# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=300)
def load_dashboard_data(asset, sym, timeframe, data_limit):
    """Tải dữ liệu cho dashboard một cách an toàn."""
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=data_limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            period = "2y" if timeframe == "1d" else "1y"
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
st.title("📊 Tổng quan Thị trường")
st.markdown("### Kiểm tra và trực quan hóa dữ liệu thị trường thô.")

data = load_dashboard_data(asset_class, symbol, tf, limit)

if data is not None and not data.empty:
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Dữ liệu cho {symbol} không có đủ các cột cần thiết.")
    else:
        st.success(f"Đã tải thành công {len(data)} dòng dữ liệu cho {symbol}.")

        # Tính toán các chỉ báo nếu được chọn
        if show_ma:
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
        if show_bb:
            data.ta.bbands(length=20, std=2, append=True)
        if show_rsi:
            data.ta.rsi(length=14, append=True)

        # Hiển thị các chỉ số chính
        st.subheader("Tổng quan Dữ liệu Mới nhất")
        latest_data = data.iloc[-1]
        change = latest_data['Close'] - data.iloc[-2]['Close']
        change_pct = (change / data.iloc[-2]['Close']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá Đóng cửa", f"${latest_data['Close']:,.2f}", f"{change:,.2f} ({change_pct:.2f}%)")
        col2.metric("Giá Cao nhất", f"${latest_data['High']:,.2f}")
        col3.metric("Giá Thấp nhất", f"${latest_data['Low']:,.2f}")
        col4.metric("Khối lượng", f"{latest_data['Volume']:,.0f}")


        # Vẽ biểu đồ
        st.subheader(f"Biểu đồ giá và chỉ báo cho {symbol}")
        fig = go.Figure()

        # Thêm biểu đồ nến
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Giá'
        ))

        # Thêm các chỉ báo vào biểu đồ
        if show_ma:
            fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_20'), mode='lines', name='SMA 20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_50'), mode='lines', name='SMA 50', line=dict(color='blue')))
        
        if show_bb:
            fig.add_trace(go.Scatter(x=data.index, y=data.get('BBU_20_2.0'), mode='lines', name='Upper Band', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data.get('BBL_20_2.0'), mode='lines', name='Lower Band', line=dict(color='gray', dash='dash')))

        fig.update_layout(
            yaxis_title='Giá',
            template='plotly_dark',
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

        if show_rsi:
            st.subheader("Chỉ báo RSI")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data.get('RSI_14'), mode='lines', name='RSI', line=dict(color='purple')))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(template='plotly_dark', height=250)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with st.expander("🔬 Xem Dữ liệu thô"):
            st.dataframe(data.tail(100))

else:
    st.warning("Không thể hiển thị biểu đồ do không tải được dữ liệu.")

