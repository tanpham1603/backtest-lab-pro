import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ccxt
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta

# --- Cấu hình trang ---
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# --- Tùy chỉnh CSS ---
st.markdown("""
    <style>
        .main { background-color: #0E1117; }
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

    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    if asset_class == "Crypto":
        symbol = st.text_input("Cặp giao dịch:", "BTC/USDT", key="dashboard_crypto_symbol")
        tf = st.selectbox("Khung thời gian:", common_timeframes, index=4, key="dashboard_crypto_tf")
        limit = st.slider("Số nến:", 100, 2000, 500, key="dashboard_crypto_limit")
        start_date = None # Crypto không cần chọn ngày
        end_date = None
    else: # Forex và Stocks
        symbol = st.text_input("Mã giao dịch:", "EURUSD=X" if asset_class == "Forex" else "AAPL", key="dashboard_stock_symbol")
        tf = st.selectbox("Khung thời gian:", common_timeframes, index=6, key="dashboard_stock_tf")
        limit = st.slider("Số nến:", 100, 2000, 500, key="dashboard_stock_limit")
        
        # NÂNG CẤP: Tự động điều chỉnh khoảng thời gian cho yfinance
        st.subheader("Khoảng thời gian (Tùy chọn)")
        yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
        
        end_date_default = datetime.now().date()
        start_date_default = end_date_default - timedelta(days=365*2) # Mặc định 2 năm
        info_message = ""

        if tf in yf_timeframe_limits:
            day_limit = yf_timeframe_limits[tf]
            start_date_default = end_date_default - timedelta(days=day_limit - 1)
            info_message = f"Gợi ý: Khung {tf} giới hạn trong {day_limit} ngày."

        end_date = st.date_input("Ngày kết thúc", value=end_date_default)
        start_date = st.date_input("Ngày bắt đầu", value=start_date_default)
        
        if info_message:
            st.caption(info_message)

    st.sidebar.subheader("📈 Chỉ báo kỹ thuật")
    show_ma = st.sidebar.checkbox("Moving Averages (20, 50)", value=True)
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)")
    show_rsi = st.sidebar.checkbox("RSI (14)")

# --- Hàm tải dữ liệu ---
@st.cache_data(ttl=300)
def load_dashboard_data(asset, sym, timeframe, data_limit, start_dt, end_dt):
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=data_limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else:
            yf_timeframe_map = {"1w": "1wk"}
            interval = yf_timeframe_map.get(timeframe, timeframe)
            data = yf.download(sym, start=start_dt, end=end_dt, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]
            data = data.tail(data_limit)

        if data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}.")
            return None
        return data
    except Exception as e:
        st.error(f"Lỗi hệ thống khi tải dữ liệu cho {sym}: {e}")
        return None

# --- Giao diện chính ---
st.title("📊 Tổng quan Thị trường")

data = load_dashboard_data(asset_class, symbol, tf, limit, start_date, end_date)

if data is not None and not data.empty:
    st.success(f"Đã tải và hiển thị {len(data)} nến gần nhất cho {symbol}.")
    if show_ma: data.ta.sma(length=20, append=True); data.ta.sma(length=50, append=True)
    if show_bb: data.ta.bbands(length=20, std=2, append=True)
    if show_rsi: data.ta.rsi(length=14, append=True)

    st.subheader("Tổng quan Chu kỳ")
    latest_data = data.iloc[-1]
    change = latest_data['Close'] - data.iloc[-2]['Close']
    change_pct = (change / data.iloc[-2]['Close']) * 100
    
    # NÂNG CẤP: Tính toán giá cao nhất và thấp nhất trong toàn chu kỳ
    period_high = data['High'].max()
    period_low = data['Low'].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá Đóng cửa (Mới nhất)", f"${latest_data['Close']:,.4f}", f"{change:,.4f} ({change_pct:.2f}%)")
    col2.metric(f"Giá Cao nhất ({len(data)} nến)", f"${period_high:,.4f}")
    col3.metric(f"Giá Thấp nhất ({len(data)} nến)", f"${period_low:,.4f}")
    col4.metric("Khối lượng (Mới nhất)", f"{latest_data['Volume']:,.0f}")

    st.subheader(f"Biểu đồ giá và chỉ báo cho {symbol}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Giá'))
    
    if show_ma:
        fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_20'), mode='lines', name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_50'), mode='lines', name='SMA 50', line=dict(color='blue')))
    if show_bb:
        fig.add_trace(go.Scatter(x=data.index, y=data.get('BBU_20_2.0'), mode='lines', name='Upper Band', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data.get('BBL_20_2.0'), mode='lines', name='Lower Band', line=dict(color='gray', dash='dash')))

    fig.update_layout(yaxis_title='Giá', template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        st.subheader("Chỉ báo RSI")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data.get('RSI_14'), mode='lines', name='RSI', line=dict(color='purple')))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red"); rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(template='plotly_dark', height=250)
        st.plotly_chart(rsi_fig, use_container_width=True)

    with st.expander("🔬 Xem Dữ liệu thô"):
        st.dataframe(data.tail(100))
else:
    st.info("👈 Vui lòng cấu hình ở thanh bên trái để xem dữ liệu.")