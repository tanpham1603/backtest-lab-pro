import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Tổng quan Thị trường")

# --- Sidebar để người dùng tùy chỉnh ---
st.sidebar.header("🎛️ Tùy chỉnh Dashboard")

asset_class = st.sidebar.selectbox("Loại tài sản:", ["Stocks", "Forex", "Crypto"])
if asset_class == "Crypto":
    symbol = st.sidebar.text_input("Cặp giao dịch:", "BTC/USDT")
    tf = st.sidebar.selectbox("Khung thời gian:", ["15m", "1h", "4h", "1d"], index=1)
    limit = st.sidebar.slider("Số nến:", 200, 1000, 500)
elif asset_class == "Forex":
    symbol = st.sidebar.text_input("Cặp giao dịch:", "EURUSD=X")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0)
    limit = st.sidebar.slider("Số nến:", 200, 1000, 500)
else: # Stocks
    symbol = st.sidebar.text_input("Mã cổ phiếu:", "AAPL")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1d", "1h"], index=0)
    limit = st.sidebar.slider("Số nến:", 200, 1000, 500)

st.sidebar.subheader("📊 Chỉ báo kỹ thuật")
selected_indicators = []
if st.sidebar.checkbox("Moving Averages (20, 50)", value=True):
    selected_indicators.append("MA")
if st.sidebar.checkbox("Bollinger Bands (20, 2)"):
    selected_indicators.append("BB")
if st.sidebar.checkbox("RSI (14)"):
    selected_indicators.append("RSI")


# --- Hàm tải dữ liệu với cache ---
@st.cache_data(ttl=300) # Cache kết quả trong 5 phút
def load_data(asset, sym, timeframe, data_limit):
    df = pd.DataFrame()
    if asset == "Crypto":
        df = CryptoLoader().fetch(sym, timeframe, data_limit)
    else: # Forex và Stocks đều dùng ForexLoader (yfinance)
        if timeframe == '1d':
            period = f"{data_limit}d"
        else:
            period = f"{min(data_limit // 8, 729)}d" 
        df = ForexLoader().fetch(sym, timeframe, period)
    
    # --- SỬA LỖI Ở ĐÂY: Xử lý trường hợp yfinance trả về MultiIndex ---
    if not df.empty:
        # Nếu tên cột là MultiIndex (ví dụ: [('Open', 'AAPL'), ...]), hãy làm phẳng nó
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Đảm bảo tất cả tên cột đều là string trước khi capitalize
        df.columns = [str(col).capitalize() for col in df.columns]
    # --------------------------------------------------------------------
    return df

# --- Hàm vẽ biểu đồ ---
def create_advanced_candlestick_chart(data, indicators=None):
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error("Dữ liệu tải về thiếu các cột cần thiết (Open, High, Low, Close, Volume).")
        return go.Figure() # Trả về biểu đồ trống

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Biểu đồ giá & Chỉ báo', 'Khối lượng giao dịch', 'RSI')
    )
    
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    ), row=1, col=1)
    
    # Thêm Moving Averages
    if indicators and 'MA' in indicators:
        ma20 = data['Close'].rolling(20).mean()
        ma50 = data['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=data.index, y=ma20, name='MA20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=ma50, name='MA50', line=dict(color='cyan', width=1)), row=1, col=1)
    
    # Thêm Bollinger Bands
    if indicators and 'BB' in indicators:
        ma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        upper_bb = ma20 + (std20 * 2)
        lower_bb = ma20 - (std20 * 2)
        fig.add_trace(go.Scatter(x=data.index, y=upper_bb, name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=lower_bb, name='BB Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    
    # Khối lượng giao dịch
    colors = np.where(data['Close'] < data['Open'], 'red', 'green')
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    # RSI
    if indicators and 'RSI' in indicators:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(template="plotly_dark", height=700)
    return fig

# --- Tải và hiển thị nội dung ---
df = load_data(asset_class, symbol, tf, limit)

st.subheader("Kiểm tra Dữ liệu thô")
st.info("Bảng này giúp kiểm tra xem dữ liệu có được tải về đúng cách hay không.")
st.dataframe(df.head())

if not df.empty:
    st.plotly_chart(create_advanced_candlestick_chart(df, selected_indicators), use_container_width=True)
    
    # --- Hiển thị thống kê ---
    st.subheader("Thống kê hiệu suất")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = ((current_price - prev_price) / prev_price) * 100
    col1.metric("Giá hiện tại", f"${current_price:,.2f}", f"{change:.2f}%")
    
    high_period = df['High'].max()
    col2.metric("Cao nhất kỳ", f"${high_period:,.2f}")
    
    low_period = df['Low'].min()
    col3.metric("Thấp nhất kỳ", f"${low_period:,.2f}")
    
    avg_volume = df['Volume'].mean()
    col4.metric("KLGD Trung bình", f"{avg_volume:,.0f}")
    
    period_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    col5.metric("Lợi nhuận kỳ", f"{period_return:.2f}%")
    
    st.subheader("Dữ liệu gần nhất")
    st.dataframe(df.tail(10))
    
else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại Cặp giao dịch/Mã cổ phiếu.")

