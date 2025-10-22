import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go

# -----------------------------------------------------------
# 🟢 Giao diện chọn cổ phiếu & tham số
# -----------------------------------------------------------
st.set_page_config(page_title="Backtest Lab Pro", layout="wide")

st.title("📊 Backtest Lab Pro")
st.write("Phân tích kỹ thuật & kiểm tra tín hiệu với các chỉ báo phổ biến")

ticker = st.text_input("Nhập mã cổ phiếu hoặc symbol (VD: AAPL, BTC-USD, VNINDEX):", "AAPL")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Từ ngày", pd.to_datetime("2022-01-01"))
with col2:
    end_date = st.date_input("Đến ngày", pd.to_datetime("today"))

# -----------------------------------------------------------
# 🟢 Tải dữ liệu
# -----------------------------------------------------------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.warning("Không tìm thấy dữ liệu cho mã này.")
    st.stop()

# -----------------------------------------------------------
# 🟢 Tùy chọn chỉ báo kỹ thuật
# -----------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình chỉ báo")

sma_periods = st.sidebar.multiselect("SMA (Simple Moving Average)", [10, 20, 50, 100, 200], [20, 50])
ema_periods = st.sidebar.multiselect("EMA (Exponential Moving Average)", [10, 20, 50, 100, 200])
bb_enabled = st.sidebar.checkbox("Bollinger Bands", value=True)
rsi_enabled = st.sidebar.checkbox("RSI (Relative Strength Index)", value=True)
macd_enabled = st.sidebar.checkbox("MACD", value=True)

# -----------------------------------------------------------
# 🧮 Tính toán các chỉ báo
# -----------------------------------------------------------
if sma_periods:
    for p in sma_periods:
        data[f"SMA_{p}"] = ta.sma(data['Close'], length=p)

if ema_periods:
    for p in ema_periods:
        data[f"EMA_{p}"] = ta.ema(data['Close'], length=p)

if bb_enabled:
    bb = ta.bbands(data['Close'], length=20, std=2)
    data = pd.concat([data, bb], axis=1)

if rsi_enabled:
    data["RSI"] = ta.rsi(data['Close'], length=14)

if macd_enabled:
    macd = ta.macd(data['Close'])
    data = pd.concat([data, macd], axis=1)

# -----------------------------------------------------------
# 📈 Biểu đồ giá
# -----------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"],
    name="Giá"
))

# SMA
for p in sma_periods:
    fig.add_trace(go.Scatter(
        x=data.index, y=data[f"SMA_{p}"], mode='lines', name=f"SMA {p}"
    ))

# EMA
for p in ema_periods:
    fig.add_trace(go.Scatter(
        x=data.index, y=data[f"EMA_{p}"], mode='lines', name=f"EMA {p}"
    ))

# Bollinger Bands
if bb_enabled:
    fig.add_trace(go.Scatter(x=data.index, y=data['BBL_20_2.0'], line=dict(width=1, dash='dot'), name='BB Lower'))
    fig.add_trace(go.Scatter(x=data.index, y=data['BBU_20_2.0'], line=dict(width=1, dash='dot'), name='BB Upper'))

fig.update_layout(
    title=f"Biểu đồ {ticker}",
    xaxis_title="Ngày",
    yaxis_title="Giá",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# 📊 Chỉ báo phụ (RSI & MACD)
# -----------------------------------------------------------
if rsi_enabled:
    st.subheader("RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(height=250, xaxis_title="Ngày", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

if macd_enabled:
    st.subheader("MACD (Moving Average Convergence Divergence)")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD_12_26_9"], mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACDs_12_26_9"], mode="lines", name="Signal"))
    fig_macd.add_trace(go.Bar(x=data.index, y=data["MACDh_12_26_9"], name="Histogram"))
    fig_macd.update_layout(height=250, xaxis_title="Ngày", yaxis_title="Giá trị MACD")
    st.plotly_chart(fig_macd, use_container_width=True)

# -----------------------------------------------------------
# 📋 Hiển thị dữ liệu
# -----------------------------------------------------------
with st.expander("📄 Xem dữ liệu gốc"):
    st.dataframe(data.tail(100))
