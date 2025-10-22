import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ccxt
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import requests
import os

# ===========================
# ⚙️ CONFIGURATION
# ===========================
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

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

# ===========================
# 🧭 SIDEBAR
# ===========================
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("⚙️ Dashboard setting")

    asset_class = st.selectbox("Asset Class:", ["Crypto", "Forex", "Stocks"], key="dashboard_asset")
    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    if asset_class == "Crypto":
        symbol = st.text_input("Pairs:", "BTC/USDT", key="dashboard_crypto_symbol")
        tf = st.selectbox("Timeframe:", common_timeframes, index=4, key="dashboard_crypto_tf")
        limit = st.slider("Number of candles:", 100, 2000, 500, key="dashboard_crypto_limit")
        start_date = None
        end_date = None
    else:
        symbol = st.text_input("Pairs:", "EURUSD=X" if asset_class == "Forex" else "AAPL", key="dashboard_stock_symbol")
        tf = st.selectbox("Timeframe:", common_timeframes, index=6, key="dashboard_stock_tf")
        limit = st.slider("Number of candles:", 100, 2000, 500, key="dashboard_stock_limit")

        st.subheader("Date Range (Optional)")
        yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
        
        end_date_default = datetime.now().date()
        start_date_default = end_date_default - timedelta(days=365*2)
        info_message = ""

        if tf in yf_timeframe_limits:
            day_limit = yf_timeframe_limits[tf]
            start_date_default = end_date_default - timedelta(days=day_limit - 1)
            info_message = f"Suggestion: {tf} timeframe is limited to {day_limit} days."

        end_date = st.date_input("End date", value=end_date_default)
        start_date = st.date_input("Start date", value=start_date_default)
        if info_message:
            st.caption(info_message)

    st.sidebar.subheader("📈 Technical Indicators")
    show_ma = st.sidebar.checkbox("Moving Averages (20, 50)", value=True)
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)")
    show_rsi = st.sidebar.checkbox("RSI (14)")

# ===========================
# 📦 LOAD DATA FUNCTION
# ===========================
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
            st.error(f"Cannot retrieve data for {sym}.")
            return None
        return data
    except Exception as e:
        st.error(f"System error while loading data for {sym}: {e}")
        return None

# ===========================
# 🧮 DUNE API FETCH FUNCTION
# ===========================
def fetch_dune_query_results(query_id: int):
    """Lấy kết quả từ Dune API"""
    try:
        DUNE_API_KEY = st.secrets["DUNE_API_KEY"]
    except Exception:
        st.error("❌ Missing DUNE_API_KEY in Streamlit secrets.")
        return None

    url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    headers = {"x-dune-api-key": DUNE_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"Dune API error: {response.status_code}")
        return None

    data = response.json()
    if "result" not in data or "rows" not in data["result"]:
        st.warning("⚠️ No data found in Dune query result.")
        return None

    return pd.DataFrame(data["result"]["rows"])

# ===========================
# 🖥️ MAIN UI
# ===========================
st.title("📊 Market Overview")

data = load_dashboard_data(asset_class, symbol, tf, limit, start_date, end_date)

if data is not None and not data.empty:
    st.success(f"✅ Loaded {len(data)} candles for {symbol}.")
    
    # Indicators
    if show_ma:
        data.ta.sma(length=20, append=True)
        data.ta.sma(length=50, append=True)
    if show_bb:
        data.ta.bbands(length=20, std=2, append=True)
    if show_rsi:
        data.ta.rsi(length=14, append=True)

    # Overview
    st.subheader("Cycle Overview")
    latest_data = data.iloc[-1]
    change = latest_data['Close'] - data.iloc[-2]['Close']
    change_pct = (change / data.iloc[-2]['Close']) * 100
    period_high = data['High'].max()
    period_low = data['Low'].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Price", f"${latest_data['Close']:,.4f}", f"{change:,.4f} ({change_pct:.2f}%)")
    col2.metric(f"Highest Price ({len(data)} candles)", f"${period_high:,.4f}")
    col3.metric(f"Lowest Price ({len(data)} candles)", f"${period_low:,.4f}")
    col4.metric("Latest Volume", f"{latest_data['Volume']:,.0f}")

    # Chart
    st.subheader(f"Price Chart and Indicators for {symbol}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Price'))
    if show_ma:
        fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_20'), mode='lines', name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_50'), mode='lines', name='SMA 50', line=dict(color='blue')))
    if show_bb:
        fig.add_trace(go.Scatter(x=data.index, y=data.get('BBU_20_2.0'), mode='lines', name='Upper Band', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data.get('BBL_20_2.0'), mode='lines', name='Lower Band', line=dict(color='gray', dash='dash')))
    fig.update_layout(yaxis_title='Price', template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        st.subheader("RSI Indicator")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data.get('RSI_14'), mode='lines', name='RSI', line=dict(color='purple')))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(template='plotly_dark', height=250)
        st.plotly_chart(rsi_fig, use_container_width=True)

    with st.expander("🔬 View Historical Data"):
        st.dataframe(data.tail(100))

    # ===========================
    # 🐋 WHALE RATIO SECTION
    # ===========================
    st.subheader("🐋 Whale Ratio (Real Data from Dune)")

    dune_query_id = st.text_input("Enter Dune Query ID:", value="", placeholder="e.g. 1234567")

    if dune_query_id:
        df_dune = fetch_dune_query_results(int(dune_query_id))

        if df_dune is not None and not df_dune.empty:
            try:
                total_inflow = df_dune['total_inflow'].sum()
                top10_inflow = df_dune['top10_inflow'].sum()
                whale_ratio = top10_inflow / total_inflow

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Inflow", f"{total_inflow:,.0f}")
                c2.metric("Top 10 Inflow", f"{top10_inflow:,.0f}")
                c3.metric("Whale Ratio", f"{whale_ratio*100:.2f}%")

                st.caption("Whale Ratio = inflow(top 10 wallets) / inflow(total network)")
            except Exception as e:
                st.error(f"Error calculating Whale Ratio: {e}")
        else:
            st.warning("No data retrieved from Dune query.")

else:
    st.info("👈 Please configure the left sidebar to view data.")
