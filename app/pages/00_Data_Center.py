import streamlit as st
import pandas as pd
import sys
import os
import ccxt 
import yfinance as yf
import plotly.graph_objects as go

# --- Cấu hình trang ---
st.set_page_config(page_title=" Let TanPham down data", page_icon="🗃️", layout="wide")

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
    st.header("⚙️ Data")
    asset_class = st.radio("Assets:", ["Crypto", "Forex", "Stocks"])

    # Khung thời gian phổ biến
    crypto_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    yfinance_timeframes = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']


    if asset_class == "Crypto":
        symbol = st.text_input("Pairs:", "BTC/USDT")
        tf = st.selectbox("Timeframe:", crypto_timeframes, index=4) 
        limit = st.slider("Number of candles:", 200, 2000, 500)

    elif asset_class == "Forex":
        symbol = st.text_input("Pairs:", "EURUSD=X")
        tf = st.selectbox("Timeframe:", yfinance_timeframes, index=4) 
        limit = st.slider("Number of candles:", 200, 1000, 500)

    else: # Stocks
        symbol = st.text_input("Pairs:", "AAPL")
        tf = st.selectbox("Timeframe:", yfinance_timeframes, index=5)
        limit = st.slider("Number of candles:", 200, 1000, 500)


# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=300)
def load_data(asset, sym, timeframe, data_limit):
    """Dowloading safety."""
    with st.spinner(f"Dowloading for {sym}..."):
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
                st.error(f"Cannot dowload {sym}. API may wrong or you choose wrong parameters.")
                return None
            
            return data.tail(data_limit)
        
        except Exception as e:
            st.error(f"System error when dowloading data for {sym}: {e}")
            return None


# --- Giao diện chính ---
st.title("🗃️ Data Center")
st.markdown("### Dowloading and view raw data.")

if st.sidebar.button("Dowloading data", type="primary"):
    df = load_data(asset_class, symbol, tf, limit)
    
    if df is not None and not df.empty and len(df) > 1:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Data for {symbol} does not have all the required columns. Current columns: {list(df.columns)}")
        else:
            st.success(f"Successfully dowloaded {len(df)} rows of data for {symbol}.")

            st.subheader("Data Overview")

            latest_data = df.iloc[-1]
            previous_data = df.iloc[-2]
            change = latest_data['Close'] - previous_data['Close']
            change_pct = (change / previous_data['Close']) * 100
            
            period_high = df['High'].max()
            period_low = df['Low'].min()
            period_avg_volume = df['Volume'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Close Price", f"${latest_data['Close']:,.4f}", f"{change:,.4f} ({change_pct:.2f}%)")
            col2.metric("Period High", f"${period_high:,.4f}")
            col3.metric("Period Low", f"${period_low:,.4f}")
            col4.metric("Average Volume", f"{period_avg_volume:,.0f}")

            st.subheader("Candlestick Chart")
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol
            )])
            
            fig.update_layout(
                title=f'Candlestick Chart for {symbol}',
                yaxis_title='Price',
                template='plotly_dark',
                height=500,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🔬 Raw data"):
                st.dataframe(df)
    
    elif df is not None and len(df) <= 1:
        st.warning("Not enough data to display (at least 2 rows are required).")

    else:
        st.info("Data dowloading process has completed. If there are errors, notifications will be displayed above.")
else:
    st.info("Please configure and click 'Dowloading data' in the left sidebar.")