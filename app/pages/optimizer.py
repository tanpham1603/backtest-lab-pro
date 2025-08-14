import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import vectorbt as vbt
from itertools import product
from datetime import datetime, timedelta
import ccxt
import yfinance as yf

# --- Cấu hình trang ---
st.set_page_config(page_title="Optimizer", page_icon="⚡", layout="wide")

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
        .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Optimizer")
    asset_class = st.selectbox("Asset Class:", ["Crypto", "Forex", "Stocks"])
    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    if asset_class == "Crypto":
        symbol = st.text_input("Symbol:", "BTC/USDT")
        tf = st.selectbox("Timeframe:", common_timeframes, index=4)
    else:
        symbol = st.text_input("Symbol:", "EURUSD=X" if asset_class == "Forex" else "AAPL")
        tf = st.selectbox("Timeframe:", common_timeframes, index=6)
    st.subheader("Optimal Time Range")
    yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
    end_date_default = datetime.now().date()
    start_date_default = end_date_default - timedelta(days=730)
    info_message = ""
    if asset_class != 'Crypto' and tf in yf_timeframe_limits:
        limit = yf_timeframe_limits[tf]
        start_date_default = end_date_default - timedelta(days=limit - 1)
        info_message = f"Suggestion: {tf} timeframe is limited to {limit} days."
    end_date_input = st.date_input("End date", value=end_date_default)
    start_date_input = st.date_input("Start date", value=start_date_default)
    if info_message: st.caption(info_message)
    st.subheader("Parameter Range")
    fasts = st.multiselect("Fast MA List:", list(range(5, 51, 5)), default=[10, 20])
    slows = st.multiselect("Slow MA List:", list(range(30, 201, 10)), default=[50, 100])
    target_metric = st.selectbox("Target Metric:", ["Total Return [%]", "Sharpe Ratio", "Win Rate [%]", "Profit Factor"])

# --- Hàm tải dữ liệu ---
@st.cache_data(ttl=600)
def load_price_data(asset, sym, timeframe, start, end):
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            since = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=5000)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data = data[data.index <= pd.to_datetime(end)]
        else:
            yf_timeframe_map = {"1w": "1wk"}
            interval = yf_timeframe_map.get(timeframe, timeframe)
            data = yf.download(sym, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]
        if data.empty: return None
        return data["Close"]
    except Exception as e:
        st.error(f"System error while loading data for {sym}: {e}")
        return None

# --- Giao diện chính ---
st.title("⚡ Grid-Search Optimizer for MA-Cross")
st.markdown("### Find the most effective parameter set for the moving average crossover strategy.")
if st.sidebar.button("🚀 Optimize", type="primary"):
    if start_date_input >= end_date_input:
        st.error("Error: Start date must be before end date.")
    else:
        price = load_price_data(asset_class, symbol, tf, start_date_input, end_date_input)
        
        if price is not None and not price.empty:
            if not fasts or not slows:
                 st.warning("Please select at least one value for Fast MA and Slow MA.")
            else:
                with st.spinner(f"Checking scenarios..."):
                    fast_ma = vbt.MA.run(price, window=fasts, short_name='fast')
                    slow_ma = vbt.MA.run(price, window=slows, short_name='slow')
                    entries = fast_ma.ma_crossed_above(slow_ma)
                    exits = fast_ma.ma_crossed_below(slow_ma)
                    vbt_freq = tf.upper().replace('M', 'T')
                    if vbt_freq == '1W': vbt_freq = 'W-MON'
                    pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.001, freq=vbt_freq)
                    
                    # SỬA LỖI: Lấy chỉ số mục tiêu một cách an toàn
                    metric_map = {"Sharpe Ratio": "sharpe_ratio", "Total Return [%]": "total_return", "Win Rate [%]": "trades.win_rate", "Profit Factor": "trades.profit_factor"}
                    perf = pf.deep_getattr(metric_map[target_metric])

                st.header("🏆 Kết quả Tốt nhất")
                best_params_col, best_value_col = perf.idxmax(), perf.max()
                best_stats = pf[best_params_col].stats()
                
                # SỬA LỖI: Lấy giá trị Total Return một cách an toàn
                total_return_val = best_stats.get('Total Return [%]', best_stats.get('Total Return', 0))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Best MA Pair", f"{int(best_params_col[0])} / {int(best_params_col[1])}")
                col2.metric(f"{target_metric}", f"{best_value_col:.2f}")
                col3.metric("Total Return", f"{total_return_val:.2f}%")
                col4.metric("Total Trades", f"{best_stats.get('Total Trades', 0):.0f}")

                st.subheader("📈 Heatmap visualization")
                fig = pf.total_return().vbt.heatmap(x_level='fast_window', y_level='slow_window', title=f"Heatmap of Total Return (%)")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔬 View Detailed Results Table for All Scenarios"):
                    # SỬA LỖI: Hiển thị toàn bộ stats có sẵn, không chỉ định tên cụ thể
                    all_stats_df = pf.stats()
                    st.dataframe(all_stats_df.astype(str))
        else:
            st.warning("No data available to run optimization.")
else:
    st.info("👈 Please configure the parameters and click 'Run Optimization' in the left sidebar.")