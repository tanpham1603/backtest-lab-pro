# Tên tệp: pages/02_backtest.py
import streamlit as st
import vectorbt as vbt
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import yfinance as yf
import time
import pandas_ta as ta
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Backtest with TanPham", page_icon="📈", layout="wide")

# --- TÙY CHỈNH CSS ---
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
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- HÀM TẢI DỮ LIỆU ---
@st.cache_data(ttl=600)
def load_price_data(asset_type, sym, timeframe, start_date, end_date):
    try:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        if asset_type == "Crypto":
            exchange = ccxt.kucoin()
            all_ohlcv = []
            since = int(start_datetime.timestamp() * 1000)
            end_ts = int(end_datetime.timestamp() * 1000)
            
            while since < end_ts:
                ohlcv = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=1000)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(0.2)
            
            if not all_ohlcv: return None
            data = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data = data[~data.index.duplicated(keep='first')]
            data = data.loc[start_datetime:end_datetime]
        
        else: # Forex và Stocks
            yf_timeframe_map = {"1w": "1wk"}
            interval = yf_timeframe_map.get(timeframe, timeframe)
            data = yf.download(sym, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]

        if data.empty: return None
        # SỬA LỖI: Thêm dòng này để xóa thông tin múi giờ
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- GIAO DIỆN CHÍNH ---
st.title("📈 Dashboard Backtest with TanPham")

run_ml_backtest = st.session_state.get('run_ml_backtest', False)

# ==============================================================================
# CHẾ ĐỘ 1: KIỂM CHỨNG CHIẾN LƯỢC ML
# ==============================================================================
if run_ml_backtest:
    st.success("**Function:** Backtest ML signals nowww .")
    info = st.session_state.get('ml_signal_info')
    
    if info and info.get('model'):
        asset = info['asset_class']
        symbol = info['symbol']
        tf = info['timeframe']
        model = info['model']
        # Lấy ngày tháng đã lưu từ trang Tín hiệu ML
        start_date_bt = info['start_date']
        end_date_bt = info['end_date']
        
        with st.sidebar:
            st.header("Parameters for Backtest")
            st.info(f"Asset: **{asset}**\nSymbol: **{symbol}**\nTimeframe: **{tf}**")
            st.write(f"**Backtest Period:**")
            st.write(f"{start_date_bt.strftime('%Y-%m-%d')} to {end_date_bt.strftime('%Y-%m-%d')}")

            if st.button("Back to MA-Cross mode"):
                st.session_state['run_ml_backtest'] = False
                st.rerun()

        with st.spinner("Loading data and validating ML model..."):
            # Use the saved dates to load data
            full_data = load_price_data(asset, symbol, tf, start_date_bt, end_date_bt)

            if full_data is not None and not full_data.empty:
                df = full_data.copy()
                
                # Tạo lại chính xác các đặc trưng như lúc huấn luyện
                df.ta.rsi(length=14, append=True)
                df.ta.sma(length=50, append=True)
                df.ta.sma(length=200, append=True)
                df.ta.adx(length=14, append=True)
                df.rename(columns={"RSI_14": "RSI", "SMA_50": "MA50", "SMA_200": "MA200", "ADX_14": "ADX"}, inplace=True)
                df['Price_vs_MA200'] = np.where(df['Close'] > df['MA200'], 1, -1)
                df.dropna(inplace=True)
                
                features = ['RSI', 'MA50', 'Price_vs_MA200', 'ADX']
                if all(f in df.columns for f in features):
                    predictions = model.predict(df[features])
                    entries = pd.Series(predictions, index=df.index).astype(bool)
                    exits = ~entries
                    
                    pf = vbt.Portfolio.from_signals(df['Close'], entries, exits, fees=0.001, freq=tf.upper().replace('M','T'))
                    stats = pf.stats()
                    
                    st.header("📊 Result ML signals?")
                    st.info(f"**Backtesting periods:** {stats['Start'].strftime('%Y-%m-%d')} to {stats['End'].strftime('%Y-%m-%d')}")
                    st.subheader("Performance Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return [%]", f"{stats.get('Total Return [%]', 0):.2f}")
                    col2.metric("Win Rate [%]", f"{stats.get('Win Rate [%]', 0):.2f}")
                    col3.metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio', 0):.2f}")
                    col4.metric("Max Drawdown [%]", f"{stats.get('Max Drawdown [%]', 0):.2f}")
                    st.divider()
                    st.subheader("Detailed Analysis")
                    plot_col, stats_col = st.columns([2, 1])
                    with plot_col:
                        st.markdown("##### Equity Curve")
                        fig = pf.plot()
                        st.plotly_chart(fig, use_container_width=True)
                    with stats_col:
                        st.markdown("##### Detailed Statistics")
                        stats_display = stats.astype(str)
                        st.dataframe(stats_display)
                else:
                    st.error("Data is insufficient.")
            else:
                st.warning("Cannot load data for the selected period.")
    else:
        st.error("ML model not found. Please return to the 'ML Signals' page and retrain.")
        if st.button("Back to MA-Cross mode"):
            st.session_state['run_ml_backtest'] = False
            st.rerun()
            
    if st.button("End of test"):
        st.session_state['run_ml_backtest'] = False
        st.rerun()

# ==============================================================================
# CHẾ ĐỘ 2: BACKTEST MA-CROSS MẶC ĐỊNH
# ==============================================================================
else:
    with st.sidebar:
        st.header("🎛️ Backtest MA-Cross Configuration")
        asset = st.selectbox("Asset Type:", ["Crypto", "Forex", "Stocks"])
        if asset == "Crypto":
            symbol = st.text_input("Pairs (CCXT):", "BTC/USDT")
            tf = st.selectbox("Timeframe:", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"], index=4)
        else:
            default_symbol = "AAPL" if asset == "Stocks" else "EURUSD=X"
            symbol = st.text_input("Symbol (Yahoo Finance):", default_symbol)
            tf = st.selectbox("Timeframe:", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"], index=6)
        st.subheader("Backtest Period")
        end_date = st.date_input("End Date", value=datetime.now())
        start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))
        st.subheader("Strategy Parameters")
        fast_ma = st.slider("Fast MA", 5, 100, 20)
        slow_ma = st.slider("Slow MA", 20, 250, 50)
        st.subheader("Risk Management")
        initial_cash = st.number_input("Initial Capital", min_value=100, value=10000, step=1000)
        sl_pct = st.slider("Stop Loss (%)", 0.5, 20.0, 2.0, 0.5)
        tp_pct = st.slider("Take Profit (%)", 0.5, 50.0, 4.0, 0.5)
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    if not run_button:
        st.info("👈 Please configure the parameters and click 'Run Backtest' to see the results.")

    if run_button:
        if start_date >= end_date:
            st.error("Error: Start date must be before end date.")
        elif fast_ma >= slow_ma:
            st.error("Error: Fast MA must be less than Slow MA.")
        else:
            with st.spinner("⏳ Loading data and running backtest..."):
                warmup_candles = slow_ma
                time_delta_map = {'1m': timedelta(minutes=1), '5m': timedelta(minutes=5), '15m': timedelta(minutes=15), '30m': timedelta(minutes=30), '1h': timedelta(hours=1), '4h': timedelta(hours=4), '1d': timedelta(days=1), '1w': timedelta(weeks=1)}
                candle_duration = time_delta_map.get(tf, timedelta(days=1))
                estimated_warmup_duration = max(candle_duration * warmup_candles * 1.7, timedelta(days=1))
                data_start_date = start_date - estimated_warmup_duration
                
                full_price_data = load_price_data(asset, symbol, tf, data_start_date, end_date)
                
                if full_price_data is not None and not full_price_data.empty:
                    price = full_price_data['Close']
                    fast_ma_series = price.rolling(fast_ma).mean()
                    slow_ma_series = price.rolling(slow_ma).mean()
                    entries = fast_ma_series > slow_ma_series
                    exits = fast_ma_series < slow_ma_series
                    
                    backtest_price = price.loc[start_date:end_date]
                    backtest_entries = entries.loc[start_date:end_date]
                    backtest_exits = exits.loc[start_date:end_date]
                    
                    if backtest_price.empty:
                        st.error("No data available for the selected period.")
                    else:
                        vbt_freq = tf.upper().replace('M', 'T')
                        if vbt_freq == '1W': vbt_freq = 'W-MON'
                        portfolio = vbt.Portfolio.from_signals(
                            close=backtest_price, entries=backtest_entries, exits=backtest_exits,
                            freq=vbt_freq, init_cash=initial_cash, sl_stop=sl_pct / 100,
                            tp_stop=tp_pct / 100, fees=0.001
                        )
                        stats = portfolio.stats()
                        st.header("📊 Overall Results")
                        st.info(f"**Backtest Period:** {stats['Start'].strftime('%Y-%m-%d')} to {stats['End'].strftime('%Y-%m-%d')}")
                        st.subheader("Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Return [%]", f"{stats.get('Total Return [%]', 0):.2f}")
                        col2.metric("Win Rate [%]", f"{stats.get('Win Rate [%]', 0):.2f}")
                        col3.metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio', 0):.2f}")
                        col4.metric("Max Drawdown [%]", f"{stats.get('Max Drawdown [%]', 0):.2f}")
                        st.divider()
                        st.subheader("Detailed Analysis")
                        plot_col, stats_col = st.columns([2, 1])
                        with plot_col:
                            st.markdown("##### Equity Curve")
                            fig = portfolio.plot()
                            st.plotly_chart(fig, use_container_width=True)
                        with stats_col:
                            st.markdown("##### Detailed Statistics")
                            stats_display = stats.astype(str)
                            st.dataframe(stats_display)
                else:
                    st.warning("No data available to run backtest.")