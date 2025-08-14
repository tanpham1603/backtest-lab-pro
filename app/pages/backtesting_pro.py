# 2_📈_Backtesting_Pro.py

import streamlit as st
import pandas as pd
import yfinance as yf
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas_ta as ta

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Par replay", page_icon="📈", layout="wide")

# --- CÁC HÀM XỬ LÝ LOGIC ---

@st.cache_data
def load_and_prepare_data(asset_class, symbol, timeframe):
    # ... (Hàm này giữ nguyên)
    st.write(f"Đang tải dữ liệu cho {symbol}...")
    try:
        if asset_class == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=2000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            interval_map = {'1m':'1m', '5m':'5m', '15m':'15m', '30m':'30m', '1h':'1h', '4h':'4h', '1d':'1d', '1w':'1wk', '1M':'1mo'}
            yf_interval = interval_map.get(timeframe, '1d')
            if yf_interval in ['1m', '5m', '15m', '30m']: period = "60d"
            elif yf_interval == '1h': period = '730d'
            else: period = "10y"
            df = yf.download(symbol, period=period, interval=yf_interval, progress=False)
        
        if df.empty:
            st.error("Cannot load data."); return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).capitalize() for col in df.columns]
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing columns in data. Current columns: {list(df.columns)}"); return None
        return df.iloc[-2000:]
    except Exception as e:
        st.error(f"Error loading data: {e}"); return None

def calculate_indicators(df, selected_indicators):
    # ... (Hàm này giữ nguyên)
    if 'SMA' in selected_indicators: df.ta.sma(length=20, append=True)
    if 'EMA' in selected_indicators: df.ta.ema(length=50, append=True)
    if 'RSI' in selected_indicators: df.ta.rsi(length=14, append=True)
    return df

def init_session_state():
    # ... (Hàm này giữ nguyên)
    st.session_state.replay_index = 50
    st.session_state.is_playing = False
    st.session_state.position_type = None
    st.session_state.entry_price = 0
    st.session_state.stop_loss = 0
    st.session_state.take_profit = 0
    st.session_state.trade_size = 0
    st.session_state.balance = 10000.0
    st.session_state.equity = 10000.0

def handle_open_position(pos_type, current_price, trade_size, sl_pct, tp_pct):
    # ... (Hàm này giữ nguyên)
    st.session_state.position_type = pos_type
    st.session_state.entry_price = current_price
    st.session_state.trade_size = trade_size
    if pos_type == 'long':
        st.session_state.stop_loss = current_price * (1 - sl_pct / 100)
        st.session_state.take_profit = current_price * (1 + tp_pct / 100)
        st.toast(f"Open BUY {trade_size} units", icon="🟢")
    else:
        st.session_state.stop_loss = current_price * (1 + sl_pct / 100)
        st.session_state.take_profit = current_price * (1 - tp_pct / 100)
        st.toast(f"Open SELL {trade_size} units", icon="🔴")

def handle_close_position(exit_price, asset_class):
    # ... (Hàm này giữ nguyên)
    entry_price = st.session_state.entry_price
    pos_size = st.session_state.trade_size
    profit = 0
    contract_size = 100000 if asset_class == 'Forex' else 1
    if st.session_state.position_type == 'long':
        profit = (exit_price - entry_price) * pos_size * contract_size
    elif st.session_state.position_type == 'short':
        profit = (entry_price - exit_price) * pos_size * contract_size
    st.session_state.balance += profit
    st.session_state.equity = st.session_state.balance
    st.session_state.position_type = None
    return profit

# --- KHỞI TẠO ---
if 'balance' not in st.session_state: init_session_state()

st.title("📈 Par replay")
st.markdown("Visually backtest your trading strategy.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Backtest parameters")
    asset_class = st.radio("Asset Class:", ["Crypto", "Forex", "Stocks"], on_change=init_session_state, key="asset_class")

    # ... (Phần cấu hình tài sản giữ nguyên)
    if asset_class == "Crypto":
        symbol = st.text_input("Pairs (CCXT):", "BTC/USDT")
        timeframe = st.selectbox("Timeframe:", ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'], index=4)
    elif asset_class == "Forex":
        symbol = st.text_input("Pairs (Yahoo):", "EURUSD=X")
        timeframe = st.selectbox("Timeframe:", ['1m', '5m', '15m', '30m', '1h', '1d', '1w'], index=4)
    else:
        symbol = st.text_input("Stock Symbol (Yahoo):", "AAPL")
        timeframe = st.selectbox("Timeframe:", ['1h', '1d', '1w'], index=1)

    selected_indicators = st.multiselect("Select Indicators:", ['SMA', 'EMA', 'RSI'], default=['SMA', 'EMA'])

    if st.button("Start / Reload Backtest", type="primary"):
        # Save asset_class to session state for later use
        st.session_state.current_asset_class = asset_class
        st.session_state.full_df = load_and_prepare_data(asset_class, symbol, timeframe)
        if st.session_state.full_df is not None:
            st.session_state.full_df = calculate_indicators(st.session_state.full_df, selected_indicators)
            init_session_state()
            st.success("Data is ready!")

    st.divider()

    if 'full_df' in st.session_state and st.session_state.full_df is not None:
        st.header("📊 Account Management")
        st.metric("Balance", f"${st.session_state.balance:,.2f}")
        st.metric("Equity", f"${st.session_state.equity:,.2f}")
        st.divider()
        st.header("TRADE PANEL")
        
        if asset_class == 'Forex':
            trade_size = st.number_input("Trade Size (Lot)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
        elif asset_class == 'Stocks':
            trade_size = st.number_input("Number of Shares", min_value=1, value=10, step=1)
        else:
            trade_size = st.number_input("Volume (Amount)", min_value=0.00001, value=0.01, step=0.0001, format="%.5f")

        sl_pct = st.number_input("Stop Loss (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        tp_pct = st.number_input("Take Profit (%)", min_value=0.1, max_value=50.0, value=4.0, step=0.1)
        
        current_price = st.session_state.full_df['Close'].iloc[st.session_state.replay_index]

        if not st.session_state.position_type:
            trade_cols = st.columns(2)
            if trade_cols[0].button("BUY", use_container_width=True):
                handle_open_position('long', current_price, trade_size, sl_pct, tp_pct)
                st.rerun()
            if trade_cols[1].button("SELL (SHORT)", use_container_width=True):
                handle_open_position('short', current_price, trade_size, sl_pct, tp_pct)
                st.rerun()
        else:
            if st.button("CLOSE POSITION", use_container_width=True, type="primary"):
                profit = handle_close_position(current_price, st.session_state.current_asset_class)
                st.toast(f"Manually close the order. Profit/Loss: ${profit:,.2f}", icon="💰")
                st.rerun()

# --- MAIN LOOP ---
if 'full_df' in st.session_state and st.session_state.full_df is not None:
    current_index = st.session_state.replay_index
    full_df = st.session_state.full_df

    # FIX 1: Logic tính toán Equity mới, chính xác và liên tục
    if st.session_state.position_type:
        current_price = full_df['Close'].iloc[current_index]
        entry_price = st.session_state.entry_price
        trade_size = st.session_state.trade_size
        asset_class_in_trade = st.session_state.get('current_asset_class', 'Crypto') # Lấy asset class của trade hiện tại
        contract_size = 100000 if asset_class_in_trade == 'Forex' else 1
        
        if st.session_state.position_type == 'long':
            unrealized_pnl = (current_price - entry_price) * trade_size * contract_size
        else: # short
            unrealized_pnl = (entry_price - current_price) * trade_size * contract_size
        
        st.session_state.equity = st.session_state.balance + unrealized_pnl
    else:
        st.session_state.equity = st.session_state.balance

    # Logic kiểm tra TP/SL ở mỗi nến
    if st.session_state.position_type:
        current_candle = full_df.iloc[current_index]
        exit_price = 0; reason = ""
        if st.session_state.position_type == 'long':
            if current_candle['Low'] <= st.session_state.stop_loss: exit_price, reason = st.session_state.stop_loss, "Stop Loss"
            elif current_candle['High'] >= st.session_state.take_profit: exit_price, reason = st.session_state.take_profit, "Take Profit"
        elif st.session_state.position_type == 'short':
            if current_candle['High'] >= st.session_state.stop_loss: exit_price, reason = st.session_state.stop_loss, "Stop Loss"
            elif current_candle['Low'] <= st.session_state.take_profit: exit_price, reason = st.session_state.take_profit, "Take Profit"
        
        if exit_price > 0:
            profit = handle_close_position(exit_price, st.session_state.current_asset_class)
            st.toast(f"Manually close the order due to {reason}. Profit/Loss: ${profit:,.2f}", icon="🤖")

    # Control Panel
    ctrl_cols = st.columns([1, 1, 1, 4])
    if ctrl_cols[0].button("▶️", help="Start"): st.session_state.is_playing = True
    if ctrl_cols[1].button("⏸️", help="Pause"): st.session_state.is_playing = False
    if ctrl_cols[2].button("⏩", help="Next"):
        if current_index < len(full_df) - 1: st.session_state.replay_index += 1
        else: st.toast("Reached the end of the data!")
    # FEATURE 2: Add a toggle button to control the chart view mode
    zoom_on_candles = ctrl_cols[3].toggle("Zoom In on Candles", value=True, help="On: Zoom in on candles. Off: View both SL/TP.")

    speed = st.slider("Speed", 0.05, 1.0, 0.2, 0.05)

    # Draw chart
    replay_df = full_df.iloc[:current_index + 1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=replay_df.index, open=replay_df['Open'], high=replay_df['High'], low=replay_df['Low'], close=replay_df['Close'], name='OHLC'), row=1, col=1)
    
    if 'SMA_20' in replay_df.columns: fig.add_trace(go.Scatter(x=replay_df.index, y=replay_df['SMA_20'], mode='lines', name='SMA 20'), row=1, col=1)
    if 'EMA_50' in replay_df.columns: fig.add_trace(go.Scatter(x=replay_df.index, y=replay_df['EMA_50'], mode='lines', name='EMA 50'), row=1, col=1)
    if 'RSI_14' in replay_df.columns: fig.add_trace(go.Scatter(x=replay_df.index, y=replay_df['RSI_14'], mode='lines', name='RSI 14'), row=2, col=1)
    
    if st.session_state.position_type:
        fig.add_hline(y=st.session_state.take_profit, line_dash="dot", line_color="green", annotation_text="Take Profit")
        fig.add_hline(y=st.session_state.stop_loss, line_dash="dot", line_color="red", annotation_text="Stop Loss")
        fig.add_hline(y=st.session_state.entry_price, line_dash="dash", line_color="blue", annotation_text="Entry Price")

    fig.update_layout(title_text=f"Replay for {symbol}", template='plotly_dark', xaxis_rangeslider_visible=False, height=500)

    # FEATURE 2: Apply chart zoom logic
    if zoom_on_candles:
        visible_low = replay_df['Low'].iloc[-100:].min() # Only consider the last 100 candles for smoothness
        visible_high = replay_df['High'].iloc[-100:].max()
        padding = (visible_high - visible_low) * 0.1
        fig.update_layout(yaxis_range=[visible_low - padding, visible_high + padding])
    # If not enabled, Plotly will automatically adjust (autorange) to see both SL/TP

    st.plotly_chart(fig, use_container_width=True)

    # Auto-play logic   
    if st.session_state.is_playing:
        if current_index < len(full_df) - 1:
            st.session_state.replay_index += 1
            time.sleep(speed); st.rerun()
        else:
            st.session_state.is_playing = False; st.toast("Reached the end of the data!"); st.rerun()
else:
    st.info("Please configure and press 'Start / Reload Backtest' to use.")