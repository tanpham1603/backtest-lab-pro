import streamlit as st
import pandas as pd
import yfinance as yf
import time
import ccxt
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
import plotly.graph_objects as go
import traceback

# --- Cấu hình trang ---
st.set_page_config(page_title="Live Trading with TanPham", page_icon="🛰️", layout="wide")

# --- Tùy chỉnh CSS ---
st.markdown("""
    <style>
        .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 10px; text-align: center; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Live Trading Station")

# --- Lớp AlpacaTrader ---
class AlpacaTrader:
    def __init__(self, api_key, api_secret, paper=True):
        self.api, self.account, self.connected = None, None, False
        try:
            self.api = TradingClient(api_key, api_secret, paper=paper)
            self.account = self.api.get_account()
            self.connected = True
        except Exception as e:
            st.error(f"Error connecting to Alpaca: {e}")

    def get_account_info(self): return self.api.get_account()
    def get_positions(self): return self.api.get_all_positions()

    def place_order(self, symbol, qty, side, asset_type):
        try:
            if asset_type == "Crypto":
                symbol = symbol.replace('USDT', 'USD')
                if '/' not in symbol and len(symbol) > 3:
                    symbol = f"{symbol.replace('USD', '')}/USD"
            elif asset_type == "Forex":
                if '/' not in symbol and len(symbol) == 6:
                    symbol = f"{symbol[:3]}/{symbol[3:]}"
            
            market_order_data = MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            order = self.api.submit_order(order_data=market_order_data)
            st.success(f"Successfully submitted {side.upper()} order for {qty} units of {symbol}!")
            return order
        except Exception as e:
            st.error(f"Error placing order: {e}")
            return None

# --- Các hàm khác (Giữ nguyên) ---
@st.cache_data(ttl=300)
def load_data_for_live(symbol, asset_type):
    try:
        if asset_type == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=500)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms'); data.set_index('timestamp', inplace=True)
        else:
            data = yf.download(symbol, period="2y", interval='1d', progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu cho {symbol}: {e}")
        return None

@st.cache_resource
def train_model_on_the_fly(data):
    # ... (Giữ nguyên)
    pass

def get_ml_signal(data, model):
    # ... (Giữ nguyên)
    pass

# --- Giao diện Streamlit ---
# Khởi tạo session state
if 'trader' not in st.session_state:
    st.session_state.trader = None

# Sidebar luôn hiển thị
with st.sidebar:
    st.header("🔌 Connect to Trading Platform")
    account_type = st.radio("Select Account Type:", ["Paper Trading", "Live Trading"])
    api_key = st.text_input("API Key", type="password", key="api_key_input")
    api_secret = st.text_input("API Secret", type="password", key="api_secret_input")

    if st.button("Connect", use_container_width=True):
        if api_key and api_secret:
            with st.spinner("Connecting..."):
                st.session_state.trader = AlpacaTrader(api_key.strip(), api_secret.strip(), paper=(account_type == 'Paper Trading'))
            # Không cần rerun() ở đây, Streamlit sẽ tự động chạy lại sau khi button click
        else:
            st.warning("Vui lòng nhập đủ API Key và Secret.")

    # Hiển thị trạng thái kết nối
    if st.session_state.trader and st.session_state.trader.connected:
        st.success(f"✅ Connected to {account_type} account!")
    else:
        st.info("Enter your Alpaca API Key and Secret to get started.")

# FIX: Restructure the main display section
if st.session_state.trader and st.session_state.trader.connected:
    trader = st.session_state.trader

    # Create tabs
    tab_titles = ["📊 Overview", "📈 Positions", "🛠️ Manual Trading", "🤖 Automated Trading"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # Tab 1: Overview
    with tab1:
        st.subheader("Account Overview")
        try:
            account = trader.get_account_info()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio Value", f"${float(account.portfolio_value):,.2f}")
            col2.metric("Buying Power", f"${float(account.buying_power):,.2f}")
            col3.metric("Cash", f"${float(account.cash):,.2f}")
            col4.metric("Status", account.status.value.upper())
            # ... (Phần biểu đồ giữ nguyên)
        except Exception as e:
            st.error(f"Cannot load account information: {e}")

    # Tab 2: Positions
    with tab2:
        st.subheader("Current Positions")
        if st.button("Refresh Positions", key="refresh_positions"):
            pass # Streamlit sẽ tự rerun và lấy dữ liệu mới
        try:
            positions = trader.get_positions()
            if positions:
                positions_data = [{"Symbol": p.symbol, "Qty": float(p.qty), "Average Entry Price": f"{float(p.avg_entry_price):,.2f}", "Current Price": f"{float(p.current_price):,.2f}", "Unrealized P/L ($)": f"{float(p.unrealized_pl):,.2f}"} for p in positions]
                st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
            else:
                st.info("No open positions.")
        except Exception as e:
            st.error(f"Cannot load positions list: {e}")

    # Tab 3: Manual Trading
    with tab3:
        st.subheader("Market Order")

        signal_to_execute = st.session_state.get('trade_signal_to_execute', None)
        default_asset_index = 0
        default_sym = "AAPL"
        
        if signal_to_execute:
            asset_map = {"Stocks": 0, "Crypto": 1, "Forex": 2}
            default_asset_index = asset_map.get(signal_to_execute['asset_class'], 0)
            default_sym = signal_to_execute['symbol']

        manual_asset_type = st.radio("Asset Type:", ["Stocks", "Crypto", "Forex"], index=default_asset_index, horizontal=True)

        if not signal_to_execute:
            if manual_asset_type == "Crypto": default_sym = "BTC/USDT"
            elif manual_asset_type == "Forex": default_sym = "EUR/USD"
            else: default_sym = "AAPL"

        manual_symbol = st.text_input("Symbol:", value=default_sym, key="manual_symbol_input").upper()
        manual_qty = st.number_input("Quantity:", min_value=0.00001, value=1.0, step=1.0, format="%.5f")

        btn_col1, btn_col2 = st.columns(2)
        
        buy_type = "primary" if (signal_to_execute and signal_to_execute['side'] == 'MUA') else "secondary"
        sell_type = "primary" if (signal_to_execute and signal_to_execute['side'] == 'BÁN') else "secondary"

        if btn_col1.button("MUA (BUY)", use_container_width=True, type=buy_type):
            trader.place_order(symbol=manual_symbol, qty=manual_qty, side="buy", asset_type=manual_asset_type)
        if btn_col2.button("BÁN (SELL)", use_container_width=True, type=sell_type):
            trader.place_order(symbol=manual_symbol, qty=manual_qty, side="sell", asset_type=manual_asset_type)

        if signal_to_execute:
            st.session_state['trade_signal_to_execute'] = None
            st.success(f"Trade information has been pre-filled. Please confirm the order.")

    # Tab 4: Automated Trading
    with tab4:
        # ... (code tab Automated Trading giữ nguyên)
        pass
else:
    # Thông báo này chỉ hiển thị khi chưa kết nối
    st.info("👈 Please connect to the Alpaca Trading API on the left sidebar to get started.")