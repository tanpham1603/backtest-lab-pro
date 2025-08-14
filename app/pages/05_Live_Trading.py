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
st.set_page_config(page_title="Live Trading Station", page_icon="🛰️", layout="wide")

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
            st.error(f"Lỗi kết nối Alpaca: {e}")

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
            st.success(f"Đã gửi yêu cầu lệnh {side.upper()} {qty} đơn vị {symbol} thành công!")
            return order
        except Exception as e:
            st.error(f"Lỗi khi đặt lệnh: {e}")
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
    st.header("🔌 Kết nối Sàn Giao dịch")
    account_type = st.radio("Chọn loại tài khoản:", ["Paper Trading", "Live Trading"])
    api_key = st.text_input("API Key", type="password", key="api_key_input")
    api_secret = st.text_input("API Secret", type="password", key="api_secret_input")
    
    if st.button("Kết nối", use_container_width=True):
        if api_key and api_secret:
            with st.spinner("Đang kết nối..."):
                st.session_state.trader = AlpacaTrader(api_key.strip(), api_secret.strip(), paper=(account_type == 'Paper Trading'))
            # Không cần rerun() ở đây, Streamlit sẽ tự động chạy lại sau khi button click
        else:
            st.warning("Vui lòng nhập đủ API Key và Secret.")

    # Hiển thị trạng thái kết nối
    if st.session_state.trader and st.session_state.trader.connected:
        st.success(f"✅ Đã kết nối với tài khoản {account_type}!")
    else:
        st.info("Nhập API Key và Secret của Alpaca để bắt đầu.")

# SỬA LỖI: Cấu trúc lại phần hiển thị chính
if st.session_state.trader and st.session_state.trader.connected:
    trader = st.session_state.trader
    
    # Tạo các tab
    tab_titles = ["📊 Tổng quan", "📈 Vị thế", "🛠️ Giao dịch Thủ công", "🤖 Giao dịch Tự động"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
    
    # Tab 1: Tổng quan
    with tab1:
        st.subheader("Tổng quan tài khoản")
        try:
            account = trader.get_account_info()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Giá trị danh mục", f"${float(account.portfolio_value):,.2f}")
            col2.metric("Sức mua", f"${float(account.buying_power):,.2f}")
            col3.metric("Tiền mặt", f"${float(account.cash):,.2f}")
            col4.metric("Trạng thái", account.status.value.upper())
            # ... (Phần biểu đồ giữ nguyên)
        except Exception as e:
            st.error(f"Không thể tải thông tin tài khoản: {e}")

    # Tab 2: Vị thế
    with tab2:
        st.subheader("Các vị thế hiện tại")
        if st.button("Làm mới Vị thế", key="refresh_positions"):
            pass # Streamlit sẽ tự rerun và lấy dữ liệu mới
        try:
            positions = trader.get_positions()
            if positions:
                positions_data = [{"Symbol": p.symbol, "Qty": float(p.qty), "Giá vào lệnh TB": f"{float(p.avg_entry_price):,.2f}", "Giá hiện tại": f"{float(p.current_price):,.2f}", "Lời/Lỗ ($)": f"{float(p.unrealized_pl):,.2f}"} for p in positions]
                st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
            else:
                st.info("Không có vị thế nào đang mở.")
        except Exception as e:
            st.error(f"Không thể tải danh sách vị thế: {e}")
            
    # Tab 3: Giao dịch Thủ công
    with tab3:
        st.subheader("Đặt lệnh Thị trường (Market Order)")
        
        signal_to_execute = st.session_state.get('trade_signal_to_execute', None)
        default_asset_index = 0
        default_sym = "AAPL"
        
        if signal_to_execute:
            asset_map = {"Stocks": 0, "Crypto": 1, "Forex": 2}
            default_asset_index = asset_map.get(signal_to_execute['asset_class'], 0)
            default_sym = signal_to_execute['symbol']

        manual_asset_type = st.radio("Loại tài sản:", ["Stocks", "Crypto", "Forex"], index=default_asset_index, horizontal=True)
        
        if not signal_to_execute:
            if manual_asset_type == "Crypto": default_sym = "BTC/USDT"
            elif manual_asset_type == "Forex": default_sym = "EUR/USD"
            else: default_sym = "AAPL"

        manual_symbol = st.text_input("Mã giao dịch:", value=default_sym, key="manual_symbol_input").upper()
        manual_qty = st.number_input("Số lượng:", min_value=0.00001, value=1.0, step=1.0, format="%.5f")
        
        btn_col1, btn_col2 = st.columns(2)
        
        buy_type = "primary" if (signal_to_execute and signal_to_execute['side'] == 'MUA') else "secondary"
        sell_type = "primary" if (signal_to_execute and signal_to_execute['side'] == 'BÁN') else "secondary"

        if btn_col1.button("MUA (BUY)", use_container_width=True, type=buy_type):
            trader.place_order(symbol=manual_symbol, qty=manual_qty, side="buy", asset_type=manual_asset_type)
        if btn_col2.button("BÁN (SELL)", use_container_width=True, type=sell_type):
            trader.place_order(symbol=manual_symbol, qty=manual_qty, side="sell", asset_type=manual_asset_type)

        if signal_to_execute:
            st.session_state['trade_signal_to_execute'] = None
            st.success(f"Đã điền sẵn thông tin. Vui lòng xác nhận lệnh.")
    
    # Tab 4: Giao dịch Tự động
    with tab4:
        # ... (code tab Giao dịch Tự động giữ nguyên)
        pass
else:
    # Thông báo này chỉ hiển thị khi chưa kết nối
    st.info("👈 Vui lòng kết nối với Sàn Giao dịch Alpaca ở thanh bên trái để bắt đầu.")