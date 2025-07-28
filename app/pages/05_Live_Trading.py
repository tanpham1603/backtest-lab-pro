import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import os
import time
import numpy as np

# --- Import các thành phần cần thiết từ thư viện alpaca-py MỚI ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError # Lớp xử lý lỗi mới

# --- Cấu hình trang ---
st.set_page_config(page_title="Live Trading", page_icon="📈", layout="wide")
st.title("📈 Live Trading (Paper Trading demo)")

# --- Lớp AlpacaTrader để quản lý kết nối và giao dịch ---
class AlpacaTrader:
    def __init__(self, api_key, api_secret):
        try:
            self.api = TradingClient(api_key, api_secret, paper=True)
            self.account = self.api.get_account()
            self.connected = True
        except APIError as e:
            self.api = None
            self.account = None
            self.connected = False
            st.error(f"Lỗi kết nối Alpaca: {e}")
        except Exception as e:
            self.api = None
            self.account = None
            self.connected = False
            st.error(f"Đã xảy ra lỗi không xác định: {e}")

    def get_account_info(self):
        return self.api.get_account()

    def get_positions(self):
        return self.api.get_all_positions()

    def place_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        try:
            order_side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            time_in_force_enum = TimeInForce.GTC if time_in_force.lower() == 'gtc' else TimeInForce.DAY

            if order_type.lower() == 'market':
                market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side_enum,
                    time_in_force=time_in_force_enum
                )
                order = self.api.submit_order(order_data=market_order_data)
                st.success(f"Đã đặt lệnh {side.upper()} {qty} cổ phiếu {symbol} thành công!")
                return order
            else:
                st.error(f"Loại lệnh '{order_type}' chưa được hỗ trợ trong phiên bản này.")
                return None
        except APIError as e:
            st.error(f"Lỗi khi đặt lệnh: {e}")
            return None

# --- Hàm để lấy tín hiệu từ mô hình ML ---
@st.cache_data(ttl=60)
def get_ml_signal(symbol):
    try:
        model_path = "app/ml_signals/rf_signal.pkl"
        if not os.path.exists(model_path):
            st.warning(f"Không tìm thấy mô hình tại: {model_path}. Vui lòng kiểm tra lại đường dẫn.")
            return "HOLD"
        model = joblib.load(model_path)
        data = yf.download(symbol, period="100d", interval="1d", progress=False)
        if data is None or data.empty:
            st.error(f"Không tải được dữ liệu cho {symbol} từ yfinance.")
            return "ERROR"
        data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
        avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data["MA20"] = data["Close"].rolling(20).mean()
        data.dropna(inplace=True)
        if data.empty:
            st.warning("Không đủ dữ liệu để tính toán chỉ báo.")
            return "HOLD"
        latest_features = data[["RSI", "MA20"]].iloc[-1:]
        prediction = model.predict(latest_features)[0]
        return "BUY" if prediction == 1 else "SELL"
    except Exception as e:
        st.error(f"Lỗi khi lấy tín hiệu ML: {e}")
        return "ERROR"

# --- Giao diện Streamlit (TỐI ƯU HÓA VỚI ST.SECRETS) ---

if 'trader' not in st.session_state:
    st.session_state.trader = None

# Cố gắng kết nối tự động bằng st.secrets nếu chưa kết nối
if not st.session_state.trader:
    try:
        if "ALPACA_API_KEY" in st.secrets and "ALPACA_API_SECRET" in st.secrets:
            with st.spinner("Đang kết nối tự động bằng secrets..."):
                st.session_state.trader = AlpacaTrader(st.secrets["ALPACA_API_KEY"], st.secrets["ALPACA_API_SECRET"])
    except Exception:
        pass

# Sidebar để quản lý kết nối
with st.sidebar:
    st.header("Kết nối Alpaca")
    if st.session_state.trader and st.session_state.trader.connected:
        st.success("✅ Đã kết nối với Alpaca!")
    else:
        st.warning("⚠️ Chưa kết nối.")
        st.info("Nhập API Key và Secret để kết nối thủ công.")
        api_key = st.text_input("Nhập Alpaca API Key", type="password", key="manual_api_key")
        api_secret = st.text_input("Nhập Alpaca API Secret", type="password", key="manual_api_secret")
        if st.button("Kết nối thủ công"):
            if api_key and api_secret:
                with st.spinner("Đang kết nối..."):
                    st.session_state.trader = AlpacaTrader(api_key.strip(), api_secret.strip())
                    st.rerun()
            else:
                st.warning("Vui lòng nhập đủ API key và Secret!")

# --- Nội dung chính của trang ---

if st.session_state.trader and st.session_state.trader.connected:
    trader = st.session_state.trader
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "📈 Vị thế", "🤖 Giao dịch tự động"])

    with tab1:
        st.subheader("Tổng quan tài khoản")
        account = trader.get_account_info()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá trị danh mục", f"${float(account.portfolio_value):,.2f}")
        col2.metric("Sức mua", f"${float(account.buying_power):,.2f}")
        col3.metric("Tiền mặt", f"${float(account.cash):,.2f}")
        col4.metric("Trạng thái", account.status.value.upper())

    with tab2:
        st.subheader("Các vị thế hiện tại")
        positions = trader.get_positions()
        if positions:
            positions_data = [{
                "Symbol": p.symbol, "Qty": float(p.qty), "Avg Entry Price": float(p.avg_entry_price),
                "Current Price": float(p.current_price), "P/L": float(p.unrealized_pl)
            } for p in positions]
            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, use_container_width=True)
        else:
            st.info("Không có vị thế nào đang mở.")

    with tab3:
        st.subheader("🤖 Giao dịch tự động dựa trên tín hiệu ML")
        symbol_to_trade = st.text_input("Nhập mã cổ phiếu để theo dõi:", "SPY").upper()
        trade_qty = st.number_input("Số lượng mỗi lệnh:", min_value=1, value=10)
        if st.button(f"Kiểm tra và Giao dịch cho {symbol_to_trade}"):
            with st.spinner(f"Đang lấy tín hiệu cho {symbol_to_trade}..."):
                signal = get_ml_signal(symbol_to_trade)
                st.metric(f"Tín hiệu mới nhất cho {symbol_to_trade}", signal)
                if signal == "BUY":
                    st.info(f"Tín hiệu MUA được phát hiện. Đang đặt lệnh...")
                    trader.place_order(symbol=symbol_to_trade, qty=trade_qty, side="buy")
                elif signal == "SELL":
                    st.warning(f"Tín hiệu BÁN được phát hiện. Đang đặt lệnh bán...")
                    current_positions = [p.symbol for p in trader.get_positions()]
                    if symbol_to_trade in current_positions:
                        trader.place_order(symbol=symbol_to_trade, qty=trade_qty, side="sell")
                    else:
                        st.warning(f"Không có vị thế {symbol_to_trade} để bán.")
                elif signal == "ERROR":
                    st.error("Không thể thực hiện giao dịch do lỗi lấy tín hiệu.")
                else: # HOLD
                    st.write("Tín hiệu là GIỮ. Không có hành động nào được thực hiện.")
else:
    st.info("👈 Vui lòng kết nối với Alpaca bằng cách sử dụng secrets hoặc nhập thông tin vào thanh bên trái.")
