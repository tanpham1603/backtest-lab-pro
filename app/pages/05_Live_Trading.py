import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import os
import time
import numpy as np
import ccxt
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

# --- Import các thành phần cần thiết từ thư viện alpaca-py ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

# --- Cấu hình trang ---
st.set_page_config(page_title="Live Trading", page_icon="📈", layout="wide")

# --- Tùy chỉnh CSS ---
st.markdown("""
    <style>
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


st.title("📈 Live Trading (Paper Trading demo)")

# --- Lớp AlpacaTrader (Không thay đổi) ---
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

    def place_order(self, symbol, qty, side):
        try:
            # Đối với Alpaca, symbol crypto cần được định dạng lại (ví dụ: BTC/USDT -> BTCUSD)
            if '/' in symbol:
                symbol = symbol.replace('/', '')

            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            order = self.api.submit_order(order_data=market_order_data)
            st.success(f"Đã đặt lệnh {side.upper()} {qty} cổ phiếu {symbol} thành công!")
            return order
        except APIError as e:
            st.error(f"Lỗi khi đặt lệnh: {e}")
            return None

# --- Hàm tải dữ liệu thông minh ---
@st.cache_data(ttl=600)
def load_data_for_live(symbol):
    """Tự động nhận diện và tải dữ liệu cho Crypto, Forex, Stocks."""
    try:
        if '/' in symbol: # Crypto
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=500)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            data = yf.download(symbol, period="2y", interval='1d', progress=False, auto_adjust=True)
            data.columns = [col.capitalize() for col in data.columns]
        
        if data.empty: return None
        return data
    except Exception:
        return None

# --- Huấn luyện mô hình tự động ---
@st.cache_resource
def train_model_on_the_fly(data):
    """Huấn luyện mô hình mới, đảm bảo tương thích 100%."""
    with st.spinner("Đang huấn luyện mô hình ML lần đầu..."):
        df = data.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.sma(length=20, append=True)
        df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 20: return None
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(df[['RSI', 'MA20']], df['target'])
    return model

# --- Hàm tạo tín hiệu ML ---
def get_ml_signal(data, model):
    if model is None: return "ERROR", "Mô hình chưa được huấn luyện"
    df = data.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
    df.dropna(inplace=True)
    if df.empty: return "HOLD", "Không đủ dữ liệu để tính toán"
    
    latest_features = df[["RSI", "MA20"]].iloc[-1:]
    prediction = model.predict(latest_features)[0]
    return ("BUY", "Tín hiệu MUA") if prediction == 1 else ("SELL", "Tín hiệu BÁN")

# --- Giao diện Streamlit ---
if 'trader' not in st.session_state:
    st.session_state.trader = None

# Kết nối tự động qua secrets
if not st.session_state.trader:
    try:
        if "ALPACA_API_KEY" in st.secrets and "ALPACA_API_SECRET" in st.secrets:
            st.session_state.trader = AlpacaTrader(st.secrets["ALPACA_API_KEY"], st.secrets["ALPACA_API_SECRET"])
    except Exception: pass

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("🔌 Kết nối Alpaca")
    if st.session_state.trader and st.session_state.trader.connected:
        st.success("✅ Đã kết nối!")
    else:
        st.warning("⚠️ Chưa kết nối.")
        api_key = st.text_input("API Key", type="password", key="manual_api_key")
        api_secret = st.text_input("API Secret", type="password", key="manual_api_secret")
        if st.button("Kết nối thủ công"):
            if api_key and api_secret:
                st.session_state.trader = AlpacaTrader(api_key.strip(), api_secret.strip())
                st.rerun()

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
            positions_data = [{"Symbol": p.symbol, "Qty": float(p.qty), "Avg Entry Price": float(p.avg_entry_price), "Current Price": float(p.current_price), "P/L": f"{float(p.unrealized_pl):,.2f}"} for p in positions]
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("Không có vị thế nào đang mở.")

    with tab3:
        st.subheader("🤖 Giao dịch tự động dựa trên tín hiệu ML")
        st.info("Lưu ý: Chức năng này chỉ dùng cho mục đích demo trên tài khoản Paper Trading.")
        
        symbol_to_trade = st.text_input("Nhập mã giao dịch (ví dụ: AAPL, EURUSD=X, BTC/USDT):", "AAPL").upper()
        trade_qty = st.number_input("Số lượng mỗi lệnh:", min_value=0.001, value=10.0, step=1.0, format="%.3f")
        
        if st.button(f"Kiểm tra và Giao dịch cho {symbol_to_trade}", type="primary"):
            data = load_data_for_live(symbol_to_trade)
            if data is not None:
                model = train_model_on_the_fly(data)
                signal, message = get_ml_signal(data, model)
                
                st.metric(f"Tín hiệu cho {symbol_to_trade}", signal)
                
                if signal == "BUY":
                    st.success(f"Phát hiện tín hiệu MUA. Đang đặt lệnh...")
                    trader.place_order(symbol=symbol_to_trade, qty=trade_qty, side="buy")
                elif signal == "SELL":
                    st.warning(f"Phát hiện tín hiệu BÁN. Đang đặt lệnh...")
                    trader.place_order(symbol=symbol_to_trade, qty=trade_qty, side="sell")
                else:
                    st.error("Không thể thực hiện giao dịch do lỗi tín hiệu.")
            else:
                st.error(f"Không thể tải dữ liệu cho {symbol_to_trade} để tạo tín hiệu.")
else:
    st.warning("👈 Vui lòng kết nối với Alpaca để sử dụng các tính năng.")
