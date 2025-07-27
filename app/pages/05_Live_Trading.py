import streamlit as st
from alpaca_trade_api.rest import REST, APIError
import pandas as pd
import joblib
import yfinance as yf
import os
import time

# --- Cấu hình trang ---
st.set_page_config(page_title="Live Trading", page_icon="📈", layout="wide")
st.title("📈 Live Trading (Paper Trading demo)")

# --- Lớp AlpacaTrader để quản lý kết nối và giao dịch ---
class AlpacaTrader:
    def __init__(self, api_key, api_secret):
        try:
            # Kết nối đến môi trường paper trading
            self.api = REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets', api_version='v2')
            self.account = self.api.get_account()
            self.connected = True
        except APIError as e:
            self.api = None
            self.account = None
            self.connected = False
            st.error(f"Lỗi kết nối Alpaca: {e}")

    def get_account_info(self):
        return self.account

    def get_positions(self):
        return self.api.list_positions()

    def place_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        """Đặt lệnh giao dịch"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            st.success(f"Đã đặt lệnh {side.upper()} {qty} cổ phiếu {symbol} thành công!")
            return order
        except APIError as e:
            st.error(f"Lỗi khi đặt lệnh: {e}")
            return None

# --- Hàm để lấy tín hiệu từ mô hình ML ---
@st.cache_data(ttl=60) # Cache kết quả trong 60 giây
def get_ml_signal(symbol):
    """
    Hàm này tải dữ liệu mới nhất, tính toán chỉ báo,
    và dùng mô hình ML đã lưu để đưa ra dự đoán.
    """
    try:
        # 1. Tải mô hình đã được huấn luyện
        model_path = "app/ml_signals/rf_signal.pkl"
        if not os.path.exists(model_path):
            st.warning("Chưa có mô hình ML. Vui lòng chạy train_model.py trước.")
            return "HOLD"
            
        model = joblib.load(model_path)

        # 2. Tải dữ liệu 100 ngày gần nhất để tính chỉ báo
        data = yf.download(symbol, period="100d", interval="1d", progress=False)
        if data.empty:
            return "HOLD"

        # 3. Tính toán các đặc trưng (features) giống như lúc train
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
        avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data["MA20"] = data["Close"].rolling(20).mean()
        data.dropna(inplace=True)

        # 4. Lấy dòng dữ liệu cuối cùng để dự đoán
        latest_features = data[["RSI", "MA20"]].iloc[-1:]
        
        # 5. Đưa ra dự đoán
        prediction = model.predict(latest_features)[0]
        
        if prediction == 1:
            return "BUY"
        else:
            return "SELL" # Hoặc "HOLD" tùy logic của bạn

    except Exception as e:
        st.error(f"Lỗi khi lấy tín hiệu ML: {e}")
        return "ERROR"

# --- Giao diện Streamlit ---

# Dùng session_state để lưu trữ kết nối
if 'trader' not in st.session_state:
    st.session_state.trader = None

# Sidebar để nhập API key
with st.sidebar:
    st.header("Kết nối Alpaca")
    api_key = st.text_input("Nhập Alpaca API Key", type="password")
    api_secret = st.text_input("Nhập Alpaca API Secret", type="password")

    if st.button("Kết nối"):
        if api_key and api_secret:
            with st.spinner("Đang kết nối..."):
                # --- SỬA LỖI Ở ĐÂY ---
                # Tự động xóa khoảng trắng ở đầu và cuối key
                cleaned_api_key = api_key.strip()
                cleaned_api_secret = api_secret.strip()
                st.session_state.trader = AlpacaTrader(cleaned_api_key, cleaned_api_secret)
                # -------------------------
        else:
            st.warning("Vui lòng nhập đủ API key và Secret!")

# --- Nội dung chính của trang ---

if st.session_state.trader and st.session_state.trader.connected:
    trader = st.session_state.trader
    account = trader.get_account_info()

    # Tab để hiển thị các thông tin khác nhau
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "📈 Vị thế", "🤖 Giao dịch tự động"])

    with tab1:
        st.subheader("Tổng quan tài khoản")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá trị danh mục", f"${float(account.portfolio_value):,.2f}")
        col2.metric("Sức mua", f"${float(account.buying_power):,.2f}")
        col3.metric("Tiền mặt", f"${float(account.cash):,.2f}")
        col4.metric("Trạng thái", account.status)

    with tab2:
        st.subheader("Các vị thế hiện tại")
        positions = trader.get_positions()
        if positions:
            positions_data = [{"Symbol": p.symbol, "Qty": p.qty, "Avg Entry Price": p.avg_entry_price, "Current Price": p.current_price, "P/L": p.unrealized_pl} for p in positions]
            st.dataframe(positions_data, use_container_width=True)
        else:
            st.info("Không có vị thế nào đang mở.")

    with tab3:
        st.subheader("🤖 Giao dịch tự động dựa trên tín hiệu ML")
        
        symbol_to_trade = st.text_input("Nhập mã cổ phiếu để theo dõi:", "SPY").upper()
        
        if st.button(f"Kiểm tra tín hiệu cho {symbol_to_trade}"):
            with st.spinner(f"Đang lấy tín hiệu cho {symbol_to_trade}..."):
                signal = get_ml_signal(symbol_to_trade)
                st.metric(f"Tín hiệu mới nhất cho {symbol_to_trade}", signal)

                # --- ĐOẠN CODE CỦA BẠN ĐƯỢỢC ĐẶT Ở ĐÂY ---
                if signal == "BUY":
                    st.info(f"Tín hiệu MUA được phát hiện. Đang đặt lệnh...")
                    trader.place_order(symbol=symbol_to_trade, qty=10, side="buy")
                elif signal == "SELL":
                    st.warning(f"Tín hiệu BÁN được phát hiện. Đang đặt lệnh bán...")
                    trader.place_order(symbol=symbol_to_trade, qty=10, side="sell")
                else:
                    st.write("Không có hành động nào được thực hiện.")
                # ----------------------------------------

else:
    st.info("👈 Vui lòng nhập API Key và Secret vào thanh bên trái để kết nối.")
