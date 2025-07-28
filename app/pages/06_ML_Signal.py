import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import ccxt
import yfinance as yf

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình Random Forest")

# --- Sidebar để người dùng tùy chỉnh ---
st.sidebar.header("⚙️ Cấu hình Tín hiệu")

asset_class = st.sidebar.radio("Loại tài sản:", ["Crypto", "Forex", "Stocks"])

if asset_class == "Crypto":
    symbol = st.sidebar.text_input("Nhập mã giao dịch:", "BTC/USDT")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=2)
else: # Forex và Stocks
    symbol = st.sidebar.text_input("Nhập mã giao dịch:", "EURUSD=X" if asset_class == "Forex" else "AAPL")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1d"], index=0)


# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=600)
def load_data_for_signal(asset, sym, timeframe):
    """Tải dữ liệu để tạo tín hiệu một cách an toàn."""
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=100) # Cần khoảng 100 nến để tính chỉ báo
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            data = yf.download(sym, period="1y", interval=timeframe, progress=False)
            data.columns = [col.capitalize() for col in data.columns]

        if data is None or data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}.")
            return None
        
        return data

    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu cho {sym}: {e}")
        return None

# --- Hàm tạo tín hiệu ML ---
def get_ml_signal(data):
    """Tạo tín hiệu từ dữ liệu đã được tải."""
    try:
        # 1. Tải mô hình đã được huấn luyện
        # Đảm bảo đường dẫn này đúng trong cấu trúc thư mục của bạn
        model_path = "app/ml_signals/rf_signal.pkl"
        if not os.path.exists(model_path):
            st.warning(f"Không tìm thấy mô hình tại: {model_path}. Vui lòng kiểm tra lại.")
            return "ERROR", "Không tìm thấy tệp mô hình"
            
        model = joblib.load(model_path)

        # 2. Tính toán các đặc trưng (features) giống như lúc train
        # Cần đảm bảo dữ liệu đầu vào có cột 'Close'
        if 'Close' not in data.columns:
            return "ERROR", "Dữ liệu thiếu cột 'Close'"

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
            return "HOLD", "Không đủ dữ liệu để tính toán chỉ báo"

        # 3. Lấy dòng dữ liệu cuối cùng để dự đoán
        latest_features = data[["RSI", "MA20"]].iloc[-1:]
        
        # 4. Đưa ra dự đoán
        prediction = model.predict(latest_features)[0]
        
        if prediction == 1:
            return "BUY", "Tín hiệu MUA được phát hiện"
        else:
            return "SELL", "Tín hiệu BÁN được phát hiện"

    except Exception as e:
        # Bắt lỗi chi tiết hơn
        error_message = f"Lỗi trong quá trình xử lý: {e}"
        st.error(error_message)
        return "ERROR", str(e)


# --- Giao diện chính ---
if st.sidebar.button("Lấy tín hiệu", type="primary"):
    data = load_data_for_signal(asset_class, symbol, tf)
    
    if data is not None:
        with st.spinner(f"Đang phân tích và tạo tín hiệu cho {symbol}..."):
            signal, message = get_ml_signal(data)
            
            st.subheader(f"Kết quả cho {symbol}")
            
            if signal == "BUY":
                st.success(f"TÍN HIỆU: {signal}")
            elif signal == "SELL":
                st.warning(f"TÍN HIỆU: {signal}")
            elif signal == "HOLD":
                st.info(f"TÍN HIỆU: {signal}")
            else: # ERROR
                st.error(f"TÍN HIỆU: {signal}")

            st.write(f"Chi tiết: {message}")
    else:
        st.warning("Không thể tạo tín hiệu do không tải được dữ liệu.")
else:
    st.info("👈 Vui lòng cấu hình và nhấn 'Lấy tín hiệu' ở thanh bên trái.")
