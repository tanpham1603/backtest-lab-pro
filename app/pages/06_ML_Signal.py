import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import ccxt
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình được huấn luyện tự động.")

# --- Sidebar để người dùng tùy chỉnh ---
st.sidebar.header("⚙️ Cấu hình Tín hiệu")
asset_class = st.sidebar.radio("Loại tài sản:", ["Crypto", "Forex", "Stocks"], key="ml_asset")

if asset_class == "Crypto":
    symbol = st.sidebar.text_input("Nhập mã giao dịch:", "BTC/USDT", key="ml_crypto_symbol")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=2, key="ml_crypto_tf")
else: # Forex và Stocks
    symbol = st.sidebar.text_input("Nhập mã giao dịch:", "EURUSD=X" if asset_class == "Forex" else "AAPL", key="ml_stock_symbol")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1d"], index=0, key="ml_stock_tf")

# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=600)
def load_data_for_signal(asset, sym, timeframe):
    """Tải dữ liệu để tạo tín hiệu một cách an toàn."""
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=500) # Lấy nhiều dữ liệu hơn để huấn luyện
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            data = yf.download(sym, period="2y", interval=timeframe, progress=False)
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

        if data is None or data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}.")
            return None
        
        return data
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu cho {sym}: {e}")
        return None

# --- GIẢI PHÁP MỚI: Huấn luyện mô hình ngay khi chạy ---
@st.cache_resource
def train_model_on_the_fly(data):
    """
    Hàm này sẽ tự động huấn luyện một mô hình mới mỗi khi ứng dụng khởi động.
    Điều này đảm bảo mô hình luôn tương thích 100% với môi trường.
    """
    st.info("Đang huấn luyện mô hình ML mới...")
    df = data.copy()
    
    # 1. Tính toán các đặc trưng (features)
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
    
    # 2. Tạo mục tiêu dự đoán (target)
    # Nếu giá đóng cửa ngày mai > giá đóng cửa hôm nay -> 1 (Mua), ngược lại -> 0 (Bán)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    if len(df) < 20:
        st.warning("Không đủ dữ liệu để huấn luyện mô hình.")
        return None

    # 3. Chuẩn bị dữ liệu
    features = ['RSI', 'MA20']
    X = df[features]
    y = df['target']
    
    # 4. Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    st.success("Huấn luyện mô hình thành công!")
    return model

# --- Hàm tạo tín hiệu ML (đã được cập nhật) ---
def get_ml_signal(data, model):
    """Tạo tín hiệu từ dữ liệu và mô hình đã được huấn luyện."""
    if model is None:
        return "ERROR", "Mô hình chưa được huấn luyện"

    df = data.copy()
    
    # Tính toán các đặc trưng cho dữ liệu mới nhất
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        return "HOLD", "Không đủ dữ liệu để tính toán chỉ báo"

    # Lấy dòng dữ liệu cuối cùng để dự đoán
    latest_features = df[["RSI", "MA20"]].iloc[-1:]
    
    # Đưa ra dự đoán
    prediction = model.predict(latest_features)[0]
    
    if prediction == 1:
        return "BUY", "Tín hiệu MUA được phát hiện"
    else:
        return "SELL", "Tín hiệu BÁN được phát hiện"

# --- Giao diện chính ---
data = load_data_for_signal(asset_class, symbol, tf)

if data is not None:
    # Huấn luyện hoặc lấy mô hình từ cache
    model = train_model_on_the_fly(data)
    
    if st.sidebar.button("Lấy tín hiệu", type="primary"):
        with st.spinner(f"Đang phân tích và tạo tín hiệu cho {symbol}..."):
            signal, message = get_ml_signal(data, model)
            
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

