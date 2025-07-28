import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import ccxt
import yfinance as yf
import pandas_ta as ta

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình Random Forest")

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
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=100)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            data = yf.download(sym, period="1y", interval=timeframe, progress=False)
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

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
        model_path = "app/ml_signals/rf_signal.pkl"
        if not os.path.exists(model_path):
            st.warning(f"Không tìm thấy mô hình tại: {model_path}.")
            return "ERROR", "Không tìm thấy tệp mô hình"
            
        model = joblib.load(model_path)

        if 'Close' not in data.columns:
            return "ERROR", "Dữ liệu thiếu cột 'Close'"

        data.ta.rsi(length=14, append=True)
        data.ta.sma(length=20, append=True)
        
        data.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
        data.dropna(inplace=True)

        if data.empty:
            return "HOLD", "Không đủ dữ liệu để tính toán chỉ báo"

        latest_features = data[["RSI", "MA20"]].iloc[-1:]
        
        prediction = model.predict(latest_features)[0]
        
        if prediction == 1:
            return "BUY", "Tín hiệu MUA được phát hiện"
        else:
            return "SELL", "Tín hiệu BÁN được phát hiện"

    except (ModuleNotFoundError, AttributeError) as e:
        st.error(f"Lỗi Tương Thích Thư Viện: Không thể tải mô hình `{model_path}`.")
        st.warning(f"Lỗi này xảy ra vì phiên bản của `scikit-learn` hoặc `numpy` trên server không khớp với phiên bản bạn đã dùng để tạo mô hình. Hãy đảm bảo tệp `requirements.txt` đã được cập nhật chính xác và ứng dụng đã được Reboot. Chi tiết lỗi: {e}")
        return "ERROR", "Lỗi tương thích thư viện"
    except Exception as e:
        st.error(f"Lỗi không xác định trong quá trình xử lý: {e}")
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
