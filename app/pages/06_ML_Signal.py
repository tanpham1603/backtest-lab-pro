import streamlit as st
import pandas as pd
import joblib
import sys
import os
import pandas_ta as ta

# --- Thêm đường dẫn để import CryptoLoader ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from loaders.crypto_loader import CryptoLoader
except ImportError as e:
    st.error("🚨 Không tìm thấy module 'CryptoLoader'. Hãy kiểm tra lại cấu trúc thư mục dự án.")
    st.stop()

# --- Cấu hình giao diện Streamlit ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình Random Forest")

# --- Load mô hình chỉ 1 lần (cache) ---
@st.cache_resource
def load_model():
    model_path = os.path.join("app", "ml_signals", "rf_signal.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
    return joblib.load(model_path)

# --- Hàm tính tín hiệu và dữ liệu ---
@st.cache_data(ttl=300, show_spinner="Đang xử lý dữ liệu...")  # Cache dữ liệu trong 5 phút
def get_signal_and_data(symbol: str, timeframe: str, limit: int = 200):
    try:
        # 1. Load model
        model = load_model()

        # 2. Tải dữ liệu
        df = CryptoLoader().fetch(symbol, timeframe, limit)
        if df.empty:
            return None, None

        # 3. Tính chỉ báo kỹ thuật
        df.ta.rsi(length=14, append=True)
        df.ta.sma(length=20, append=True)
        df.rename(columns={'RSI_14': 'RSI', 'SMA_20': 'MA20'}, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            return None, None

        # 4. Dự đoán bằng mô hình
        df['Prediction'] = model.predict(df[["RSI", "MA20"]])
        df['Prediction'] = df['Prediction'].map({1: "BUY", 0: "SELL"})
        signal = df['Prediction'].iloc[-1]

        return signal, df.tail(30)

    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình xử lý: {e}")
        return None, None

# --- Giao diện người dùng ---
st.sidebar.header("⚙️ Cấu hình Tín hiệu")
symbol_input = st.sidebar.text_input("Nhập mã giao dịch:", "BTC/USDT").upper()
tf_input = st.sidebar.selectbox("Khung thời gian:", ["15m", "1h", "4h", "1d"], index=1)

if st.sidebar.button("📈 Lấy tín hiệu"):
    with st.spinner(f"🔍 Phân tích tín hiệu cho {symbol_input}..."):
        current_signal, latest_data = get_signal_and_data(symbol_input, tf_input)

        if current_signal:
            st.subheader(f"Tín hiệu hiện tại cho {symbol_input}")

            if current_signal == "BUY":
                st.success("🟢 **TÍN HIỆU: MUA (BUY)**")
            else:
                st.error("🔴 **TÍN HIỆU: BÁN (SELL)**")

            st.markdown("---")
            st.subheader("📊 Dữ liệu và dự đoán gần nhất")

            def highlight_prediction(s):
                return ['background-color: #2E8B57' if v == "BUY" else 'background-color: #B22222' for v in s]

            st.dataframe(
                latest_data[['Close', 'RSI', 'MA20', 'Prediction']]
                .style.apply(highlight_prediction, subset=['Prediction'])
            )
        else:
            st.warning("⚠️ Không thể tạo tín hiệu cho mã này.")
else:
    st.info("👈 Vui lòng chọn mã và nhấn 'Lấy tín hiệu' để xem dự đoán.")
