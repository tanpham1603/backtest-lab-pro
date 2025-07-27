import streamlit as st
import pandas as pd
import joblib
import sys
import os
import pandas_ta as ta # Import thư viện mới

# Thêm đường dẫn để import các module từ thư mục app
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from loaders.crypto_loader import CryptoLoader
except ImportError:
    st.error("Lỗi import: Không tìm thấy CryptoLoader. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Xem dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình Random Forest.")

# --- Hàm tải model và dữ liệu, tính toán tín hiệu ---
@st.cache_data(ttl=300) # Cache kết quả trong 5 phút
def get_signal_and_data(symbol, timeframe, limit):
    """
    Tải model, dữ liệu, tính toán chỉ báo và trả về tín hiệu cùng với dữ liệu.
    """
    try:
        # 1. Tải mô hình
        model_path = os.path.join("app", "ml_signals", "rf_signal.pkl")
        if not os.path.exists(model_path):
            st.error(f"Lỗi: Không tìm thấy file model tại '{model_path}'. Vui lòng chạy 'train_model.py' trước.")
            return None, None
            
        model = joblib.load(model_path)

        # 2. Tải dữ liệu
        df = CryptoLoader().fetch(symbol, timeframe, limit)
        if df.empty:
            st.warning(f"Không có dữ liệu cho mã {symbol}.")
            return None, None

        # 3. Tính toán các đặc trưng (features) bằng pandas-ta
        df.ta.rsi(length=14, append=True)
        df.ta.sma(length=20, append=True)
        df.rename(columns={'RSI_14': 'RSI', 'SMA_20': 'MA20'}, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            st.warning("Không đủ dữ liệu để tính toán chỉ báo.")
            return None, None

        # 4. Đưa ra dự đoán
        predictions = model.predict(df[["RSI", "MA20"]])
        df['Prediction'] = ["BUY" if p == 1 else "SELL" for p in predictions]
        signal = df['Prediction'].iloc[-1]
        
        return signal, df.tail(30)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy tín hiệu ML: {e}")
        return None, None

# --- Giao diện Streamlit ---
st.sidebar.header("Cấu hình Tín hiệu")
symbol_input = st.sidebar.text_input("Nhập mã giao dịch:", "BTC/USDT").upper()
tf_input = st.sidebar.selectbox("Khung thời gian:", ["15m", "1h", "4h", "1d"], index=1)

if st.sidebar.button("Lấy tín hiệu"):
    with st.spinner(f"Đang phân tích và dự đoán cho {symbol_input}..."):
        current_signal, latest_data = get_signal_and_data(symbol_input, tf_input, 200)
        
        if current_signal:
            st.subheader(f"Dự đoán cho {symbol_input}")
            
            if current_signal == "BUY":
                st.success(f"🟢 **TÍN HIỆU HIỆN TẠI: MUA (BUY)**")
            else:
                st.error(f"🔴 **TÍN HIỆU HIỆN TẠI: BÁN (SELL)**")
            
            st.markdown("---")
            st.subheader("Dữ liệu và Dự đoán gần nhất")
            
            def highlight_signal(s):
                return ['background-color: #2E8B57' if v == "BUY" else 'background-color: #B22222' for v in s]

            st.dataframe(latest_data[['Close', 'RSI', 'MA20', 'Prediction']].style.apply(highlight_signal, subset=['Prediction']))
        else:
            st.warning("Không thể tạo tín hiệu.")
else:
    st.info("👈 Vui lòng cấu hình và nhấn 'Lấy tín hiệu' ở thanh bên trái.")
