import streamlit as st
import pandas as pd
import joblib
import sys
import os

# --- Thêm đường dẫn để import các module từ thư mục app ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from loaders.crypto_loader import CryptoLoader
    from loaders.forex_loader import ForexLoader
except ImportError:
    st.error("Lỗi import: Không tìm thấy các file loader. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Xem dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình Random Forest.")

# --- Hàm tải model và dữ liệu, tính toán tín hiệu ---
@st.cache_data(ttl=300) # Cache kết quả trong 5 phút
def get_signal_and_data(asset_class, symbol, timeframe):
    """
    Tải model, dữ liệu, tính toán chỉ báo và trả về tín hiệu cùng với dữ liệu.
    """
    try:
        # 1. Tải mô hình đã được huấn luyện
        model_path = os.path.join(project_root, "ml_signals", "rf_signal.pkl")
        if not os.path.exists(model_path):
            st.error(f"Lỗi: Không tìm thấy file model tại '{model_path}'. Vui lòng chạy 'train_model.py' trước.")
            return None, None
            
        model = joblib.load(model_path)

        # 2. Tải dữ liệu bằng loader phù hợp
        df = pd.DataFrame()
        if asset_class == "Crypto":
            df = CryptoLoader().fetch(symbol, timeframe, 200)
        else: # Forex và Stocks
            df = ForexLoader().fetch(symbol, timeframe, "100d")

        if df.empty:
            st.warning(f"Không có dữ liệu cho mã {symbol}.")
            return None, None

        # 3. Tính toán các đặc trưng (features)
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
        avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df["MA20"] = df['Close'].rolling(20).mean()
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
asset_class_input = st.sidebar.selectbox("Loại tài sản:", ["Stocks", "Crypto", "Forex"])

if asset_class_input == "Crypto":
    symbol_input = st.sidebar.text_input("Mã giao dịch:", "BTC/USDT").upper()
    tf_input = st.sidebar.selectbox("Khung thời gian:", ["15m", "1h", "4h", "1d"], index=1)
elif asset_class_input == "Forex":
    symbol_input = st.sidebar.text_input("Mã giao dịch:", "EURUSD=X").upper()
    tf_input = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=1)
else: # Stocks
    symbol_input = st.sidebar.text_input("Mã giao dịch:", "SPY").upper()
    tf_input = st.sidebar.selectbox("Khung thời gian:", ["1d"], index=0)


if st.sidebar.button("Lấy tín hiệu"):
    with st.spinner(f"Đang phân tích và dự đoán cho {symbol_input}..."):
        # Thêm cảnh báo về mô hình
        st.warning("Lưu ý: Mô hình ML hiện tại được huấn luyện trên dữ liệu chứng khoán (SPY) và có thể không chính xác cho các loại tài sản khác.")
        
        current_signal, latest_data = get_signal_and_data(asset_class_input, symbol_input, tf_input)
        
        if current_signal:
            st.subheader(f"Dự đoán cho {symbol_input}")
            
            if current_signal == "BUY":
                st.success(f"🟢 **TÍN HIỆU HIỆN TẠI: MUA (BUY)**")
            else:
                st.error(f"🔴 **TÍN HIỆN HIỆN TẠI: BÁN (SELL)**")
            
            st.markdown("---")
            st.subheader("Dữ liệu và Dự đoán gần nhất")
            
            def highlight_signal(s):
                return ['background-color: #2E8B57' if v == "BUY" else 'background-color: #B22222' for v in s]

            st.dataframe(latest_data[['Close', 'RSI', 'MA20', 'Prediction']].style.apply(highlight_signal, subset=['Prediction']))
        else:
            st.warning("Không thể tạo tín hiệu.")
else:
    st.info("👈 Vui lòng cấu hình và nhấn 'Lấy tín hiệu' ở thanh bên trái.")
