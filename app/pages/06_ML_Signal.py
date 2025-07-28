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
import plotly.graph_objects as go

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal", page_icon="🤖", layout="wide")

# --- TÙY CHỈNH GIAO DIỆN VỚI CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        /* Card styles */
        .signal-card {
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
            text-align: center;
            border: 1px solid #30363D;
        }
        .buy-signal {
            background-color: rgba(40, 167, 69, 0.2);
            border-left: 5px solid #28a745;
        }
        .sell-signal {
            background-color: rgba(220, 53, 69, 0.2);
            border-left: 5px solid #dc3545;
        }
        .hold-signal {
            background-color: rgba(255, 193, 7, 0.2);
            border-left: 5px solid #ffc107;
            color: white;
        }
        .error-signal {
            background-color: rgba(108, 117, 125, 0.2);
            border-left: 5px solid #6c757d;
        }
        .signal-text {
            font-size: 20px;
            font-weight: bold;
        }
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")
st.markdown("### Dự đoán MUA/BÁN cho ngày tiếp theo dựa trên mô hình được huấn luyện tự động.")


# --- Sidebar để người dùng tùy chỉnh ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("⚙️ Cấu hình Tín hiệu")
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
            data = yf.download(sym, period="2y", interval=timeframe, progress=False, auto_adjust=True)
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
    Hàm này sẽ tự động huấn luyện một mô hình mới, đảm bảo tương thích 100%.
    """
    with st.spinner("Đang huấn luyện mô hình ML lần đầu..."):
        df = data.copy()
        
        df.ta.rsi(length=14, append=True)
        df.ta.sma(length=20, append=True)
        df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
        
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 20:
            st.warning("Không đủ dữ liệu để huấn luyện mô hình.")
            return None

        features = ['RSI', 'MA20']
        X = df[features]
        y = df['target']
        
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
    
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        return "HOLD", "Không đủ dữ liệu để tính toán chỉ báo"

    latest_features = df[["RSI", "MA20"]].iloc[-1:]
    
    prediction = model.predict(latest_features)[0]
    
    if prediction == 1:
        return "BUY", "Tín hiệu MUA được phát hiện"
    else:
        return "SELL", "Tín hiệu BÁN được phát hiện"

# --- Giao diện chính ---
data = load_data_for_signal(asset_class, symbol, tf)

if data is not None:
    model = train_model_on_the_fly(data)
    
    if st.sidebar.button("Lấy tín hiệu", type="primary"):
        with st.spinner(f"Đang phân tích và tạo tín hiệu cho {symbol}..."):
            signal, message = get_ml_signal(data, model)
            
            st.subheader(f"Kết quả cho {symbol}")
            
            col1, col2 = st.columns([1, 2])

            with col1:
                if signal == "BUY":
                    st.metric(label="Tín hiệu 📈", value="MUA", delta="Tích cực")
                elif signal == "SELL":
                    st.metric(label="Tín hiệu 📉", value="BÁN", delta="Tiêu cực", delta_color="inverse")
                elif signal == "HOLD":
                    st.metric(label="Tín hiệu ⏸️", value="GIỮ", delta="Trung tính", delta_color="off")
                else: # ERROR
                    st.metric(label="Tín hiệu ⚠️", value="LỖI", delta="Không xác định", delta_color="off")
                
                signal_class = f"{signal.lower()}-signal"
                st.markdown(f"""
                    <div class="signal-card {signal_class}">
                        <div class="signal-text">{message}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.write("**Biểu đồ giá 30 nến gần nhất**")
                recent_data = data.tail(30)
                fig = go.Figure(data=[go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close']
                )])
                fig.update_layout(
                    template="plotly_dark", 
                    xaxis_rangeslider_visible=False,
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Không thể tạo tín hiệu do không tải được dữ liệu.")

