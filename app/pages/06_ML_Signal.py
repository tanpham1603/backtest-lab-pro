import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import requests
import time

# --- Cấu hình trang ---
st.set_page_config(page_title="ML Signal Pro", page_icon="🤖", layout="wide")
st.markdown("""<style>.main { background-color: #0E1117; } .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 10px; text-align: center; }</style>""", unsafe_allow_html=True)
st.title("🤖 Tín hiệu Giao dịch từ Machine Learning")

# --- Danh sách CỐ ĐỊNH (Cho Forex/Stocks) ---
FOREX_LIST = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'GC=F', 'CL=F']
STOCKS_LIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'SPY', 'QQQ']

# --- CÁC HÀM XỬ LÝ ---
@st.cache_data(ttl=1800) # Cache 30 phút
def get_market_scan_list(asset_class):
    if asset_class == "Crypto":
        with st.spinner("Đang tải danh sách coin từ CoinGecko..."):
            try:
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                coin_list = [f"{coin.get('symbol', '').upper()}/USDT" for coin in data if coin.get('market_cap') and coin['market_cap'] > 10_000_000 and coin.get('symbol')]
                st.success(f"Đã tải {len(coin_list)} mã Crypto có vốn hóa > $10M.")
                return coin_list
            except Exception as e:
                st.error(f"Lỗi gọi API CoinGecko: {e}. Sử dụng danh sách dự phòng.")
                return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']
    elif asset_class == "Forex": return FOREX_LIST
    else: return STOCKS_LIST

@st.cache_data(ttl=300)
def load_data_for_signal(asset, sym, timeframe, start, end):
    try:
        if asset == "Crypto":
            exchange = ccxt.binance()
            since = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=2000)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data = data[data.index <= pd.to_datetime(end)]
        else:
            yf_timeframe_map = {"1w": "1wk"}
            interval = yf_timeframe_map.get(timeframe, timeframe)
            data = yf.download(sym, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]
        if data.empty: return None
        return data
    except Exception:
        return None

# NÂNG CẤP: Tách riêng hàm huấn luyện để cache
@st.cache_resource
def get_trained_model(asset, sym, timeframe, start, end):
    data = load_data_for_signal(asset, sym, timeframe, start, end)
    if data is None or data.empty or len(data) < 100:
        return None

    df = data.copy()
    df.ta.rsi(length=14, append=True); df.ta.sma(length=20, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    if len(df) < 100: return None
    features = ['RSI', 'MA20']; X = df[features]; y = df['target']
    
    # Sử dụng toàn bộ dữ liệu để huấn luyện (trừ 1 nến cuối)
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    
    return {"model": model, "features": features}

# Hàm lấy tín hiệu (không cache)
def get_ml_signal(data, model_results):
    if model_results is None: return "LỖI", "Mô hình lỗi", None, 0
    model, features = model_results['model'], model_results['features']
    
    df = data.copy()
    df.ta.rsi(length=14, append=True); df.ta.sma(length=20, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_20": "MA20"}, inplace=True)
    latest_features = df[features].dropna().iloc[-1:]
    
    if latest_features.empty: return "GIỮ", "Không đủ dữ liệu mới", None, 0
    
    prediction = model.predict(latest_features)[0]
    proba = model.predict_proba(latest_features)[0]
    confidence = proba[prediction]
    signal_text = "MUA" if prediction == 1 else "BÁN"
    message = f"Tín hiệu {signal_text} (Độ tin cậy: {confidence*100:.1f}%)"
    return signal_text, message, latest_features, confidence

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Cấu hình Tín hiệu")
    scan_mode = st.radio("Chế độ:", ["Tín hiệu Đơn lẻ", "Quét Thị trường"], horizontal=True)
    asset_class = st.radio("Loại tài sản:", ["Crypto", "Forex", "Stocks"])
    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    if scan_mode == "Tín hiệu Đơn lẻ":
        if asset_class == "Crypto":
            symbol = st.text_input("Mã (CCXT):", "BTC/USDT")
            tf = st.selectbox("Khung TG:", common_timeframes, index=6)
        else:
            default_symbol = "EURUSD=X" if asset_class == "Forex" else "AAPL"
            symbol = st.text_input("Mã (Yahoo):", default_symbol)
            tf = st.selectbox("Khung TG:", common_timeframes, index=6)
        st.subheader("Khoảng thời gian")
        yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
        end_date = datetime.now().date()
        start_date_default = end_date - timedelta(days=730)
        info_message = ""
        if asset_class != 'Crypto' and tf in yf_timeframe_limits:
            limit = yf_timeframe_limits[tf]
            start_date_default = end_date - timedelta(days=limit - 1)
            info_message = f"Gợi ý: Khung {tf} giới hạn trong {limit} ngày."
        end_date_input = st.date_input("Ngày kết thúc", value=end_date)
        start_date_input = st.date_input("Ngày bắt đầu", value=start_date_default)
        if info_message: st.caption(info_message)
        run_button_label = "Lấy tín hiệu"
    else: # Chế độ Quét
        tf = st.selectbox("Khung TG để quét:", ["1h", "4h", "1d", "1w"], index=2)
        st.info("Chế độ quét sẽ tự động chạy trên danh sách các tài sản được lọc theo vốn hóa.")
        end_date_input = datetime.now().date()
        start_date_input = end_date_input - timedelta(days=730)
        run_button_label = "Bắt đầu Quét"

# --- Giao diện chính ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []

run_button = st.sidebar.button(run_button_label, type="primary", use_container_width=True)

if scan_mode == "Tín hiệu Đơn lẻ":
    if run_button:
        if start_date_input >= end_date_input:
            st.error("Lỗi: Ngày bắt đầu phải trước ngày kết thúc.")
        else:
            data = load_data_for_signal(asset_class, symbol, tf, start_date_input, end_date_input)
            if data is not None and not data.empty:
                model_results = get_trained_model(asset_class, symbol, tf, start_date_input, end_date_input)
                if model_results:
                    signal, message, latest_features, _ = get_ml_signal(data, model_results)
                    st.session_state['ml_signal_info'] = {
                        "signal": signal, "symbol": symbol, "asset_class": asset_class,
                        "timeframe": tf, "model": model_results['model'],
                        "start_date": start_date_input, "end_date": end_date_input
                    }
                    st.subheader(f"Kết quả cho {symbol} - Khung {tf}")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if signal == "MUA": st.metric("Tín hiệu 📈", "MUA")
                        else: st.metric("Tín hiệu 📉", "BÁN")
                        st.info(message); st.caption("Cơ sở tín hiệu:"); st.dataframe(latest_features)
                    with col2:
                        st.caption("Biểu đồ 60 nến gần nhất")
                        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
                        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=280, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig.update_xaxes(range=[data.index[-60], data.index[-1]]), use_container_width=True)
                    st.subheader("Hành động Tiếp theo"); act_col1, act_col2 = st.columns(2)
                    if act_col1.button("Kiểm chứng Lịch sử (Backtest) 🧪", use_container_width=True):
                        st.session_state['run_ml_backtest'] = True; st.success("Đã lưu! Hãy chuyển sang trang 'Backtest'.")
                    if act_col2.button("Thực hiện Giao dịch 🛰️", use_container_width=True):
                        st.session_state['trade_signal_to_execute'] = {'symbol': symbol, 'side': signal, 'asset_class': asset_class}
                        st.success("Đã lưu! Hãy chuyển sang trang 'Live Trading'.")

elif scan_mode == "Quét Thị trường":
    if run_button:
        asset_list = get_market_scan_list(asset_class)
        results = []
        progress_bar = st.progress(0, text="Bắt đầu quét...")
        for i, sym in enumerate(asset_list):
            progress_bar.progress((i+1)/len(asset_list), text=f"Đang quét {sym} ({i+1}/{len(asset_list)})...")
            
            # 1. Huấn luyện mô hình (sẽ dùng cache)
            model_results = get_trained_model(asset_class, sym, tf, start_date_input, end_date_input)
            if model_results is None: continue
            
            # 2. Tải dữ liệu mới nhất (không dùng cache) để lấy tín hiệu
            # (Chúng ta có thể tối ưu hơn, nhưng tạm thời dùng lại hàm load)
            data = load_data_for_signal(asset_class, sym, tf, start_date_input, end_date_input)
            if data is None or data.empty: continue
            
            signal, message, features, confidence = get_ml_signal(data, model_results)
            if signal in ["BUY", "SELL"]:
                results.append({"Mã": sym, "Tín hiệu": signal, "Độ tin cậy": confidence, "Mô hình": model_results['model']})
        
        progress_bar.empty()
        st.success(f"Đã quét xong {len(asset_list)} mã tài sản!")
        st.session_state.scan_results = results

    # NÂNG CẤP: Hiển thị kết quả và các hành động
    if st.session_state.scan_results:
        df_results = pd.DataFrame(st.session_state.scan_results).sort_values(by="Độ tin cậy", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Top 10 Tín hiệu MUA")
            top_buys = df_results[df_results['Tín hiệu'] == "MUA"].head(10).reset_index(drop=True)
            st.dataframe(top_buys[['Mã', 'Độ tin cậy']], use_container_width=True)
        with col2:
            st.markdown("#### 📉 Top 10 Tín hiệu BÁN")
            top_sells = df_results[df_results['Tín hiệu'] == "BÁN"].head(10).reset_index(drop=True)
            st.dataframe(top_sells[['Mã', 'Độ tin cậy']], use_container_width=True)

        st.divider()
        st.subheader("Hành động với Tín hiệu đã Quét")
        
        # Tạo danh sách các mã có tín hiệu để chọn
        actionable_symbols = list(df_results['Mã'])
        if actionable_symbols:
            selected_symbol = st.selectbox("Chọn một mã để hành động:", actionable_symbols)
            
            # Tìm tín hiệu tương ứng
            selected_row = df_results[df_results['Mã'] == selected_symbol].iloc[0]
            selected_signal = selected_row['Tín hiệu']
            selected_model = selected_row['Mô hình']
            
            st.info(f"Bạn đã chọn: **{selected_symbol}** (Tín hiệu: **{selected_signal}**)")
            
            act_col1, act_col2 = st.columns(2)
            if act_col1.button(f"Kiểm chứng {selected_symbol} 🧪", use_container_width=True):
                st.session_state['run_ml_backtest'] = True
                st.session_state['ml_signal_info'] = {
                    "model": selected_model, "symbol": selected_symbol, 
                    "asset_class": asset_class, "timeframe": tf,
                    "start_date": start_date_input, "end_date": end_date_input
                }
                st.success("Đã lưu! Hãy chuyển sang trang 'Backtest'.")
            
            if act_col2.button(f"Giao dịch {selected_signal} {selected_symbol} 🛰️", use_container_width=True):
                st.session_state['trade_signal_to_execute'] = {
                    'symbol': selected_symbol, 'side': selected_signal, 'asset_class': asset_class
                }
                st.success("Đã lưu! Hãy chuyển sang trang 'Live Trading'.")