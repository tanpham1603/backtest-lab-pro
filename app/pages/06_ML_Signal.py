# Tên tệp: pages/01_ML_Signals.py
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

st.set_page_config(page_title="ML Signal by TanPham", page_icon="🤖", layout="wide")
st.markdown("""<style>.main { background-color: #0E1117; } .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 10px; text-align: center; }</style>""", unsafe_allow_html=True)
st.title("🤖 Trading signals from Machine Learning")

@st.cache_data(ttl=300)
def load_data_for_signal(asset, sym, timeframe, start, end):
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
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
    except Exception as e:
        st.error(f"Error loading data for {sym}: {e}"); return None

@st.cache_resource
def train_and_evaluate_model(data):
    with st.spinner("Preparing data and training advanced model..."):
        df = data.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.adx(length=14, append=True)
        df.rename(columns={"RSI_14": "RSI", "SMA_50": "MA50", "SMA_200": "MA200", "ADX_14": "ADX"}, inplace=True)
        df['Price_vs_MA200'] = np.where(df['Close'] > df['MA200'], 1, -1)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        
        if len(df) < 250: return None
        features = ['RSI', 'MA50', 'Price_vs_MA200', 'ADX']
        X = df[features]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
        model.fit(X_train, y_train)
        
        return {"model": model, "accuracy": accuracy_score(y_test, model.predict(X_test)),
                "confusion_matrix": confusion_matrix(y_test, model.predict(X_test)),
                "feature_importances": model.feature_importances_, "feature_names": features}

def get_ml_signal(data, model_results):
    model, features = model_results['model'], model_results['feature_names']
    df = data.copy()
    df.ta.rsi(length=14, append=True); df.ta.sma(length=50, append=True); df.ta.sma(length=200, append=True); df.ta.adx(length=14, append=True)
    df.rename(columns={"RSI_14": "RSI", "SMA_50": "MA50", "SMA_200": "MA200", "ADX_14": "ADX"}, inplace=True)
    df['Price_vs_MA200'] = np.where(df['Close'] > df['MA200'], 1, -1)
    latest_features = df[features].dropna().iloc[-1:]
    if latest_features.empty: return "GIỮ", "Không đủ dữ liệu mới", None
    prediction = model.predict(latest_features)[0]
    signal = "BUY" if prediction == 1 else "SELL"
    return signal, f"Expected: {signal}", latest_features

with st.sidebar:
    st.header("⚙️ Signal Configuration")
    asset_class = st.radio("Asset Class:", ["Crypto", "Forex", "Stocks"])
    tf = st.selectbox("Timeframe:", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"], index=6)
    if asset_class == "Crypto": symbol = st.text_input("Symbol (CCXT):", "BTC/USDT")
    else: symbol = st.text_input("Symbol (Yahoo):", "AAPL" if asset_class == "Stocks" else "EURUSD=X")

    st.subheader("Time Period")
    yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
    end_date = datetime.now().date()
    start_date_default = end_date - timedelta(days=730)
    info_message = ""
    if asset_class != 'Crypto' and tf in yf_timeframe_limits:
        limit = yf_timeframe_limits[tf]
        start_date_default = end_date - timedelta(days=limit - 1)
        info_message = f"Suggestion: {tf} range limited to {limit} days."
    end_date_input = st.date_input("End Date", value=end_date)
    start_date_input = st.date_input("Start Date", value=start_date_default)
    if info_message: st.caption(info_message)

if start_date_input >= end_date_input:
    st.error("Error: Start date must be before end date.")
else:
    data = load_data_for_signal(asset_class, symbol, tf, start_date_input, end_date_input)
    if data is not None and not data.empty:
        model_results = train_and_evaluate_model(data)
        if model_results:
            st.success("Training and evaluating model successful!")
            signal, message, latest_features = get_ml_signal(data, model_results)

            st.subheader(f"Results for {symbol} - Timeframe {tf}")
            col1, col2 = st.columns([1, 2])
            with col1:
                if signal == "BUY": st.metric("Signal 📈", "BUY")
                else: st.metric("Signal 📉", "SELL")
                st.info(message)
                st.caption("Data base for signals:")
                st.dataframe(latest_features)
            with col2:
                st.caption("Last 60 candles chart")
                fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=280, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig.update_xaxes(range=[data.index[-60], data.index[-1]]), use_container_width=True)

            st.subheader("Next Steps")
            act_col1, act_col2 = st.columns(2)
            if act_col1.button("Backtest 🧪", use_container_width=True):
                st.session_state['run_ml_backtest'] = True
                st.session_state['ml_signal_info'] = {"model": model_results['model'], "symbol": symbol, "asset_class": asset_class, "timeframe": tf, "start_date": start_date_input, "end_date": end_date_input}
                st.success("Saved! Please go to the 'Backtest' page to verify.")
            if act_col2.button("Execute Trade 🛰️", use_container_width=True):
                st.session_state['trade_signal_to_execute'] = {'symbol': symbol, 'side': signal, 'asset_class': asset_class}
                st.success("Saved! Please go to the 'Live Trading' page to execute.")

            st.divider()
            st.subheader("Training & Evaluation Results")
            acc_col, fi_col = st.columns(2)
            with acc_col:
                st.metric("Accuracy (on test data)", f"{model_results['accuracy'] * 100:.2f}%")
                st.write("**Confusion Matrix**")
                cm = model_results['confusion_matrix']; x = ['Predicted SELL', 'Predicted BUY']; y = ['Actual SELL', 'Actual BUY']
                fig_cm = ff.create_annotated_heatmap(cm, x=x, y=y, colorscale='Blues', showscale=False)
                fig_cm.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)
            with fi_col:
                st.write("**Feature Importance**")
                fi_df = pd.DataFrame({'Feature': model_results['feature_names'], 'Importance': model_results['feature_importances']}).sort_values(by='Importance', ascending=False)
                st.dataframe(fi_df)
    else:
        st.warning("Please reconfigure, unable to load data.")