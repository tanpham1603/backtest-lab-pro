import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import warnings
import traceback
warnings.filterwarnings('ignore')

# --- Cấu hình trang ---
st.set_page_config(
    page_title="🚀 ML Signal Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS tùy chỉnh ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .welcome-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #8898aa;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        height: 280px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: 0.5s;
    }
    .feature-card:hover::before {
        left: 100%;
    }
    .feature-card:hover {
        transform: translateY(-10px);
        border-color: #667eea;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
    }
    .feature-desc {
        color: #8898aa;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .status-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #8898aa;
        font-size: 0.9rem;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 3rem 0;
        border: none;
    }
    .config-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .config-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .signal-hold {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .feature-importance {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>🤖</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>ML Signal Pro</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>AI Trading Platform</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("✨ Select analysis mode to begin")

# --- Header ---
st.markdown('<div class="welcome-header">🚀 ML Signal Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced AI-Powered Trading Signal Platform</div>', unsafe_allow_html=True)

# --- Status Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🤖</div>
        <div class="metric-value">Active</div>
        <div class="metric-label">AI Engine</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📊</div>
        <div class="metric-value">Live</div>
        <div class="metric-label">Data Feed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">⚡</div>
        <div class="metric-value">Ready</div>
        <div class="metric-label">Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🎯</div>
        <div class="metric-value">Pro</div>
        <div class="metric-label">Signals</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Danh sách CỐ ĐỊNH ---
FOREX_LIST = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'GC=F', 'CL=F']
STOCKS_LIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'SPY', 'QQQ']

# --- CÁC HÀM XỬ LÝ NÂNG CAO ---
@st.cache_data(ttl=1800)
def get_market_scan_list(asset_class):
    if asset_class == "Crypto":
        with st.spinner("🔄 Loading coin list from CoinGecko..."):
            try:
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                coin_list = [f"{coin.get('symbol', '').upper()}/USDT" for coin in data if coin.get('market_cap') and coin['market_cap'] > 10_000_000 and coin.get('symbol')]
                return coin_list
            except Exception as e:
                st.error(f"❌ CoinGecko API Error: {e}. Using backup list.")
                return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']
    elif asset_class == "Forex": 
        return FOREX_LIST
    else: 
        return STOCKS_LIST

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
        if data.empty: 
            return None
        return data
    except Exception:
        return None

def create_advanced_features(data):
    """Create advanced feature set for ML model"""
    return create_technical_indicators(data)

def create_technical_indicators(data):
    df = data.copy()
    try:
        # Trend indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)
        
        # MACD
        macd_data = ta.macd(df['Close'])
        if macd_data is not None:
            df['MACD'] = macd_data.get('MACD_12_26_9', np.nan)
            df['MACD_Signal'] = macd_data.get('MACDs_12_26_9', np.nan)
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        else:
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
            df['MACD_Histogram'] = np.nan
        
        # RSI
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_7'] = ta.rsi(df['Close'], length=7)
        
        # Stochastic
        stoch_data = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch_data is not None:
            df['Stoch_K'] = stoch_data.get('STOCHk_14_3_3', np.nan)
            df['Stoch_D'] = stoch_data.get('STOCHd_14_3_3', np.nan)
        else:
            df['Stoch_K'] = np.nan
            df['Stoch_D'] = np.nan
            
        # Williams %R
        df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
        
        # Bollinger Bands
        bb_data = ta.bbands(df['Close'])
        if bb_data is not None:
            bb_columns = bb_data.columns
            if 'BBU_20_2.0' in bb_columns:
                df['BB_Upper'] = bb_data['BBU_20_2.0']
                df['BB_Lower'] = bb_data['BBL_20_2.0']
                df['BB_Middle'] = bb_data['BBM_20_2.0']
            elif 'BBU_20_2' in bb_columns:
                df['BB_Upper'] = bb_data['BBU_20_2']
                df['BB_Lower'] = bb_data['BBL_20_2'] 
                df['BB_Middle'] = bb_data['BBM_20_2']
            else:
                df['BB_Upper'] = bb_data.iloc[:, 2] if len(bb_columns) > 2 else np.nan
                df['BB_Lower'] = bb_data.iloc[:, 0] if len(bb_columns) > 0 else np.nan
                df['BB_Middle'] = bb_data.iloc[:, 1] if len(bb_columns) > 1 else np.nan
            
        # ATR
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Volume indicators
        df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        # Price changes
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Time features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        # Handle NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
    except Exception as e:
        st.error(f"Technical indicator error: {e}")
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df = df.fillna(method='bfill').fillna(method='ffill')
    return df

@st.cache_resource
def get_advanced_trained_model(_data, asset, sym, timeframe, model_type):
    data = _data.copy()
    if data is None or data.empty or len(data) < 100: 
        return None
        
    try:
        df = create_advanced_features(data)
        df['Future_Return_5'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target'] = (df['Future_Return_5'] > 0).astype(int)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in ['target', 'Future_Return_5', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        feature_columns = feature_columns[:20]
        
        df_clean = df[feature_columns + ['target']].dropna()
        
        if len(df_clean) < 100: 
            return None
        
        X = df_clean[feature_columns]
        y = df_clean['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=10, 
                min_samples_leaf=5, 
                random_state=42, 
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                random_state=42
            )
        
        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
        model.fit(X_scaled, y)
        
        return {
            "model": model, 
            "features": feature_columns, 
            "scaler": scaler,
            "cv_accuracy": cv_scores.mean(), 
            "data": df_clean
        }
    except Exception as e:
        st.error(f"Model training error: {e}")
        traceback.print_exc()
        return None

def get_advanced_ml_signal(data, model_results):
    if model_results is None: 
        return "ERROR", "Model error", None, 0, {}
        
    try:
        model = model_results['model']
        features = model_results['features']
        scaler = model_results['scaler']
        
        df = create_advanced_features(data)
        latest_features = df[features].dropna().iloc[-1:]
        
        if latest_features.empty: 
            return "HOLD", "Insufficient data", None, 0, {}
        
        X_latest = scaler.transform(latest_features)
        prediction = model.predict(X_latest)[0]
        proba = model.predict_proba(X_latest)[0]
        confidence = proba[prediction]
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(features, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []
        
        signal_text = "BUY" if prediction == 1 else "SELL"
        
        analysis = {
            "confidence": confidence,
            "top_features": top_features,
            "probability_buy": proba[1],
            "probability_sell": proba[0],
            "signal_strength": "STRONG" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "WEAK"
        }
        
        message = f"🎯 {signal_text} | Confidence: {confidence*100:.1f}% | Strength: {analysis['signal_strength']}"
        
        return signal_text, message, latest_features, confidence, analysis
        
    except Exception as e:
        return "ERROR", f"Prediction error: {e}", None, 0, {}

def create_signal_radar_chart(analysis):
    if not analysis:
        return None
        
    categories = ['Confidence', 'Buy Probability', 'Sell Probability', 'Signal Strength', 'Model Quality']
    
    values = [
        analysis['confidence'] * 100,
        analysis['probability_buy'] * 100,
        analysis['probability_sell'] * 100,
        (analysis['confidence'] - 0.5) * 200,
        85
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Signal Analysis',
        line=dict(color='#00D4AA')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def run_backtest_analysis(data, signal_info):
    try:
        df = data.copy()
        
        df['Signal'] = 'HOLD'
        df['Return'] = df['Close'].pct_change()
        
        if signal_info['signal'] == 'BUY':
            df['Position'] = 1
        else:
            df['Position'] = -1
            
        df['Strategy_Return'] = df['Position'].shift(1) * df['Return']
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        df['Benchmark_Return'] = (1 + df['Return']).cumprod()
        
        total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
        benchmark_return = (df['Benchmark_Return'].iloc[-1] - 1) * 100
        win_rate = (df['Strategy_Return'] > 0).mean() * 100
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'win_rate': win_rate,
            'max_return': df['Cumulative_Return'].max() * 100,
            'min_return': df['Cumulative_Return'].min() * 100,
            'data': df
        }
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None

def create_trading_setup(symbol, signal, current_price, confidence):
    st.subheader("💰 Trading Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Symbol:** {symbol}")
        st.write(f"**Signal:** {signal}")
        st.write(f"**Current Price:** ${current_price:.4f}")
        st.write(f"**Confidence:** {confidence*100:.1f}%")
    
    with col2:
        quantity = st.number_input("Quantity:", min_value=0.001, value=1.0, step=0.1, key="trade_quantity")
        risk_percent = st.slider("Risk per Trade (%):", 1.0, 10.0, 2.0, 0.5)
        
        order_value = quantity * current_price
        risk_amount = order_value * risk_percent / 100
        
        st.metric("💰 Order Value", f"${order_value:.2f}")
        st.metric("⚠️ Risk Amount", f"${risk_amount:.2f}")
    
    return quantity, order_value

def get_technical_analysis_data(data):
    try:
        tech_data = create_technical_indicators(data)
        analysis = {}
        
        analysis['rsi'] = tech_data['RSI_14'].iloc[-1] if 'RSI_14' in tech_data.columns else np.nan
        
        analysis['macd_hist'] = tech_data['MACD_Histogram'].iloc[-1] if 'MACD_Histogram' in tech_data.columns else np.nan
        
        if all(col in tech_data.columns for col in ['BB_Upper', 'BB_Lower']):
            current_price = data['Close'].iloc[-1]
            bb_upper = tech_data['BB_Upper'].iloc[-1]
            bb_lower = tech_data['BB_Lower'].iloc[-1]
            
            if bb_upper != bb_lower:
                analysis['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                analysis['bb_position'] = 0.5
        else:
            analysis['bb_position'] = 0.5
        
        analysis['volume_trend'] = "📈 High" if data['Volume'].iloc[-1] > data['Volume'].tail(20).mean() else "📉 Low"
        
        if all(col in tech_data.columns for col in ['SMA_20', 'SMA_50']):
            analysis['sma_20'] = tech_data['SMA_20'].iloc[-1]
            analysis['sma_50'] = tech_data['SMA_50'].iloc[-1]
            analysis['trend'] = "📈 Bullish" if analysis['sma_20'] > analysis['sma_50'] else "📉 Bearish" if analysis['sma_20'] < analysis['sma_50'] else "➡️ Sideways"
            analysis['price_vs_sma20'] = (data['Close'].iloc[-1] / analysis['sma_20'] - 1) * 100
            analysis['price_vs_sma50'] = (data['Close'].iloc[-1] / analysis['sma_50'] - 1) * 100
        else:
            analysis['trend'] = "Unknown"
            analysis['price_vs_sma20'] = 0
            analysis['price_vs_sma50'] = 0
        
        analysis['atr'] = tech_data['ATR'].iloc[-1] if 'ATR' in tech_data.columns else np.nan
        analysis['volatility'] = data['Close'].pct_change().std() * 100
        
        analysis['support'] = data['Low'].tail(20).min()
        analysis['resistance'] = data['High'].tail(20).max()
        
        analysis['rsi_data'] = tech_data['RSI_14'] if 'RSI_14' in tech_data.columns else pd.Series([50] * len(data))
        
        return analysis
        
    except Exception as e:
        st.error(f"Technical analysis error: {e}")
        return {
            'rsi': 50, 'macd_hist': 0, 'bb_position': 0.5, 'volume_trend': "Unknown",
            'trend': "Unknown", 'price_vs_sma20': 0, 'price_vs_sma50': 0,
            'atr': 0, 'volatility': 0, 'support': data['Close'].min() if not data.empty else 0,
            'resistance': data['Close'].max() if not data.empty else 0,
            'rsi_data': pd.Series([50] * len(data))
        }

# --- CONFIGURATION SECTION ---
st.markdown("## ⚙️ ANALYSIS CONFIGURATION")

# Section 1: Basic Configuration
st.markdown('<div class="config-section">', unsafe_allow_html=True)
st.markdown('<div class="config-header">📊 BASIC CONFIGURATION</div>', unsafe_allow_html=True)

col_basic1, col_basic2, col_basic3 = st.columns(3)

with col_basic1:
    scan_mode = st.radio("**Analysis Mode:**", ["Single Signal", "Market Scan"], horizontal=True)

with col_basic2:
    asset_class = st.radio("**Asset Class:**", ["Crypto", "Forex", "Stocks"], horizontal=True)

with col_basic3:
    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    if scan_mode == "Single Signal":
        tf = st.selectbox("**Timeframe:**", common_timeframes, index=6)
    else:
        tf = st.selectbox("**Timeframe:**", ["4h", "1d", "1w"], index=1)

st.markdown('</div>', unsafe_allow_html=True)

# Section 2: ML Configuration
st.markdown('<div class="config-section">', unsafe_allow_html=True)
st.markdown('<div class="config-header">🧠 MACHINE LEARNING CONFIGURATION</div>', unsafe_allow_html=True)

col_ml1, col_ml2, col_ml3 = st.columns(3)

with col_ml1:
    model_type = st.selectbox("**Model Type:**", ["RandomForest", "GradientBoosting"])

with col_ml2:
    confidence_threshold = st.slider("**Confidence Threshold:**", 0.5, 0.95, 0.65, 0.05)

with col_ml3:
    if scan_mode == "Market Scan":
        max_symbols = st.slider("**Max Symbols:**", 10, 100, 50)

st.markdown('</div>', unsafe_allow_html=True)

# Section 3: Time Range
st.markdown('<div class="config-section">', unsafe_allow_html=True)
st.markdown('<div class="config-header">📅 ANALYSIS TIME RANGE</div>', unsafe_allow_html=True)

col_time1, col_time2, col_time3 = st.columns([2, 1, 1])

with col_time1:
    if scan_mode == "Single Signal":
        if asset_class == "Crypto":
            symbol = st.text_input("**Trading Symbol (CCXT):**", "BTC/USDT", help="Example: BTC/USDT, ETH/USDT")
        else:
            default_symbol = "EURUSD=X" if asset_class == "Forex" else "AAPL"
            symbol = st.text_input("**Trading Symbol (Yahoo Finance):**", default_symbol, help="Example: AAPL, MSFT, EURUSD=X")

with col_time2:
    end_date = datetime.now().date()
    start_date_default = end_date - timedelta(days=365)
    
    yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
    if asset_class != 'Crypto' and tf in yf_timeframe_limits:
        limit = yf_timeframe_limits[tf]
        start_date_default = end_date - timedelta(days=limit - 1)
        st.info(f"{tf} timeframe limited to {limit} days")
    
    start_date_input = st.date_input("**Start Date**", value=start_date_default)

with col_time3:
    end_date_input = st.date_input("**End Date**", value=end_date)
    if scan_mode == "Market Scan":
        st.info(f"Scanning {max_symbols} symbols")

st.markdown('</div>', unsafe_allow_html=True)

# Analysis Button
if scan_mode == "Single Signal":
    run_button = st.button("🚀 START ADVANCED ANALYSIS", type="primary", use_container_width=True, key="analyze_single")
else:
    run_button = st.button("🎯 START MARKET SCAN", type="primary", use_container_width=True, key="analyze_scan")

# --- MAIN INTERFACE ---
if 'scan_results' not in st.session_state: 
    st.session_state.scan_results = []
if 'model_performance' not in st.session_state: 
    st.session_state.model_performance = {}
if 'current_analysis' not in st.session_state: 
    st.session_state.current_analysis = None

if run_button or st.session_state.current_analysis or st.session_state.scan_results:
    st.markdown("---")
    st.markdown("## 📈 ANALYSIS RESULTS")

if scan_mode == "Single Signal":
    if run_button:
        if start_date_input >= end_date_input:
            st.error("❌ Start date must be before end date")
        else:
            with st.spinner("🔄 Running advanced analysis..."):
                data = load_data_for_signal(asset_class, symbol, tf, start_date_input, end_date_input)
                if data is not None and not data.empty:
                    model_results = get_advanced_trained_model(data, asset_class, symbol, tf, model_type)
                    
                    if model_results:
                        signal, message, latest_features, confidence, analysis = get_advanced_ml_signal(data, model_results)
                        tech_analysis = get_technical_analysis_data(data)
                        
                        st.session_state.current_analysis = {
                            "signal": signal, "symbol": symbol, "asset_class": asset_class, 
                            "timeframe": tf, "model": model_results['model'],
                            "model_results": model_results, "analysis": analysis, 
                            "tech_analysis": tech_analysis, "data": data, 
                            "start_date": start_date_input, "end_date": end_date_input, 
                            "confidence": confidence
                        }
                    else:
                        st.error("❌ Could not train model. Try longer time range or different symbol.")
                else:
                    st.error("❌ Could not load data. Check symbol and time range.")
    
    if st.session_state.current_analysis:
        analysis_data = st.session_state.current_analysis
        signal = analysis_data['signal']
        symbol = analysis_data['symbol']
        tf = analysis_data['timeframe']
        confidence = analysis_data['confidence']
        data = analysis_data['data']
        analysis = analysis_data['analysis']
        tech_analysis = analysis_data['tech_analysis']
        model_results = analysis_data['model_results']

        st.subheader(f"📊 Detailed Analysis: {symbol} - {tf}")
        
        # Row 1: Main Signal and Confidence
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if signal == "BUY":
                st.markdown('<div class="signal-buy"><h3>🎯 BUY SIGNAL</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-sell"><h3>🎯 SELL SIGNAL</h3></div>', unsafe_allow_html=True)
            
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.metric("Signal Strength", analysis.get('signal_strength', 'N/A'))
        
        with col2:
            radar_chart = create_signal_radar_chart(analysis)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
        
        with col3:
            st.metric("Model Quality (CV)", f"{model_results['cv_accuracy']*100:.1f}%")
            st.metric("Feature Count", len(model_results['features']))
        
        # Row 2: Feature Importance and Chart
        col4, col5 = st.columns([1, 2])
        
        with col4:
            st.subheader("🔍 Top Feature Importance")
            top_features = analysis.get('top_features', [])
            if top_features:
                for feature, importance in top_features:
                    st.markdown(f'<div class="feature-importance">{feature}: {importance:.3f}</div>', unsafe_allow_html=True)
            else:
                st.info("No feature importance data")
        
        with col5:
            st.subheader(f"📈 Price & Signal Chart")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index[-60:], 
                open=data['Open'][-60:], 
                high=data['High'][-60:], 
                low=data['Low'][-60:], 
                close=data['Close'][-60:], 
                name="Price"
            ))
            
            last_price = data['Close'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[last_price],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green' if signal == "BUY" else 'red',
                    symbol='triangle-up' if signal == "BUY" else 'triangle-down'
                ),
                name=f"Signal: {signal}"
            ))
            
            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=400,
                title=f"{symbol} Chart - {signal} Signal"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # --- ACTION SECTION ---
        st.subheader("🎮 Next Actions")
        
        action_tab1, action_tab2, action_tab3 = st.tabs([
            "🧪 Advanced Validation", 
            "💰 Trade Now", 
            "📊 Technical Analysis"
        ])
        
        with action_tab1:
            st.subheader("Backtest & Performance Validation")
            
            if st.button("🚀 Run Detailed Backtest", key="run_backtest"):
                with st.spinner("Running backtest..."):
                    backtest_results = run_backtest_analysis(data, {
                        'signal': signal,
                        'confidence': confidence
                    })
                    
                    if backtest_results:
                        st.success("✅ Backtest completed!")
                        
                        col_bt1, col_bt2, col_bt3 = st.columns(3)
                        
                        with col_bt1:
                            st.metric(
                                "Strategy Profit", 
                                f"{backtest_results['total_return']:.1f}%",
                                delta=f"{backtest_results['total_return'] - backtest_results['benchmark_return']:.1f}% vs Benchmark"
                            )
                        
                        with col_bt2:
                            st.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
                        
                        with col_bt3:
                            st.metric("Benchmark Return", f"{backtest_results['benchmark_return']:.1f}%")
                        
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(
                            x=backtest_results['data'].index,
                            y=backtest_results['data']['Cumulative_Return'] * 100,
                            name='ML Strategy',
                            line=dict(color='#00D4AA', width=3)
                        ))
                        fig_bt.add_trace(go.Scatter(
                            x=backtest_results['data'].index,
                            y=backtest_results['data']['Benchmark_Return'] * 100,
                            name='Buy & Hold',
                            line=dict(color='#666666', width=2, dash='dash')
                        ))
                        fig_bt.update_layout(
                            title='Backtest Performance: ML Strategy vs Buy & Hold',
                            xaxis_title='Time',
                            yaxis_title='Return (%)',
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_bt, use_container_width=True)
                    else:
                        st.error("❌ Could not run backtest")
            
            st.info("""
            **📈 Backtest will check:**
            - Strategy performance on historical data
            - Comparison with Buy & Hold  
            - Win rate and drawdown
            - Signal stability
            """)
        
        with action_tab2:
            st.subheader("Trading Setup")
            
            current_price = data['Close'].iloc[-1]
            quantity, order_value = create_trading_setup(symbol, signal, current_price, confidence)
            
            col_trade1, col_trade2 = st.columns(2)
            
            with col_trade1:
                if st.button("📊 Risk Analysis", key="risk_analysis"):
                    st.success("🔍 Risk analysis:")
                    
                    col_risk1, col_risk2, col_risk3 = st.columns(3)
                    
                    with col_risk1:
                        st.metric("Daily Volatility", f"{tech_analysis['volatility']:.2f}%")
                    
                    with col_risk2:
                        st.metric("ATR (14)", f"{tech_analysis['atr']:.4f}")
                    
                    with col_risk3:
                        st.metric("Support/Resistance", f"${tech_analysis['support']:.2f}/${tech_analysis['resistance']:.2f}")
            
            with col_trade2:
                if st.button("🛰️ Save Trade Order", type="primary", key="save_trade"):
                    st.session_state.trade_signal_to_execute = {
                        'symbol': symbol, 
                        'side': signal, 
                        'asset_class': asset_class,
                        'confidence': confidence,
                        'quantity': quantity,
                        'current_price': current_price,
                        'order_value': order_value,
                        'timeframe': tf
                    }
                    st.success("✅ Trade order saved!")
        
        with action_tab3:
            st.subheader("Detailed Technical Analysis")
            
            tech = tech_analysis
            
            tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
            
            with tech_col1:
                rsi_color = "red" if tech['rsi'] > 70 else "green" if tech['rsi'] < 30 else "orange"
                st.metric("RSI (14)", f"{tech['rsi']:.1f}", 
                         delta="Overbought" if tech['rsi'] > 70 else "Oversold" if tech['rsi'] < 30 else "Neutral",
                         delta_color="inverse" if tech['rsi'] > 70 else "normal")
            
            with tech_col2:
                st.metric("MACD Histogram", f"{tech['macd_hist']:.4f}",
                         delta="Bullish" if tech['macd_hist'] > 0 else "Bearish")
            
            with tech_col3:
                st.metric("Bollinger Position", f"{(tech['bb_position']*100):.1f}%",
                         delta="Upper" if tech['bb_position'] > 0.8 else "Lower" if tech['bb_position'] < 0.2 else "Middle")
            
            with tech_col4:
                st.metric("Volume", tech['volume_trend'])
            
            st.subheader("RSI Indicator")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=tech['rsi_data'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought 70")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold 30")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="yellow", annotation_text="Neutral 50")
            fig_rsi.update_layout(height=300, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            st.subheader("Trend Analysis")
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                st.metric("SMA Trend", tech['trend'])
                st.write(f"SMA20: ${tech['sma_20']:.2f}")
                st.write(f"SMA50: ${tech['sma_50']:.2f}")
            
            with trend_col2:
                st.metric("Price vs SMA20", f"{tech['price_vs_sma20']:+.1f}%")
                st.metric("Price vs SMA50", f"{tech['price_vs_sma50']:+.1f}%")

elif scan_mode == "Market Scan":
    if run_button:
        with st.spinner("🔄 Scanning market..."):
            asset_list = get_market_scan_list(asset_class)
            asset_list = asset_list[:max_symbols]
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, sym in enumerate(asset_list):
                status_text.text(f"📡 Analyzing {sym} ({i+1}/{len(asset_list)})...")
                progress_bar.progress((i + 1) / len(asset_list))
                
                try:
                    data = load_data_for_signal(asset_class, sym, tf, start_date_input, end_date_input)
                    if data is None or data.empty:
                        continue
                        
                    model_results = get_advanced_trained_model(data, asset_class, sym, tf, model_type)
                    if model_results is None:
                        continue
                    
                    signal, message, features, confidence, analysis = get_advanced_ml_signal(data, model_results)
                    
                    if confidence >= confidence_threshold:
                        results.append({
                            "Symbol": sym, 
                            "Signal": signal, 
                            "Confidence": confidence,
                            "Strength": analysis.get('signal_strength', 'N/A'),
                            "ML Quality": model_results['cv_accuracy']
                        })
                        
                except Exception as e:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                st.session_state.scan_results = results
                st.success(f"✅ Scanned {len(asset_list)} symbols! Found {len(results)} quality signals.")
            else:
                st.warning("⚠️ No signals found above confidence threshold.")
    
    if st.session_state.scan_results:
        df_results = pd.DataFrame(st.session_state.scan_results)
        
        st.subheader("📊 Market Scan Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", len(df_results))
        with col2:
            st.metric("BUY Signals", len(df_results[df_results['Signal'] == 'BUY']))
        with col3:
            st.metric("SELL Signals", len(df_results[df_results['Signal'] == 'SELL']))
        with col4:
            avg_conf = df_results['Confidence'].mean() * 100
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        st.dataframe(
            df_results.sort_values('Confidence', ascending=False),
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("### 📊 System Statistics")
if st.session_state.model_performance:
    perf_df = pd.DataFrame(st.session_state.model_performance)
    avg_accuracy = perf_df['CV Accuracy'].mean() * 100
    st.metric("Average Model Accuracy", f"{avg_accuracy:.1f}%")
else:
    st.info("Run analysis to see model performance statistics")

st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Advanced AI Trading Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>ML Signal Pro v4.0</p>
</div>
""", unsafe_allow_html=True)