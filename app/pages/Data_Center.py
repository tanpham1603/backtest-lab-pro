import streamlit as st
import pandas as pd
import ccxt 
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 Data Center Pro", 
    page_icon="🗃️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING - ĐỒNG BỘ THIẾT KẾ ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .main-header {
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
        transition: all 0.3s ease;
    }
    .status-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
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
        text-transform: uppercase;
        letter-spacing: 1px;
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
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .analysis-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    .data-source-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        color: #8898aa;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .quick-preset-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        color: #8898aa;
        transition: all 0.3s ease;
        width: 100%;
    }
    .quick-preset-btn:hover {
        border-color: #667eea;
        color: white;
    }
    .asset-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section - MỚI ---
st.markdown('<div class="main-header">🚀 Data Center Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Multi-Asset Market Data Platform</div>', unsafe_allow_html=True)

# --- Status Cards - MỚI ---
col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📊</div>
        <div class="metric-value">3-in-1</div>
        <div class="metric-label">Asset Classes</div>
    </div>
    """, unsafe_allow_html=True)
with col2: 
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">⚡</div>
        <div class="metric-value">Real</div>
        <div class="metric-label">Time Data</div>
    </div>
    """, unsafe_allow_html=True)
with col3: 
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📈</div>
        <div class="metric-value">Advanced</div>
        <div class="metric-label">Charts</div>
    </div>
    """, unsafe_allow_html=True)
with col4: 
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🔧</div>
        <div class="metric-value">Pro</div>
        <div class="metric-label">Tools</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Feature Cards - ĐÃ CẢI TIẾN ---
st.markdown('<div class="section-header">🌟 CORE FEATURES</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Multi-Asset Data</div>
        <div class="feature-desc">Access comprehensive market data across Crypto, Forex, and Stocks with real-time updates from multiple reliable sources</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Advanced Charts</div>
        <div class="feature-desc">Interactive candlestick charts with volume analysis and professional technical indicators for in-depth market analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📥</div>
        <div class="feature-title">Export Data</div>
        <div class="feature-desc">Download clean, formatted data in CSV format for further analysis, backtesting, and integration with other tools</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Configuration Section - ĐÃ TÁI CẤU TRÚC ---
st.markdown('<div class="config-section">', unsafe_allow_html=True)
st.markdown('<div class="config-header">🎯 DATA CONFIGURATION</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📦 Asset Type")
    asset_class = st.radio(
        "Select Asset Class:",
        ["Crypto", "Forex", "Stocks"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Asset-specific configuration với badge
    if asset_class == "Crypto":
        st.markdown('<div class="asset-badge">💱 Crypto Assets</div>', unsafe_allow_html=True)
        symbol = st.text_input(
            "**Trading Pair**",
            value="BTC/USDT",
            placeholder="BTC/USDT, ETH/USDT, ADA/USDT...",
            help="Enter cryptocurrency trading pair"
        )
        tf = st.selectbox("**Timeframe**", ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'], index=4)
        st.markdown('<div class="data-source-badge">Data Source: CCXT Multi-Exchange</div>', unsafe_allow_html=True)
        
    elif asset_class == "Forex":
        st.markdown('<div class="asset-badge">🌍 Forex Pairs</div>', unsafe_allow_html=True)
        symbol = st.text_input(
            "**Forex Pair**",
            value="EURUSD=X",
            placeholder="EURUSD=X, GBPUSD=X, USDJPY=X...",
            help="Enter Forex pair symbol"
        )
        tf = st.selectbox("**Timeframe**", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], index=4)
        st.markdown('<div class="data-source-badge">Data Source: Yahoo Finance</div>', unsafe_allow_html=True)
        
    else:  # Stocks
        st.markdown('<div class="asset-badge">📈 Stocks</div>', unsafe_allow_html=True)
        symbol = st.text_input(
            "**Stock Symbol**",
            value="AAPL",
            placeholder="AAPL, TSLA, MSFT, SPY...",
            help="Enter stock ticker symbol"
        )
        tf = st.selectbox("**Timeframe**", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], index=5)
        st.markdown('<div class="data-source-badge">Data Source: Yahoo Finance</div>', unsafe_allow_html=True)

with col2:
    st.markdown("#### 📅 Date Range")
    
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(years=1)
    
    start_date = st.date_input("**Start Date**", start_date)
    end_date = st.date_input("**End Date**", end_date)
    
    # Quick date presets với design mới
    st.markdown("#### ⚡ Quick Presets")
    preset_cols = st.columns(3)
    with preset_cols[0]:
        if st.button("**1 Month**", use_container_width=True, type="secondary"):
            start_date = end_date - pd.DateOffset(months=1)
            st.rerun()
    with preset_cols[1]:
        if st.button("**3 Months**", use_container_width=True, type="secondary"):
            start_date = end_date - pd.DateOffset(months=3)
            st.rerun()
    with preset_cols[2]:
        if st.button("**1 Year**", use_container_width=True, type="secondary"):
            start_date = end_date - pd.DateOffset(years=1)
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Download Button với design mới
download_btn = st.button("🚀 Download Market Data", type="primary", use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- GIỮ NGUYÊN CÁC HÀM DATA LOADING ---
def get_crypto_data_simple(symbol='BTC/USDT', timeframe='1h', limit=500):
    """Simple data fetcher using multiple exchanges - THAY THẾ BINANCE API"""
    
    exchanges = [
        {'name': 'bybit', 'class': ccxt.bybit},
        {'name': 'okx', 'class': ccxt.okx},
        {'name': 'kucoin', 'class': ccxt.kucoin},
        {'name': 'gateio', 'class': ccxt.gateio},
        {'name': 'htx', 'class': ccxt.htx},
    ]
    
    for exchange_info in exchanges:
        try:
            exchange = exchange_info['class']({
                'timeout': 30000,
                'enableRateLimit': True,
            })
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            continue
    
    return get_yahoo_fallback(symbol)

def get_yahoo_fallback(symbol):
    """Fallback to Yahoo Finance"""
    symbol_map = {
        'BTC/USDT': 'BTC-USD',
        'ETH/USDT': 'ETH-USD', 
        'BNB/USDT': 'BNB-USD',
        'ADA/USDT': 'ADA-USD',
        'XRP/USDT': 'XRP-USD',
        'DOT/USDT': 'DOT-USD',
        'LINK/USDT': 'LINK-USD',
        'LTC/USDT': 'LTC-USD',
        'BCH/USDT': 'BCH-USD',
        'SOL/USDT': 'SOL-USD'
    }
    
    yahoo_symbol = symbol_map.get(symbol, 'BTC-USD')
    try:
        data = yf.download(yahoo_symbol, period='6mo', interval='1h')
        return data
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_data(asset, sym, timeframe, start_dt, end_dt):
    """
    Tải dữ liệu an toàn dựa trên khoảng thời gian (Start Date & End Date).
    """
    with st.spinner(f"🔍 Downloading {asset} data for {sym}..."):
        try:
            start_datetime = pd.to_datetime(start_dt)
            end_datetime = pd.to_datetime(end_dt)

            if asset == "Crypto":
                df = get_crypto_data_simple(sym, timeframe, 1000)
                
                if df is not None and not df.empty:
                    df = df.loc[start_datetime:end_datetime]
                    return df
                else:
                    st.error(f"❌ Cannot load Crypto data for {sym}")
                    return None
                    
            else:
                data = yf.download(sym, start=start_datetime, end=end_datetime, interval=timeframe, progress=False, auto_adjust=True)
                
                new_columns = []
                for col in data.columns:
                    if isinstance(col, tuple):
                        new_columns.append(col[0].capitalize())
                    else:
                        new_columns.append(str(col).capitalize())
                data.columns = new_columns

            if data is None or data.empty:
                st.error(f"❌ Cannot download {sym}. Check symbol or parameters.")
                return None
            
            return data
        
        except Exception as e:
            st.error(f"❌ System error downloading {sym}: {e}")
            return None

# --- Results Section - ĐÃ TÁI THIẾT ---
if download_btn:
    if start_date >= end_date:
        st.error("""
        ❌ **Invalid Date Range**
        
        Start Date must be before End Date.
        Please adjust your date selection.
        """)
    else:
        df = load_data(asset_class, symbol, tf, start_date, end_date)
        
        if df is not None and not df.empty and len(df) > 1:
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ Data for {symbol} missing required columns. Found: {list(df.columns)}")
            else:
                st.success(f"✅ Successfully downloaded {len(df)} rows for {symbol}")

                # Market Overview với design mới
                st.markdown('<div class="section-header">📊 MARKET OVERVIEW</div>', unsafe_allow_html=True)
                
                latest_data = df.iloc[-1]
                previous_data = df.iloc[-2]
                change = latest_data['Close'] - previous_data['Close']
                change_pct = (change / previous_data['Close']) * 100
                
                period_high = df['High'].max()
                period_low = df['Low'].min()
                period_avg_volume = df['Volume'].mean()
                volatility = (df['High'] - df['Low']).mean()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric(
                        "💵 Current Price", 
                        f"{latest_data['Close']:,.4f}", 
                        f"{change:+.4f} ({change_pct:+.2f}%)",
                        delta_color=delta_color
                    )
                
                with col2:
                    st.metric("🏆 Period High", f"{period_high:,.4f}")
                
                with col3:
                    st.metric("📉 Period Low", f"{period_low:,.4f}")
                
                with col4:
                    st.metric("📈 Avg Volume", f"{period_avg_volume:,.0f}")

                # Advanced Charts
                st.markdown('<div class="section-header">📈 ADVANCED CHART ANALYSIS</div>', unsafe_allow_html=True)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.1, 
                                    subplot_titles=(f'{symbol} Price Chart', 'Volume'),
                                    row_heights=[0.7, 0.3])

                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol
                ), row=1, col=1)

                # Volume chart
                colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)

                fig.update_layout(
                    yaxis_title='Price',
                    yaxis2_title='Volume',
                    template='plotly_dark',
                    height=600,
                    xaxis_rangeslider_visible=False,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data Table and Export
                st.markdown('<div class="section-header">🔬 DATA EXPLORATION</div>', unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs(["📋 Data Table", "📥 Export Data"])
                
                with tab1:
                    st.markdown("""
                    <div class="analysis-card">
                        <h4>📋 Raw Data Preview</h4>
                        <p style="color: #8898aa;">Complete dataset with formatted values for detailed analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df.style.format({
                        'Open': '{:.4f}',
                        'High': '{:.4f}', 
                        'Low': '{:.4f}',
                        'Close': '{:.4f}',
                        'Volume': '{:,.0f}'
                    }), use_container_width=True)
                
                with tab2:
                    st.markdown("""
                    <div class="analysis-card">
                        <div class="feature-icon">📥</div>
                        <h4>Export Market Data</h4>
                        <p style="color: #8898aa;">Download the complete dataset for further analysis in your preferred tools</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    csv = df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"{symbol.replace('/', '_')}_{tf}_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
        
        elif df is not None and len(df) <= 1:
            st.warning("""
            ⚠️ **Insufficient Data**
            
            Not enough data points to display meaningful analysis.
            Try adjusting your date range or timeframe.
            """)
        else:
            st.info("📊 Data process completed. Check notifications above for any issues.")
else:
    # Welcome state với design mới
    st.markdown('<div class="section-header">💡 GETTING STARTED</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <h4>📊 Supported Assets</h4>
            <div class="feature-desc">
            <strong>💱 Crypto:</strong> BTC/USDT, ETH/USDT, ADA/USDT...<br>
            <strong>🌍 Forex:</strong> EURUSD=X, GBPUSD=X, USDJPY=X...<br>
            <strong>📈 Stocks:</strong> AAPL, TSLA, MSFT, SPY...<br>
            <strong>⏰ Timeframes:</strong> 1m to 1Month
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-card">
            <h4>⚡ Quick Tips</h4>
            <div class="feature-desc">
            • Use quick date presets for common periods<br>
            • Higher timeframes = faster loading<br>
            • Crypto data via CCXT (multi-exchange)<br>
            • Stocks/Forex via Yahoo Finance<br>
            • Export for backtesting & analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer với design mới
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Market Data Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Data Center Pro v3.0 - Multi-Asset Analysis</p>
</div>
""", unsafe_allow_html=True)