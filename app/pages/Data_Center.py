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

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
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
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        margin: 2rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header-gradient">🚀 Data Center Pro</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #8898aa; font-size: 1.2rem; margin-bottom: 3rem;">Professional Market Data Platform</div>', unsafe_allow_html=True)

# --- HÀM TẢI DỮ LIỆU (GIỮ NGUYÊN LOGIC) ---
@st.cache_data(ttl=300)
def load_data(asset, sym, timeframe, start_dt, end_dt):
    """
    Tải dữ liệu an toàn dựa trên khoảng thời gian (Start Date & End Date).
    """
    with st.spinner(f"Downloading data for {sym} from {start_dt} to {end_dt}..."):
        try:
            # Chuyển đổi datetime để đảm bảo tính chính xác
            start_datetime = pd.to_datetime(start_dt)
            end_datetime = pd.to_datetime(end_dt)

            if asset == "Crypto":
                exchange = ccxt.kucoin()
                if not exchange.has['fetchOHLCV']:
                    st.error(f"Sàn {exchange.id} không hỗ trợ tải dữ liệu OHLCV.")
                    return None
                    
                since = int(start_datetime.timestamp() * 1000)
                end_ts = int(end_datetime.timestamp() * 1000)
                
                all_ohlcv = []
                # Dùng vòng lặp để tải toàn bộ lịch sử trong khoảng thời gian
                while since < end_ts:
                    ohlcv = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1 # +1ms để tránh lặp
                
                if not all_ohlcv:
                    st.error(f"Không thể tải dữ liệu Crypto cho {sym}.")
                    return None
                    
                data = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
                # Lọc lại chính xác theo ngày
                data = data.loc[start_datetime:end_datetime]

            else: # Forex và Stocks
                # yfinance tải trực tiếp bằng start và end, chính xác hơn
                data = yf.download(sym, start=start_datetime, end=end_datetime, interval=timeframe, progress=False, auto_adjust=True)
                
                # Logic xử lý tên cột phức tạp từ yfinance (Giữ nguyên logic của bạn vì nó tốt)
                new_columns = []
                for col in data.columns:
                    if isinstance(col, tuple):
                        new_columns.append(col[0].capitalize())
                    else:
                        new_columns.append(str(col).capitalize())
                data.columns = new_columns

            if data is None or data.empty:
                st.error(f"Cannot download {sym}. API may wrong or you choose wrong parameters.")
                return None
            
            return data
        
        except Exception as e:
            st.error(f"System error when downloading data for {sym}: {e}")
            return None

# --- Welcome Section ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3>Multi-Asset Data</h3>
        <p style="color: #8898aa;">Access comprehensive market data across Crypto, Forex, and Stocks with real-time updates</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h3>Advanced Charts</h3>
        <p style="color: #8898aa;">Interactive candlestick charts with volume analysis and professional technical indicators</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📥</div>
        <h3>Export Data</h3>
        <p style="color: #8898aa;">Download clean, formatted data in CSV format for further analysis and backtesting</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Configuration Section ---
with st.container():
    st.markdown("### ⚙️ Data Configuration")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Asset & Symbol")
        asset_class = st.radio("Asset Type:", ["Crypto", "Forex", "Stocks"], 
                              horizontal=True, label_visibility="collapsed")
        
        # Asset-specific configuration
        if asset_class == "Crypto":
            symbol = st.text_input("Trading Pair:", "BTC/USDT", 
                                 placeholder="BTC/USDT, ETH/USDT, ADA/USDT...")
            tf = st.selectbox("Timeframe:", ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'], index=4)
            st.info("💱 **Crypto Data**: Using CCXT for major exchanges")
            
        elif asset_class == "Forex":
            symbol = st.text_input("Forex Pair:", "EURUSD=X", 
                                 placeholder="EURUSD=X, GBPUSD=X, USDJPY=X...")
            tf = st.selectbox("Timeframe:", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], index=4)
            st.info("🌍 **Forex Data**: Yahoo Finance integration")
            
        else: # Stocks
            symbol = st.text_input("Stock Symbol:", "AAPL", 
                                 placeholder="AAPL, TSLA, MSFT, SPY...")
            tf = st.selectbox("Timeframe:", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'], index=5)
            st.info("📈 **Stock Data**: Yahoo Finance integration")

    with col2:
        st.markdown("##### Date Range")
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=1)
        
        start_date = st.date_input("Start Date", start_date)
        end_date = st.date_input("End Date", end_date)
        
        # Quick date presets
        st.markdown("##### Quick Presets")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("1 Month", use_container_width=True):
                start_date = end_date - pd.DateOffset(months=1)
                st.rerun()
        with col_p2:
            if st.button("3 Months", use_container_width=True):
                start_date = end_date - pd.DateOffset(months=3)
                st.rerun()
        with col_p3:
            if st.button("1 Year", use_container_width=True):
                start_date = end_date - pd.DateOffset(years=1)
                st.rerun()

# Download Button
run_button = st.button("🚀 Download Market Data", type="primary", use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Results Section ---
if run_button:
    # Validation
    if start_date >= end_date:
        st.error("❌ Start Date must be before End Date.")
    else:
        # Gọi hàm load_data đã nâng cấp
        df = load_data(asset_class, symbol, tf, start_date, end_date)
        
        if df is not None and not df.empty and len(df) > 1:
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Data for {symbol} does not have all the required columns. Current columns: {list(df.columns)}")
            else:
                st.success(f"✅ Successfully downloaded {len(df)} rows of data for {symbol}")

                # Data Overview with new UI
                st.markdown("### 📊 Market Overview")
                
                latest_data = df.iloc[-1]
                previous_data = df.iloc[-2]
                change = latest_data['Close'] - previous_data['Close']
                change_pct = (change / previous_data['Close']) * 100
                
                period_high = df['High'].max()
                period_low = df['Low'].min()
                period_avg_volume = df['Volume'].mean()
                volatility = (df['High'] - df['Low']).mean()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"{latest_data['Close']:,.4f}", 
                           f"{change:+.4f} ({change_pct:+.2f}%)")
                col2.metric("Period High", f"{period_high:,.4f}")
                col3.metric("Period Low", f"{period_low:,.4f}")
                col4.metric("Avg Volume", f"{period_avg_volume:,.0f}")

                # Advanced Charts
                st.markdown("### 📈 Advanced Chart Analysis")
                
                # Tạo 2 biểu đồ con, chia sẻ trục X
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.1, 
                                    subplot_titles=(f'{symbol} Price Chart', 'Volume'),
                                    row_heights=[0.7, 0.3])

                # 1. Biểu đồ nến
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol
                ), row=1, col=1)

                # 2. Biểu đồ Volume
                colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)

                # Cập nhật layout
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
                st.markdown("### 🔬 Raw Data & Export")
                
                tab1, tab2 = st.tabs(["📋 Data Table", "📥 Export Data"])
                
                with tab1:
                    st.dataframe(df.style.format({
                        'Open': '{:.4f}',
                        'High': '{:.4f}', 
                        'Low': '{:.4f}',
                        'Close': '{:.4f}',
                        'Volume': '{:,.0f}'
                    }), use_container_width=True)
                
                with tab2:
                    st.markdown("""
                    <div class="feature-card">
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
            st.warning("⚠️ Not enough data to display (at least 2 rows are required).")

        else:
            st.info("📊 Data downloading process has completed. If there are errors, notifications will be displayed above.")
else:
    # Welcome message when no data loaded
    st.markdown("### 💡 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Supported Assets</h4>
            <p style="color: #8898aa;">
            <strong>Crypto:</strong> BTC/USDT, ETH/USDT, ADA/USDT...<br>
            <strong>Forex:</strong> EURUSD=X, GBPUSD=X, USDJPY=X...<br>
            <strong>Stocks:</strong> AAPL, TSLA, MSFT, SPY...<br>
            <strong>Timeframes:</strong> 1m to 1Month
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Quick Tips</h4>
            <p style="color: #8898aa;">
            • Use quick date presets for common periods<br>
            • Higher timeframes = faster loading<br>
            • Crypto data via CCXT (real-time)<br>
            • Stocks/Forex via Yahoo Finance<br>
            • Export for backtesting & analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Data Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Data Center Pro v2.0</p>
</div>
""", unsafe_allow_html=True)