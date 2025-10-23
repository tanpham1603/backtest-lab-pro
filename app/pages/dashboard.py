import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ccxt
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import requests
import os
import time

# ===========================
# ⚙️ CONFIGURATION
# ===========================
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================
# 🧭 SIDEBAR
# ===========================
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("⚙️ Dashboard setting")

    asset_class = st.selectbox("Asset Class:", ["Crypto", "Forex", "Stocks"], key="dashboard_asset")
    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    if asset_class == "Crypto":
        symbol = st.text_input("Pairs:", "BTC/USDT", key="dashboard_crypto_symbol")
        tf = st.selectbox("Timeframe:", common_timeframes, index=4, key="dashboard_crypto_tf")
        limit = st.slider("Number of candles:", 100, 2000, 500, key="dashboard_crypto_limit")
        start_date = None
        end_date = None
    else:
        symbol = st.text_input("Pairs:", "EURUSD=X" if asset_class == "Forex" else "AAPL", key="dashboard_stock_symbol")
        tf = st.selectbox("Timeframe:", common_timeframes, index=6, key="dashboard_stock_tf")
        limit = st.slider("Number of candles:", 100, 2000, 500, key="dashboard_stock_limit")

        st.subheader("Date Range (Optional)")
        yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
        
        end_date_default = datetime.now().date()
        start_date_default = end_date_default - timedelta(days=365*2)
        info_message = ""

        if tf in yf_timeframe_limits:
            day_limit = yf_timeframe_limits[tf]
            start_date_default = end_date_default - timedelta(days=day_limit - 1)
            info_message = f"Suggestion: {tf} timeframe is limited to {day_limit} days."

        end_date = st.date_input("End date", value=end_date_default)
        start_date = st.date_input("Start date", value=start_date_default)
        if info_message:
            st.caption(info_message)

    st.sidebar.subheader("📈 Technical Indicators")
    show_ma = st.sidebar.checkbox("Moving Averages (20, 50)", value=True)
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)")
    show_rsi = st.sidebar.checkbox("RSI (14)")

# ===========================
# 📦 LOAD DATA FUNCTION
# ===========================
@st.cache_data(ttl=300)
def load_dashboard_data(asset, sym, timeframe, data_limit, start_dt, end_dt):
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=data_limit)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else:
            yf_timeframe_map = {"1w": "1wk"}
            interval = yf_timeframe_map.get(timeframe, timeframe)
            data = yf.download(sym, start=start_dt, end=end_dt, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]
            data = data.tail(data_limit)

        if data.empty:
            st.error(f"Cannot retrieve data for {sym}.")
            return None
        return data
    except Exception as e:
        st.error(f"System error while loading data for {sym}: {e}")
        return None

# ===========================
# 🧮 DUNE API FETCH FUNCTION
# ===========================
def fetch_dune_query_results(query_id: int):
    """Lấy kết quả từ Dune API"""
    try:
        DUNE_API_KEY = st.secrets["DUNE_API_KEY"]
    except Exception:
        st.error("❌ Missing DUNE_API_KEY in Streamlit secrets.")
        return None

    url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    headers = {"x-dune-api-key": DUNE_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"Dune API error: {response.status_code}")
        return None

    data = response.json()
    if "result" not in data or "rows" not in data["result"]:
        st.warning("⚠️ No data found in Dune query result.")
        return None

    return pd.DataFrame(data["result"]["rows"])

# ===========================
# 🐋 WHALE RATIO FUNCTIONS - ETHERSCAN API FIXED
# ===========================
def get_etherscan_api_key():
    """Lấy Etherscan API Key từ secrets"""
    try:
        return st.secrets["ETHERSCAN_API_KEY"]
    except:
        st.error("❌ Missing ETHERSCAN_API_KEY in Streamlit secrets.")
        return None

def debug_etherscan_api(token_address):
    """Debug function to check Etherscan API"""
    api_key = get_etherscan_api_key()
    
    st.write("🔧 **Debug Information:**")
    st.write(f"- API Key: {'✅ Found' if api_key else '❌ Missing'}")
    st.write(f"- Token Address: {token_address}")
    
    # Test multiple endpoints
    endpoints = {
        "tokenSupply": f"https://api.etherscan.io/api?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}",
        "tokenBalance": f"https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress={token_address}&address=0x0000000000000000000000000000000000000000&tag=latest&apikey={api_key}",
        "getSourceCode": f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={token_address}&apikey={api_key}"
    }
    
    for endpoint_name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=10)
            st.write(f"\n**{endpoint_name}:**")
            st.write(f"- Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                st.write(f"- API Status: {data.get('status')}")
                st.write(f"- Message: {data.get('message')}")
                if data.get('result'):
                    result_str = str(data['result'])
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    st.write(f"- Result: {result_str}")
        except Exception as e:
            st.write(f"- Error: {e}")

def get_token_supply_etherscan(token_address):
    """Lấy total supply từ Etherscan - endpoint đơn giản hơn"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        url = f"https://api.etherscan.io/api?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                supply = int(data['result'])
                st.success(f"✅ Total Supply: {supply:,}")
                return supply
            else:
                st.warning(f"⚠️ Etherscan API: {data.get('message', 'Unknown error')}")
        return None
    except Exception as e:
        st.error(f"Lỗi lấy token supply: {e}")
        return None

def get_token_holders_etherscan(token_address, page=1, offset=100):
    """Lấy danh sách holders từ Etherscan API - FIXED VERSION"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        url = f"https://api.etherscan.io/api?module=token&action=tokenholderlist&contractaddress={token_address}&page={page}&offset={offset}&apikey={api_key}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                return data['result']
            else:
                st.warning(f"Etherscan API: {data.get('message', 'Unknown error')}")
                return None
        else:
            st.error(f"Lỗi HTTP: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Timeout khi kết nối đến Etherscan API")
        return None
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")
        return None

def get_token_holders_with_retry(token_address, max_holders=100):
    """
    Lấy holders với retry logic và giới hạn số lượng
    """
    all_holders = []
    page = 1
    offset = 100  # Etherscan limit per page
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while len(all_holders) < max_holders:
        status_text.text(f"Đang tải trang {page}... ({len(all_holders)} holders)")
        progress_bar.progress(min(len(all_holders) / max_holders, 1.0))
        
        holders = get_token_holders_etherscan(token_address, page, offset)
        
        if not holders:
            break
            
        all_holders.extend(holders)
        
        # Nếu số holders trả về ít hơn offset, có nghĩa là đã hết
        if len(holders) < offset:
            break
            
        page += 1
        
        # Rate limiting - tránh bị block
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    return all_holders[:max_holders]

def calculate_whale_metrics(holders_data, total_supply):
    """
    Tính toán các chỉ số whale từ dữ liệu holders
    """
    if not holders_data or total_supply == 0:
        return None
    
    # Sắp xếp holders theo balance giảm dần
    sorted_holders = sorted(holders_data, key=lambda x: float(x.get('value', 0)), reverse=True)
    
    # Tính toán các metrics
    top_10_balance = sum(float(holder.get('value', 0)) for holder in sorted_holders[:10])
    top_20_balance = sum(float(holder.get('value', 0)) for holder in sorted_holders[:20])
    top_50_balance = sum(float(holder.get('value', 0)) for holder in sorted_holders[:50])
    
    metrics = {
        'total_holders': len(holders_data),
        'total_supply': total_supply,
        'whale_ratio_10': (top_10_balance / total_supply) * 100,
        'whale_ratio_20': (top_20_balance / total_supply) * 100,
        'whale_ratio_50': (top_50_balance / total_supply) * 100,
        'top_10_holders': sorted_holders[:10],
        'top_20_holders': sorted_holders[:20],
        'gini_coefficient': calculate_gini_coefficient(sorted_holders, total_supply)
    }
    
    return metrics

def calculate_gini_coefficient(holders, total_supply):
    """
    Tính hệ số Gini - đo lường độ tập trung
    """
    if total_supply == 0:
        return 0
    
    balances = [float(holder.get('value', 0)) for holder in holders]
    balances.sort()
    
    n = len(balances)
    if n == 0:
        return 0
        
    cumulative_balances = [sum(balances[:i+1]) for i in range(n)]
    
    # Area under Lorenz curve
    area_under_curve = sum(cumulative_balances) / cumulative_balances[-1] if cumulative_balances[-1] > 0 else 0
    
    # Gini coefficient
    gini = (n + 1 - 2 * area_under_curve) / n
    return max(0, min(1, gini))  # Đảm bảo trong khoảng 0-1

def create_whale_chart(metrics):
    """
    Tạo biểu đồ whale distribution
    """
    if not metrics:
        return None
    
    # Data for chart
    categories = ['Top 10', 'Top 20', 'Top 50']
    percentages = [
        metrics['whale_ratio_10'],
        metrics['whale_ratio_20'], 
        metrics['whale_ratio_50']
    ]
    
    fig = go.Figure()
    
    # Whale ratio bars
    fig.add_trace(go.Bar(
        x=categories,
        y=percentages,
        name='Whale Ratio',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text=[f'{p:.1f}%' for p in percentages],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Whale Concentration Ratios',
        xaxis_title='Holder Groups',
        yaxis_title='Percentage of Total Supply (%)',
        showlegend=False,
        height=400,
        template='plotly_dark'
    )
    
    return fig

def create_distribution_pie(metrics):
    """
    Tạo pie chart phân bổ supply
    """
    if not metrics:
        return None
    
    top_10_supply = metrics['whale_ratio_10']
    top_11_20_supply = metrics['whale_ratio_20'] - metrics['whale_ratio_10']
    top_21_50_supply = metrics['whale_ratio_50'] - metrics['whale_ratio_20']
    rest_supply = 100 - metrics['whale_ratio_50']
    
    labels = ['Top 10 Whales', 'Top 11-20', 'Top 21-50', 'Rest Holders']
    values = [top_10_supply, top_11_20_supply, top_21_50_supply, rest_supply]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = px.pie(
        names=labels, 
        values=values,
        title='Token Supply Distribution',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_dark')
    
    return fig

# ===========================
# 🔧 CALLBACK FUNCTIONS FOR SESSION STATE
# ===========================
def set_token_address(token_addr):
    """Callback function to set token address in session state"""
    st.session_state.whale_token_address = token_addr
    st.rerun()

# Initialize session state for whale token address
if "whale_token_address" not in st.session_state:
    st.session_state.whale_token_address = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

# ===========================
# 🖥️ MAIN UI
# ===========================
st.title("📊 Market Overview")

data = load_dashboard_data(asset_class, symbol, tf, limit, start_date, end_date)

if data is not None and not data.empty:
    st.success(f"✅ Loaded {len(data)} candles for {symbol}.")
    
    # Indicators
    if show_ma:
        data.ta.sma(length=20, append=True)
        data.ta.sma(length=50, append=True)
    if show_bb:
        data.ta.bbands(length=20, std=2, append=True)
    if show_rsi:
        data.ta.rsi(length=14, append=True)

    # Overview
    st.subheader("Cycle Overview")
    latest_data = data.iloc[-1]
    change = latest_data['Close'] - data.iloc[-2]['Close']
    change_pct = (change / data.iloc[-2]['Close']) * 100
    period_high = data['High'].max()
    period_low = data['Low'].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Price", f"${latest_data['Close']:,.4f}", f"{change:,.4f} ({change_pct:.2f}%)")
    col2.metric(f"Highest Price ({len(data)} candles)", f"${period_high:,.4f}")
    col3.metric(f"Lowest Price ({len(data)} candles)", f"${period_low:,.4f}")
    col4.metric("Latest Volume", f"{latest_data['Volume']:,.0f}")

    # Chart
    st.subheader(f"Price Chart and Indicators for {symbol}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Price'))
    if show_ma:
        fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_20'), mode='lines', name='SMA 20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data.index, y=data.get('SMA_50'), mode='lines', name='SMA 50', line=dict(color='blue')))
    if show_bb:
        fig.add_trace(go.Scatter(x=data.index, y=data.get('BBU_20_2.0'), mode='lines', name='Upper Band', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data.get('BBL_20_2.0'), mode='lines', name='Lower Band', line=dict(color='gray', dash='dash')))
    fig.update_layout(yaxis_title='Price', template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        st.subheader("RSI Indicator")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data.get('RSI_14'), mode='lines', name='RSI', line=dict(color='purple')))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(template='plotly_dark', height=250)
        st.plotly_chart(rsi_fig, use_container_width=True)

    with st.expander("🔬 View Historical Data"):
        st.dataframe(data.tail(100))

    # ===========================
    # 🐋 WHALE RATIO SECTION - ETHERSCAN API FIXED
    # ===========================
    st.markdown("---")
    st.header("🐋 Whale Ratio Analysis - Etherscan API")
    
    # Tabs for different whale ratio methods
    tab1, tab2 = st.tabs(["🔍 Etherscan API", "📊 Dune Analytics"])
    
    with tab1:
        st.subheader("Token Holder Concentration Analysis")
        
        # Debug button
        if st.button("🔧 Debug Etherscan API"):
            debug_etherscan_api(st.session_state.whale_token_address)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            token_address = st.text_input(
                "Token Contract Address (ETH):",
                placeholder="0x...",
                value=st.session_state.whale_token_address,
                key="whale_token_address_input"
            )
        
        with col2:
            st.markdown("###")
            analyze_btn = st.button("🚀 Analyze Whale Ratio", type="primary", key="whale_analyze_btn")
        
        if analyze_btn and token_address:
            with st.spinner("Đang phân tích whale ratio..."):
                # Validate token address format
                if not token_address.startswith("0x") or len(token_address) != 42:
                    st.error("❌ Địa chỉ token không hợp lệ. Phải bắt đầu bằng 0x và có 42 ký tự.")
                else:
                    # Lấy total supply trước
                    total_supply = get_token_supply_etherscan(token_address)
                    
                    if total_supply and total_supply > 0:
                        # Lấy danh sách holders
                        holders_data = get_token_holders_with_retry(token_address, max_holders=100)
                        
                        if holders_data:
                            # Tính toán metrics
                            metrics = calculate_whale_metrics(holders_data, total_supply)
                            
                            if metrics:
                                # Hiển thị KPI cards
                                st.success("✅ Phân tích thành công!")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        label="Top 10 Whale Ratio", 
                                        value=f"{metrics['whale_ratio_10']:.1f}%",
                                        delta="High" if metrics['whale_ratio_10'] > 50 else "Medium"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Total Holders", 
                                        value=f"{metrics['total_holders']:,}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        label="Gini Coefficient", 
                                        value=f"{metrics['gini_coefficient']:.3f}",
                                        delta="Concentrated" if metrics['gini_coefficient'] > 0.6 else "Distributed"
                                    )
                                
                                with col4:
                                    risk_level = "HIGH" if metrics['whale_ratio_10'] > 60 else "MEDIUM" if metrics['whale_ratio_10'] > 30 else "LOW"
                                    st.metric(
                                        label="Whale Risk Level", 
                                        value=risk_level
                                    )
                                
                                # Charts
                                col_chart1, col_chart2 = st.columns(2)
                                
                                with col_chart1:
                                    whale_chart = create_whale_chart(metrics)
                                    if whale_chart:
                                        st.plotly_chart(whale_chart, use_container_width=True)
                                
                                with col_chart2:
                                    pie_chart = create_distribution_pie(metrics)
                                    if pie_chart:
                                        st.plotly_chart(pie_chart, use_container_width=True)
                                
                                # Top whales table
                                st.subheader("🏆 Top 10 Whale Holders")
                                
                                top_whales_data = []
                                for i, holder in enumerate(metrics['top_10_holders'], 1):
                                    balance = float(holder.get('value', 0))
                                    percentage = (balance / metrics['total_supply']) * 100
                                    
                                    top_whales_data.append({
                                        'Rank': i,
                                        'Address': holder.get('TokenHolderAddress', '')[:20] + '...',
                                        'Balance': f"{balance:,.2f}",
                                        'Percentage': f"{percentage:.2f}%"
                                    })
                                
                                df_whales = pd.DataFrame(top_whales_data)
                                st.dataframe(df_whales, use_container_width=True, hide_index=True)
                                
                                # Risk assessment
                                st.subheader("📊 Đánh giá rủi ro")
                                
                                if metrics['whale_ratio_10'] > 70:
                                    st.error("**CẢNH BÁO CAO**: Token rất tập trung, top 10 holders nắm giữ hơn 70% supply!")
                                elif metrics['whale_ratio_10'] > 50:
                                    st.warning("**CẢNH BÁO**: Token khá tập trung, top 10 holders nắm giữ hơn 50% supply")
                                elif metrics['whale_ratio_10'] > 30:
                                    st.info("**TRUNG BÌNH**: Token có mức độ tập trung vừa phải")
                                else:
                                    st.success("**TỐT**: Token phân bổ khá đồng đều")
                                    
                            else:
                                st.error("Không thể tính toán metrics từ dữ liệu")
                        else:
                            st.error("Không thể lấy dữ liệu holders từ Etherscan. Token có thể không có holders hoặc API limit.")
                    else:
                        st.error("Không thể lấy total supply từ Etherscan. Token address có thể không phải ERC-20 contract.")
        
        # Quick test buttons với các token phổ biến - SỬ DỤNG CALLBACK
        st.subheader("🚀 Test nhanh với các token phổ biến")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("USDT", use_container_width=True, key="usdt_btn"):
                set_token_address("0xdAC17F958D2ee523a2206206994597C13D831ec7")
        
        with col2:
            if st.button("USDC", use_container_width=True, key="usdc_btn"):
                set_token_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
        
        with col3:
            if st.button("UNI", use_container_width=True, key="uni_btn"):
                set_token_address("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984")
        
        with col4:
            if st.button("SHIB", use_container_width=True, key="shib_btn"):
                set_token_address("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE")
        
        # Hiển thị token address hiện tại
        st.info(f"🔍 Token address hiện tại: `{st.session_state.whale_token_address}`")
        
        # Hướng dẫn sử dụng
        with st.expander("ℹ️ Hướng dẫn sử dụng Etherscan API"):
            st.markdown("""
            **Cách sử dụng:**
            1. Nhập địa chỉ token Ethereum cần phân tích (bắt đầu bằng 0x)
            2. Click "Analyze Whale Ratio" hoặc chọn token test nhanh
            3. Xem kết quả phân tích
            
            **Cần có Etherscan API Key:**
            - Đăng ký miễn phí tại: https://etherscan.io/register
            - Lấy API Key tại: https://etherscan.io/myapikey
            - Thêm vào Streamlit secrets với key: `ETHERSCAN_API_KEY`
            
            **Chỉ số quan trọng:**
            - **Top 10 Whale Ratio**: % supply mà top 10 holders nắm giữ
            - **Gini Coefficient**: Độ tập trung (0 = hoàn toàn phân tán, 1 = hoàn toàn tập trung)
            - **Total Holders**: Tổng số địa chỉ nắm giữ token
            
            **Lưu ý:**
            - Etherscan có rate limits (5 calls/sec cho free tier)
            - Một số token mới có thể chưa có đủ dữ liệu
            - Dùng nút **Debug** để kiểm tra API
            """)
    
    with tab2:
        st.subheader("Dune Analytics Query")
        
        dune_query_id = st.text_input("Enter Dune Query ID:", value="", placeholder="e.g. 1234567", key="dune_query_input")

        if dune_query_id:
            df_dune = fetch_dune_query_results(int(dune_query_id))

            if df_dune is not None and not df_dune.empty:
                try:
                    total_inflow = df_dune['total_inflow'].sum()
                    top10_inflow = df_dune['top10_inflow'].sum()
                    whale_ratio = top10_inflow / total_inflow

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Inflow", f"{total_inflow:,.0f}")
                    c2.metric("Top 10 Inflow", f"{top10_inflow:,.0f}")
                    c3.metric("Whale Ratio", f"{whale_ratio*100:.2f}%")

                    st.caption("Whale Ratio = inflow(top 10 wallets) / inflow(total network)")
                    
                    # Hiển thị dữ liệu thô từ Dune
                    with st.expander("View Dune Data"):
                        st.dataframe(df_dune)
                        
                except Exception as e:
                    st.error(f"Error calculating Whale Ratio: {e}")
            else:
                st.warning("No data retrieved from Dune query.")

else:
    st.info("👈 Please configure the left sidebar to view data.")

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit • Etherscan API • Dune Analytics*")