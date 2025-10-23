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
        DUNE_API_KEY = "64Yf8r2u9IZd0PAQJp23w4VHkL3RvIi0"
    except Exception:
        st.error("❌ Missing DUNE_API_KEY")
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
# 🐋 WHALE RATIO FUNCTIONS - COVALENT API
# ===========================
def get_covalent_api_key():
    """Lấy Covalent API Key"""
    try:
        # Ưu tiên từ secrets
        return st.secrets["COVALENT_API_KEY"]
    except:
        try:
            # Fallback: demo key (có giới hạn)
            return "ckey_6c1e3e7e3e3e3e3e3e3e3e3e3e"
        except:
            st.error("❌ Missing COVALENT_API_KEY")
            return None

def debug_covalent_api(token_address, chain_id=1):
    """Debug Covalent API"""
    api_key = get_covalent_api_key()
    
    st.write("🔧 **Debug Information - Covalent API:**")
    st.write(f"- API Key: {'✅ Found' if api_key else '❌ Missing'}")
    st.write(f"- Token Address: {token_address}")
    st.write(f"- Chain ID: {chain_id} (1 = Ethereum Mainnet)")
    
    # Test các endpoint khác nhau
    endpoints = {
        "Token Holders": f"https://api.covalenthq.com/v1/{chain_id}/tokens/{token_address}/token_holders/",
        "Token Info": f"https://api.covalenthq.com/v1/{chain_id}/tokens/{token_address}/",
        "Token Balances": f"https://api.covalenthq.com/v1/{chain_id}/address/{token_address}/balances_v2/"
    }
    
    for endpoint_name, url in endpoints.items():
        try:
            st.write(f"\n**{endpoint_name}:**")
            
            params = {"key": api_key, "page-size": 10}
            debug_url = f"{url}?key=API_KEY_HIDDEN&page-size=10"
            st.write(f"- URL: {debug_url}")
            
            response = requests.get(url, params=params, timeout=15)
            st.write(f"- HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                st.write(f"- API Status: {data.get('error', False)}")
                
                if data.get('data'):
                    data_info = data['data']
                    st.write(f"- Items Count: {len(data_info.get('items', []))}")
                    st.write(f"- Updated At: {data_info.get('updated_at', 'N/A')}")
                    
                    if data_info.get('items') and len(data_info['items']) > 0:
                        sample_item = data_info['items'][0]
                        st.write(f"- Sample Item Keys: {list(sample_item.keys())[:5]}...")
                else:
                    st.write("- Data: None or Empty")
                    
            elif response.status_code == 402:
                st.error("❌ Quota exceeded - Hết lượt free tier")
            elif response.status_code == 429:
                st.error("❌ Rate limit exceeded")
            else:
                st.error(f"❌ HTTP Error: {response.status_code}")
                st.write(f"- Response: {response.text[:200]}...")
                
        except Exception as e:
            st.error(f"❌ Exception: {e}")

def get_token_holders_covalent(token_address, chain_id=1, page_size=1000):
    """Lấy token holders từ Covalent API"""
    api_key = get_covalent_api_key()
    if not api_key:
        return None
        
    try:
        url = f"https://api.covalenthq.com/v1/{chain_id}/tokens/{token_address}/token_holders/"
        params = {
            "key": api_key,
            "page-size": page_size,
            "page-number": 0
        }
        
        with st.spinner(f"📥 Đang lấy dữ liệu {page_size} holders từ Covalent API..."):
            response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('data') and data['data'].get('items'):
                holders = data['data']['items']
                st.success(f"✅ Lấy được {len(holders)} holders từ Covalent API")
                return holders
            else:
                st.warning("⚠️ Không tìm thấy holder data trong response")
                return None
                
        elif response.status_code == 402:
            st.error("❌ Covalent API: Quota exceeded - Hết lượt free tier")
            return None
        elif response.status_code == 429:
            st.error("❌ Covalent API: Rate limit exceeded")
            return None
        else:
            st.error(f"❌ Covalent API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi Covalent API: {e}")
        return None

def get_token_info_covalent(token_address, chain_id=1):
    """Lấy thông tin token từ Covalent"""
    api_key = get_covalent_api_key()
    if not api_key:
        return None
        
    try:
        url = f"https://api.covalenthq.com/v1/{chain_id}/tokens/{token_address}/"
        params = {"key": api_key}
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('data') and data['data'].get('items') and len(data['data']['items']) > 0:
                token_data = data['data']['items'][0]
                return {
                    'name': token_data.get('contract_name', 'Unknown'),
                    'symbol': token_data.get('contract_ticker_symbol', 'UNKN'),
                    'decimals': token_data.get('contract_decimals', 18),
                    'total_supply': int(token_data.get('total_supply', 0)) if token_data.get('total_supply') else 0
                }
        return None
    except Exception as e:
        st.error(f"❌ Lỗi lấy token info: {e}")
        return None

def calculate_gini_coefficient(holders, total_supply, decimals=18):
    """Tính hệ số Gini"""
    if total_supply == 0 or not holders:
        return 0
    
    # Convert balances to actual values
    balances = [float(holder.get('balance', 0)) / (10 ** decimals) for holder in holders]
    balances.sort()
    
    n = len(balances)
    if n == 0:
        return 0
        
    cumulative_balances = [sum(balances[:i+1]) for i in range(n)]
    
    # Area under Lorenz curve
    area_under_curve = sum(cumulative_balances) / cumulative_balances[-1] if cumulative_balances[-1] > 0 else 0
    
    # Gini coefficient
    gini = (n + 1 - 2 * area_under_curve) / n
    return max(0, min(1, gini))

def calculate_whale_metrics_covalent(holders_data, total_supply, decimals=18):
    """Tính whale metrics từ Covalent data"""
    if not holders_data:
        return None
    
    # Convert total supply
    total_supply_actual = total_supply / (10 ** decimals) if total_supply > 0 else 0
    
    # Sắp xếp holders
    sorted_holders = sorted(holders_data, key=lambda x: float(x.get('balance', 0)), reverse=True)
    
    # Tính tổng balance thực tế
    total_balance_actual = sum(float(h.get('balance', 0)) / (10 ** decimals) for h in sorted_holders)
    
    # Tính ratios
    top_10_balance = sum(float(h.get('balance', 0)) / (10 ** decimals) for h in sorted_holders[:10])
    top_20_balance = sum(float(h.get('balance', 0)) / (10 ** decimals) for h in sorted_holders[:20])
    top_50_balance = sum(float(h.get('balance', 0)) / (10 ** decimals) for h in sorted_holders[:50])
    
    # Sử dụng total_supply_actual hoặc total_balance_actual
    supply_to_use = total_supply_actual if total_supply_actual > 0 else total_balance_actual
    
    if supply_to_use == 0:
        st.warning("⚠️ Total supply = 0, không thể tính ratios")
        return None
    
    metrics = {
        'total_holders': len(holders_data),
        'total_supply': supply_to_use,
        'whale_ratio_10': (top_10_balance / supply_to_use) * 100,
        'whale_ratio_20': (top_20_balance / supply_to_use) * 100,
        'whale_ratio_50': (top_50_balance / supply_to_use) * 100,
        'top_10_holders': sorted_holders[:10],
        'gini_coefficient': calculate_gini_coefficient(sorted_holders, supply_to_use, decimals)
    }
    
    return metrics

def create_whale_chart(metrics):
    """Tạo biểu đồ whale distribution"""
    if not metrics:
        return None
    
    categories = ['Top 10', 'Top 20', 'Top 50']
    percentages = [
        metrics['whale_ratio_10'],
        metrics['whale_ratio_20'], 
        metrics['whale_ratio_50']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=percentages,
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
    """Tạo pie chart phân bổ supply"""
    if not metrics:
        return None
    
    top_10_supply = metrics['whale_ratio_10']
    top_11_20_supply = max(0, metrics['whale_ratio_20'] - metrics['whale_ratio_10'])
    top_21_50_supply = max(0, metrics['whale_ratio_50'] - metrics['whale_ratio_20'])
    rest_supply = max(0, 100 - metrics['whale_ratio_50'])
    
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

def get_realistic_mock_data(token_address):
    """Mock data fallback"""
    mock_data = {
        "0xdac17f958d2ee523a2206206994597c13d831ec7": {
            'name': 'Tether USD',
            'symbol': 'USDT',
            'total_holders': 4500000,
            'total_supply': 100000000000,
            'whale_ratio_10': 15.2,
            'whale_ratio_20': 22.7,
            'whale_ratio_50': 35.1,
            'gini_coefficient': 0.68,
            'source': 'mock'
        },
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
            'name': 'USD Coin',
            'symbol': 'USDC', 
            'total_holders': 2800000,
            'total_supply': 50000000000,
            'whale_ratio_10': 18.5,
            'whale_ratio_20': 26.3,
            'whale_ratio_50': 38.9,
            'gini_coefficient': 0.62,
            'source': 'mock'
        },
        "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": {
            'name': 'Uniswap',
            'symbol': 'UNI',
            'total_holders': 320000,
            'total_supply': 1000000000,
            'whale_ratio_10': 42.1,
            'whale_ratio_20': 55.6,
            'whale_ratio_50': 68.9,
            'gini_coefficient': 0.81,
            'source': 'mock'
        }
    }
    
    normalized_address = token_address.lower()
    return mock_data.get(normalized_address, {
        'name': 'Unknown Token',
        'symbol': 'UNKN',
        'total_holders': 150000,
        'total_supply': 1000000000,
        'whale_ratio_10': 25.5,
        'whale_ratio_20': 38.2,
        'whale_ratio_50': 52.7,
        'gini_coefficient': 0.65,
        'source': 'mock'
    })

def get_whale_data_covalent(token_address):
    """Lấy dữ liệu whale từ Covalent API"""
    
    # Lấy thông tin token
    token_info = get_token_info_covalent(token_address)
    
    # Lấy holders
    holders_data = get_token_holders_covalent(token_address, page_size=1000)
    
    if holders_data:
        # Tính toán metrics
        total_supply = token_info.get('total_supply', 0) if token_info else 0
        decimals = token_info.get('decimals', 18) if token_info else 18
        
        metrics = calculate_whale_metrics_covalent(holders_data, total_supply, decimals)
        
        if metrics:
            if token_info:
                metrics.update({
                    'name': token_info['name'],
                    'symbol': token_info['symbol'],
                    'source': 'covalent'
                })
            else:
                metrics.update({
                    'name': 'Unknown Token',
                    'symbol': 'UNKN', 
                    'source': 'covalent'
                })
            return metrics
    
    # Fallback to mock data
    st.warning("🔄 Đang sử dụng mock data (Covalent API có thể cần key hoặc đang load)")
    return get_realistic_mock_data(token_address)

# ===========================
# 🔧 CALLBACK FUNCTIONS
# ===========================
def set_token_address(token_addr):
    st.session_state.whale_token_address = token_addr
    st.rerun()

# Initialize session state
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
# 🐋 WHALE RATIO SECTION - COVALENT API
# ===========================
st.markdown("---")
st.header("🐋 Whale Ratio Analysis - Covalent API")

tab1, tab2 = st.tabs(["🔍 Covalent API", "📊 Dune Analytics"])

with tab1:
    st.subheader("Token Holder Concentration - Covalent API")
    
    # Debug section
    with st.expander("🔧 Debug Covalent API", expanded=False):
        if st.button("Run Debug Covalent"):
            debug_covalent_api(st.session_state.whale_token_address)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        token_address = st.text_input(
            "Token Contract Address (ETH):",
            value=st.session_state.whale_token_address,
            key="whale_token_address_input"
        )
    
    with col2:
        st.markdown("###")
        analyze_btn = st.button("🚀 Analyze with Covalent", type="primary", key="whale_analyze_btn_covalent")
    
    if analyze_btn and token_address:
        if not token_address.startswith("0x") or len(token_address) != 42:
            st.error("❌ Địa chỉ token không hợp lệ.")
        else:
            with st.spinner("Đang phân tích với Covalent API..."):
                # Sử dụng Covalent API
                metrics = get_whale_data_covalent(token_address)
                
                if metrics:
                    source_badge = "🔴 Live Data (Covalent)" if metrics.get('source') == 'covalent' else "🟡 Demo Data"
                    st.success(f"✅ Phân tích thành công! {source_badge}")
                    st.info(f"💰 Token: {metrics['name']} ({metrics['symbol']})")
                    
                    # Hiển thị KPI cards
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
    
    # Quick test buttons
    st.subheader("🚀 Test với token phổ biến")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("USDT", use_container_width=True, key="usdt_btn_covalent"):
            set_token_address("0xdAC17F958D2ee523a2206206994597C13D831ec7")
    
    with col2:
        if st.button("USDC", use_container_width=True, key="usdc_btn_covalent"):
            set_token_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    
    with col3:
        if st.button("UNI", use_container_width=True, key="uni_btn_covalent"):
            set_token_address("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984")
    
    st.info(f"🔍 Token address hiện tại: `{st.session_state.whale_token_address}`")

with tab2:
    st.subheader("Dune Analytics")
    st.info("Chức năng Dune Analytics sẽ hoạt động khi có API Key hợp lệ")

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit • Covalent API • Dune Analytics*")