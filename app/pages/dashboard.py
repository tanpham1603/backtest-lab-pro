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
# 🐋 WHALE RATIO FUNCTIONS - ETHERSCAN API FIXED FOR DEPLOY
# ===========================
def get_etherscan_api_key():
    """Lấy Etherscan API Key an toàn cho deploy"""
    try:
        # Streamlit secrets (khi deploy)
        api_key = st.secrets["ETHERSCAN_API_KEY"]
        if api_key and len(api_key) > 10:  # Validate API key format
            return api_key
        else:
            st.error("❌ Etherscan API Key không hợp lệ trong secrets")
            return None
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy Etherscan API Key: {e}")
        return None

def debug_etherscan_api(token_address):
    """Debug function chi tiết hơn"""
    api_key = get_etherscan_api_key()
    
    st.write("🔧 **Debug Information:**")
    st.write(f"- API Key: {'✅ Found' if api_key else '❌ Missing'}")
    if api_key:
        st.write(f"- API Key (first/last 4 chars): {api_key[:4]}...{api_key[-4:]}")
    st.write(f"- Token Address: {token_address}")
    
    # Test với các endpoint khác nhau
    endpoints = {
        "tokenSupply": {
            "url": f"https://api.etherscan.io/api?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}",
            "description": "Lấy total supply"
        },
        "tokenHolders": {
            "url": f"https://api.etherscan.io/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=10&apikey={api_key}",
            "description": "Lấy top 10 holders"
        }
    }
    
    for endpoint_name, endpoint_info in endpoints.items():
        try:
            st.write(f"\n**{endpoint_name} - {endpoint_info['description']}:**")
            
            response = requests.get(endpoint_info['url'], timeout=15)
            st.write(f"- HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                st.write(f"- API Status: {data.get('status')}")
                st.write(f"- Message: {data.get('message')}")
                
                if data.get('result'):
                    result = data['result']
                    if isinstance(result, list) and len(result) > 0:
                        st.write(f"- Result Sample: {str(result[0])[:100]}...")
                    else:
                        st.write(f"- Result: {str(result)[:200]}...")
                
                # Phân tích lỗi NOTOK
                if data.get('message') == 'NOTOK':
                    st.error("❌ Lỗi NOTOK - Nguyên nhân có thể:")
                    st.write("  - API Key không hợp lệ")
                    st.write("  - Rate limit vượt quá")
                    st.write("  - Contract address không tồn tại")
                    
            else:
                st.error(f"❌ Lỗi HTTP: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

def get_token_supply_etherscan(token_address):
    """Lấy total supply từ Etherscan với error handling tốt hơn"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        url = f"https://api.etherscan.io/api?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}"
        
        # Thêm delay để tránh rate limit
        time.sleep(0.3)
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == '1':
                supply = int(data['result'])
                return supply
            else:
                error_msg = data.get('message', 'Unknown error')
                st.warning(f"⚠️ Etherscan API Error: {error_msg}")
                return None
        else:
            st.error(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
        return None

def get_token_holders_etherscan_safe(token_address, max_holders=50):
    """Lấy holders an toàn với rate limiting"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        # Chỉ lấy 50 holders đầu tiên để test
        url = f"https://api.etherscan.io/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset={max_holders}&apikey={api_key}"
        
        # Thêm delay để tránh rate limit
        time.sleep(0.5)
        
        response = requests.get(url, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == '1':
                holders = data['result']
                if holders and len(holders) > 0:
                    return holders
                else:
                    st.warning("⚠️ Token có vẻ như không có holders hoặc dữ liệu trống")
                    return None
            else:
                error_msg = data.get('message', 'Unknown error')
                st.warning(f"⚠️ Không thể lấy holders: {error_msg}")
                return None
        else:
            st.error(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy holders: {e}")
        return None

def calculate_whale_metrics(holders_data, total_supply):
    """Tính toán các chỉ số whale từ dữ liệu holders"""
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
    """Tính hệ số Gini - đo lường độ tập trung"""
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
# 🧪 TEST WITH MOCK DATA AS FALLBACK
# ===========================
def get_whale_data_with_fallback(token_address):
    """Lấy dữ liệu whale với fallback sang mock data nếu API fail"""
    
    # Thử lấy dữ liệu thật từ Etherscan
    total_supply = get_token_supply_etherscan(token_address)
    holders_data = get_token_holders_etherscan_safe(token_address)
    
    if total_supply and holders_data:
        # Dữ liệu thật thành công
        metrics = calculate_whale_metrics(holders_data, total_supply)
        if metrics:
            metrics['source'] = 'etherscan'
            return metrics
    
    # Fallback sang mock data để demo UI
    st.warning("🔄 Đang sử dụng mock data để demo (API có thể bị limit)")
    
    # Mock data cho các token phổ biến
    mock_data = {
        "0xdAC17F958D2ee523a2206206994597C13D831ec7": {  # USDT
            'total_holders': 4500000,
            'total_supply': 100000000000,
            'whale_ratio_10': 35.2,
            'whale_ratio_20': 48.7,
            'whale_ratio_50': 62.1,
            'gini_coefficient': 0.78,
            'source': 'mock'
        },
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": {  # USDC
            'total_holders': 2800000,
            'total_supply': 50000000000,
            'whale_ratio_10': 28.5,
            'whale_ratio_20': 42.3,
            'whale_ratio_50': 58.9,
            'gini_coefficient': 0.72,
            'source': 'mock'
        },
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": {  # UNI
            'total_holders': 320000,
            'total_supply': 1000000000,
            'whale_ratio_10': 42.1,
            'whale_ratio_20': 55.6,
            'whale_ratio_50': 68.9,
            'gini_coefficient': 0.81,
            'source': 'mock'
        }
    }
    
    # Mock data mặc định cho token khác
    default_mock = {
        'total_holders': 150000,
        'total_supply': 1000000000,
        'whale_ratio_10': 25.5,
        'whale_ratio_20': 38.2,
        'whale_ratio_50': 52.7,
        'gini_coefficient': 0.65,
        'source': 'mock'
    }
    
    return mock_data.get(token_address, default_mock)

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
# 🖥️ MAIN UI - SIMPLIFIED FOR DEPLOY
# ===========================
st.title("📊 Market Overview")

# ===========================
# 🐋 WHALE RATIO SECTION - DEPLOY FRIENDLY
# ===========================
st.markdown("---")
st.header("🐋 Whale Ratio Analysis")

tab1, tab2 = st.tabs(["🔍 Token Analysis", "📊 Dune Analytics"])

with tab1:
    st.subheader("Token Holder Concentration")
    
    # Debug section
    with st.expander("🔧 Debug Etherscan API", expanded=False):
        if st.button("Run Debug"):
            debug_etherscan_api(st.session_state.whale_token_address)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        token_address = st.text_input(
            "Token Contract Address (ETH):",
            value=st.session_state.whale_token_address,
            key="whale_token_address_input"
        )
    
    with col2:
        st.markdown("###")
        analyze_btn = st.button("🚀 Analyze", type="primary", key="whale_analyze_btn")
    
    if analyze_btn and token_address:
        if not token_address.startswith("0x") or len(token_address) != 42:
            st.error("❌ Địa chỉ token không hợp lệ. Phải bắt đầu bằng 0x và có 42 ký tự.")
        else:
            with st.spinner("Đang phân tích whale ratio..."):
                # Sử dụng hàm có fallback
                metrics = get_whale_data_with_fallback(token_address)
                
                if metrics:
                    source_badge = "🔴 Live Data" if metrics.get('source') == 'etherscan' else "🟡 Demo Data"
                    st.success(f"✅ Phân tích thành công! {source_badge}")
                    
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
    st.subheader("🚀 Test nhanh")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("USDT", use_container_width=True):
            set_token_address("0xdAC17F958D2ee523a2206206994597C13D831ec7")
    
    with col2:
        if st.button("USDC", use_container_width=True):
            set_token_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    
    with col3:
        if st.button("UNI", use_container_width=True):
            set_token_address("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984")
    
    st.info(f"🔍 Token address hiện tại: `{st.session_state.whale_token_address}`")

with tab2:
    st.subheader("Dune Analytics")
    st.info("Chức năng Dune Analytics sẽ hoạt động khi có API Key hợp lệ")

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit • Etherscan API • Dune Analytics*")