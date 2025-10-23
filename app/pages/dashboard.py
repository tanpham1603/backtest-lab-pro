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
# 🐋 WHALE RATIO FUNCTIONS - ETHERSCAN API V2
# ===========================
def get_etherscan_api_key():
    """Lấy Etherscan API Key an toàn cho deploy"""
    try:
        # Streamlit secrets (khi deploy)
        api_key = st.secrets["ETHERSCAN_API_KEY"]
        if api_key and len(api_key) > 10:
            return api_key
        else:
            st.error("❌ Etherscan API Key không hợp lệ trong secrets")
            return None
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy Etherscan API Key: {e}")
        return None

def debug_etherscan_api_v2(token_address):
    """Debug function cho API V2"""
    api_key = get_etherscan_api_key()
    
    st.write("🔧 **Debug Information - Etherscan API V2:**")
    st.write(f"- API Key: {'✅ Found' if api_key else '❌ Missing'}")
    if api_key:
        st.write(f"- API Key (first/last 4 chars): {api_key[:4]}...{api_key[-4:]}")
    st.write(f"- Token Address: {token_address}")
    
    # Test với các endpoint V2
    endpoints = {
        "tokenSupply V2": {
            "url": f"https://api.etherscan.io/v2/api?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}",
            "description": "Lấy total supply (V2)"
        },
        "tokenInfo V2": {
            "url": f"https://api.etherscan.io/v2/api?module=token&action=tokeninfo&contractaddress={token_address}&apikey={api_key}",
            "description": "Lấy thông tin token (V2)"
        },
        "tokenHolders V2": {
            "url": f"https://api.etherscan.io/v2/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=10&apikey={api_key}",
            "description": "Lấy top holders (V2)"
        }
    }
    
    for endpoint_name, endpoint_info in endpoints.items():
        try:
            st.write(f"\n**{endpoint_name} - {endpoint_info['description']}:**")
            
            # Ẩn API key trong log
            debug_url = endpoint_info['url'].replace(api_key, "API_KEY_HIDDEN") if api_key else endpoint_info['url']
            st.write(f"- URL: {debug_url}")
            
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
                else:
                    st.write("- Result: None or Empty")
                
            else:
                st.error(f"❌ HTTP Error: {response.status_code}")
                st.write(f"- Response: {response.text[:200]}...")
                
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

def get_token_supply_etherscan_v2(token_address):
    """Lấy total supply từ Etherscan API V2"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        # API V2 endpoint
        url = f"https://api.etherscan.io/v2/api?module=stats&action=tokensupply&contractaddress={token_address}&apikey={api_key}"
        
        # Thêm delay để tránh rate limit
        time.sleep(0.3)
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == '1':
                supply = int(data['result'])
                st.success(f"✅ Total Supply (V2): {supply:,}")
                return supply
            else:
                error_msg = data.get('message', 'Unknown error')
                st.warning(f"⚠️ Etherscan API V2 Error: {error_msg}")
                return None
        else:
            st.error(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi lấy token supply V2: {e}")
        return None

def get_token_holders_etherscan_v2(token_address, max_holders=50):
    """Lấy holders từ Etherscan API V2"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        # API V2 endpoint
        url = f"https://api.etherscan.io/v2/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset={max_holders}&apikey={api_key}"
        
        # Thêm delay để tránh rate limit
        time.sleep(0.5)
        
        response = requests.get(url, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == '1':
                holders = data.get('result', [])
                if holders and len(holders) > 0:
                    st.success(f"✅ Lấy được {len(holders)} holders từ API V2")
                    return holders
                else:
                    st.warning("⚠️ Token có vẻ như không có holders hoặc dữ liệu trống")
                    return None
            else:
                error_msg = data.get('message', 'Unknown error')
                st.warning(f"⚠️ Không thể lấy holders V2: {error_msg}")
                return None
        else:
            st.error(f"❌ HTTP Error V2: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy holders V2: {e}")
        return None

def get_token_info_etherscan_v2(token_address):
    """Lấy thông tin token từ Etherscan API V2"""
    api_key = get_etherscan_api_key()
    if not api_key:
        return None
        
    try:
        # API V2 endpoint for token info
        url = f"https://api.etherscan.io/v2/api?module=token&action=tokeninfo&contractaddress={token_address}&apikey={api_key}"
        
        time.sleep(0.3)
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == '1' and data.get('result'):
                token_data = data['result'][0]  # Lấy token đầu tiên
                return {
                    'name': token_data.get('tokenName', 'Unknown'),
                    'symbol': token_data.get('tokenSymbol', 'UNKN'),
                    'decimals': int(token_data.get('divisor', 18)),
                    'total_supply': int(token_data.get('totalSupply', 0))
                }
            else:
                st.warning(f"⚠️ Không thể lấy token info V2: {data.get('message', 'Unknown error')}")
                return None
        else:
            st.error(f"❌ HTTP Error V2: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi lấy token info V2: {e}")
        return None

def calculate_whale_metrics(holders_data, total_supply, decimals=18):
    """Tính toán các chỉ số whale từ dữ liệu holders"""
    if not holders_data or total_supply == 0:
        return None
    
    # Convert total supply to actual value (considering decimals)
    total_supply_actual = total_supply / (10 ** decimals)
    
    # Sắp xếp holders theo balance giảm dần
    sorted_holders = sorted(holders_data, key=lambda x: float(x.get('value', 0)), reverse=True)
    
    # Tính toán các metrics
    top_10_balance = sum(float(holder.get('value', 0)) for holder in sorted_holders[:10])
    top_20_balance = sum(float(holder.get('value', 0)) for holder in sorted_holders[:20])
    top_50_balance = sum(float(holder.get('value', 0)) for holder in sorted_holders[:50])
    
    metrics = {
        'total_holders': len(holders_data),
        'total_supply': total_supply_actual,
        'whale_ratio_10': (top_10_balance / total_supply_actual) * 100 if total_supply_actual > 0 else 0,
        'whale_ratio_20': (top_20_balance / total_supply_actual) * 100 if total_supply_actual > 0 else 0,
        'whale_ratio_50': (top_50_balance / total_supply_actual) * 100 if total_supply_actual > 0 else 0,
        'top_10_holders': sorted_holders[:10],
        'top_20_holders': sorted_holders[:20],
        'gini_coefficient': calculate_gini_coefficient(sorted_holders, total_supply_actual)
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
    return max(0, min(1, gini))

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
    top_11_20_supply = metrics['whale_ratio_20'] - metrics['whale_ratio_10']
    top_21_50_supply = metrics['whale_ratio_50'] - metrics['whale_ratio_20']
    rest_supply = 100 - metrics['whale_ratio_50']
    
    # Đảm bảo không có giá trị âm
    top_11_20_supply = max(0, top_11_20_supply)
    top_21_50_supply = max(0, top_21_50_supply)
    rest_supply = max(0, rest_supply)
    
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

def get_whale_data_etherscan_v2(token_address):
    """Lấy dữ liệu whale từ Etherscan API V2"""
    
    # Thử lấy thông tin token trước
    token_info = get_token_info_etherscan_v2(token_address)
    
    if token_info:
        st.success(f"✅ Token: {token_info['name']} ({token_info['symbol']})")
        
        # Lấy total supply
        total_supply = get_token_supply_etherscan_v2(token_address)
        
        if total_supply:
            # Lấy holders
            holders_data = get_token_holders_etherscan_v2(token_address, max_holders=100)
            
            if holders_data:
                # Tính toán metrics
                metrics = calculate_whale_metrics(holders_data, total_supply, token_info.get('decimals', 18))
                if metrics:
                    metrics.update({
                        'name': token_info['name'],
                        'symbol': token_info['symbol'],
                        'source': 'etherscan_v2'
                    })
                    return metrics
    
    # Fallback sang mock data nếu API fail
    st.warning("🔄 Đang sử dụng mock data (API V2 có thể cần time để active)")
    return get_realistic_mock_data(token_address)

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
# 🖥️ MAIN UI - ETHERSCAN API V2
# ===========================
st.title("📊 Market Overview")

# ===========================
# 🐋 WHALE RATIO SECTION - ETHERSCAN API V2
# ===========================
st.markdown("---")
st.header("🐋 Whale Ratio Analysis - Etherscan API V2")

tab1, tab2 = st.tabs(["🔍 Etherscan API V2", "📊 Dune Analytics"])

with tab1:
    st.subheader("Token Holder Concentration - API V2")
    
    # Debug section
    with st.expander("🔧 Debug Etherscan API V2", expanded=False):
        if st.button("Run Debug V2"):
            debug_etherscan_api_v2(st.session_state.whale_token_address)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        token_address = st.text_input(
            "Token Contract Address (ETH):",
            value=st.session_state.whale_token_address,
            key="whale_token_address_input"
        )
    
    with col2:
        st.markdown("###")
        analyze_btn = st.button("🚀 Analyze with V2 API", type="primary", key="whale_analyze_btn_v2")
    
    if analyze_btn and token_address:
        if not token_address.startswith("0x") or len(token_address) != 42:
            st.error("❌ Địa chỉ token không hợp lệ.")
        else:
            with st.spinner("Đang phân tích với Etherscan API V2..."):
                # Sử dụng API V2
                metrics = get_whale_data_etherscan_v2(token_address)
                
                if metrics:
                    source_badge = "🔴 Live Data (V2)" if metrics.get('source') == 'etherscan_v2' else "🟡 Demo Data"
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
    
    # Quick test buttons
    st.subheader("🚀 Test với token phổ biến")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("USDT", use_container_width=True, key="usdt_btn_v2"):
            set_token_address("0xdAC17F958D2ee523a2206206994597C13D831ec7")
    
    with col2:
        if st.button("USDC", use_container_width=True, key="usdc_btn_v2"):
            set_token_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    
    with col3:
        if st.button("UNI", use_container_width=True, key="uni_btn_v2"):
            set_token_address("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984")
    
    st.info(f"🔍 Token address hiện tại: `{st.session_state.whale_token_address}`")

with tab2:
    st.subheader("Dune Analytics")
    st.info("Chức năng Dune Analytics sẽ hoạt động khi có API Key hợp lệ")

# Footer
st.markdown("---")
st.markdown("*Dashboard built with Streamlit • Etherscan API V2 • Dune Analytics*")