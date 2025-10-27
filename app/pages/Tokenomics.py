import streamlit as st
import requests
from web3 import Web3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="🚀 Crypto Analysis Pro", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# RPC endpoints
RPC_URLS = {
    'Ethereum': 'https://eth.llamarpc.com',
    'BSC': 'https://bsc-dataseed.binance.org',
    'Polygon': 'https://polygon-rpc.com',
    'Arbitrum': 'https://arb1.arbitrum.io/rpc'
}

PLATFORM_MAP = {
    'Ethereum': 'ethereum',
    'BSC': 'binance-smart-chain',
    'Polygon': 'polygon-pos',
    'Arbitrum': 'arbitrum-one'
}

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}
]

KNOWN_COINS = {
    'DOGE': 'dogecoin',
    'USDT': 'tether',
    'USDC': 'usd-coin',
    'UNI': 'uniswap',
    'LINK': 'chainlink',
    'AAVE': 'aave',
    'SHIB': 'shiba-inu'
}

# --- CUSTOM CSS STYLING ---
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
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .token-badge {
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

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>🚀</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>Crypto Analysis Pro</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>Advanced Crypto Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("✨ Enter token address to begin analysis")

# --- HEADER ---
st.markdown('<div class="welcome-header">🚀 Crypto Analysis Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Cryptocurrency Analytics with Auto Chain Detection</div>', unsafe_allow_html=True)

# --- STATUS CARDS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🔍</div>
        <div class="metric-value">Auto</div>
        <div class="metric-label">Chain Detection</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📊</div>
        <div class="metric-value">Live</div>
        <div class="metric-label">Market Data</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">⚡</div>
        <div class="metric-value">Pro</div>
        <div class="metric-label">Risk Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🛡️</div>
        <div class="metric-value">Any</div>
        <div class="metric-label">Token Support</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ==================================================
# AUTO-DETECTION FUNCTIONS
# ==================================================

def detect_chain_and_get_data(token_address):
    """Auto detect chain and get token data"""
    detected_chain = None
    token_info = None
    for chain_name in RPC_URLS.keys():
        try:
            w3 = Web3(Web3.HTTPProvider(RPC_URLS[chain_name]))
            if w3.is_connected():
                # Try to get token info
                contract = w3.eth.contract(
                    address=w3.to_checksum_address(token_address),
                    abi=ERC20_ABI
                )
                name = contract.functions.name().call()
                symbol = contract.functions.symbol().call()
                decimals = contract.functions.decimals().call()
                total_supply = contract.functions.totalSupply().call()
                
                token_info = {
                    'name': name,
                    'symbol': symbol,
                    'decimals': decimals,
                    'total_supply': total_supply,
                    'total_supply_formatted': total_supply / (10 ** decimals)
                }
                detected_chain = chain_name
                break  # If info retrieved, stop and use this chain
        except Exception as e:
            continue
    
    if not token_info:
        return None, None, None
    
    # Get market data from CoinGecko global
    market_data = get_global_data_from_coingecko(token_info['symbol'], detected_chain, token_address)
    
    if market_data and market_data.get('price', 0) > 0:
        return detected_chain, token_info, market_data
    
    return detected_chain, token_info, None

def get_price_from_dexscreener_auto(token_address):
    """Get price from DexScreener without chain - using new API"""
    try:
        url = f"https://api.dexscreener.com/latest/dex/search/?q={token_address}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'pairs' in data and len(data['pairs']) > 0:
                valid_pairs = [
                    p for p in data['pairs']
                    if float(p.get('priceUsd', 0)) > 0
                    and p.get('liquidity', {}).get('usd', 0) >= 1000
                    and p.get('volume', {}).get('h24', 0) > 0
                ]
                if not valid_pairs:
                    return None
                pair = max(valid_pairs, key=lambda x: x.get('volume', {}).get('h24', 0))
                circulating_supply = pair.get('circulatingSupply')
                total_supply_api = pair.get('totalSupply')
                fdv = pair.get('fdv')
                return {
                    'price': float(pair['priceUsd']),
                    'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                    'volume_24h': pair.get('volume', {}).get('h24', 0),
                    'liquidity': pair.get('liquidity', {}).get('usd', 0),
                    'circulating_supply': float(circulating_supply) if circulating_supply else None,
                    'fdv': float(fdv) if fdv else None,
                    'total_supply_api': float(total_supply_api) if total_supply_api else None,
                    'dex_id': pair.get('dexId'),
                    'chain_id': pair.get('chainId'),
                    'pair_address': pair.get('pairAddress'),
                    'base_token': pair.get('baseToken', {}),
                    'quote_token': pair.get('quoteToken', {}),
                    'price_changes': pair.get('priceChange', {}),
                    'volumes': pair.get('volume', {}),
                    'txns': pair.get('txns', {})
                }
    except Exception as e:
        st.error(f"DexScreener error: {e}")
        return None

def get_global_data_from_coingecko(symbol, chain, token_address):
    """Get price from CoinGecko global"""
    coingecko_id = get_coingecko_id(symbol, chain, token_address)
    if coingecko_id:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('market_data', {})
                return {
                    'price': market_data.get('current_price', {}).get('usd', 0.0),
                    'price_change_24h': market_data.get('price_change_percentage_24h', 0.0),
                    'volume_24h': market_data.get('total_volume', {}).get('usd', 0.0),
                    'liquidity': market_data.get('liquidity_score', 0.0),  # If available, or fallback
                    'market_cap': market_data.get('market_cap', {}).get('usd', 0.0),
                    'circulating_supply': market_data.get('circulating_supply'),
                    'fdv': market_data.get('fully_diluted_valuation', {}).get('usd'),
                    'source': 'CoinGecko',
                    'reliability': "✅ HIGH",
                    'price_changes': {  # Fallback, not as complete as DexScreener
                        'h24': market_data.get('price_change_percentage_24h', 0.0),
                    },
                    'volumes': {  # Fallback
                        'h24': market_data.get('total_volume', {}).get('usd', 0.0),
                    },
                    'txns': {},  # Not available from CoinGecko
                    'dex_id': 'N/A',
                    'chain_id': chain,
                    'pair_address': 'N/A',
                    'base_token': {'symbol': symbol},
                    'quote_token': {'symbol': 'USD'}
                }
        except Exception as e:
            st.error(f"CoinGecko error: {e}")
    return None

def get_circulating_from_coingecko(symbol):
    if symbol not in KNOWN_COINS:
        return None
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={KNOWN_COINS[symbol]}&locale=en"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        if data:
            return data[0]['circulating_supply']
        return None
    except:
        return None

def get_token_logo(symbol, chain, token_address):
    coingecko_id = get_coingecko_id(symbol, chain, token_address)
    if coingecko_id:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}?localization=false"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('image', {}).get('large')
        except:
            pass
    return None

def get_accurate_market_cap_auto(price_data, contract_supply, symbol):
    """Calculate market cap with fallback options"""
    cg_supply = get_circulating_from_coingecko(symbol) if symbol in KNOWN_COINS else None
    if cg_supply and cg_supply <= contract_supply:
        circulating_supply = cg_supply
        supply_type = "circulating (CoinGecko)"
        reliability = "✅ HIGH"
        market_cap = circulating_supply * price_data['price']
        supply_used = circulating_supply
        price_data['market_cap'] = market_cap
        price_data['circulating_supply'] = circulating_supply
    else:
        # Prioritize market cap from DexScreener
        if price_data.get('market_cap'):
            supply_used = price_data['market_cap'] / price_data['price'] if price_data['price'] > 0 else contract_supply
            supply_type = "circulating"
            reliability = "✅ HIGH"
        
        # Fallback to circulating supply
        elif price_data.get('circulating_supply'):
            market_cap = price_data['circulating_supply'] * price_data['price']
            supply_used = price_data['circulating_supply']
            supply_type = "circulating"
            reliability = "✅ HIGH"
            price_data['market_cap'] = market_cap
        
        # Fallback to FDV
        elif price_data.get('fdv'):
            market_cap = price_data.get('fdv')
            supply_used = market_cap / price_data['price'] if price_data['price'] > 0 else contract_supply
            supply_type = "fully diluted"
            reliability = "⚠️ MEDIUM"
            price_data['market_cap'] = market_cap
        
        # Final fallback: contract supply
        else:
            market_cap = contract_supply * price_data['price']
            supply_used = contract_supply
            supply_type = "total_supply"
            reliability = "❌ LOW"
            price_data['market_cap'] = market_cap
    
    return {
        'market_cap': price_data['market_cap'],
        'supply_used': supply_used,
        'supply_type': supply_type,
        'reliability': reliability
    }

# ==================================================
# COINGECKO FUNCTIONS FOR HISTORICAL DATA
# ==================================================

def get_coingecko_id(symbol, chain, token_address):
    """Find CoinGecko ID based on symbol, chain and address, or just symbol if not available"""
    if symbol.upper() in KNOWN_COINS:
        return KNOWN_COINS[symbol.upper()]
    
    try:
        url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json().get('coins', [])
        for coin in data:
            if coin.get('symbol', '').upper() == symbol.upper():
                # Prioritize address match if available
                platforms = coin.get('platforms', {})
                platform = PLATFORM_MAP.get(chain)
                if platform and platforms.get(platform, '').lower() == token_address.lower():
                    return coin['id']
                # Otherwise, return first ID matching symbol
                return coin['id']
        return None
    except Exception as e:
        return None

def get_historical_data(symbol, chain, token_address, days=30):
    """Get real historical data from CoinGecko if available, fallback to simulation"""
    coingecko_id = get_coingecko_id(symbol, chain, token_address)
    
    if coingecko_id:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception("API error")
            
            data = response.json()
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                raise Exception("No data")
            
            dates = [datetime.fromtimestamp(ts / 1000) for ts, _ in prices]
            price_values = [p for _, p in prices]
            volume_values = [v for _, v in volumes]
            
            df = pd.DataFrame({
                'date': dates,
                'price': price_values,
                'volume': volume_values
            })
            st.info("✅ Using real historical data from CoinGecko")
            return df
        except Exception as e:
            st.warning("⚠️ Could not fetch real historical data, using simulated data")
    
    # Fallback simulation
    dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
    base_price = np.random.normal(100, 30)
    prices = []
    volumes = []
    
    current_price = base_price
    for i in range(days):
        change = np.random.normal(0.002, 0.03)
        current_price = current_price * (1 + change)
        prices.append(current_price)
        volumes.append(abs(np.random.normal(1000000, 300000)))
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': volumes
    })

# ==================================================
# EXISTING FUNCTIONS (with minor modifications)
# ==================================================

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df['sma_7'] = df['price'].rolling(window=7).mean()
    df['sma_21'] = df['price'].rolling(window=21).mean()
    
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['volume_sma'] = df['volume'].rolling(window=7).mean()
    
    return df

def get_token_risk_metrics(market_data, token_info):
    """Overall risk assessment"""
    risk_score = 0
    warnings = []
    recommendations = []
    
    liquidity = market_data.get('liquidity', 0)
    if liquidity < 1000:
        risk_score += 2
        warnings.append("💰 Low liquidity")
        recommendations.append("Consider higher liquidity pools")
    elif liquidity < 10000:
        risk_score += 1
        warnings.append("💰 Medium liquidity")
    else:
        recommendations.append("✅ Good liquidity levels")
    
    volume = market_data.get('volume_24h', 0)
    market_cap = market_data.get('market_cap', 1)
    volume_ratio = (volume / market_cap) if market_cap > 0 else 0
    
    if volume_ratio < 0.001:
        risk_score += 2
        warnings.append("📊 Low trading activity")
        recommendations.append("Low volume may indicate illiquidity")
    elif volume_ratio < 0.01:
        risk_score += 1
        warnings.append("📊 Moderate trading activity")
    else:
        recommendations.append("✅ Healthy trading volume")
    
    reliability = market_data.get('reliability', '❌ VERY LOW')
    if '❌' in reliability:
        risk_score += 2
        warnings.append("🔍 Unreliable data sources")
        recommendations.append("Verify data from multiple sources")
    elif '⚠️' in reliability:
        risk_score += 1
        warnings.append("🔍 Medium data reliability")
    else:
        recommendations.append("✅ Reliable data sources")
    
    if risk_score >= 4:
        risk_level = "🔴 HIGH RISK"
        color = "#e74c3c"
    elif risk_score >= 2:
        risk_level = "🟡 MEDIUM RISK" 
        color = "#f39c12"
    else:
        risk_level = "🟢 LOW RISK"
        color = "#27ae60"
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_color': color,
        'warnings': warnings,
        'recommendations': recommendations,
        'volume_ratio': round(volume_ratio * 100, 2),
        'liquidity_score': "Low" if liquidity < 1000 else "Medium" if liquidity < 10000 else "High"
    }

def create_financial_metrics(market_data, token_info):
    """Create comprehensive financial metrics"""
    market_cap = market_data.get('market_cap', 0)
    volume = market_data.get('volume_24h', 0)
    liquidity = market_data.get('liquidity', 0)
    
    metrics = {
        'volume_mcap_ratio': (volume / market_cap * 100) if market_cap > 0 else 0,
        'liquidity_mcap_ratio': (liquidity / market_cap * 100) if market_cap > 0 else 0,
        'circulation_ratio': (market_data.get('supply_used', 0) / token_info['total_supply_formatted'] * 100) if token_info['total_supply_formatted'] > 0 else 0
    }
    
    return metrics

def create_enhanced_price_chart(historical_data):
    """Enhanced price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['price'],
        name='Price',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    if 'sma_7' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['sma_7'],
            name='SMA 7',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
    
    if 'sma_21' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['sma_21'],
            name='SMA 21',
            line=dict(color='#2ecc71', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title='📈 Price History with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=400,
        showlegend=True
    )
    
    return fig

def create_volume_analysis_chart(historical_data):
    """Volume analysis chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=historical_data['date'],
        y=historical_data['volume'],
        name='Daily Volume',
        marker_color='#4ecdc4',
        opacity=0.7
    ))
    
    if 'volume_sma' in historical_data.columns:
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['volume_sma'],
            name='Volume SMA 7',
            line=dict(color='#2c3e50', width=2)
        ))
    
    fig.update_layout(
        title='📊 Trading Volume Analysis',
        xaxis_title='Date',
        yaxis_title='Volume (USD)',
        template='plotly_dark',
        height=350
    )
    
    return fig

def create_rsi_chart(historical_data):
    """RSI chart"""
    if 'rsi' not in historical_data.columns:
        return None
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['rsi'],
        name='RSI',
        line=dict(color='#fd7e14', width=2)
    ))
    
    fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Overbought")
    fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Oversold")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title='🎯 RSI Indicator',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        height=300,
        yaxis_range=[0, 100]
    )
    
    return fig

def create_supply_distribution_chart(market_data, token_info):
    """Supply distribution chart"""
    if market_data.get('circulating_supply'):
        circulating = market_data['circulating_supply']
        total = token_info['total_supply_formatted']
        locked = max(0, total - circulating)
        
        labels = ['Circulating', 'Locked/Vesting'] if locked > 0 else ['Circulating']
        values = [circulating, locked] if locked > 0 else [circulating]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=['#667eea', '#ff6b6b'],
            textinfo='percent+label'
        )])
        
        fig.update_layout(
            title='🥧 Supply Distribution',
            height=350,
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig
    return None

def format_large_number(num):
    if num is None or num <= 0:
        return "N/A"
    suffixes = ['', 'K', 'M', 'B', 'T']
    suffix_index = 0
    while num >= 1000 and suffix_index < len(suffixes) - 1:
        num /= 1000
        suffix_index += 1
    return f"{num:.2f}{suffixes[suffix_index]}"

def format_price(price):
    if price is None or price <= 0:
        return "N/A"
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    elif price >= 0.01:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"

# ==================================================
# MAIN UI - ENHANCED WITH NEW STYLING
# ==================================================

with st.sidebar:
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="config-header">🔧 TOKEN ANALYSIS</div>', unsafe_allow_html=True)
    
    # Quick access tokens
    st.markdown("#### Popular Tokens")
    popular_tokens = {
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA"
    }
    
    for name, address in popular_tokens.items():
        if st.button(f"🔹 {name}", key=f"quick_{name}", use_container_width=True):
            st.session_state.token_address = address
    
    st.markdown("---")
    
    token_input = st.text_input(
        "**Enter Token Contract Address or Symbol (e.g., SHIB/USDT)**",
        value=getattr(st.session_state, 'token_address', ""),
        placeholder="0x... or SHIB/USDT",
        key="token_input"
    )
    
    analyze_btn = st.button("🚀 Analyze Token", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main analysis section
if analyze_btn and token_input:
    # Process input as address or symbol/pair
    if '/' in token_input:
        # Assume pair like X/USDT, take X as main symbol, ignore USDT (since global data doesn't need pair)
        symbol = token_input.split('/')[0].upper()
        token_address = None
        chain = None
        token_info = {
            'name': symbol,
            'symbol': symbol,
            'decimals': 18,  # Default
            'total_supply_formatted': 0  # Will update from data
        }
    else:
        token_address = token_input if token_input.startswith("0x") else None
        chain, token_info, _ = detect_chain_and_get_data(token_address) if token_address else (None, None, None)
        if token_info:
            symbol = token_info['symbol']
    
    if symbol:
        # Get global market data from CoinGecko
        market_data = get_global_data_from_coingecko(symbol, chain, token_address)
        
        if market_data:
            # Update token_info if needed
            if 'total_supply_formatted' in token_info and token_info['total_supply_formatted'] == 0:
                token_info['total_supply_formatted'] = market_data.get('fdv', 0) / market_data['price'] if market_data['price'] > 0 else 0
            
            # Calculate accurate market cap
            market_cap_info = get_accurate_market_cap_auto(market_data, token_info['total_supply_formatted'], symbol)
            market_data.update(market_cap_info)
            
            # Get token logo
            logo_url = get_token_logo(symbol, chain, token_address)
            
            # Display token header
            col_logo, col_info = st.columns([1, 4])
            with col_logo:
                if logo_url:
                    st.image(logo_url, width=100)
                else:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem;'>
                        <div style='font-size: 3rem;'>💰</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"<div class='token-badge'>{symbol}</div>", unsafe_allow_html=True)
            
            with col_info:
                detected_chain = chain if chain else market_data.get('chain_id', 'Unknown')
                st.markdown(f"### {token_info.get('name', symbol)} ({symbol})")
                st.caption(f"Chain: {detected_chain}")
            
            # ==================================================
            # MARKET OVERVIEW SECTION
            # ==================================================
            st.markdown('<div class="section-header">📊 MARKET OVERVIEW</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_color = "normal" if market_data['price_change_24h'] >= 0 else "inverse"
                st.metric(
                    label="💵 Current Price",
                    value=format_price(market_data['price']),
                    delta=f"{market_data['price_change_24h']:.2f}%",
                    delta_color=delta_color
                )
            
            with col2:
                st.metric(
                    label="🏦 Market Cap",
                    value=f"${format_large_number(market_data['market_cap'])}"
                )
            
            with col3:
                st.metric(
                    label="📈 24h Volume", 
                    value=f"${format_large_number(market_data['volume_24h'])}"
                )
            
            with col4:
                st.metric(
                    label="💧 Liquidity",
                    value=f"${format_large_number(market_data['liquidity'])}"
                )
            
            # ==================================================
            # FINANCIALS OVERVIEW SECTION
            # ==================================================
            st.markdown('<div class="section-header">💹 FINANCIALS OVERVIEW</div>', unsafe_allow_html=True)
            
            financial_metrics = create_financial_metrics(market_data, token_info)
            risk_metrics = get_token_risk_metrics(market_data, token_info)
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.markdown(f"""
                <div class="status-card">
                    <div class="metric-value">{financial_metrics['volume_mcap_ratio']:.2f}%</div>
                    <div class="metric-label">Volume/MCap Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div class="status-card">
                    <div class="metric-value">{financial_metrics['circulation_ratio']:.1f}%</div>
                    <div class="metric-label">Circulation Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                st.markdown(f"""
                <div class="status-card">
                    <div class="metric-value">{market_data.get('reliability', 'Unknown')}</div>
                    <div class="metric-label">Data Reliability</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div class="status-card">
                    <div class="metric-value" style="color: {risk_metrics['risk_color']}">{risk_metrics['risk_level'].split(' ')[1]}</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ==================================================
            # TECHNICAL ANALYSIS SECTION
            # ==================================================
            st.markdown('<div class="section-header">🔗 TECHNICAL ANALYSIS</div>', unsafe_allow_html=True)
            
            historical_data = get_historical_data(symbol, chain, token_address)
            historical_data = calculate_technical_indicators(historical_data)
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                price_chart = create_enhanced_price_chart(historical_data)
                st.plotly_chart(price_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_chart2:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                volume_chart = create_volume_analysis_chart(historical_data)
                st.plotly_chart(volume_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            col_chart3, col_chart4 = st.columns(2)
            
            with col_chart3:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                rsi_chart = create_rsi_chart(historical_data)
                if rsi_chart:
                    st.plotly_chart(rsi_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_chart4:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                supply_chart = create_supply_distribution_chart(market_data, token_info)
                if supply_chart:
                    st.plotly_chart(supply_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ==================================================
            # RISK & RECOMMENDATIONS SECTION
            # ==================================================
            if risk_metrics['warnings'] or risk_metrics['recommendations']:
                st.markdown('<div class="section-header">⚠️ RISK ANALYSIS & RECOMMENDATIONS</div>', unsafe_allow_html=True)
                
                col_warn, col_rec = st.columns(2)
                
                with col_warn:
                    if risk_metrics['warnings']:
                        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                        st.markdown('<div class="config-header">🚨 POTENTIAL RISKS</div>', unsafe_allow_html=True)
                        for warning in risk_metrics['warnings']:
                            st.write(f"• {warning}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col_rec:
                    if risk_metrics['recommendations']:
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        st.markdown('<div class="config-header">💡 RECOMMENDATIONS</div>', unsafe_allow_html=True)
                        for recommendation in risk_metrics['recommendations']:
                            st.write(f"• {recommendation}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # ==================================================
            # TOKEN DETAILS SECTION
            # ==================================================
            st.markdown('<div class="section-header">🔍 TOKEN DETAILS</div>', unsafe_allow_html=True)
            
            col9, col10, col11, col12 = st.columns(4)
            
            with col9:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<div class="config-header">📝 TOKEN INFO</div>', unsafe_allow_html=True)
                st.write(f"**Token Name:** {token_info.get('name', 'N/A')}")
                st.write(f"**Symbol:** {symbol}")
                st.write(f"**Decimals:** {token_info.get('decimals', 'N/A')}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col10:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<div class="config-header">🔗 NETWORK INFO</div>', unsafe_allow_html=True)
                base_token = market_data.get('base_token', {})
                quote_token = market_data.get('quote_token', {})
                st.write(f"**Primary Pair:** {base_token.get('symbol', 'N/A')}/{quote_token.get('symbol', 'N/A')}")
                st.write(f"**DEX:** {market_data.get('dex_id', 'N/A')}")
                st.write(f"**Chain:** {market_data.get('chain_id', 'N/A')}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col11:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<div class="config-header">💰 SUPPLY INFO</div>', unsafe_allow_html=True)
                st.write(f"**Total Supply:** {format_large_number(token_info['total_supply_formatted'])}")
                st.write(f"**Supply Used:** {format_large_number(market_data.get('supply_used', 0))}")
                st.write(f"**Supply Type:** {market_data.get('supply_type', 'Unknown')}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col12:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<div class="config-header">📊 METRICS</div>', unsafe_allow_html=True)
                st.write(f"**Liquidity Score:** {risk_metrics['liquidity_score']}")
                st.write(f"**Volume/MCap:** {risk_metrics['volume_ratio']}%")
                st.write(f"**Risk Score:** {risk_metrics['risk_score']}/6")
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.error("❌ Could not find token data. Please check the contract address or symbol and try again.")

# Footer
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Crypto Analytics Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Crypto Analysis Pro v2.0</p>
</div>
""", unsafe_allow_html=True)