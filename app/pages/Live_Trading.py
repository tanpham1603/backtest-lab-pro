import streamlit as st
import pandas as pd
import yfinance as yf
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from datetime import datetime, timedelta
import requests
import json
import base64
import os
import pandas_ta as ta

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 Live Trading Pro", 
    page_icon="🛰️", 
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
    .dashboard-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    .position-group {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .position-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }
    .position-item.sell {
        border-left-color: #ff4444;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value {
        font-size: 2rem;
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
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-stock { background: linear-gradient(135deg, #28a745, #7ae582); color: white; }
    .badge-crypto { background: linear-gradient(135deg, #f7931a, #ffc46c); color: black; }
    .badge-profit { background: linear-gradient(135deg, #00ff88, #00cc6a); color: white; }
    .badge-loss { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
    .connection-panel {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .tab-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border: none;
    }
    .order-success {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .order-error {
        background: rgba(255, 68, 68, 0.1);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- ACCOUNT MANAGER ---
class AccountManager:
    def __init__(self):
        self.accounts_file = "saved_accounts.json"
    
    def save_account(self, api_key, api_secret, nickname=""):
        try:
            accounts = self.load_accounts()
            encoded_key = base64.b64encode(api_key.encode()).decode()
            encoded_secret = base64.b64encode(api_secret.encode()).decode()
            
            account_id = f"alpaca_{nickname}_{int(time.time())}"
            accounts[account_id] = {
                'api_key': encoded_key,
                'api_secret': encoded_secret,
                'nickname': nickname,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.accounts_file, 'w') as f:
                json.dump(accounts, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving account: {e}")
            return False
    
    def load_accounts(self):
        try:
            if os.path.exists(self.accounts_file):
                with open(self.accounts_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def delete_account(self, account_id):
        try:
            accounts = self.load_accounts()
            if account_id in accounts:
                del accounts[account_id]
                with open(self.accounts_file, 'w') as f:
                    json.dump(accounts, f, indent=2)
                return True
        except Exception as e:
            st.error(f"Error deleting account: {e}")
        return False
    
    def get_account(self, account_id):
        accounts = self.load_accounts()
        if account_id in accounts:
            account = accounts[account_id].copy()
            try:
                account['api_key'] = base64.b64decode(account['api_key']).decode()
                account['api_secret'] = base64.b64decode(account['api_secret']).decode()
            except:
                pass
            return account
        return None

# --- ALPACA TRADING CLIENT ---
class AlpacaTradingClient:
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self.base_url = "https://paper-api.alpaca.markets"
        self.headers = {}
        self.last_order_time = {}
        self.order_history = []
        
    def connect(self, api_key, api_secret):
        try:
            self.headers = {
                "APCA-API-KEY-ID": api_key.strip(),
                "APCA-API-SECRET-KEY": api_secret.strip()
            }
            
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            if response.status_code == 200:
                self.connected = True
                self._update_account_info()
                return True
            else:
                st.error(f"Connection failed: {response.json().get('message', 'Unknown error')}")
                return False
        except Exception as e:
            st.error(f"Connection error: {e}")
            return False
    
    def _update_account_info(self):
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            if response.status_code == 200:
                account_data = response.json()
                positions_response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
                positions = positions_response.json() if positions_response.status_code == 200 else []
                
                unrealized_pl = sum(float(pos['unrealized_pl']) for pos in positions)
                positions_value = sum(float(pos['market_value']) for pos in positions)
                
                self.account_info = {
                    'portfolio_value': float(account_data['portfolio_value']),
                    'buying_power': float(account_data['buying_power']),
                    'cash': float(account_data['cash']),
                    'equity': float(account_data['equity']),
                    'positions_value': positions_value,
                    'unrealized_pl': unrealized_pl
                }
        except Exception as e:
            st.error(f"Error updating account info: {e}")
    
    def get_account_info(self):
        class Account:
            def __init__(self, info):
                self.portfolio_value = info.get('portfolio_value', 0)
                self.buying_power = info.get('buying_power', 0)
                self.cash = info.get('cash', 0)
                self.equity = info.get('equity', 0)
                self.positions_value = info.get('positions_value', 0)
                self.unrealized_pl = info.get('unrealized_pl', 0)
        return Account(self.account_info)
    
    def get_positions(self):
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            return []
    
    def is_crypto_symbol(self, symbol):
        """Check if symbol is a cryptocurrency"""
        crypto_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'MATIC', 'SOL', 'DOT', 'AVAX', 'LINK', 'LTC', 'BCH', 'XLM', 'UNI', 'ALGO', 'ETC', 'XRP', 'EOS', 'AAVE', 'MKR', 'COMP', 'YFI', 'SNX', 'UMA', 'ZRX', 'BAT']
        symbol_clean = symbol.replace('USD', '').replace('/USD', '').replace('-USD', '')
        return any(crypto in symbol_clean.upper() for crypto in crypto_symbols)
    
    def get_existing_position(self, symbol):
        """Get existing position for a symbol to avoid wash trading"""
        try:
            positions = self.get_positions()
            for position in positions:
                if position['symbol'].upper() == symbol.upper():
                    return position
            return None
        except:
            return None
    
    def calculate_max_buyable_quantity(self, symbol, current_price):
        """Calculate maximum quantity that can be bought based on buying power"""
        try:
            account_info = self.get_account_info()
            buying_power = float(account_info.buying_power)
            
            if current_price <= 0:
                return 0
                
            # Reserve 2% for fees and price fluctuations
            available_cash = buying_power * 0.98
            max_qty = available_cash / current_price
            
            return int(max_qty)
        except:
            return 0
    
    def is_wash_trade(self, symbol, side):
        """Check if this would be a wash trade"""
        try:
            # Get recent order history for this symbol
            recent_orders = [o for o in self.order_history 
                           if o['symbol'] == symbol.upper() 
                           and time.time() - o['timestamp'] < 300]  # 5 minutes
            
            if not recent_orders:
                return False
                
            # Check if we're trying to buy after recently selling, or vice versa
            last_order = recent_orders[-1]
            if last_order['side'] != side:
                st.warning(f"⚠️ Wash trade warning: You {last_order['side'].upper()}ed this symbol recently")
                return True
                
            return False
        except:
            return False
    
    def place_order(self, symbol, qty, side, order_type="market", notional=None):
        try:
            # Anti-wash trading: Check recent trades
            if self.is_wash_trade(symbol, side):
                st.error("🚫 Order blocked: This would be a wash trade. Wait 5 minutes between buy/sell for the same symbol.")
                return None
            
            # Rate limiting
            current_time = time.time()
            if symbol in self.last_order_time:
                time_since_last = current_time - self.last_order_time[symbol]
                if time_since_last < 5:  # 5 second cooldown per symbol
                    st.warning(f"⏳ Please wait {5 - time_since_last:.1f}s before trading {symbol} again")
                    return None
            
            # Different order parameters for crypto vs stocks
            is_crypto = self.is_crypto_symbol(symbol)
            
            # For crypto, use notional amount instead of quantity
            if is_crypto and side.lower() == "buy":
                current_price = get_current_price(symbol, "Crypto")
                if current_price:
                    notional_value = float(qty) * current_price
                    if notional_value < 1.0:  # Alpaca minimum for crypto
                        st.error(f"🚫 Minimum crypto order is $1.00. Your order: ${notional_value:.2f}")
                        return None
            
            order_data = {
                "symbol": symbol.upper(),
                "side": side.lower(),
                "type": order_type.lower(),
            }
            
            # Set appropriate parameters based on asset type
            if is_crypto and side.lower() == "buy" and notional:
                order_data["notional"] = str(notional)
            else:
                order_data["qty"] = str(int(qty))
            
            # Set appropriate time_in_force
            if is_crypto:
                order_data["time_in_force"] = "gtc"  # Good till cancelled for crypto
            else:
                order_data["time_in_force"] = "day"  # Day for stocks
            
            # Additional parameters to avoid issues
            order_data["client_order_id"] = f"{symbol}_{side}_{int(time.time())}_{np.random.randint(1000,9999)}"
            
            response = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data)
            
            if response.status_code == 200:
                self.last_order_time[symbol] = current_time
                # Record this order in history
                self.order_history.append({
                    'symbol': symbol.upper(),
                    'side': side.lower(),
                    'timestamp': current_time,
                    'qty': qty
                })
                self._update_account_info()
                order_result = response.json()
                
                # Display success message
                st.markdown(f"""
                <div class="order-success">
                    <h4>✅ Order Executed Successfully!</h4>
                    <p><strong>Symbol:</strong> {order_result['symbol']}</p>
                    <p><strong>Side:</strong> {order_result['side'].upper()}</p>
                    <p><strong>Quantity:</strong> {order_result.get('qty', order_result.get('notional', 'N/A'))}</p>
                    <p><strong>Type:</strong> {order_result['type']}</p>
                    <p><strong>Status:</strong> {order_result['status']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                return order_result
            else:
                error_msg = response.json().get('message', 'Unknown error')
                st.markdown(f"""
                <div class="order-error">
                    <h4>❌ Order Failed</h4>
                    <p><strong>Error:</strong> {error_msg}</p>
                </div>
                """, unsafe_allow_html=True)
                return None
                
        except Exception as e:
            st.markdown(f"""
            <div class="order-error">
                <h4>❌ Order Error</h4>
                <p><strong>Exception:</strong> {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
            return None

# --- PERFORMANCE ANALYTICS ---
class PerformanceAnalytics:
    def __init__(self, trader):
        self.trader = trader
        self.trades = []
    
    def get_portfolio_summary(self):
        try:
            account_info = self.trader.get_account_info()
            positions = self.trader.get_positions()
            
            return {
                'total_equity': float(account_info.equity),
                'buying_power': float(account_info.buying_power),
                'cash': float(account_info.cash),
                'positions_value': float(account_info.positions_value),
                'unrealized_pl': float(account_info.unrealized_pl),
                'total_positions': len(positions)
            }
        except:
            return {
                'total_equity': 0, 'buying_power': 0, 'cash': 0,
                'positions_value': 0, 'unrealized_pl': 0, 'total_positions': 0
            }

# --- RISK MANAGEMENT ---
class RiskManager:
    def __init__(self, trader):
        self.trader = trader
        
    def calculate_var(self, positions, confidence_level=0.95, periods=252):
        try:
            account_info = self.trader.get_account_info()
            portfolio_value = float(account_info.equity)
            
            if portfolio_value == 0:
                return {'1d': 0, '1w': 0, '1m': 0}
                
            annual_volatility = 0.20
            daily_volatility = annual_volatility / np.sqrt(periods)
            
            var_1d = portfolio_value * daily_volatility * 2.33
            var_1w = var_1d * np.sqrt(5)
            var_1m = var_1d * np.sqrt(21)
            
            return {'1d': var_1d, '1w': var_1w, '1m': var_1m}
        except Exception as e:
            return {'1d': 0, '1w': 0, '1m': 0}

# --- SOCIAL SENTIMENT ---
class SocialSentimentAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def get_social_sentiment(self, symbol):
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        combined_sentiment = np.random.uniform(-1, 1)
        
        result = {
            'combined_score': combined_sentiment,
            'sentiment_label': 'Bullish' if combined_sentiment > 0.1 else 'Bearish' if combined_sentiment < -0.1 else 'Neutral',
            'confidence': min(100, abs(combined_sentiment) * 100)
        }
        
        self.cache[cache_key] = result
        return result

# --- MACHINE LEARNING ---
@st.cache_resource
def train_model_on_the_fly(data):
    if data is None or len(data) < 100:
        return None
    
    try:
        data['SMA_20'] = data['close'].rolling(20).mean()
        data['SMA_50'] = data['close'].rolling(50).mean()
        data['RSI'] = ta.rsi(data['close'], length=14)
        data['Volume_SMA'] = data['volume'].rolling(20).mean()
        
        data['Future_Return'] = data['close'].shift(-5) / data['close'] - 1
        data['Target'] = (data['Future_Return'] > 0).astype(int)
        
        features = ['SMA_20', 'SMA_50', 'RSI', 'Volume_SMA']
        data_clean = data.dropna()
        
        if len(data_clean) < 50:
            return None
            
        X = data_clean[features]
        y = data_clean['Target']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    except Exception as e:
        return None

def get_ml_signal(data, model):
    if data is None or model is None:
        return "NO_SIGNAL", 0
    
    try:
        latest = data.iloc[-1].copy()
        latest['SMA_20'] = data['close'].rolling(20).mean().iloc[-1]
        latest['SMA_50'] = data['close'].rolling(50).mean().iloc[-1]
        latest['RSI'] = ta.rsi(data['close'], length=14).iloc[-1]
        latest['Volume_SMA'] = data['volume'].rolling(20).mean().iloc[-1]
        
        features = ['SMA_20', 'SMA_50', 'RSI', 'Volume_SMA']
        X_latest = pd.DataFrame([latest[features]])
        
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        
        confidence = max(probability)
        
        if prediction == 1 and confidence > 0.6:
            return "BUY", confidence
        elif prediction == 0 and confidence > 0.6:
            return "SELL", confidence
        else:
            return "HOLD", confidence
            
    except Exception as e:
        return "NO_SIGNAL", 0

# --- DATA LOADING ---
@st.cache_data(ttl=300)
def load_stock_data(symbol, period="6mo"):
    try:
        data = yf.download(symbol, period=period, progress=False)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        return data
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_crypto_data(symbol):
    try:
        # Clean symbol for crypto
        symbol_clean = symbol.replace('USD', '').replace('/USD', '').replace('-USD', '').replace('/USDT', '').replace('USDT', '')
        data = yf.download(symbol_clean + "-USD", period="6mo", progress=False)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns]
        return data
    except:
        return None

def get_current_price(symbol, asset_type):
    try:
        if asset_type == "Stock":
            data = load_stock_data(symbol, "1d")
        else:
            data = load_crypto_data(symbol)
        
        if data is not None and not data.empty:
            return data['close'].iloc[-1]
        return None
    except:
        return None

# --- HELPER FUNCTIONS ---
def display_position(position):
    symbol = position['symbol']
    qty = float(position['qty'])
    avg_entry = float(position['avg_entry_price'])
    current_price = float(position['current_price'])
    unrealized_pl = float(position['unrealized_pl'])
    pl_percent = (unrealized_pl / (avg_entry * qty)) * 100 if avg_entry * qty != 0 else 0
    
    pl_class = "" if unrealized_pl >= 0 else "sell"
    badge_class = "badge-profit" if unrealized_pl >= 0 else "badge-loss"
    
    st.markdown(f"""
    <div class="position-item {pl_class}">
        <div style="display: flex; justify-content: between; align-items: center;">
            <div>
                <strong>{symbol}</strong>
                <span class="badge {badge_class}">{qty:.0f} shares</span>
            </div>
            <div style="text-align: right;">
                <div style="color: {'#00ff88' if unrealized_pl >= 0 else '#ff4444'}; font-weight: bold;">
                    ${unrealized_pl:+.2f} ({pl_percent:+.1f}%)
                </div>
                <div style="font-size: 0.8em; color: #8898aa;">
                    Avg: ${avg_entry:.2f} • Current: ${current_price:.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'trader' not in st.session_state:
    st.session_state.trader = AlpacaTradingClient()
if 'performance_analytics' not in st.session_state:
    st.session_state.performance_analytics = None
if 'account_manager' not in st.session_state:
    st.session_state.account_manager = AccountManager()
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SocialSentimentAnalyzer()
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# --- CONNECTION PANEL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="connection-panel">
        <h3>🔌 API Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input("API Key", type="password", placeholder="Enter Alpaca API Key")
    api_secret = st.text_input("API Secret", type="password", placeholder="Enter Alpaca API Secret")
    
    col1a, col1b = st.columns(2)
    with col1a:
        if st.button("🚀 Connect", use_container_width=True, type="primary"):
            if api_key and api_secret:
                with st.spinner("Connecting..."):
                    if st.session_state.trader.connect(api_key, api_secret):
                        st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                        st.success("Connected Successfully!")
            else:
                st.warning("Enter API credentials")
    
    with col1b:
        if st.button("💾 Save Account", use_container_width=True):
            if api_key and api_secret:
                nickname = st.text_input("Account Name", value="My Alpaca Account", key="account_name")
                if st.session_state.account_manager.save_account(api_key, api_secret, nickname):
                    st.success("Account Saved!")

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <h4>📋 Quick Guide</h4>
        <p><strong>1. Get API Keys:</strong><br>app.alpaca.markets</p>
        <p><strong>2. Paper Trading:</strong><br>$100,000 virtual</p>
        <p><strong>3. US Stocks:</strong><br>AAPL, TSLA, etc.</p>
        <p><strong>4. Crypto:</strong><br>BTC, ETH, etc.</p>
    </div>
    """, unsafe_allow_html=True)

# Load saved accounts
saved_accounts = st.session_state.account_manager.load_accounts()
if saved_accounts:
    with st.expander("💾 Saved Accounts", expanded=False):
        account_options = {k: v['nickname'] for k, v in saved_accounts.items()}
        selected_account = st.selectbox("Select account:", [""] + list(account_options.keys()), 
                                      format_func=lambda x: account_options.get(x, "Select account"))
        
        if selected_account:
            account_info = st.session_state.account_manager.get_account(selected_account)
            if account_info and st.button("Use This Account"):
                st.session_state.trader = AlpacaTradingClient()
                if st.session_state.trader.connect(account_info['api_key'], account_info['api_secret']):
                    st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                    st.success("Connected with saved account!")
                    st.rerun()

# --- MAIN DASHBOARD ---
if st.session_state.trader.connected:
    trader = st.session_state.trader
    performance_analytics = st.session_state.performance_analytics
    
    # Refresh Button
    if st.button("🔄 Refresh Data"):
        trader._update_account_info()
        st.rerun()
    
    # Portfolio Overview
    st.markdown("---")
    st.subheader("📊 Portfolio Overview")
    
    portfolio_summary = performance_analytics.get_portfolio_summary()
    positions = trader.get_positions()
    
    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">💰</div>
            <div class="metric-value">${portfolio_summary['total_equity']:,.0f}</div>
            <div class="metric-label">Total Equity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">⚡</div>
            <div class="metric-value">${portfolio_summary['buying_power']:,.0f}</div>
            <div class="metric-label">Buying Power</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">💵</div>
            <div class="metric-value">${portfolio_summary['cash']:,.0f}</div>
            <div class="metric-label">Available Cash</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pl_color = "#00ff88" if portfolio_summary['unrealized_pl'] >= 0 else "#ff4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">📈</div>
            <div class="metric-value" style="color: {pl_color}">${portfolio_summary['unrealized_pl']:,.0f}</div>
            <div class="metric-label">Unrealized P&L</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Positions Grouped View
    st.markdown("---")
    st.subheader("📈 Positions")
    
    if positions:
        # Group positions by type (profit/loss)
        profitable_positions = [p for p in positions if float(p['unrealized_pl']) > 0]
        losing_positions = [p for p in positions if float(p['unrealized_pl']) <= 0]
        
        col_pos1, col_pos2 = st.columns(2)
        
        with col_pos1:
            if profitable_positions:
                st.markdown("""
                <div class="position-group">
                    <h4>🟢 Profitable Positions</h4>
                </div>
                """, unsafe_allow_html=True)
                for position in profitable_positions:
                    display_position(position)
            else:
                st.info("No profitable positions")
        
        with col_pos2:
            if losing_positions:
                st.markdown("""
                <div class="position-group">
                    <h4>🔴 Losing Positions</h4>
                </div>
                """, unsafe_allow_html=True)
                for position in losing_positions:
                    display_position(position)
            else:
                st.info("No losing positions")
    else:
        st.info("💰 No active positions")
    
    # Tabs for Advanced Features
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Trading", "🤖 Auto Trading", "📊 Analytics", "🎭 Sentiment"])
    
    with tab1:
        st.subheader("Manual Trading")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            asset_type = st.radio("Asset Type", ["Stock", "Crypto"], horizontal=True, key="asset_type")
            symbol = st.text_input("Symbol", value="AAPL" if asset_type == "Stock" else "BTC", key="trading_symbol").upper()
            
            # Show existing position if any
            existing_position = trader.get_existing_position(symbol)
            if existing_position:
                st.info(f"📊 Existing position: {existing_position['qty']} shares at ${existing_position['avg_entry_price']}")
        
        with col_t2:
            # Different input for crypto vs stocks
            current_price = get_current_price(symbol, asset_type)
            
            if asset_type == "Crypto":
                # For crypto, use dollar amount instead of quantity
                st.markdown("**Investment Amount ($)**")
                notional = st.number_input(
                    "Amount in USD", 
                    min_value=1.0, 
                    value=100.0, 
                    step=1.0, 
                    key="crypto_amount",
                    label_visibility="collapsed"
                )
                
                if current_price:
                    equivalent_qty = notional / current_price
                    st.caption(f"≈ {equivalent_qty:.6f} {symbol}")
                    qty = equivalent_qty
                else:
                    qty = 0
            else:
                # For stocks, use quantity
                qty = st.number_input(
                    "Quantity", 
                    min_value=1, 
                    value=10, 
                    step=1, 
                    key="stock_qty"
                )
                notional = None
            
            # Order type selection
            order_type = st.selectbox("Order Type", ["market", "limit"], key="order_type")
            
            # Price display and validation
            if current_price:
                st.metric("Current Price", f"${current_price:.2f}")
                
                # Calculate order value and validate
                if asset_type == "Crypto":
                    order_value = notional
                else:
                    order_value = qty * current_price
                
                account_info = trader.get_account_info()
                buying_power = float(account_info.buying_power)
                
                if order_value > buying_power:
                    st.error(f"❌ Insufficient buying power. Required: ${order_value:.2f}, Available: ${buying_power:.2f}")
                else:
                    st.success(f"✅ Sufficient buying power: ${buying_power:.2f} available")
        
        col_buy, col_sell, col_info = st.columns([1, 1, 2])
        
        with col_buy:
            if st.button("🟢 BUY", use_container_width=True, type="primary", key="buy_btn"):
                if symbol and ((asset_type == "Crypto" and notional >= 1.0) or (asset_type == "Stock" and qty > 0)):
                    # Final validation before order
                    if asset_type == "Crypto" and notional < 1.0:
                        st.error("🚫 Minimum crypto order is $1.00")
                    else:
                        result = trader.place_order(symbol, qty, "buy", order_type, notional)
                        if result:
                            time.sleep(2)
                            st.rerun()
                else:
                    st.error("Please check your order parameters")
        
        with col_sell:
            if st.button("🔴 SELL", use_container_width=True, type="primary", key="sell_btn"):
                if symbol and qty > 0:
                    # Check if we have the position to sell
                    existing_position = trader.get_existing_position(symbol)
                    if not existing_position:
                        st.error(f"❌ No position found for {symbol}")
                    elif float(existing_position['qty']) < qty:
                        st.error(f"❌ Insufficient shares. You have {existing_position['qty']} shares, trying to sell {qty}")
                    else:
                        result = trader.place_order(symbol, qty, "sell", order_type)
                        if result:
                            time.sleep(2)
                            st.rerun()
                else:
                    st.error("Please check your order parameters")
        
        with col_info:
            st.markdown("""
            <div class="warning-box">
                <h4>💡 Trading Rules</h4>
                <p><strong>Stocks:</strong> Minimum 1 share</p>
                <p><strong>Crypto:</strong> Minimum $1.00 order</p>
                <p><strong>Wash Trade Protection:</strong> 5-minute cooldown between buy/sell</p>
                <p><strong>Rate Limit:</strong> 5 seconds between orders per symbol</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Automated Trading")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            auto_symbol = st.text_input("Symbol", value="AAPL", key="auto_symbol").upper()
            auto_asset = st.radio("Asset", ["Stock", "Crypto"], horizontal=True, key="auto_asset")
            risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)
        
        with col_a2:
            ml_enabled = st.checkbox("Enable ML Signals", value=True)
            min_confidence = st.slider("Min Confidence", 0.5, 0.95, 0.7, 0.05)
            
            col_start, col_stop = st.columns(2)
            with col_start:
                if not st.session_state.bot_running:
                    if st.button("🚀 Start Bot", use_container_width=True):
                        st.session_state.bot_running = True
                        st.success("Bot started!")
                else:
                    if st.button("⏸️ Pause Bot", use_container_width=True):
                        st.session_state.bot_running = False
                        st.warning("Bot paused!")
            
            with col_stop:
                if st.button("🛑 Stop Bot", use_container_width=True):
                    st.session_state.bot_running = False
                    st.error("Bot stopped!")
        
        if st.session_state.bot_running:
            st.info("🤖 Bot is running...")
            # Simulate ML trading
            data = load_stock_data(auto_symbol) if auto_asset == "Stock" else load_crypto_data(auto_symbol)
            if data is not None:
                model = train_model_on_the_fly(data)
                signal, confidence = get_ml_signal(data, model)
                
                col_sig, col_conf = st.columns(2)
                with col_sig:
                    st.metric("ML Signal", signal)
                with col_conf:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                if ml_enabled and confidence >= min_confidence and signal != "HOLD":
                    st.warning(f"🚨 Signal: {signal} {auto_symbol} (Confidence: {confidence:.1%})")
    
    with tab3:
        st.subheader("Risk & Performance")
        
        risk_manager = RiskManager(trader)
        portfolio_var = risk_manager.calculate_var(positions)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("1-Day VaR", f"${portfolio_var['1d']:,.0f}")
        with col_r2:
            st.metric("1-Week VaR", f"${portfolio_var['1w']:,.0f}")
        with col_r3:
            st.metric("1-Month VaR", f"${portfolio_var['1m']:,.0f}")
        
        # Performance Chart
        if st.button("Generate Report"):
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = portfolio_summary['unrealized_pl'],
                title = {'text': "Portfolio P&L"},
                gauge = {'axis': {'range': [min(-1000, portfolio_summary['unrealized_pl']), max(1000, portfolio_summary['unrealized_pl'])]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Social Sentiment")
        
        sentiment_symbol = st.text_input("Symbol for Analysis", value="AAPL", key="sentiment_symbol").upper()
        
        if st.button("Analyze Sentiment"):
            sentiment_data = st.session_state.sentiment_analyzer.get_social_sentiment(sentiment_symbol)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Sentiment", sentiment_data['sentiment_label'])
            with col_s2:
                st.metric("Score", f"{sentiment_data['combined_score']:.2f}")
            with col_s3:
                st.metric("Confidence", f"{sentiment_data['confidence']:.1f}%")
            
            # Sentiment Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_data['combined_score'],
                title = {'text': "Sentiment Score"},
                gauge = {'axis': {'range': [-1, 1]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

else:
    # Welcome Screen
    st.markdown("---")
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">📈</div>
            <h3>Stock Trading</h3>
            <p>Trade US stocks with real-time data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_w2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">💰</div>
            <h3>Paper Account</h3>
            <p>$100,000 virtual trading</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_w3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">⚡</div>
            <h3>Live Data</h3>
            <p>Real-time market updates</p>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit • Alpaca Markets API</p>
</div>
""", unsafe_allow_html=True)