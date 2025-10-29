import streamlit as st
import pandas as pd
import yfinance as yf
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
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
    .badge-profit { background: linear-gradient(135deg, #00ff88, #00cc6a); color: white; }
    .badge-loss { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
    .connection-panel {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .sentiment-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .sentiment-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    .sentiment-positive {
        border-left: 4px solid #00ff88;
    }
    .sentiment-negative {
        border-left: 4px solid #ff4444;
    }
    .sentiment-neutral {
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header-gradient">🚀 Live Trading Pro</div>', unsafe_allow_html=True)

# --- ALPACA TRADING CLIENT ---
class AlpacaTradingClient:
    def __init__(self):
        self.connected = False
        self.base_url = "https://paper-api.alpaca.markets"
        self.headers = {}
        self.account_info = {}
        self.positions = []
        
    def connect(self, api_key, api_secret):
        try:
            self.headers = {
                "APCA-API-KEY-ID": api_key.strip(),
                "APCA-API-SECRET-KEY": api_secret.strip()
            }
            
            # Test connection
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if response.status_code == 200:
                self.connected = True
                self._update_account_data()
                return True
            else:
                error_msg = response.json().get('message', 'Unknown error')
                st.error(f"❌ Connection failed: {error_msg}")
                return False
                
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
            return False
    
    def _update_account_data(self):
        """Update both account info and positions"""
        try:
            # Get account info
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
            if response.status_code == 200:
                account_data = response.json()
                
                # Get positions
                positions_response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
                if positions_response.status_code == 200:
                    self.positions = positions_response.json()
                else:
                    self.positions = []
                
                # Calculate metrics
                positions_value = sum(float(pos.get('market_value', 0)) for pos in self.positions)
                unrealized_pl = sum(float(pos.get('unrealized_pl', 0)) for pos in self.positions)
                
                self.account_info = {
                    'equity': float(account_data.get('equity', 0)),
                    'buying_power': float(account_data.get('buying_power', 0)),
                    'cash': float(account_data.get('cash', 0)),
                    'portfolio_value': float(account_data.get('portfolio_value', 0)),
                    'positions_value': positions_value,
                    'unrealized_pl': unrealized_pl,
                    'positions_count': len(self.positions)
                }
                return True
            return False
        except Exception as e:
            st.error(f"Error updating account data: {str(e)}")
            return False
    
    def get_account_info(self):
        return self.account_info
    
    def get_positions(self):
        return self.positions
    
    def place_order(self, symbol, qty, side):
        try:
            symbol = symbol.upper().strip()
            order_data = {
                "symbol": symbol,
                "qty": str(int(qty)),
                "side": side.lower(),
                "type": "market",
                "time_in_force": "day"
            }
            
            response = requests.post(
                f"{self.base_url}/v2/orders", 
                headers=self.headers, 
                json=order_data, 
                timeout=10
            )
            
            if response.status_code == 200:
                # Wait a moment for order to process
                time.sleep(2)
                # Update account data after successful order
                self._update_account_data()
                return response.json()
            else:
                error_msg = response.json().get('message', 'Unknown error')
                st.error(f"❌ Order failed: {error_msg}")
                return None
                
        except Exception as e:
            st.error(f"❌ Order error: {str(e)}")
            return None

# --- AI TRADING ASSISTANT ---
class AITradingAssistant:
    def __init__(self):
        self.history = []
    
    def analyze_stock(self, symbol):
        try:
            # Get stock data
            stock_data = yf.download(symbol, period='3mo', progress=False)
            if stock_data.empty:
                return self._get_fallback_analysis(symbol, "No data available for symbol")
            
            # Calculate technical indicators
            current_price = float(stock_data['Close'].iloc[-1])
            sma_20 = float(stock_data['Close'].rolling(20).mean().iloc[-1])
            sma_50 = float(stock_data['Close'].rolling(50).mean().iloc[-1])
            
            # Calculate RSI safely
            rsi_series = ta.rsi(stock_data['Close'])
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50.0
            
            # Price vs moving averages (with zero check)
            price_vs_sma_20 = ((current_price - sma_20) / sma_20) * 100 if sma_20 != 0 else 0
            price_vs_sma_50 = ((current_price - sma_50) / sma_50) * 100 if sma_50 != 0 else 0
            
            # Determine sentiment
            if price_vs_sma_20 > 5 and price_vs_sma_50 > 10:
                sentiment = "STRONG_BULLISH"
                recommendation = "🟢 STRONG BUY"
                confidence = 85
                risk = "LOW"
            elif price_vs_sma_20 > 2 and price_vs_sma_50 > 5:
                sentiment = "BULLISH"
                recommendation = "🟢 BUY"
                confidence = 75
                risk = "MEDIUM"
            elif abs(price_vs_sma_20) < 2 and abs(price_vs_sma_50) < 5:
                sentiment = "NEUTRAL"
                recommendation = "🟡 HOLD"
                confidence = 60
                risk = "MEDIUM"
            elif price_vs_sma_20 < -2 and price_vs_sma_50 < -5:
                sentiment = "BEARISH"
                recommendation = "🔴 SELL"
                confidence = 75
                risk = "HIGH"
            else:
                sentiment = "STRONG_BEARISH"
                recommendation = "🔴 STRONG SELL"
                confidence = 85
                risk = "VERY_HIGH"
            
            # Generate insights
            insights = []
            if price_vs_sma_20 > 0:
                insights.append(f"📈 Trading {price_vs_sma_20:.1f}% above 20-day average")
            else:
                insights.append(f"📉 Trading {abs(price_vs_sma_20):.1f}% below 20-day average")
            
            if rsi > 70:
                insights.append("🚨 RSI indicates overbought conditions")
            elif rsi < 30:
                insights.append("💎 RSI indicates oversold conditions")
            
            # Price targets
            if "BULLISH" in sentiment:
                targets = {
                    'short_term': current_price * 1.05,
                    'medium_term': current_price * 1.15,
                    'long_term': current_price * 1.25
                }
            elif "BEARISH" in sentiment:
                targets = {
                    'short_term': current_price * 0.95,
                    'medium_term': current_price * 0.85,
                    'long_term': current_price * 0.75
                }
            else:
                targets = {
                    'short_term': current_price * 1.02,
                    'medium_term': current_price * 1.08,
                    'long_term': current_price * 1.12
                }
            
            # Support and resistance
            support = float(stock_data['Low'].rolling(20).min().iloc[-1])
            resistance = float(stock_data['High'].rolling(20).max().iloc[-1])
            
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'sentiment': sentiment,
                'recommendation': recommendation,
                'confidence': confidence,
                'risk': risk,
                'insights': insights,
                'targets': targets,
                'support': support,
                'resistance': resistance,
                'rsi': rsi,
                'timestamp': datetime.now().isoformat()
            }
            
            self.history.append(analysis)
            return analysis
            
        except Exception as e:
            return self._get_fallback_analysis(symbol, str(e))
    
    def _get_fallback_analysis(self, symbol, error_msg):
        return {
            'symbol': symbol,
            'current_price': 100.0,
            'sentiment': 'NEUTRAL',
            'recommendation': '🟡 HOLD',
            'confidence': 50,
            'risk': 'MEDIUM',
            'insights': [
                f"⚠️ Analysis limited: {error_msg}",
                "💡 Try popular symbols: AAPL, TSLA, MSFT, GOOGL, SPY"
            ],
            'targets': {
                'short_term': 105.0,
                'medium_term': 110.0, 
                'long_term': 115.0
            },
            'support': 95.0,
            'resistance': 105.0,
            'rsi': 50.0,
            'timestamp': datetime.now().isoformat()
        }

# --- DATA FUNCTIONS ---
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="3mo"):
    try:
        data = yf.download(symbol, period=period, progress=False)
        return data if not data.empty else None
    except:
        return None

def get_current_price(symbol):
    try:
        data = yf.download(symbol, period='1d', progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else None
    except:
        return None

# --- UI COMPONENTS ---
def display_portfolio_overview(client):
    account_info = client.get_account_info()
    
    st.subheader("📊 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">💰</div>
            <div class="metric-value">${account_info.get('equity', 0):,.0f}</div>
            <div class="metric-label">Portfolio Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">⚡</div>
            <div class="metric-value">${account_info.get('buying_power', 0):,.0f}</div>
            <div class="metric-label">Buying Power</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">💵</div>
            <div class="metric-value">${account_info.get('cash', 0):,.0f}</div>
            <div class="metric-label">Available Cash</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unrealized_pl = account_info.get('unrealized_pl', 0)
        pl_color = "#00ff88" if unrealized_pl >= 0 else "#ff4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="feature-icon">📈</div>
            <div class="metric-value" style="color: {pl_color}">${unrealized_pl:,.2f}</div>
            <div class="metric-label">Unrealized P&L</div>
        </div>
        """, unsafe_allow_html=True)

def display_positions(client):
    positions = client.get_positions()
    
    st.subheader("📈 Current Positions")
    
    if not positions:
        st.info("💰 No active positions - Start trading to build your portfolio!")
        return
    
    # Group positions by profit/loss
    profitable = []
    losing = []
    
    for pos in positions:
        try:
            unrealized_pl = float(pos.get('unrealized_pl', 0))
            if unrealized_pl > 0:
                profitable.append(pos)
            else:
                losing.append(pos)
        except (ValueError, TypeError):
            continue
    
    col1, col2 = st.columns(2)
    
    with col1:
        if profitable:
            st.markdown("""
            <div class="position-group">
                <h4>🟢 Profitable Positions</h4>
            </div>
            """, unsafe_allow_html=True)
            for pos in profitable:
                display_position_card(pos)
        else:
            st.info("No profitable positions")
    
    with col2:
        if losing:
            st.markdown("""
            <div class="position-group">
                <h4>🔴 Losing Positions</h4>
            </div>
            """, unsafe_allow_html=True)
            for pos in losing:
                display_position_card(pos)
        else:
            st.info("No losing positions")

def display_position_card(position):
    try:
        symbol = position.get('symbol', 'Unknown')
        qty = float(position.get('qty', 0))
        avg_price = float(position.get('avg_entry_price', 0))
        current_price = float(position.get('current_price', 0))
        unrealized_pl = float(position.get('unrealized_pl', 0))
        
        # Calculate P&L percentage safely
        cost_basis = avg_price * qty
        if cost_basis > 0:
            pl_percent = (unrealized_pl / cost_basis) * 100
        else:
            pl_percent = 0.0
        
        pl_class = "" if unrealized_pl >= 0 else "sell"
        badge_class = "badge-profit" if unrealized_pl >= 0 else "badge-loss"
        
        st.markdown(f"""
        <div class="position-item {pl_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{symbol}</strong>
                    <span class="badge {badge_class}">{qty:.0f} shares</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {'#00ff88' if unrealized_pl >= 0 else '#ff4444'}; font-weight: bold;">
                        ${unrealized_pl:+.2f} ({pl_percent:+.1f}%)
                    </div>
                    <div style="font-size: 0.8em; color: #8898aa;">
                        Avg: ${avg_price:.2f} • Current: ${current_price:.2f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        return

def display_ai_assistant():
    st.subheader("🤖 AI Trading Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter stock symbol:", "AAPL", key="ai_symbol").upper()
        
        if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing..."):
                assistant = AITradingAssistant()
                analysis = assistant.analyze_stock(symbol)
                display_ai_analysis(analysis)
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h4>🎯 AI Capabilities</h4>
            <p>• Technical Analysis</p>
            <p>• Price Action</p>
            <p>• RSI Indicators</p>
            <p>• Support/Resistance</p>
            <p>• Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)

def display_ai_analysis(analysis):
    sentiment_color = {
        "STRONG_BULLISH": "#00ff88",
        "BULLISH": "#7ae582", 
        "NEUTRAL": "#667eea",
        "BEARISH": "#ff6b6b",
        "STRONG_BEARISH": "#ff4444"
    }
    
    color = sentiment_color.get(analysis['sentiment'], "#667eea")
    
    st.markdown(f"""
    <div class="sentiment-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: {color}; font-size: 1.5rem;">
                    {analysis['recommendation']}
                </h3>
                <p style="margin: 0.2rem 0 0 0; color: #8898aa;">
                    {analysis['symbol']} • Confidence: {analysis['confidence']}% • Risk: {analysis['risk']}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.2rem; font-weight: bold; color: {color};">
                    {analysis['sentiment'].replace('_', ' ').title()}
                </div>
                <div style="color: #8898aa; font-size: 0.8rem;">
                    Current: ${analysis['current_price']:.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 💡 Key Insights")
    for insight in analysis['insights']:
        st.markdown(f"- {insight}")
    
    st.markdown("#### 🎯 Price Targets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Short Term", f"${analysis['targets']['short_term']:.2f}")
    with col2:
        st.metric("Medium Term", f"${analysis['targets']['medium_term']:.2f}")
    with col3:
        st.metric("Long Term", f"${analysis['targets']['long_term']:.2f}")
    
    st.markdown("#### 📊 Support & Resistance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Support Level", f"${analysis['support']:.2f}")
    with col2:
        st.metric("Resistance Level", f"${analysis['resistance']:.2f}")

# --- SESSION STATE ---
if 'trading_client' not in st.session_state:
    st.session_state.trading_client = AlpacaTradingClient()
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# --- MAIN APP ---
def main():
    # Connection Panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="connection-panel">
            <h3>🔌 Alpaca Trading API</h3>
            <p>Connect your Alpaca Paper Trading account to start live trading</p>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input("API Key", type="password", placeholder="Enter Alpaca API Key")
        api_secret = st.text_input("API Secret", type="password", placeholder="Enter Alpaca API Secret")
        
        col_connect, col_refresh = st.columns(2)
        with col_connect:
            if st.button("🚀 Connect to Alpaca", type="primary", use_container_width=True):
                if api_key and api_secret:
                    with st.spinner("Connecting to Alpaca..."):
                        if st.session_state.trading_client.connect(api_key, api_secret):
                            st.success("✅ Connected to Alpaca successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to connect. Check your credentials.")
                else:
                    st.warning("⚠️ Please enter both API Key and Secret")
        
        with col_refresh:
            if st.session_state.trading_client.connected:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    with st.spinner("Refreshing..."):
                        st.session_state.trading_client._update_account_data()
                        st.rerun()
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h4>📋 Quick Guide</h4>
            <p><strong>1. Get API Keys:</strong><br>paper.alpaca.markets</p>
            <p><strong>2. Paper Trading:</strong><br>$100,000 virtual</p>
            <p><strong>3. Supported:</strong><br>US Stocks & ETFs</p>
            <p><strong>4. Real-time:</strong><br>Live market data</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Dashboard
    if st.session_state.trading_client.connected:
        client = st.session_state.trading_client
        
        # Portfolio Overview
        display_portfolio_overview(client)
        st.markdown("---")
        
        # Positions
        display_positions(client)
        st.markdown("---")
        
        # Trading Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Manual Trading", "📊 Analytics", "🤖 AI Assistant"])
        
        with tab1:
            st.subheader("Manual Trading")
            
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.text_input("Stock Symbol:", "AAPL", key="trade_symbol").upper()
                qty = st.number_input("Quantity:", min_value=1, value=10, key="trade_qty")
                
                current_price = get_current_price(symbol)
                if current_price:
                    st.metric("Current Price", f"${current_price:.2f}")
                    total_cost = current_price * qty
                    st.metric("Total Cost", f"${total_cost:,.2f}")
                else:
                    st.warning("Could not fetch current price")
            
            with col2:
                st.markdown("#### Place Order")
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if st.button("🟢 BUY", use_container_width=True, type="primary"):
                        if symbol and qty > 0:
                            result = client.place_order(symbol, qty, "buy")
                            if result:
                                st.success(f"✅ Buy order executed for {qty} {symbol}")
                                time.sleep(2)
                                st.rerun()
                
                with col_sell:
                    if st.button("🔴 SELL", use_container_width=True, type="primary"):
                        if symbol and qty > 0:
                            result = client.place_order(symbol, qty, "sell")
                            if result:
                                st.success(f"✅ Sell order executed for {qty} {symbol}")
                                time.sleep(2)
                                st.rerun()
        
        with tab2:
            st.subheader("Portfolio Analytics")
            
            account_info = client.get_account_info()
            positions = client.get_positions()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                portfolio_value = account_info.get('portfolio_value', 0)
                var_1d = portfolio_value * 0.02
                st.metric("1-Day VaR (95%)", f"${var_1d:,.0f}")
            
            with col2:
                diversification = len(positions)
                st.metric("Diversification", f"{diversification} positions")
            
            with col3:
                cash = account_info.get('cash', 0)
                portfolio_value = account_info.get('portfolio_value', 1)
                cash_ratio = (cash / portfolio_value) * 100 if portfolio_value > 0 else 0
                st.metric("Cash Allocation", f"{cash_ratio:.1f}%")
        
        with tab3:
            display_ai_assistant()
    
    else:
        # Welcome screen when not connected
        st.markdown("---")
        st.subheader("🎯 Welcome to Live Trading Pro")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">📈</div>
                <h3>Alpaca Integration</h3>
                <p>Connect to Alpaca Paper Trading for real trading experience</p>
                <p><strong>Features:</strong> Live orders, portfolio tracking</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">🤖</div>
                <h3>AI Assistant</h3>
                <p>Advanced technical analysis and trading insights</p>
                <p><strong>Features:</strong> Price targets, risk assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">⚡</div>
                <h3>Real-time Data</h3>
                <p>Live market data and portfolio updates</p>
                <p><strong>Features:</strong> Yahoo Finance, Alpaca API</p>
            </div>
            """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

# --- FOOTER ---
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit • Alpaca Markets API • Real Paper Trading</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>Live Market Data • AI Analysis • Risk Management</p>
</div>
""", unsafe_allow_html=True)