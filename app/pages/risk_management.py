import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ALPACA IMPORTS
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🛡️ Advanced Risk Manager Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CSS STYLING ---
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
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
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
    .section-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .section-card:hover {
        border-color: #667eea;
    }
    .position-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
        transition: all 0.3s ease;
    }
    .position-card:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateX(5px);
    }
    .position-card.negative {
        border-left-color: #ff4444;
    }
    .badge {
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-success { background: linear-gradient(135deg, #00b894, #00a085); color: white; }
    .badge-danger { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .badge-warning { background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }
    .badge-info { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
    .badge-secondary { background: rgba(255, 255, 255, 0.1); color: #8898aa; }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border: none;
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
    .config-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .warning-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- ALPACA CLIENT CLASS - FIXED ATTRIBUTES ---
class AlpacaRiskClient:
    def __init__(self):
        self.api = None
        self.connected = False
        self.account = None
        self.positions = []
    
    def connect(self, api_key, api_secret, paper=True):
        try:
            if not ALPACA_AVAILABLE:
                st.error("❌ Alpaca SDK not available. Please install: pip install alpaca-trade-api")
                return False
                
            api_key = api_key.strip()
            api_secret = api_secret.strip()
            
            if paper and not api_key.startswith('PK'):
                st.error("❌ Paper Trading keys must start with 'PK'")
                return False
                
            if not paper and not api_key.startswith('AK'):
                st.error("❌ Live Trading keys must start with 'AK'")
                return False
                
            with st.spinner("🔗 Connecting to Alpaca..."):
                self.api = TradingClient(api_key, api_secret, paper=paper)
                self.account = self.api.get_account()
                self.positions = self.api.get_all_positions()
                self.connected = True
                
            st.success("✅ Successfully connected to Alpaca!")
            return True
            
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
            return False
    
    def get_account_info(self):
        if self.connected and self.account:
            try:
                # Safe attribute access với fallback values
                account_data = {
                    'portfolio_value': float(getattr(self.account, 'portfolio_value', 0)),
                    'buying_power': float(getattr(self.account, 'buying_power', 0)),
                    'cash': float(getattr(self.account, 'cash', 0)),
                    'equity': float(getattr(self.account, 'equity', 0)),
                }
                
                # Thêm các attributes optional với safe access
                optional_attrs = [
                    'day_trading_buying_power', 'regt_buying_power', 
                    'initial_margin', 'maintenance_margin', 'last_equity',
                    'long_market_value', 'short_market_value'
                ]
                
                for attr in optional_attrs:
                    if hasattr(self.account, attr):
                        try:
                            account_data[attr] = float(getattr(self.account, attr))
                        except (ValueError, TypeError):
                            account_data[attr] = 0.0
                    else:
                        account_data[attr] = 0.0
                
                return account_data
                
            except Exception as e:
                st.error(f"Error getting account info: {e}")
                # Return basic info even if there's an error
                return {
                    'portfolio_value': float(getattr(self.account, 'portfolio_value', 0)),
                    'buying_power': float(getattr(self.account, 'buying_power', 0)),
                    'cash': float(getattr(self.account, 'cash', 0)),
                    'equity': float(getattr(self.account, 'equity', 0)),
                }
        return None
    
    def get_positions(self):
        if self.connected:
            try:
                positions = self.api.get_all_positions()
                formatted_positions = []
                
                for pos in positions:
                    try:
                        position_data = {
                            'symbol': getattr(pos, 'symbol', 'Unknown'),
                            'qty': float(getattr(pos, 'qty', 0)),
                            'market_value': float(getattr(pos, 'market_value', 0)),
                            'avg_entry_price': float(getattr(pos, 'avg_entry_price', 0)),
                            'current_price': float(getattr(pos, 'current_price', 0)),
                            'unrealized_pl': float(getattr(pos, 'unrealized_pl', 0)),
                            'unrealized_plpc': float(getattr(pos, 'unrealized_plpc', 0)),
                            'side': 'LONG' if float(getattr(pos, 'qty', 0)) > 0 else 'SHORT'
                        }
                        formatted_positions.append(position_data)
                    except Exception as e:
                        st.warning(f"Could not process position: {e}")
                        continue
                        
                return formatted_positions
                
            except Exception as e:
                st.error(f"Error getting positions: {e}")
                return []
        return []

# --- ADVANCED RISK MANAGER CLASS ---
class AdvancedRiskManager:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def calculate_position_size(self, price, stop_loss_price, risk_percent=2, max_position_percent=10):
        """Calculate optimal position size with multiple constraints"""
        if price <= stop_loss_price:
            return 0, 0, "Stop loss must be below entry price"
            
        # Risk-based position sizing
        risk_per_share = price - stop_loss_price
        risk_amount = self.current_capital * (risk_percent / 100)
        risk_based_shares = int(risk_amount / risk_per_share)
        
        # Capital-based position sizing
        max_position_value = self.current_capital * (max_position_percent / 100)
        capital_based_shares = int(max_position_value / price)
        
        # Use the more conservative approach
        position_size = min(risk_based_shares, capital_based_shares)
        actual_risk = position_size * risk_per_share
        
        if position_size == 0:
            return 0, 0, "Position size too small - adjust parameters"
            
        return position_size, actual_risk, "Optimal position calculated"
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss == 0:
            return 0
        win_ratio = avg_win / abs(avg_loss)
        kelly_f = (win_rate * win_ratio - (1 - win_rate)) / win_ratio
        return max(0, min(kelly_f, 0.25))  # Cap at 25% for safety
    
    def simulate_portfolio_monte_carlo(self, expected_return, volatility, days=252, simulations=1000):
        """Enhanced Monte Carlo simulation with multiple distributions"""
        # Student's t-distribution for fat tails
        t_df = 5
        t_returns = stats.t.rvs(t_df, expected_return/days, volatility/np.sqrt(days), (days, simulations))
        t_paths = np.cumprod(1 + t_returns, axis=0) * self.initial_capital
        
        # Normal distribution for comparison
        normal_returns = np.random.normal(expected_return/days, volatility/np.sqrt(days), (days, simulations))
        normal_paths = np.cumprod(1 + normal_returns, axis=0) * self.initial_capital
        
        return t_paths, normal_paths
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns, confidence_level=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return (np.mean(returns) - self.risk_free_rate/252) / np.std(returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)

# --- HELPER FUNCTIONS ---
def display_position(position):
    """Display position card with enhanced styling"""
    symbol = position['symbol']
    qty = position['qty']
    market_value = position['market_value']
    unrealized_pl = position['unrealized_pl']
    pl_percent = position['unrealized_plpc'] * 100
    side = position.get('side', 'LONG')
    
    pl_class = "" if unrealized_pl >= 0 else "negative"
    pl_color = "#00ff88" if unrealized_pl >= 0 else "#ff4444"
    side_badge = "badge-success" if side == 'LONG' else "badge-warning"
    
    st.markdown(f"""
    <div class="position-card {pl_class}">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="font-size: 1.1rem;">{symbol}</strong>
                    <span class="badge {side_badge}" style="margin-left: 0.5rem;">{side}</span>
                    <span class="badge badge-secondary">{qty:.0f} shares</span>
                </div>
                <div style="color: #8898aa; font-size: 0.9rem;">
                    Value: ${market_value:,.2f} • Avg: ${position['avg_entry_price']:.2f}
                </div>
            </div>
            <div style="text-align: right; min-width: 120px;">
                <div style="color: {pl_color}; font-weight: bold; font-size: 1rem;">
                    ${unrealized_pl:+.2f}
                </div>
                <div style="color: {pl_color}; font-size: 0.9rem;">
                    {pl_percent:+.1f}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_performance_chart(portfolio_values, benchmark_values=None):
    """Create performance comparison chart"""
    fig = go.Figure()
    
    # Portfolio line
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_values))),
        y=portfolio_values,
        name='Portfolio',
        line=dict(color='#00ff88', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    # Benchmark line (if provided)
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        fig.add_trace(go.Scatter(
            x=list(range(len(benchmark_values))),
            y=benchmark_values,
            name='Benchmark',
            line=dict(color='#667eea', width=2, dash='dash'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Portfolio Performance",
        template="plotly_dark",
        height=400,
        xaxis_title="Time Period",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    return fig

# --- MAIN STREAMLIT INTERFACE ---
def main():
    # Initialize session state
    if 'alpaca_risk_client' not in st.session_state:
        st.session_state.alpaca_risk_client = AlpacaRiskClient()
    
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = AdvancedRiskManager()
    
    # Header Section
    st.markdown('<div class="main-header">🛡️ Advanced Risk Manager Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Professional Portfolio Risk Analysis & Management</div>', unsafe_allow_html=True)
    
    # Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <div class="metric-value">Real-Time</div>
            <div class="metric-label">Portfolio Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
            <div class="metric-value">Advanced</div>
            <div class="metric-label">Risk Metrics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
            <div class="metric-value">Smart</div>
            <div class="metric-label">Position Sizing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛡️</div>
            <div class="metric-value">Pro</div>
            <div class="metric-label">Protection</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Connection Section
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('### 🔌 Connect to Alpaca')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        account_type = st.radio("Account Type:", ["Paper Trading", "Live Trading"], 
                               horizontal=True, key="account_type")
        api_key = st.text_input("API Key", type="password", 
                               placeholder="Enter your API key (starts with PK for paper)",
                               key="api_key")
        api_secret = st.text_input("API Secret", type="password", 
                                  placeholder="Enter your API secret",
                                  key="api_secret")
        
        col_connect, col_disconnect, col_refresh = st.columns(3)
        
        with col_connect:
            if st.button("🚀 Connect", use_container_width=True, type="primary"):
                if api_key and api_secret:
                    if st.session_state.alpaca_risk_client.connect(
                        api_key, api_secret, paper=(account_type == "Paper Trading")):
                        st.rerun()
                else:
                    st.warning("Please enter both API credentials")
        
        with col_disconnect:
            if st.button("🔌 Disconnect", use_container_width=True):
                st.session_state.alpaca_risk_client = AlpacaRiskClient()
                st.rerun()
        
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h4>📋 Quick Guide</h4>
            <p><strong>Paper Trading:</strong> Use PK... keys</p>
            <p><strong>Live Trading:</strong> Use AK... keys</p>
            <p><strong>Get Keys:</strong> alpaca.markets</p>
            <p><strong>Features:</strong> Real-time risk analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Content - Only show when connected
    if st.session_state.alpaca_risk_client.connected:
        try:
            trader = st.session_state.alpaca_risk_client
            risk_manager = st.session_state.risk_manager
            
            # Get account data
            account_info = trader.get_account_info()
            positions = trader.get_positions()
            
            if account_info:
                # Portfolio Overview
                st.markdown("### 📊 Portfolio Overview")
                
                total_unrealized_pl = sum(pos['unrealized_pl'] for pos in positions) if positions else 0
                total_market_value = sum(pos['market_value'] for pos in positions) if positions else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    pl_color = "#00ff88" if total_unrealized_pl >= 0 else "#ff4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {pl_color}">${total_unrealized_pl:+,.0f}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${account_info['portfolio_value']:,.0f}</div>
                        <div class="metric-label">Portfolio Value</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${account_info['buying_power']:,.0f}</div>
                        <div class="metric-label">Buying Power</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(positions)}</div>
                        <div class="metric-label">Active Positions</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional account metrics
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${account_info['cash']:,.0f}</div>
                        <div class="metric-label">Available Cash</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${account_info['equity']:,.0f}</div>
                        <div class="metric-label">Account Equity</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col7:
                    day_trading_bp = account_info.get('day_trading_buying_power', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${day_trading_bp:,.0f}</div>
                        <div class="metric-label">Day Trading BP</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col8:
                    maintenance_margin = account_info.get('maintenance_margin', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${maintenance_margin:,.0f}</div>
                        <div class="metric-label">Maintenance Margin</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                
                # Positions and Risk Analysis
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Positions", "🎯 Risk Tools", "📊 Analytics", "⚡ Stress Test"])
                
                with tab1:
                    st.markdown("#### 📈 Your Positions")
                    
                    if positions:
                        # Sort positions by P&L
                        profitable = [p for p in positions if p['unrealized_pl'] > 0]
                        losing = [p for p in positions if p['unrealized_pl'] <= 0]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if profitable:
                                st.markdown("**🟢 Profitable Positions**")
                                for position in sorted(profitable, key=lambda x: x['unrealized_pl'], reverse=True):
                                    display_position(position)
                            else:
                                st.info("No profitable positions")
                        
                        with col2:
                            if losing:
                                st.markdown("**🔴 Losing Positions**")
                                for position in sorted(losing, key=lambda x: x['unrealized_pl']):
                                    display_position(position)
                            else:
                                st.info("No losing positions")
                        
                        # Position Summary
                        st.markdown("#### 📋 Position Summary")
                        summary_cols = st.columns(4)
                        
                        with summary_cols[0]:
                            st.metric("Total Value", f"${total_market_value:,.0f}")
                        with summary_cols[1]:
                            st.metric("Total P&L", f"${total_unrealized_pl:+,.0f}")
                        with summary_cols[2]:
                            avg_pl_percent = (total_unrealized_pl / total_market_value * 100) if total_market_value > 0 else 0
                            st.metric("Avg Return", f"{avg_pl_percent:+.1f}%")
                        with summary_cols[3]:
                            concentration = (total_market_value / account_info['portfolio_value'] * 100) if account_info['portfolio_value'] > 0 else 0
                            st.metric("Concentration", f"{concentration:.1f}%")
                    
                    else:
                        st.info("💰 No active positions found")
                
                with tab2:
                    st.markdown("#### 🎯 Advanced Position Sizing")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Risk-Based Sizing")
                        symbol = st.text_input("Symbol", "AAPL", key="sizing_symbol")
                        entry_price = st.number_input("Entry Price ($)", 1.0, 10000.0, 150.0, key="entry_price")
                        stop_loss = st.number_input("Stop Loss ($)", 1.0, 10000.0, 140.0, key="stop_loss")
                        risk_percent = st.slider("Risk %", 0.1, 10.0, 2.0, 0.1, key="risk_percent")
                        max_position_percent = st.slider("Max Position %", 1.0, 50.0, 10.0, 0.5, key="max_position_percent")
                        
                        if st.button("Calculate Position", key="calc_position"):
                            position_size, actual_risk, message = risk_manager.calculate_position_size(
                                entry_price, stop_loss, risk_percent, max_position_percent)
                            
                            st.markdown("##### 📊 Results")
                            st.metric("Position Size", f"{position_size:,} shares")
                            st.metric("Position Value", f"${position_size * entry_price:,.0f}")
                            st.metric("Risk Amount", f"${actual_risk:,.0f}")
                            
                            if "Optimal" in message:
                                st.success(message)
                            else:
                                st.warning(message)
                    
                    with col2:
                        st.markdown("##### Kelly Criterion")
                        win_rate = st.slider("Win Rate (%)", 1, 99, 60, key="win_rate") / 100
                        avg_win = st.number_input("Average Win (%)", 0.1, 100.0, 15.0, key="avg_win") / 100
                        avg_loss = st.number_input("Average Loss (%)", 0.1, 100.0, 10.0, key="avg_loss") / 100
                        
                        kelly_f = risk_manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                        
                        st.markdown("##### 🎲 Kelly Results")
                        st.metric("Kelly Fraction", f"{kelly_f:.1%}")
                        st.metric("Suggested Position", f"${risk_manager.current_capital * kelly_f:,.0f}")
                        
                        if kelly_f > 0.2:
                            st.warning("High risk - consider using half Kelly")
                        elif kelly_f > 0.1:
                            st.info("Moderate risk - standard position")
                        else:
                            st.success("Low risk - conservative position")
                
                with tab3:
                    st.markdown("#### 📊 Portfolio Analytics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Monte Carlo Simulation")
                        expected_return = st.slider("Expected Return (%)", -20, 50, 12, key="mc_return") / 100
                        volatility = st.slider("Volatility (%)", 5, 80, 20, key="mc_vol") / 100
                        
                        if st.button("Run Simulation", key="run_simulation"):
                            with st.spinner("Running Monte Carlo simulation..."):
                                t_paths, normal_paths = risk_manager.simulate_portfolio_monte_carlo(
                                    expected_return, volatility, 252, 1000)
                                
                                # Create comparison chart
                                fig = go.Figure()
                                
                                # Add some sample paths from t-distribution
                                for i in range(min(20, 1000)):
                                    fig.add_trace(go.Scatter(
                                        y=t_paths[:, i],
                                        mode='lines',
                                        line=dict(width=1, color='rgba(102, 126, 234, 0.2)'),
                                        showlegend=False
                                    ))
                                
                                # Add percentiles
                                percentiles = [5, 25, 50, 75, 95]
                                for p in percentiles:
                                    fig.add_trace(go.Scatter(
                                        y=np.percentile(t_paths, p, axis=1),
                                        mode='lines',
                                        line=dict(width=2),
                                        name=f'{p}th Percentile'
                                    ))
                                
                                fig.update_layout(
                                    title="Monte Carlo Simulation - Portfolio Paths",
                                    template="plotly_dark",
                                    height=400,
                                    xaxis_title="Trading Days",
                                    yaxis_title="Portfolio Value ($)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### Risk Metrics")
                        
                        # Simulate some returns for demonstration
                        returns = np.random.normal(0.001, 0.02, 1000)  # Simulated daily returns
                        
                        col_metrics1, col_metrics2 = st.columns(2)
                        
                        with col_metrics1:
                            st.metric("VaR (95%)", f"${risk_manager.calculate_var(returns) * 100000:,.0f}")
                            st.metric("CVaR (95%)", f"${risk_manager.calculate_cvar(returns) * 100000:,.0f}")
                            st.metric("Sharpe Ratio", f"{risk_manager.calculate_sharpe_ratio(returns):.2f}")
                        
                        with col_metrics2:
                            st.metric("Max Drawdown", f"{risk_manager.calculate_max_drawdown(np.cumprod(1 + returns) * 100000) * 100:.1f}%")
                            st.metric("Volatility", f"{np.std(returns) * np.sqrt(252) * 100:.1f}%")
                            st.metric("Expected Return", f"{np.mean(returns) * 252 * 100:.1f}%")
                
                with tab4:
                    st.markdown("#### ⚡ Portfolio Stress Test")
                    
                    if positions:
                        scenarios = {
                            "Market Crash (-30%)": 0.7,
                            "Severe Correction (-20%)": 0.8,
                            "Moderate Correction (-10%)": 0.9,
                            "Normal Market (0%)": 1.0,
                            "Bull Market (+15%)": 1.15,
                            "Strong Rally (+25%)": 1.25
                        }
                        
                        selected_scenario = st.selectbox("Select Stress Scenario", list(scenarios.keys()))
                        multiplier = scenarios[selected_scenario]
                        
                        current_value = total_market_value
                        stressed_value = current_value * multiplier
                        impact = stressed_value - current_value
                        impact_percent = (impact / current_value) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Value", f"${current_value:,.0f}")
                        with col2:
                            st.metric("Stressed Value", f"${stressed_value:,.0f}")
                        with col3:
                            st.metric("Impact", f"${impact:+,.0f}", f"{impact_percent:+.1f}%")
                        
                        # Risk assessment
                        if impact_percent <= -20:
                            st.error("🚨 Extreme Risk - Consider reducing exposure")
                        elif impact_percent <= -10:
                            st.warning("⚠️ High Risk - Review portfolio allocation")
                        elif impact_percent <= -5:
                            st.info("🔶 Moderate Risk - Monitor positions")
                        else:
                            st.success("✅ Low Risk - Portfolio appears resilient")
                    
                    else:
                        st.info("No positions available for stress testing")
            
            else:
                st.error("Unable to fetch account information")
                
        except Exception as e:
            st.error(f"Error in main content: {str(e)}")
    
    else:
        # Welcome Screen
        st.markdown("---")
        st.markdown("### 🚀 Get Started with Advanced Risk Management")
        
        col1, col2, col3 = st.columns(3)
        
        features = [
            ("📊", "Real-Time Analysis", "Live portfolio monitoring with advanced risk metrics and real-time P&L tracking"),
            ("🎯", "Smart Position Sizing", "Optimal trade sizing using Kelly Criterion and risk-based calculations"),
            ("⚡", "Advanced Simulations", "Monte Carlo simulations and stress testing for comprehensive risk assessment")
        ]
        
        for i, (icon, title, desc) in enumerate(features):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        if not ALPACA_AVAILABLE:
            st.markdown("---")
            st.warning("""
            **Alpaca SDK Not Available**
            
            To use all features, install the Alpaca SDK:
            ```bash
            pip install alpaca-trade-api
            ```
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #8898aa;'>
        <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Advanced Risk Management Platform</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Risk Manager Pro v3.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()