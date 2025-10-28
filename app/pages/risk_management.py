import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ALPACA IMPORTS
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.common.exceptions import APIError

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🛡️ Advanced Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #8898aa;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .dashboard-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
        height: 100%;
    }
    .dashboard-card:hover {
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
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #667eea;
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
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-success { background: linear-gradient(135deg, #00ff88, #00cc6a); color: white; }
    .badge-warning { background: linear-gradient(135deg, #ffa726, #f57c00); color: white; }
    .badge-danger { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
    .badge-info { background: linear-gradient(135deg, #29b6f6, #0288d1); color: white; }
    .risk-indicator {
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low { background: linear-gradient(135deg, #00ff88, #00cc6a); }
    .risk-medium { background: linear-gradient(135deg, #ffa726, #f57c00); }
    .risk-high { background: linear-gradient(135deg, #ff4444, #cc0000); }
    .position-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }
    .position-item.negative {
        border-left-color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# --- ALPACA CLIENT CLASS FOR RISK MANAGEMENT ---
class AlpacaRiskClient:
    def __init__(self):
        self.api = None
        self.connected = False
        self.account = None
    
    def connect(self, api_key, api_secret, paper=True):
        """Connect to Alpaca API with detailed debug"""
        try:
            # Remove extra whitespace
            api_key = api_key.strip()
            api_secret = api_secret.strip()
            
            # Check API key format
            if paper and not api_key.startswith('PK'):
                st.error("""
                ❌ **INVALID API KEY FORMAT**
                
                Paper Trading keys must start with **'PK'**
                
                📍 **How to get correct keys:**
                1. Login to [Alpaca Dashboard](https://app.alpaca.markets/)
                2. Go to **Paper Trading** account
                3. Click **"Generate API Key"**
                4. Copy keys starting with **PK...**
                """)
                return False
                
            if not paper and not api_key.startswith('AK'):
                st.error("""
                ❌ **INVALID API KEY FORMAT**
                
                Live Trading keys must start with **'AK'**
                """)
                return False
                
            # Check secret length
            if len(api_secret) < 30:
                st.error("❌ API Secret seems too short. Please check your copy/paste.")
                return False
                
            # Try connection
            with st.spinner("🔐 Connecting to Alpaca..."):
                self.api = TradingClient(api_key, api_secret, paper=paper)
                self.account = self.api.get_account()
                self.connected = True
                
            st.success("✅ Successfully connected to Alpaca!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Connection failed: {error_msg}")
            
            if "unauthorized" in error_msg.lower():
                st.markdown("""
                ### 🚨 **UNAUTHORIZED ERROR - FIX GUIDE**
                
                **1. Check API Key Format:**
                - Paper Trading: Must start with **`PK`**
                - Live Trading: Must start with **`AK`**
                
                **2. Verify Your Keys:**
                - Login to [Alpaca Dashboard](https://app.alpaca.markets/)
                - Navigate to **API Keys** section
                - Generate NEW keys if needed
                """)
            
            return False
    
    def get_account_info(self):
        """Get account information with proper error handling"""
        if self.connected and self.account:
            return {
                'portfolio_value': float(self.account.portfolio_value),
                'buying_power': float(self.account.buying_power),
                'cash': float(self.account.cash),
                'equity': float(self.account.equity),
                'currency': self.account.currency,
                'status': self.account.status,
                'account_number': self.account.account_number
            }
        return None
    
    def get_positions(self):
        """Get positions list with unrealized P&L"""
        if self.connected:
            try:
                positions = self.api.get_all_positions()
                positions_data = []
                for position in positions:
                    positions_data.append({
                        'symbol': position.symbol,
                        'qty': float(position.qty),
                        'market_value': float(position.market_value),
                        'avg_entry_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc),
                        'side': position.side
                    })
                return positions_data
            except Exception as e:
                st.error(f"❌ Error getting positions: {e}")
                return []
        return []
    
    def get_portfolio_history(self, period="1M"):
        """Get portfolio history"""
        if self.connected:
            try:
                params = GetPortfolioHistoryRequest(period=period)
                return self.api.get_portfolio_history(params)
            except Exception as e:
                st.error(f"❌ Error getting portfolio history: {e}")
                return None
        return None

    def calculate_total_unrealized_pl(self):
        """Calculate total unrealized P&L from positions"""
        positions = self.get_positions()
        if positions:
            return sum(pos['unrealized_pl'] for pos in positions)
        return 0.0

# --- ADVANCED RISK MANAGEMENT CLASS ---
class AdvancedRiskManager:
    """Advanced risk management system with multiple methodologies"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    def calculate_position_size(self, price, stop_loss_price, risk_percent=2):
        """Calculate position size based on % risk"""
        if price <= stop_loss_price:
            return 0, 0

        risk_per_share = price - stop_loss_price
        risk_amount = self.current_capital * (risk_percent / 100)
        
        position_size = int(risk_amount / risk_per_share)
        return position_size, risk_amount
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss == 0 or avg_win <= 0:
            return 0
            
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_percent = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Conservative Kelly (half-Kelly)
        return max(0, min(kelly_percent * 0.5, 0.25))
    
    def simulate_portfolio_monte_carlo(self, expected_return, volatility, days=252, simulations=1000):
        """Run Monte Carlo simulation for portfolio value with fat tails"""
        # Student's t-distribution for fat tails (more realistic)
        t_df = 5  # Degrees of freedom for fat tails
        daily_returns = stats.t.rvs(t_df, expected_return/days, volatility/np.sqrt(days), (days, simulations))
        price_paths = np.cumprod(1 + daily_returns, axis=0) * self.initial_capital
        return price_paths
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def calculate_cvar(self, returns, confidence_level=0.95):
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= -var].mean()
        return abs(cvar) if not np.isnan(cvar) else 0
    
    def calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        if len(values) == 0:
            return 0
            
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd * 100
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def get_volatility_estimate(self, symbol):
        """Estimate volatility for symbol"""
        volatility_map = {
            'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.28, 'TSLA': 0.45, 
            'AMZN': 0.30, 'NVDA': 0.50, 'META': 0.35, 'NFLX': 0.40,
            'SPY': 0.18, 'QQQ': 0.24, 'IWM': 0.28, 'DIA': 0.20,
            'BTCUSD': 0.65, 'ETHUSD': 0.70, 'ADAUSD': 0.80, 'DOTUSD': 0.75
        }
        return volatility_map.get(symbol, 0.30)

    def get_beta_estimate(self, symbol):
        """Estimate beta for symbol"""
        beta_map = {
            'AAPL': 1.2, 'MSFT': 0.9, 'GOOGL': 1.1, 'TSLA': 2.0,
            'AMZN': 1.3, 'NVDA': 1.8, 'META': 1.4, 'NFLX': 1.6,
            'SPY': 1.0, 'QQQ': 1.05, 'IWM': 1.1, 'DIA': 0.95,
            'BTCUSD': 1.5, 'ETHUSD': 1.6, 'ADAUSD': 1.8, 'DOTUSD': 1.7
        }
        return beta_map.get(symbol, 1.0)

    def get_live_portfolio_returns(self, alpaca_client, days=252):
        """Get actual returns data from Alpaca portfolio history"""
        try:
            portfolio_history = alpaca_client.get_portfolio_history(period="1M")
            if portfolio_history and hasattr(portfolio_history, 'equity'):
                equity = portfolio_history.equity
                if equity and len(equity) > 1:
                    # Calculate returns from equity curve
                    returns = np.diff(equity) / equity[:-1]
                    return returns
            return None
        except Exception as e:
            st.error(f"Error getting portfolio returns: {e}")
            return None

    def calculate_live_portfolio_metrics(self, alpaca_client):
        """Calculate risk metrics from actual data"""
        try:
            returns = self.get_live_portfolio_returns(alpaca_client)
            if returns is not None and len(returns) > 0:
                metrics = {
                    'volatility': np.std(returns) * np.sqrt(252) * 100,
                    'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                    'max_drawdown': self.calculate_max_drawdown(returns),
                    'var_95': self.calculate_var(returns) * 100,
                    'cvar_95': self.calculate_cvar(returns) * 100,
                    'total_return': (returns[-1] - returns[0]) / returns[0] * 100 if len(returns) > 1 else 0
                }
                return metrics
            return None
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {e}")
            return None

# --- HELPER FUNCTIONS ---
def display_position(position):
    """Display individual position with proper formatting"""
    symbol = position['symbol']
    qty = position['qty']
    market_value = position['market_value']
    unrealized_pl = position['unrealized_pl']
    pl_percent = position['unrealized_plpc'] * 100
    
    pl_class = "" if unrealized_pl >= 0 else "negative"
    pl_color = "#00ff88" if unrealized_pl >= 0 else "#ff4444"
    
    st.markdown(f"""
    <div class="position-item {pl_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>{symbol}</strong>
                <span class="badge {'badge-success' if unrealized_pl >= 0 else 'badge-danger'}">{qty:.0f} shares</span>
            </div>
            <div style="text-align: right;">
                <div style="color: {pl_color}; font-weight: bold;">
                    ${unrealized_pl:+.2f} ({pl_percent:+.1f}%)
                </div>
                <div style="font-size: 0.8em; color: #8898aa;">
                    Value: ${market_value:,.2f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_sample_portfolio_data():
    """Return sample portfolio data"""
    portfolio_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'],
        'Position_Value': [25000, 20000, 15000, 18000, 12000, 8000, 10000, 7000],
        'Volatility': [0.25, 0.22, 0.28, 0.45, 0.30, 0.50, 0.35, 0.40],
        'Beta': [1.2, 0.9, 1.1, 2.0, 1.3, 1.8, 1.4, 1.6],
        'Quantity': [100, 80, 50, 60, 40, 30, 45, 35],
        'Avg_Price': [150.0, 250.0, 150.0, 300.0, 120.0, 266.67, 222.22, 200.0],
        'Current_Price': [155.0, 255.0, 155.0, 310.0, 125.0, 275.0, 230.0, 210.0],
        'Unrealized_PL': [500.0, 400.0, 250.0, 600.0, 200.0, 250.0, 350.0, 350.0],
        'PL_Percent': [2.0, 2.0, 1.67, 3.33, 1.67, 3.13, 3.5, 5.0]
    }
    return pd.DataFrame(portfolio_data)

# --- MAIN STREAMLIT INTERFACE ---
def main():
    # --- HEADER ---
    st.markdown('<div class="main-header">🛡️ Advanced Risk Management System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Professional Risk Analysis with Alpaca Integration</div>', unsafe_allow_html=True)

    # --- CONNECTION PANEL ON MAIN SCREEN ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="connection-panel">
            <h3>🔌 API Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for Alpaca client
        if 'alpaca_risk_client' not in st.session_state:
            st.session_state.alpaca_risk_client = AlpacaRiskClient()
        
        account_type = st.radio("Account Type:", ["Paper Trading", "Live Trading"], horizontal=True, key="risk_account_type")
        api_key = st.text_input("API Key", type="password", key="risk_api_key", placeholder="PK... for Paper Trading")
        api_secret = st.text_input("API Secret", type="password", key="risk_api_secret", placeholder="64-character secret key")
        
        col_connect, col_disconnect = st.columns(2)
        
        with col_connect:
            if st.button("🚀 Connect", use_container_width=True, type="primary"):
                if api_key and api_secret:
                    with st.spinner("Connecting to Alpaca..."):
                        if st.session_state.alpaca_risk_client.connect(
                            api_key.strip(), 
                            api_secret.strip(), 
                            paper=(account_type == "Paper Trading")
                        ):
                            st.rerun()
                else:
                    st.warning("Please enter API Key and Secret")
        
        with col_disconnect:
            if st.button("🔌 Disconnect", use_container_width=True):
                st.session_state.alpaca_risk_client = AlpacaRiskClient()
                st.rerun()

    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h4>📋 Quick Guide</h4>
            <p><strong>1. Get API Keys:</strong><br>app.alpaca.markets</p>
            <p><strong>2. Paper Trading:</strong><br>Use PK... keys</p>
            <p><strong>3. Risk Analysis:</strong><br>Real-time monitoring</p>
        </div>
        """, unsafe_allow_html=True)

    # Display connection status
    if st.session_state.alpaca_risk_client.connected:
        try:
            account_info = st.session_state.alpaca_risk_client.get_account_info()
            if account_info:
                st.success(f"✅ Connected to {account_type}")
                
                # Calculate total unrealized P&L from positions
                total_unrealized_pl = st.session_state.alpaca_risk_client.calculate_total_unrealized_pl()
                
                # Portfolio Overview Metrics
                st.markdown("---")
                st.subheader("📊 Portfolio Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="feature-icon">💰</div>
                        <div class="metric-value">${account_info['portfolio_value']:,.0f}</div>
                        <div class="metric-label">Portfolio Value</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="feature-icon">⚡</div>
                        <div class="metric-value">${account_info['buying_power']:,.0f}</div>
                        <div class="metric-label">Buying Power</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="feature-icon">💵</div>
                        <div class="metric-value">${account_info['cash']:,.0f}</div>
                        <div class="metric-label">Available Cash</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    pl_color = "#00ff88" if total_unrealized_pl >= 0 else "#ff4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="feature-icon">📈</div>
                        <div class="metric-value" style="color: {pl_color}">${total_unrealized_pl:,.0f}</div>
                        <div class="metric-label">Unrealized P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error getting account info: {e}")

    # --- MAIN DASHBOARD CONTENT ---
    if st.session_state.alpaca_risk_client.connected:
        trader = st.session_state.alpaca_risk_client
        risk_manager = AdvancedRiskManager(100000)  # Default capital
        
        # Refresh Button
        if st.button("🔄 Refresh Data"):
            st.rerun()

        # --- PORTFOLIO POSITIONS ---
        st.markdown("---")
        st.subheader("📈 Portfolio Positions")
        
        positions = trader.get_positions()
        if positions:
            # Display positions in a nice layout
            col1, col2 = st.columns(2)
            
            profitable_positions = [p for p in positions if p['unrealized_pl'] > 0]
            losing_positions = [p for p in positions if p['unrealized_pl'] <= 0]
            
            with col1:
                if profitable_positions:
                    st.markdown("#### 🟢 Profitable Positions")
                    for position in profitable_positions:
                        display_position(position)
                else:
                    st.info("No profitable positions")
            
            with col2:
                if losing_positions:
                    st.markdown("#### 🔴 Losing Positions")
                    for position in losing_positions:
                        display_position(position)
                else:
                    st.info("No losing positions")
            
            # Portfolio statistics
            total_positions_value = sum(pos['market_value'] for pos in positions)
            total_unrealized_pl = sum(pos['unrealized_pl'] for pos in positions)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Total Positions", f"{len(positions)}")
            col_stat2.metric("Total Value", f"${total_positions_value:,.0f}")
            col_stat3.metric("Total P&L", f"${total_unrealized_pl:,.2f}")
            
        else:
            st.info("📭 No active positions found in your portfolio")

        # --- ADVANCED FEATURES IN TABS ---
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Position Sizing", "🎲 Monte Carlo", "⚡ Stress Testing", "📊 Risk Analysis"])

        with tab1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('#### 🎯 Advanced Position Sizing')
            
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.text_input("Symbol", "AAPL", key="pos_symbol")
                entry_price = st.number_input("Entry Price ($)", 0.01, 10000.0, 150.0, 0.01, key="entry_price")
                stop_loss = st.number_input("Stop Loss ($)", 0.01, 10000.0, 140.0, 0.01, key="stop_loss")
                risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1, key="risk_pct")
            
            with col2:
                position_size, risk_amount = risk_manager.calculate_position_size(
                    entry_price, stop_loss, risk_percent)
                
                st.metric("Position Size", f"{position_size:,} shares")
                st.metric("Position Value", f"${position_size * entry_price:,.0f}")
                st.metric("Risk Amount", f"${risk_amount:,.0f}")
                
                risk_reward_ratio = (entry_price - stop_loss) / stop_loss
                st.metric("Risk/Reward", f"{risk_reward_ratio:.2f}:1")
                
                # Risk assessment
                risk_percentage = (risk_amount / 100000) * 100
                if risk_percentage > 5:
                    st.error("⚠️ High Risk: Consider reducing position size")
                elif risk_percentage < 1:
                    st.success("✅ Conservative Risk: Safe position size")
                else:
                    st.warning("📊 Moderate Risk: Appropriate position size")
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('#### 🎲 Monte Carlo Simulation')
            
            col1, col2 = st.columns(2)
            
            with col1:
                expected_return = st.slider("Expected Return (%)", -20, 50, 12, key="mc_return")
                volatility = st.slider("Volatility (%)", 5, 80, 20, key="mc_vol")
                simulations = st.select_slider("Simulations", [100, 500, 1000], 1000, key="mc_sims")
                
                if st.button("Run Simulation", key="run_mc"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        paths = risk_manager.simulate_portfolio_monte_carlo(
                            expected_return/100, volatility/100, 252, simulations)
                        
                        # Plot results
                        fig = go.Figure()
                        for i in range(min(100, simulations)):
                            fig.add_trace(go.Scatter(
                                y=paths[:, i], 
                                mode='lines', 
                                line=dict(width=1, color='rgba(100,149,237,0.1)'), 
                                showlegend=False
                            ))
                        
                        # Add percentiles
                        percentiles = [5, 25, 50, 75, 95]
                        for p in percentiles:
                            fig.add_trace(go.Scatter(
                                y=np.percentile(paths, p, axis=1),
                                mode='lines',
                                line=dict(width=2, dash='dash'),
                                name=f'{p}th Percentile'
                            ))
                        
                        fig.update_layout(
                            title=f"Monte Carlo Simulation - {simulations} Scenarios",
                            template="plotly_dark",
                            height=400,
                            xaxis_title='Days',
                            yaxis_title='Portfolio Value ($)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.info("""
                **Monte Carlo Simulation**
                - Models portfolio future values
                - Uses statistical distributions
                - Provides probability analysis
                - Helps understand potential outcomes
                
                **Interpretation:**
                - **50th percentile**: Median expected outcome
                - **5th-95th percentile**: 90% confidence interval
                - **Wider bands**: Higher uncertainty
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('#### ⚡ Portfolio Stress Testing')
            
            if positions:
                # Stress test scenarios
                stress_scenarios = {
                    'Market Crash (-20%)': 0.8,
                    'Correction (-10%)': 0.9, 
                    'Normal (0%)': 1.0,
                    'Rally (+10%)': 1.1,
                    'Bull Market (+20%)': 1.2
                }
                
                scenario = st.selectbox("Select Scenario", list(stress_scenarios.keys()))
                multiplier = stress_scenarios[scenario]
                
                current_value = sum(pos['market_value'] for pos in positions)
                stressed_value = current_value * multiplier
                impact = stressed_value - current_value
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Value", f"${current_value:,.0f}")
                col2.metric("Stressed Value", f"${stressed_value:,.0f}")
                col3.metric("Impact", f"${impact:,.0f}", f"{(impact/current_value)*100:.1f}%")
                
                # Position-level impact
                st.markdown("##### Position Impact")
                impact_data = []
                for pos in positions:
                    pos_impact = pos['market_value'] * (multiplier - 1)
                    impact_data.append({
                        'Symbol': pos['symbol'],
                        'Current Value': pos['market_value'],
                        'Impact': pos_impact,
                        'New Value': pos['market_value'] + pos_impact
                    })
                
                impact_df = pd.DataFrame(impact_data)
                st.dataframe(impact_df.style.format({
                    'Current Value': '${:,.2f}',
                    'Impact': '${:,.2f}',
                    'New Value': '${:,.2f}'
                }))
            else:
                st.info("No positions available for stress testing")
            
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('#### 📊 Detailed Risk Analysis')
            
            # Calculate basic risk metrics from positions
            if positions:
                total_value = sum(pos['market_value'] for pos in positions)
                
                # Concentration risk
                largest_position = max(positions, key=lambda x: x['market_value'])
                concentration = (largest_position['market_value'] / total_value) * 100
                
                # Volatility estimate (weighted average)
                weighted_vol = sum(pos['market_value'] * risk_manager.get_volatility_estimate(pos['symbol']) 
                                 for pos in positions) / total_value
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Number of Positions", len(positions))
                col2.metric("Largest Position", f"{concentration:.1f}%")
                col3.metric("Est. Volatility", f"{weighted_vol:.1%}")
                col4.metric("Total Exposure", f"${total_value:,.0f}")
                
                # Risk assessment
                risk_factors = []
                if concentration > 25:
                    risk_factors.append("⚠️ High concentration in single position")
                if len(positions) < 5:
                    risk_factors.append("⚠️ Low diversification")
                if weighted_vol > 0.3:
                    risk_factors.append("⚠️ High portfolio volatility")
                
                if risk_factors:
                    st.warning("#### Risk Factors Identified")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.success("#### ✅ Portfolio appears well diversified")
            else:
                st.info("No positions available for risk analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # --- WELCOME SCREEN WHEN NOT CONNECTED ---
        st.markdown("---")
        st.subheader("🚀 Get Started with Risk Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">📈</div>
                <h3>Portfolio Analysis</h3>
                <p>Comprehensive risk assessment of your investment portfolio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">🎯</div>
                <h3>Position Sizing</h3>
                <p>Advanced algorithms for optimal position sizing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">⚡</div>
                <h3>Stress Testing</h3>
                <p>Test your portfolio against various market scenarios</p>
            </div>
            """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("""
    <div style='text-align: center; padding: 3rem; color: #8898aa;'>
        <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit • Advanced Risk Management Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()