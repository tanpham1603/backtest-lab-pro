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
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
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
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .api-section {
        background: rgba(102, 126, 234, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #667eea;
        margin: 1rem 0;
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>🛡️</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>Risk Manager</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>Advanced Risk Management</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("✨ Connect to Alpaca to begin risk analysis")

# --- HEADER ---
st.markdown('<div class="welcome-header">🛡️ Advanced Risk Management System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Risk Analysis with Alpaca Integration</div>', unsafe_allow_html=True)

# --- STATUS CARDS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📊</div>
        <div class="metric-value">Live</div>
        <div class="metric-label">Portfolio Data</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">⚡</div>
        <div class="metric-value">Pro</div>
        <div class="metric-label">Risk Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🎯</div>
        <div class="metric-value">5+</div>
        <div class="metric-label">Methodologies</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🛡️</div>
        <div class="metric-value">Active</div>
        <div class="metric-label">Protection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

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
            if not api_key.startswith('PK'):
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
                
            # Check secret length
            if len(api_secret) < 30:
                st.error("❌ API Secret seems too short. Please check your copy/paste.")
                return False
                
            # Try connection
            with st.spinner("🔐 Connecting to Alpaca Paper Trading..."):
                self.api = TradingClient(api_key, api_secret, paper=True)
                self.account = self.api.get_account()
                self.connected = True
                
            st.success("✅ Successfully connected to Paper Trading!")
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
                - Login to [Alpaca Paper Trading](https://app.alpaca.markets/paper/dashboard/overview)
                - Navigate to **API Keys** section
                - Generate NEW keys if needed
                """)
            
            return False
    
    def get_account_info(self):
        """Get account information"""
        if self.connected:
            return self.account
        return None
    
    def get_positions(self):
        """Get positions list"""
        if self.connected:
            try:
                return self.api.get_all_positions()
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
    
    def calculate_optimal_f(self, win_rate, avg_win, avg_loss):
        """Calculate Optimal F (Fixed Fractional)"""
        if avg_loss == 0:
            return 0
            
        win_loss_ratio = avg_win / abs(avg_loss)
        optimal_f = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0, min(optimal_f, 0.25))
    
    def simulate_portfolio_monte_carlo(self, expected_return, volatility, days=252, simulations=1000):
        """Run Monte Carlo simulation for portfolio value with fat tails"""
        # Student's t-distribution for fat tails (more realistic)
        t_df = 5  # Degrees of freedom for fat tails
        daily_returns = stats.t.rvs(t_df, expected_return/days, volatility/np.sqrt(days), (days, simulations))
        price_paths = np.cumprod(1 + daily_returns, axis=0) * self.initial_capital
        return price_paths
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def calculate_cvar(self, returns, confidence_level=0.95):
        """Calculate Conditional VaR (Expected Shortfall)"""
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
    
    def stress_test_portfolio(self, portfolio_weights, asset_volatilities, stress_scenario):
        """Stress test portfolio under different market scenarios"""
        scenario_impact = np.dot(portfolio_weights, stress_scenario * asset_volatilities)
        return scenario_impact

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

def get_stock_category(symbol):
    """Classify stocks for stress testing"""
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'CRM']
    growth_stocks = ['TSLA', 'SNOW', 'DDOG', 'NET', 'CRWD', 'SHOP', 'SQ', 'UBER', 'LYFT', 'ROKU', 'ZM']
    large_cap = ['SPY', 'IVV', 'VOO', 'DIA', 'BRK.B', 'JPM', 'JNJ', 'XOM', 'WMT', 'PG', 'V', 'MA']
    crypto = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOTUSD', 'SOLUSD', 'DOGEUSD']
    
    if symbol in tech_stocks:
        return 'Tech'
    elif symbol in growth_stocks:
        return 'Growth'
    elif symbol in large_cap:
        return 'Large Cap'
    elif symbol in crypto:
        return 'Crypto'
    else:
        return 'Small Cap'

# --- MAIN STREAMLIT INTERFACE ---
def main():
    # --- SIDEBAR: ALPACA CONNECTION ---
    with st.sidebar:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">🔌 ALPACA CONNECTION</div>', unsafe_allow_html=True)
        
        # Initialize session state for Alpaca client
        if 'alpaca_risk_client' not in st.session_state:
            st.session_state.alpaca_risk_client = AlpacaRiskClient()
        
        # API Connection Section
        st.markdown('<div class="api-section">', unsafe_allow_html=True)
        st.subheader("API Configuration")
        
        account_type = st.radio("Account Type:", ["Paper Trading", "Live Trading"], key="risk_account_type")
        api_key = st.text_input("API Key", type="password", key="risk_api_key", placeholder="PK... for Paper Trading")
        api_secret = st.text_input("API Secret", type="password", key="risk_api_secret", placeholder="64-character secret key")
        
        col_connect, col_disconnect = st.columns(2)
        
        with col_connect:
            if st.button("Connect", use_container_width=True, type="primary"):
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
            if st.button("Disconnect", use_container_width=True):
                st.session_state.alpaca_risk_client = AlpacaRiskClient()
                st.rerun()
        
        # Display connection status
        if st.session_state.alpaca_risk_client.connected:
            st.success(f"✅ Connected to {account_type}")
            try:
                account = st.session_state.alpaca_risk_client.get_account_info()
                st.metric("Portfolio Value", f"${float(account.portfolio_value):,.2f}")
                st.metric("Buying Power", f"${float(account.buying_power):,.2f}")
            except Exception as e:
                st.error(f"Error getting account info: {e}")
        else:
            st.info("🔌 Connect to Alpaca to analyze your live portfolio")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="config-header">🎛️ RISK PARAMETERS</div>', unsafe_allow_html=True)
        initial_capital = st.number_input("💰 Initial Capital ($)", 10000, 1000000, 100000, key="risk_capital")
        
        st.markdown("---")
        st.markdown('<div class="config-header">⚙️ RISK PREFERENCES</div>', unsafe_allow_html=True)
        risk_tolerance = st.select_slider("Risk Tolerance", 
                                         ["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
                                         "Moderate", key="risk_tolerance")
        
        max_portfolio_risk = st.slider("Max Portfolio Risk (%)", 1.0, 10.0, 5.0, 0.1, key="max_risk")
        
        st.markdown("---")
        st.markdown('<div class="config-header">📈 MARKET ASSUMPTIONS</div>', unsafe_allow_html=True)
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1, key="risk_free") / 100

        st.markdown('</div>', unsafe_allow_html=True)

    risk_manager = AdvancedRiskManager(initial_capital)

    # --- ADVANCED TABS INTERFACE ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Position Sizing", "🎲 Monte Carlo Simulation", 
                                           "📊 Portfolio Analysis", "⚡ Stress Testing", "📈 Risk Metrics"])

    # --- TAB 1: ADVANCED POSITION SIZING ---
    with tab1:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">🎯 ADVANCED POSITION SIZING</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Trade Setup")
            entry_price = st.number_input("Entry Price ($)", 0.01, 10000.0, 150.0, 0.01, key="entry_price")
            stop_loss_price = st.number_input("Stop Loss Price ($)", 0.01, 10000.0, 140.0, 0.01, key="stop_loss")
            take_profit_price = st.number_input("Take Profit Price ($)", 0.01, 10000.0, 180.0, 0.01, key="take_profit")
            
            sizing_method = st.selectbox("Sizing Method:", 
                                       ["Fixed Risk %", "Kelly Criterion", "Optimal F", "Conservative Kelly"],
                                       key="sizing_method")

            # INITIALIZE VARIABLES BEFORE USE
            win_rate = 0.6
            avg_win = 50.0
            avg_loss = 30.0
            risk_percent = 2.0

            if sizing_method == "Fixed Risk %":
                risk_percent = st.slider("📉 Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1, key="risk_pct")
            else:
                st.markdown("#### Strategy Parameters")
                win_rate = st.slider("Win Rate (%)", 1, 100, 60, key="win_rate") / 100
                avg_win = st.number_input("Average Win ($)", 1.0, 1000.0, 50.0, 0.1, key="avg_win")
                avg_loss = st.number_input("Average Loss ($)", 1.0, 1000.0, 30.0, 0.1, key="avg_loss")

        with col2:
            st.markdown("#### Analysis Results")
            
            # Calculate risk metrics
            risk_per_share = entry_price - stop_loss_price
            reward_per_share = take_profit_price - entry_price
            risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
            
            position_size = 0
            risk_amount_val = 0
            
            if sizing_method == "Fixed Risk %":
                position_size, risk_amount_val = risk_manager.calculate_position_size(
                    entry_price, stop_loss_price, risk_percent)
            elif sizing_method == "Kelly Criterion":
                kelly_fraction = risk_manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                risk_amount_val = initial_capital * kelly_fraction
                position_size = int(risk_amount_val / entry_price) if entry_price > 0 else 0
            elif sizing_method == "Optimal F":
                optimal_f = risk_manager.calculate_optimal_f(win_rate, avg_win, avg_loss)
                risk_amount_val = initial_capital * optimal_f
                position_size = int(risk_amount_val / entry_price) if entry_price > 0 else 0
            elif sizing_method == "Conservative Kelly":
                kelly_fraction = risk_manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                risk_amount_val = initial_capital * kelly_fraction * 0.5
                position_size = int(risk_amount_val / entry_price) if entry_price > 0 else 0
            
            position_value = position_size * entry_price
            
            # Calculate expected value
            if sizing_method != "Fixed Risk %":
                expected_profit = position_size * reward_per_share * win_rate
                expected_loss = position_size * risk_per_share * (1 - win_rate)
                expected_value = expected_profit - expected_loss
            else:
                expected_value = position_size * reward_per_share * 0.5

            # Display metrics
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-value">{position_size:,}</div>
                <div class="metric-label">Position Size (Shares)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-value">${position_value:,.0f}</div>
                <div class="metric-label">Position Value</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-value">${risk_amount_val:,.0f}</div>
                <div class="metric-label">Risk Amount</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-value">{risk_reward_ratio:.2f}:1</div>
                <div class="metric-label">Risk/Reward Ratio</div>
            </div>
            """, unsafe_allow_html=True)

            # Risk assessment
            risk_percentage = (risk_amount_val / initial_capital) * 100
            if risk_percentage > 5:
                st.markdown('<div class="warning-card">⚠️ High Risk: Consider reducing position size</div>', unsafe_allow_html=True)
            elif risk_percentage < 1:
                st.markdown('<div class="success-card">✅ Conservative Risk: Safe position size</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-card">📊 Moderate Risk: Appropriate position size</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: MONTE CARLO SIMULATION WITH LIVE DATA ---
    with tab2:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">🎲 ADVANCED MONTE CARLO SIMULATION</div>', unsafe_allow_html=True)
        
        if st.session_state.alpaca_risk_client.connected:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Simulation Parameters")
                
                # Get estimates from actual portfolio
                live_metrics = risk_manager.calculate_live_portfolio_metrics(st.session_state.alpaca_risk_client)
                
                if live_metrics:
                    st.success("✅ Using live portfolio data for simulation")
                    default_return = max(min(live_metrics.get('total_return', 12), 50), -20)
                    default_volatility = max(min(live_metrics.get('volatility', 20), 80), 5)
                else:
                    st.info("📊 Using default parameters")
                    default_return = 12
                    default_volatility = 20
                
                expected_return = st.slider("Expected Annual Return (%)", -20, 50, int(default_return), key="mc_return") / 100
                annual_volatility = st.slider("Annual Volatility (%)", 5, 80, int(default_volatility), key="mc_vol") / 100
                time_horizon = st.slider("Time Horizon (Days)", 30, 1000, 252, key="mc_days")
                num_simulations = st.select_slider("Number of Simulations", [100, 500, 1000, 2000, 5000], 1000, key="mc_sims")
                confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="mc_conf")

                if st.button("🎲 Run Live Simulation", type="primary", key="run_mc"):
                    with st.spinner("Running Monte Carlo simulation with live data..."):
                        paths = risk_manager.simulate_portfolio_monte_carlo(
                            expected_return, annual_volatility, time_horizon, num_simulations)
                        
                        final_values = paths[-1, :]
                        returns = (final_values - initial_capital) / initial_capital
                        
                        st.session_state.mc_results = {
                            'paths': paths,
                            'final_values': final_values,
                            'returns': returns,
                            'params': {
                                'sims': num_simulations,
                                'initial': initial_capital,
                                'conf_level': confidence_level
                            }
                        }

            with col2:
                if 'mc_results' in st.session_state:
                    results = st.session_state.mc_results
                    paths = results['paths']
                    final_values = results['final_values']
                    returns = results['returns']
                    params = results['params']
                    
                    # Create advanced visualization
                    fig = go.Figure()
                    
                    # Plot all paths with transparency
                    for i in range(min(50, params['sims'])):
                        fig.add_trace(go.Scatter(
                            y=paths[:, i], 
                            mode='lines', 
                            line=dict(width=1, color='rgba(100,149,237,0.1)'), 
                            showlegend=False
                        ))
                    
                    # Plot confidence intervals
                    percentiles = [5, 25, 50, 75, 95]
                    for p in percentiles:
                        fig.add_trace(go.Scatter(
                            y=np.percentile(paths, p, axis=1),
                            mode='lines',
                            line=dict(width=2, dash='dash'),
                            name=f'{p}th Percentile'
                        ))
                    
                    fig.update_layout(
                        title=f'Monte Carlo Simulation - {params["sims"]} Scenarios',
                        template="plotly_dark",
                        height=500,
                        xaxis_title='Days',
                        yaxis_title='Portfolio Value ($)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Advanced metrics
                    var = risk_manager.calculate_var(returns, params['conf_level'])
                    cvar = risk_manager.calculate_cvar(returns, params['conf_level'])
                    max_dd = risk_manager.calculate_max_drawdown(np.percentile(paths, 50, axis=1))
                    sharpe = risk_manager.calculate_sharpe_ratio(returns, risk_free_rate)

                    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                    col_metrics1.metric("VaR (95%)", f"${var * initial_capital:,.0f}")
                    col_metrics2.metric("CVaR (95%)", f"${cvar * initial_capital:,.0f}")
                    col_metrics3.metric("Max Drawdown", f"{max_dd:.1f}%")
                    col_metrics4.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    # Probability analysis
                    prob_profit = (final_values > initial_capital).mean() * 100
                    prob_20_percent = (final_values > initial_capital * 1.2).mean() * 100
                    prob_loss = (final_values < initial_capital * 0.9).mean() * 100
                    
                    st.markdown("#### 📊 Probability Analysis")
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    prob_col1.metric("Probability of Profit", f"{prob_profit:.1f}%")
                    prob_col2.metric("Probability of 20%+ Gain", f"{prob_20_percent:.1f}%")
                    prob_col3.metric("Probability of 10%+ Loss", f"{prob_loss:.1f}%")
                else:
                    st.info("🎲 Click 'Run Live Simulation' to generate Monte Carlo analysis")
        else:
            st.info("🔌 Connect to Alpaca to run live Monte Carlo simulations with your portfolio data")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: PORTFOLIO ANALYSIS WITH ALPACA INTEGRATION ---
    with tab3:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">📊 ADVANCED PORTFOLIO ANALYSIS</div>', unsafe_allow_html=True)
        
        # ADD BUTTON TO GET ACTUAL PORTFOLIO
        col_source, col_refresh = st.columns([3, 1])
        
        with col_source:
            portfolio_source = st.radio(
                "Portfolio Data Source:",
                ["📊 Sample Data", "🔄 Live Alpaca Portfolio"],
                horizontal=True
            )
        
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_portfolio"):
                st.rerun()
        
        # GET PORTFOLIO DATA
        if portfolio_source == "🔄 Live Alpaca Portfolio":
            if st.session_state.alpaca_risk_client.connected:
                try:
                    positions = st.session_state.alpaca_risk_client.get_positions()
                    
                    if positions:
                        portfolio_data = []
                        for position in positions:
                            symbol = position.symbol
                            portfolio_data.append({
                                'Symbol': symbol,
                                'Position_Value': float(position.market_value),
                                'Volatility': risk_manager.get_volatility_estimate(symbol),
                                'Beta': risk_manager.get_beta_estimate(symbol),
                                'Quantity': float(position.qty),
                                'Avg_Price': float(position.avg_entry_price),
                                'Current_Price': float(position.current_price),
                                'Unrealized_PL': float(position.unrealized_pl),
                                'PL_Percent': (float(position.unrealized_pl) / float(position.market_value)) * 100
                            })
                        
                        portfolio_df = pd.DataFrame(portfolio_data)
                        st.success(f"✅ Loaded {len(portfolio_df)} live positions from Alpaca")
                        
                    else:
                        st.warning("📭 No positions found in Alpaca account. Using sample data.")
                        portfolio_df = get_sample_portfolio_data()
                        
                except Exception as e:
                    st.error(f"❌ Error loading Alpaca positions: {e}")
                    portfolio_df = get_sample_portfolio_data()
            else:
                st.warning("🔌 Please connect to Alpaca first. Using sample data.")
                portfolio_df = get_sample_portfolio_data()
        else:
            # Sample data
            portfolio_df = get_sample_portfolio_data()

        # PORTFOLIO ANALYSIS SECTION
        portfolio_df['Weight'] = portfolio_df['Position_Value'] / portfolio_df['Position_Value'].sum()
        portfolio_df['Risk_Contribution'] = portfolio_df['Weight'] * portfolio_df['Volatility']
        portfolio_df['Dollar_Risk'] = portfolio_df['Risk_Contribution'] * portfolio_df['Position_Value'].sum()

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Composition")
            
            # Portfolio metrics
            total_value = portfolio_df['Position_Value'].sum()
            avg_volatility = np.average(portfolio_df['Volatility'], weights=portfolio_df['Weight'])
            portfolio_beta = np.average(portfolio_df['Beta'], weights=portfolio_df['Weight'])
            
            st.metric("Total Portfolio Value", f"${total_value:,.0f}")
            st.metric("Weighted Avg Volatility", f"{avg_volatility:.1%}")
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
            
            # Display portfolio table
            display_columns = ['Symbol', 'Position_Value', 'Weight', 'Volatility', 'Beta']
            if 'Unrealized_PL' in portfolio_df.columns:
                display_columns.extend(['Unrealized_PL', 'PL_Percent'])
            
            st.dataframe(portfolio_df[display_columns].style.format({
                "Position_Value": "${:,.0f}", 
                "Weight": "{:.1%}",
                "Volatility": "{:.1%}", 
                "Beta": "{:.2f}",
                "Unrealized_PL": "${:,.2f}",
                "PL_Percent": "{:.2f}%"
            }))

        with col2:
            st.markdown("#### Risk Analysis")
            
            # Risk contribution pie chart
            fig_pie = px.pie(
                portfolio_df, 
                values='Risk_Contribution', 
                names='Symbol',
                title='Risk Contribution by Asset',
                template='plotly_dark'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Concentration analysis
            st.markdown("#### Concentration Risk")
            herfindahl = (portfolio_df['Weight'] ** 2).sum()
            st.metric("Herfindahl Index", f"{herfindahl:.3f}")
            if herfindahl > 0.25:
                st.warning("High concentration risk - consider diversification")
            else:
                st.success("Good portfolio diversification")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 4: STRESS TESTING WITH LIVE DATA ---
    with tab4:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">⚡ LIVE PORTFOLIO STRESS TESTING</div>', unsafe_allow_html=True)
        
        if st.session_state.alpaca_risk_client.connected:
            # Get actual portfolio
            positions = st.session_state.alpaca_risk_client.get_positions()
            
            if positions:
                st.success(f"✅ Stress testing {len(positions)} live positions")
                
                # Create portfolio data from actual positions
                portfolio_data = []
                for position in positions:
                    symbol = position.symbol
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Position_Value': float(position.market_value),
                        'Volatility': risk_manager.get_volatility_estimate(symbol),
                        'Beta': risk_manager.get_beta_estimate(symbol),
                        'Category': get_stock_category(symbol)
                    })
                
                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_df['Weight'] = portfolio_df['Position_Value'] / portfolio_df['Position_Value'].sum()
                
                # Stress scenarios
                scenarios = {
                    '🐻 Market Crash (2008)': {
                        'Large Cap': -0.4, 'Tech': -0.5, 'Growth': -0.6, 'Small Cap': -0.55, 'Crypto': -0.7,
                        'description': 'Global financial crisis scenario'
                    },
                    '🦠 Pandemic (COVID-19)': {
                        'Large Cap': -0.3, 'Tech': -0.2, 'Growth': -0.25, 'Small Cap': -0.4, 'Crypto': -0.35,
                        'description': 'COVID-19 market crash scenario'
                    },
                    '💸 Inflation Shock (2022)': {
                        'Large Cap': -0.2, 'Tech': -0.3, 'Growth': -0.35, 'Small Cap': -0.25, 'Crypto': -0.4,
                        'description': 'High inflation, rising rates scenario'
                    },
                    '⚡ Tech Bubble (2000)': {
                        'Large Cap': -0.15, 'Tech': -0.7, 'Growth': -0.65, 'Small Cap': -0.4, 'Crypto': -0.5,
                        'description': 'Dot-com bubble burst scenario'
                    },
                    '🛢️ Oil Price Shock': {
                        'Large Cap': -0.1, 'Tech': -0.15, 'Growth': -0.2, 'Small Cap': -0.25, 'Crypto': -0.1,
                        'description': 'Energy crisis scenario'
                    },
                    '💻 Tech Rally': {
                        'Large Cap': 0.1, 'Tech': 0.3, 'Growth': 0.25, 'Small Cap': 0.15, 'Crypto': 0.4,
                        'description': 'Technology sector bull market'
                    }
                }
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### Stress Scenarios")
                    selected_scenario = st.selectbox(
                        "Choose Stress Scenario:",
                        list(scenarios.keys()),
                        key="stress_scenario"
                    )
                    
                    scenario = scenarios[selected_scenario]
                    st.markdown(f"**Description:** {scenario['description']}")
                    
                    # Display scenario impacts
                    st.markdown("#### Scenario Impacts by Category")
                    for category, impact in scenario.items():
                        if category != 'description':
                            st.metric(f"{category} Impact", f"{impact:.1%}")
                
                with col2:
                    st.markdown("#### Portfolio Impact Analysis")
                    
                    # Calculate impact for portfolio
                    total_impact = 0
                    impact_details = []
                    
                    for _, position in portfolio_df.iterrows():
                        category = position['Category']
                        scenario_impact = scenario.get(category, -0.2)  # Default -20% if category not found
                        position_impact = position['Position_Value'] * scenario_impact
                        total_impact += position_impact
                        
                        impact_details.append({
                            'Symbol': position['Symbol'],
                            'Category': category,
                            'Current Value': position['Position_Value'],
                            'Impact %': scenario_impact,
                            'Impact $': position_impact
                        })
                    
                    impact_df = pd.DataFrame(impact_details)
                    
                    # Display results
                    current_total = portfolio_df['Position_Value'].sum()
                    new_total = current_total + total_impact
                    impact_percentage = (total_impact / current_total) * 100
                    
                    st.metric("Current Portfolio Value", f"${current_total:,.2f}")
                    st.metric("Scenario Impact", f"${total_impact:,.2f}", f"{impact_percentage:.1f}%")
                    st.metric("New Portfolio Value", f"${new_total:,.2f}")
                    
                    # Visualize impact
                    fig_impact = px.bar(
                        impact_df,
                        x='Symbol',
                        y='Impact $',
                        color='Impact %',
                        title=f'Position Impact - {selected_scenario}',
                        template='plotly_dark',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig_impact, use_container_width=True)
                    
                    # Display detailed impact
                    st.markdown("#### Detailed Impact Analysis")
                    st.dataframe(impact_df.style.format({
                        'Current Value': '${:,.2f}',
                        'Impact %': '{:.1%}',
                        'Impact $': '${:,.2f}'
                    }))
            
            else:
                st.info("📭 No positions found in your Alpaca account. Stress testing requires active positions.")
        else:
            st.info("🔌 Connect to Alpaca to perform live portfolio stress testing")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 5: RISK METRICS WITH LIVE DATA ---
    with tab5:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-header">📈 LIVE PORTFOLIO RISK METRICS</div>', unsafe_allow_html=True)
        
        if st.session_state.alpaca_risk_client.connected:
            # Get and calculate risk metrics from actual data
            live_metrics = risk_manager.calculate_live_portfolio_metrics(st.session_state.alpaca_risk_client)
            
            if live_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Portfolio Performance Metrics")
                    
                    # Key metrics
                    st.metric("Total Return", f"{live_metrics.get('total_return', 0):.2f}%")
                    st.metric("Annual Volatility", f"{live_metrics.get('volatility', 0):.2f}%")
                    st.metric("Sharpe Ratio", f"{live_metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Max Drawdown", f"{live_metrics.get('max_drawdown', 0):.2f}%")
                
                with col2:
                    st.markdown("#### 🛡️ Risk Exposure Metrics")
                    
                    # Risk metrics
                    st.metric("Value at Risk (95%)", f"{live_metrics.get('var_95', 0):.2f}%")
                    st.metric("Conditional VaR (95%)", f"{live_metrics.get('cvar_95', 0):.2f}%")
                    
                    # Additional risk calculations
                    positions = st.session_state.alpaca_risk_client.get_positions()
                    if positions:
                        portfolio_value = sum(float(pos.market_value) for pos in positions)
                        concentration_risk = max(float(pos.market_value) for pos in positions) / portfolio_value * 100
                        st.metric("Largest Position Concentration", f"{concentration_risk:.1f}%")
                        
                        # Sector concentration (simplified)
                        tech_exposure = sum(float(pos.market_value) for pos in positions if get_stock_category(pos.symbol) == 'Tech') / portfolio_value * 100
                        st.metric("Tech Sector Exposure", f"{tech_exposure:.1f}%")
                
                # Risk Assessment
                st.markdown("#### 📋 Risk Assessment")
                
                risk_score = 0
                risk_factors = []
                
                # Evaluate volatility
                volatility = live_metrics.get('volatility', 0)
                if volatility > 30:
                    risk_score += 2
                    risk_factors.append(f"⚠️ High volatility ({volatility:.1f}%)")
                elif volatility > 20:
                    risk_score += 1
                    risk_factors.append(f"📊 Moderate volatility ({volatility:.1f}%)")
                else:
                    risk_factors.append(f"✅ Low volatility ({volatility:.1f}%)")
                
                # Evaluate drawdown
                max_dd = live_metrics.get('max_drawdown', 0)
                if max_dd > 20:
                    risk_score += 2
                    risk_factors.append(f"⚠️ High max drawdown ({max_dd:.1f}%)")
                elif max_dd > 10:
                    risk_score += 1
                    risk_factors.append(f"📊 Moderate max drawdown ({max_dd:.1f}%)")
                else:
                    risk_factors.append(f"✅ Low max drawdown ({max_dd:.1f}%)")
                
                # Evaluate VaR
                var_95 = live_metrics.get('var_95', 0)
                if var_95 > 5:
                    risk_score += 2
                    risk_factors.append(f"⚠️ High VaR ({var_95:.1f}%)")
                elif var_95 > 3:
                    risk_score += 1
                    risk_factors.append(f"📊 Moderate VaR ({var_95:.1f}%)")
                else:
                    risk_factors.append(f"✅ Low VaR ({var_95:.1f}%)")
                
                # Display risk assessment
                st.markdown("##### Risk Factors:")
                for factor in risk_factors:
                    st.write(factor)
                
                # Overall risk rating
                if risk_score >= 4:
                    st.error(f"🚨 HIGH RISK PORTFOLIO (Score: {risk_score}/6)")
                    st.warning("Consider reducing position sizes and increasing diversification")
                elif risk_score >= 2:
                    st.warning(f"📊 MODERATE RISK PORTFOLIO (Score: {risk_score}/6)")
                    st.info("Portfolio risk is acceptable but monitor closely")
                else:
                    st.success(f"✅ LOW RISK PORTFOLIO (Score: {risk_score}/6)")
                    st.info("Portfolio maintains conservative risk profile")
                
                # Portfolio History Visualization
                st.markdown("#### 📈 Portfolio Performance History")
                try:
                    portfolio_history = st.session_state.alpaca_risk_client.get_portfolio_history(period="1M")
                    if portfolio_history and hasattr(portfolio_history, 'equity') and portfolio_history.equity:
                        equity_data = portfolio_history.equity
                        timestamps = portfolio_history.timestamp
                        
                        if equity_data and len(equity_data) > 1:
                            # Create DataFrame for chart
                            history_df = pd.DataFrame({
                                'timestamp': [pd.to_datetime(ts * 1000, unit='ms') for ts in timestamps],
                                'equity': equity_data
                            })
                            
                            fig_equity = px.line(
                                history_df,
                                x='timestamp',
                                y='equity',
                                title='Portfolio Equity Curve',
                                template='plotly_dark'
                            )
                            fig_equity.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Portfolio Value ($)'
                            )
                            st.plotly_chart(fig_equity, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load portfolio history: {e}")
            
            else:
                st.info("📊 No sufficient historical data available for risk metrics calculation. Trade for a while to generate data.")
        else:
            st.info("🔌 Connect to Alpaca to view live portfolio risk metrics")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #8898aa;'>
        <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Risk Management Platform</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Advanced Risk Manager v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()