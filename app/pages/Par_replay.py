import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🚀 TradingView Pro", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MINIMAL CSS STYLING ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .header {
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
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        border-color: #667eea;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border: none;
    }
    .config-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
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
    .param-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .profit { color: #00d4aa; font-weight: bold; }
    .loss { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HÀM TẢI DỮ LIỆU ĐÃ ĐƯỢC SỬA ---
def get_crypto_data_simple(symbol='BTC/USDT', timeframe='1h', limit=500):
    """Simple data fetcher using multiple exchanges - THAY THẾ BINANCE API"""
    
    exchanges = [
        {'name': 'bybit', 'class': ccxt.bybit},
        {'name': 'okx', 'class': ccxt.okx},
        {'name': 'kucoin', 'class': ccxt.kucoin},
        {'name': 'gateio', 'class': ccxt.gateio},
        {'name': 'htx', 'class': ccxt.htx},
    ]
    
    for exchange_info in exchanges:
        try:
            exchange = exchange_info['class']({
                'timeout': 30000,
                'enableRateLimit': True,
            })
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            continue
    
    return get_yahoo_fallback(symbol)

def get_yahoo_fallback(symbol):
    """Fallback to Yahoo Finance"""
    symbol_map = {
        'BTC/USDT': 'BTC-USD',
        'ETH/USDT': 'ETH-USD', 
        'BNB/USDT': 'BNB-USD',
        'ADA/USDT': 'ADA-USD',
        'XRP/USDT': 'XRP-USD',
        'DOT/USDT': 'DOT-USD',
        'LINK/USDT': 'LINK-USD',
        'LTC/USDT': 'LTC-USD',
        'BCH/USDT': 'BCH-USD',
        'SOL/USDT': 'SOL-USD'
    }
    
    yahoo_symbol = symbol_map.get(symbol, 'BTC-USD')
    try:
        data = yf.download(yahoo_symbol, period='6mo', interval='1d')
        return data
    except Exception as e:
        return None

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>📈</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>TradingView Pro</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>Advanced Backtesting</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# --- HEADER ---
st.markdown('<div class="header">🚀 TradingView Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Trading Platform with Advanced Backtesting</div>', unsafe_allow_html=True)

# --- STATUS CARDS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">Live</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Market Data</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">Pro</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Backtesting</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛡️</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">Active</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Risk Management</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">20+</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Indicators</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- CORE LOGIC FUNCTIONS (FIXED PnL) ---
@st.cache_data(ttl=3600)
def load_and_prepare_data(asset_class, symbol, timeframe, start_date=None, end_date=None):
    """Load and prepare data from multiple sources"""
    try:
        if asset_class == "Crypto":
            df = get_crypto_data_simple(symbol, timeframe, 2000)
            
            if df is not None and not df.empty:
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df.index >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df.index <= end_dt]
            else:
                symbol_map = {
                    'BTC/USDT': 'BTC-USD',
                    'ETH/USDT': 'ETH-USD',
                    'SOL/USDT': 'SOL-USD',
                    'XRP/USDT': 'XRP-USD',
                    'DOGE/USDT': 'DOGE-USD'
                }
                yahoo_symbol = symbol_map.get(symbol, symbol.replace('/USDT', '-USD'))
                df = yf.download(yahoo_symbol, period='6mo', interval='1d', progress=False, auto_adjust=True)
        else:
            interval_map = {
                '1m':'1m', '5m':'5m', '15m':'15m', '30m':'30m', 
                '1h':'1h', '4h':'4h', '1d':'1d', '1w':'1wk', '1M':'1mo'
            }
            yf_interval = interval_map.get(timeframe, '1d')
            
            if not start_date:
                if yf_interval in ['1m', '5m', '15m', '30m']: 
                    start_date = datetime.now() - timedelta(days=60)
                elif yf_interval == '1h': 
                    start_date = datetime.now() - timedelta(days=730)
                else: 
                    start_date = datetime.now() - timedelta(days=365*2)
            
            if not end_date:
                end_date = datetime.now()
                
            df = yf.download(symbol, start=start_date, end=end_date, interval=yf_interval, progress=False)
        
        if df.empty:
            st.error("Unable to load data.")
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).capitalize() for col in df.columns]
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing data columns. Current columns: {list(df.columns)}")
            return None
            
        return df.iloc[-2000:]  # Limit to 2000 candles
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_advanced_indicators(df, selected_indicators):
    """Calculate advanced technical indicators"""
    try:
        if 'SMA' in selected_indicators:
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            
        if 'EMA' in selected_indicators:
            df.ta.ema(length=12, append=True)
            df.ta.ema(length=26, append=True)
            
        if 'RSI' in selected_indicators:
            df.ta.rsi(length=14, append=True)
            
        if 'MACD' in selected_indicators:
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            
        if 'BB' in selected_indicators:
            df.ta.bbands(length=20, std=2, append=True)
            
        if 'Stoch' in selected_indicators:
            df.ta.stoch(k=14, d=3, append=True)
            
        if 'Volume' in selected_indicators:
            df.ta.volume_profile(length=20, append=True)
            
        return df
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return df

def init_session_state():
    """Initialize session state"""
    if 'replay_index' not in st.session_state:
        st.session_state.replay_index = 100
        
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
        
    if 'position_type' not in st.session_state:
        st.session_state.position_type = None
        st.session_state.entry_price = 0
        st.session_state.stop_loss = 0
        st.session_state.take_profit = 0
        st.session_state.trade_size = 0
        st.session_state.entry_time = None
        
    if 'balance' not in st.session_state:
        st.session_state.balance = 10000.0
        st.session_state.equity = 10000.0
        st.session_state.initial_balance = 10000.0
        
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
        
    if 'risk_management' not in st.session_state:
        st.session_state.risk_management = {
            'risk_per_trade': 2.0,
            'max_drawdown': 20.0,
            'daily_loss_limit': 5.0
        }

def get_contract_multiplier(asset_class):
    """Get contract multiplier for each asset class"""
    multipliers = {
        'Forex': 100000,    # 1 lot = 100,000 units
        'Crypto': 1,        # 1 unit = 1 unit
        'Stocks': 1         # 1 share = 1 share
    }
    return multipliers.get(asset_class, 1)

def calculate_position_size(entry_price, stop_loss, risk_per_trade, asset_class):
    """Calculate position size based on risk management - FIXED"""
    risk_amount = st.session_state.balance * risk_per_trade / 100
    risk_per_unit = abs(entry_price - stop_loss)
    contract_multiplier = get_contract_multiplier(asset_class)
    
    if risk_per_unit > 0:
        base_position_size = risk_amount / risk_per_unit
        
        if asset_class == 'Forex':
            position_size = base_position_size / contract_multiplier
            max_position = (st.session_state.balance * 0.1) / (entry_price * contract_multiplier)
        else:
            position_size = base_position_size
            max_position = (st.session_state.balance * 0.1) / entry_price
        
        return min(position_size, max_position)
    return 0

def handle_open_position(pos_type, current_price, sl_pct, tp_pct):
    """Handle opening position with risk management - FIXED"""
    risk_per_trade = st.session_state.risk_management['risk_per_trade']
    asset_class = st.session_state.get('current_asset_class', 'Crypto')
    
    if pos_type == 'long':
        stop_loss = current_price * (1 - sl_pct / 100)
        take_profit = current_price * (1 + tp_pct / 100)
    else:
        stop_loss = current_price * (1 + sl_pct / 100)
        take_profit = current_price * (1 - tp_pct / 100)
        
    trade_size = calculate_position_size(current_price, stop_loss, risk_per_trade, asset_class)
    
    if trade_size > 0:
        st.session_state.position_type = pos_type
        st.session_state.entry_price = current_price
        st.session_state.stop_loss = stop_loss
        st.session_state.take_profit = take_profit
        st.session_state.trade_size = trade_size
        st.session_state.entry_time = datetime.now()
        
        action = "LONG" if pos_type == 'long' else "SHORT"
        
        if asset_class == 'Forex':
            size_display = f"{trade_size:.2f} lots"
            contract_size = get_contract_multiplier(asset_class)
            notional_value = trade_size * contract_size * current_price
        else:
            size_display = f"{trade_size:.4f} units"
            notional_value = trade_size * current_price
        
        st.toast(f"🎯 Opened {action} | Size: {size_display} | Notional: ${notional_value:,.2f} | SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}", icon="✅")
        return True
    else:
        st.toast("❌ Position size too small or insufficient funds", icon="⚠️")
        return False

def handle_close_position(exit_price, reason="Manual"):
    """Handle closing position and record history - COMPLETELY FIXED"""
    if not st.session_state.position_type:
        return 0
        
    entry_price = st.session_state.entry_price
    pos_size = st.session_state.trade_size
    asset_class = st.session_state.get('current_asset_class', 'Crypto')
    
    contract_multiplier = get_contract_multiplier(asset_class)
    
    if st.session_state.position_type == 'long':
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price
    
    profit = price_diff * pos_size * contract_multiplier
    
    st.session_state.balance += profit
    st.session_state.equity = st.session_state.balance
    
    trade_record = {
        'entry_time': st.session_state.entry_time,
        'exit_time': datetime.now(),
        'symbol': st.session_state.get('current_symbol', 'N/A'),
        'type': st.session_state.position_type.upper(),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'quantity': pos_size,
        'pnl': profit,
        'reason': reason,
        'balance_after': st.session_state.balance
    }
    
    st.session_state.trade_history.append(trade_record)
    
    st.session_state.position_type = None
    st.session_state.entry_price = 0
    st.session_state.stop_loss = 0
    st.session_state.take_profit = 0
    st.session_state.trade_size = 0
    st.session_state.entry_time = None
    
    return profit

def calculate_performance_metrics():
    """Calculate performance metrics"""
    if not st.session_state.trade_history:
        return {}
    
    trades = pd.DataFrame(st.session_state.trade_history)
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] < 0]
    
    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = trades['pnl'].sum()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 else float('inf')
    
    balances = [trade['balance_after'] for trade in st.session_state.trade_history]
    running_max = pd.Series(balances).cummax()
    drawdowns = (pd.Series(balances) - running_max) / running_max * 100
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'final_balance': st.session_state.balance,
        'total_return': (st.session_state.balance - st.session_state.initial_balance) / st.session_state.initial_balance * 100
    }

def create_tradingview_chart(df, current_index, indicators):
    """Create TradingView-style chart"""
    replay_df = df.iloc[:current_index + 1]
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.6, 0.15, 0.15, 0.1],
        subplot_titles=('Price Chart', 'Volume', 'RSI', 'MACD')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=replay_df.index, 
            open=replay_df['Open'], 
            high=replay_df['High'], 
            low=replay_df['Low'], 
            close=replay_df['Close'], 
            name='Price'
        ), 
        row=1, col=1
    )
    
    if 'SMA_20' in replay_df.columns:
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['SMA_20'], 
                      line=dict(color='orange', width=1), name='SMA 20'),
            row=1, col=1
        )
    
    if 'SMA_50' in replay_df.columns:
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['SMA_50'], 
                      line=dict(color='blue', width=1), name='SMA 50'),
            row=1, col=1
        )
    
    if 'BBU_20_2.0' in replay_df.columns:
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['BBU_20_2.0'], 
                      line=dict(color='gray', width=1, dash='dash'), name='BB Upper'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['BBL_20_2.0'], 
                      line=dict(color='gray', width=1, dash='dash'), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    colors = ['red' if row['Open'] > row['Close'] else 'green' 
              for _, row in replay_df.iterrows()]
    fig.add_trace(
        go.Bar(x=replay_df.index, y=replay_df['Volume'], 
               marker_color=colors, name='Volume'),
        row=2, col=1
    )
    
    if 'RSI_14' in replay_df.columns:
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['RSI_14'], 
                      line=dict(color='purple', width=1), name='RSI'),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    if 'MACD_12_26_9' in replay_df.columns:
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['MACD_12_26_9'], 
                      line=dict(color='blue', width=1), name='MACD'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=replay_df.index, y=replay_df['MACDs_12_26_9'], 
                      line=dict(color='red', width=1), name='Signal'),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=replay_df.index, y=replay_df['MACDh_12_26_9'], 
                   marker_color=np.where(replay_df['MACDh_12_26_9'] > 0, 'green', 'red'),
                   name='Histogram'),
            row=4, col=1
        )
    
    if st.session_state.position_type:
        fig.add_hline(y=st.session_state.take_profit, line_dash="dot", 
                     line_color="green", annotation_text="TP", row=1, col=1)
        fig.add_hline(y=st.session_state.stop_loss, line_dash="dot", 
                     line_color="red", annotation_text="SL", row=1, col=1)
        fig.add_hline(y=st.session_state.entry_price, line_dash="dash", 
                     line_color="blue", annotation_text="Entry", row=1, col=1)
    
    fig.update_layout(
        title=f"TradingView Pro - {st.session_state.get('current_symbol', 'Chart')}",
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# --- INITIALIZATION ---
init_session_state()

# --- MAIN LAYOUT ---
st.markdown('<div class="section">⚙️ BACKTEST CONFIGURATION</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**📊 DATA CONFIGURATION**')
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        asset_class = st.radio(
            "Asset Class:", 
            ["Crypto", "Forex", "Stocks"], 
            key="asset_class_radio",
            horizontal=True
        )
    
    with config_col2:
        if asset_class == "Crypto":
            symbol = st.text_input("Symbol Pair:", "BTC/USDT", key="crypto_symbol")
            timeframe = st.selectbox(
                "Timeframe:", 
                ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'], 
                index=4,
                key="crypto_tf"
            )
        elif asset_class == "Forex":
            symbol = st.text_input("Currency Pair:", "EURUSD=X", key="forex_symbol")
            timeframe = st.selectbox(
                "Timeframe:", 
                ['1m', '5m', '15m', '30m', '1h', '1d', '1w'], 
                index=4,
                key="forex_tf"
            )
        else:
            symbol = st.text_input("Stock Symbol:", "AAPL", key="stock_symbol")
            timeframe = st.selectbox(
                "Timeframe:", 
                ['1h', '1d', '1w'], 
                index=1,
                key="stock_tf"
            )
    
    with config_col3:
        if st.button("🚀 START BACKTEST", type="primary", use_container_width=True):
            st.session_state.current_asset_class = asset_class
            st.session_state.current_symbol = symbol
            st.session_state.full_df = load_and_prepare_data(asset_class, symbol, timeframe)
            
            if st.session_state.full_df is not None:
                st.session_state.full_df = calculate_advanced_indicators(
                    st.session_state.full_df, st.session_state.get('selected_indicators', ['SMA', 'RSI'])
                )
                init_session_state()
                st.success("✅ Data ready!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**📈 TECHNICAL INDICATORS**')
    
    selected_indicators = st.multiselect(
        "Select Indicators:",
        ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'Stoch', 'Volume'],
        default=['SMA', 'RSI', 'MACD', 'BB'],
        key="tech_indicators"
    )
    
    st.markdown("**Selected Indicators:**")
    for indicator in selected_indicators:
        st.markdown(f'<div class="param-container">{indicator}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**🛡️ RISK MANAGEMENT**')
    
    st.session_state.risk_management['risk_per_trade'] = st.slider(
        "Risk per Trade (%):", 0.1, 10.0, 2.0, 0.1, key="risk_slider"
    )
    st.session_state.risk_management['max_drawdown'] = st.slider(
        "Max Drawdown (%):", 5.0, 50.0, 20.0, 1.0, key="drawdown_slider"
    )
    
    st.info(f"**Maximum position size:** {st.session_state.risk_management['risk_per_trade']}% of balance")
    st.markdown('</div>', unsafe_allow_html=True)

# --- ACCOUNT & TRADING PANEL ---
col4, col5 = st.columns([1, 1])

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**💰 ACCOUNT MANAGEMENT**')
    
    account_col1, account_col2 = st.columns(2)
    
    with account_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💰</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">${st.session_state.balance:,.0f}</div>
            <div style="color: #8898aa; font-size: 0.9rem;">Balance</div>
        </div>
        """, unsafe_allow_html=True)
        
    with account_col2:
        equity_color = "profit" if st.session_state.equity >= st.session_state.balance else "loss"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📈</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {equity_color}; margin: 0.5rem 0;">${st.session_state.equity:,.0f}</div>
            <div style="color: #8898aa; font-size: 0.9rem;">Equity</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.position_type:
        current_price = 0
        if 'full_df' in st.session_state and st.session_state.full_df is not None:
            current_price = st.session_state.full_df['Close'].iloc[st.session_state.replay_index]
        
        unrealized_pnl = st.session_state.equity - st.session_state.balance
        pnl_color = "profit" if unrealized_pnl >= 0 else "loss"
        
        st.markdown(f"""
        <div class="card">
            <h4>📊 CURRENT POSITION</h4>
            <p><strong>Type:</strong> {st.session_state.position_type.upper()}</p>
            <p><strong>Entry:</strong> ${st.session_state.entry_price:.4f}</p>
            <p><strong>Current:</strong> ${current_price:.4f}</p>
            <p><strong>Size:</strong> {st.session_state.trade_size:.4f}</p>
            <p><strong>Unrealized PnL:</strong> <span class="{pnl_color}">${unrealized_pnl:,.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**🎯 TRADING PANEL**')
    
    current_price = 0
    if 'full_df' in st.session_state and st.session_state.full_df is not None:
        current_price = st.session_state.full_df['Close'].iloc[st.session_state.replay_index]
        st.markdown(f'<div class="metric-card"><div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">${current_price:.4f}</div><div style="color: #8898aa; font-size: 0.9rem;">Current Price</div></div>', unsafe_allow_html=True)
    
    trade_col1, trade_col2 = st.columns(2)
    
    with trade_col1:
        if asset_class == 'Forex':
            trade_size = st.number_input("Volume (Lot)", min_value=0.01, max_value=10.0, value=0.1, step=0.01, key="forex_size")
        elif asset_class == 'Stocks':
            trade_size = st.number_input("Share Quantity", min_value=1, value=10, step=1, key="stock_size")
        else:
            trade_size = st.number_input("Volume", min_value=0.00001, value=0.01, step=0.0001, format="%.5f", key="crypto_size")
    
    with trade_col2:
        sl_pct = st.number_input("Stop Loss (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1, key="sl_input")
        tp_pct = st.number_input("Take Profit (%)", min_value=0.1, max_value=50.0, value=4.0, step=0.1, key="tp_input")
    
    if not st.session_state.position_type:
        trade_btn_col1, trade_btn_col2 = st.columns(2)
        with trade_btn_col1:
            if st.button("🟢 BUY/LONG", use_container_width=True, type="primary", key="buy_btn"):
                if handle_open_position('long', current_price, sl_pct, tp_pct):
                    st.rerun()
        with trade_btn_col2:
            if st.button("🔴 SELL/SHORT", use_container_width=True, type="secondary", key="sell_btn"):
                if handle_open_position('short', current_price, sl_pct, tp_pct):
                    st.rerun()
    else:
        if st.button("🟡 CLOSE POSITION", use_container_width=True, type="primary", key="close_btn"):
            profit = handle_close_position(current_price, "Manual Close")
            pnl_color = "profit" if profit >= 0 else "loss"
            st.toast(f"Position closed. PnL: <span class='{pnl_color}'>${profit:,.2f}</span>", icon="💰")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN CHART & CONTROLS ---
if 'full_df' in st.session_state and st.session_state.full_df is not None:
    current_index = st.session_state.replay_index
    full_df = st.session_state.full_df
    
    if st.session_state.position_type:
        current_price = full_df['Close'].iloc[current_index]
        entry_price = st.session_state.entry_price
        trade_size = st.session_state.trade_size
        asset_class_in_trade = st.session_state.get('current_asset_class', 'Crypto')
        contract_multiplier = get_contract_multiplier(asset_class_in_trade)
        
        if st.session_state.position_type == 'long':
            unrealized_pnl = (current_price - entry_price) * trade_size * contract_multiplier
        else:
            unrealized_pnl = (entry_price - current_price) * trade_size * contract_multiplier
        
        st.session_state.equity = st.session_state.balance + unrealized_pnl
    else:
        st.session_state.equity = st.session_state.balance
    
    if st.session_state.position_type:
        current_candle = full_df.iloc[current_index]
        exit_price = 0
        reason = ""
        
        if st.session_state.position_type == 'long':
            if current_candle['Low'] <= st.session_state.stop_loss:
                exit_price, reason = st.session_state.stop_loss, "Stop Loss"
            elif current_candle['High'] >= st.session_state.take_profit:
                exit_price, reason = st.session_state.take_profit, "Take Profit"
        else:
            if current_candle['High'] >= st.session_state.stop_loss:
                exit_price, reason = st.session_state.stop_loss, "Stop Loss"
            elif current_candle['Low'] <= st.session_state.take_profit:
                exit_price, reason = st.session_state.take_profit, "Take Profit"
        
        if exit_price > 0:
            profit = handle_close_position(exit_price, f"Auto {reason}")
            pnl_color = "profit" if profit >= 0 else "loss"
            st.toast(f"🤖 Auto closed due to {reason}. PnL: <span class='{pnl_color}'>${profit:,.2f}</span>")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**🎮 BACKTEST CONTROLS**')
    
    control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns([1, 1, 1, 2, 3])
    
    with control_col1:
        if st.button("▶️ PLAY", use_container_width=True, type="primary"):
            st.session_state.is_playing = True
    with control_col2:
        if st.button("⏸️ PAUSE", use_container_width=True):
            st.session_state.is_playing = False
    with control_col3:
        if st.button("⏩ NEXT CANDLE", use_container_width=True):
            if current_index < len(full_df) - 1:
                st.session_state.replay_index += 1
            else:
                st.toast("🎉 Reached end of data!")
            st.rerun()
    with control_col4:
        zoom_on_candles = st.toggle("Zoom to candles", value=True, key="zoom_toggle")
    with control_col5:
        speed = st.slider("Replay speed", 0.05, 1.0, 0.2, 0.05, key="speed_slider")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section">📊 TRADINGVIEW PRO CHART</div>', unsafe_allow_html=True)
    fig = create_tradingview_chart(full_df, current_index, selected_indicators)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section">📈 PERFORMANCE ANALYSIS</div>', unsafe_allow_html=True)
    
    if st.session_state.trade_history:
        metrics = calculate_performance_metrics()
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Total Trades", metrics['total_trades'])
            win_rate_color = "profit" if metrics['win_rate'] > 50 else "loss"
            st.markdown(f'<div class="param-container">Win Rate: <span class="{win_rate_color}">{metrics["win_rate"]:.1f}%</span></div>', unsafe_allow_html=True)
            
        with perf_col2:
            st.metric("Total PnL", f"${metrics['total_pnl']:,.2f}")
            total_return_color = "profit" if metrics['total_return'] >= 0 else "loss"
            st.markdown(f'<div class="param-container">Total Return: <span class="{total_return_color}">{metrics["total_return"]:.2f}%</span></div>', unsafe_allow_html=True)
            
        with perf_col3:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Avg Win", f"${metrics['avg_win']:,.2f}")
            
        with perf_col4:
            st.metric("Avg Loss", f"${metrics['avg_loss']:,.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        
        st.markdown("**📋 TRADE HISTORY**")
        if st.session_state.trade_history:
            trades_df = pd.DataFrame(st.session_state.trade_history)
            trades_df['entry_time'] = trades_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            trades_df['exit_time'] = trades_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                trades_df.style.format({
                    'entry_price': '{:.4f}',
                    'exit_price': '{:.4f}',
                    'pnl': '{:,.2f}',
                    'balance_after': '{:,.2f}'
                }),
                use_container_width=True,
                height=300
            )
    
    if st.session_state.is_playing:
        if current_index < len(full_df) - 1:
            st.session_state.replay_index += 1
            time.sleep(speed)
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.toast("🎉 Backtest completed!")
            st.rerun()

else:
    st.markdown("""
    <div style="text-align: center; padding: 100px; background: rgba(255, 255, 255, 0.05); border-radius: 20px; margin: 20px; border: 1px solid rgba(255, 255, 255, 0.1);">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800;">🎯 WELCOME TO TRADINGVIEW PRO</h1>
        <p style="font-size: 1.3rem; color: #8898aa; margin-top: 20px;">
            Professional backtesting platform with TradingView interface
        </p>
        <div style="margin-top: 50px;">
            <div style="display: inline-block; margin: 15px; padding: 25px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; width: 250px; vertical-align: top; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="color: #667eea;">📊 MULTI-TIMEFRAME</h3>
                <p style="color: #8898aa;">1 minute to 1 week</p>
            </div>
            <div style="display: inline-block; margin: 15px; padding: 25px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; width: 250px; vertical-align: top; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="color: #667eea;">📈 20+ INDICATORS</h3>
                <p style="color: #8898aa;">Professional indicators</p>
            </div>
            <div style="display: inline-block; margin: 15px; padding: 25px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; width: 250px; vertical-align: top; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="color: #667eea;">🛡️ RISK MANAGEMENT</h3>
                <p style="color: #8898aa;">Smart risk control</p>
            </div>
        </div>
        <p style="margin-top: 50px; font-size: 1.1rem; color: #8898aa;">
            ⚡ <strong>GET STARTED:</strong> Configure backtest above and click "START BACKTEST"
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Trading Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>TradingView Pro v2.0</p>
</div>
""", unsafe_allow_html=True)