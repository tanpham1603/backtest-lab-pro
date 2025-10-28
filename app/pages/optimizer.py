import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import vectorbt as vbt
from itertools import product
from datetime import datetime, timedelta
import ccxt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# --- Cấu hình trang ---
st.set_page_config(
    page_title="🚀 Advanced Strategy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS tùy chỉnh ---
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
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-badge {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM TẢI DỮ LIỆU ĐÃ ĐƯỢC SỬA ---
def get_crypto_data_simple(symbol='BTC/USDT', timeframe='1h', limit=500):
    """Simple data fetcher using multiple exchanges - THAY THẾ BINANCE API"""
    
    # Danh sách exchanges ít bị chặn
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
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            continue
    
    # Fallback cuối cùng: Yahoo Finance
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

# --- Sidebar ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>⚡</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>Strategy Optimizer</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>Advanced Multi-Strategy</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("✨ Configure parameters to begin optimization")

# --- Header ---
st.markdown('<div class="welcome-header">🚀 Advanced Strategy Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Multi-Strategy Optimization with Advanced Risk Analysis</div>', unsafe_allow_html=True)

# --- Status Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📈</div>
        <div class="metric-value">5+</div>
        <div class="metric-label">Strategies</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">⚡</div>
        <div class="metric-value">7+</div>
        <div class="metric-label">Metrics</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🛡️</div>
        <div class="metric-value">Pro</div>
        <div class="metric-label">Risk Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🎯</div>
        <div class="metric-value">Live</div>
        <div class="metric-label">Optimization</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- STRATEGY CONFIGURATION SECTION ---
st.markdown("## ⚙️ STRATEGY CONFIGURATION")

# Main Configuration Row
config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="config-header">📊 ASSET & DATA</div>', unsafe_allow_html=True)
    
    asset_class = st.selectbox("**Asset Class:**", ["Crypto", "Forex", "Stocks"])
    common_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    if asset_class == "Crypto":
        symbol = st.text_input("**Symbol:**", "BTC/USDT", help="Example: BTC/USDT, ETH/USDT")
        tf = st.selectbox("**Timeframe:**", common_timeframes, index=4)
    else:
        symbol = st.text_input("**Symbol:**", "EURUSD=X" if asset_class == "Forex" else "AAPL", 
                              help="Example: AAPL, MSFT, EURUSD=X")
        tf = st.selectbox("**Timeframe:**", common_timeframes, index=6)
    st.markdown('</div>', unsafe_allow_html=True)

with config_col2:
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="config-header">📅 TIME RANGE</div>', unsafe_allow_html=True)
    
    yf_timeframe_limits = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}
    end_date_default = datetime.now().date()
    start_date_default = end_date_default - timedelta(days=365)
    
    if asset_class != 'Crypto' and tf in yf_timeframe_limits:
        limit = yf_timeframe_limits[tf]
        start_date_default = end_date_default - timedelta(days=limit - 1)
        st.info(f"{tf} timeframe limited to {limit} days")
    
    end_date_input = st.date_input("**End Date**", value=end_date_default)
    start_date_input = st.date_input("**Start Date**", value=start_date_default)
    
    use_walk_forward = st.checkbox("**Use Walk-Forward Optimization**", value=False)
    if use_walk_forward:
        n_periods = st.slider("**Number of Walk-Forward Periods:**", 3, 10, 5)
    st.markdown('</div>', unsafe_allow_html=True)

with config_col3:
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="config-header">🎯 STRATEGY & TARGETS</div>', unsafe_allow_html=True)
    
    strategy_type = st.selectbox(
        "**Strategy Type:**",
        ["MA Cross", "RSI + MA", "Bollinger Bands", "MACD Cross", "Multi-Strategy"]
    )
    
    target_metric = st.selectbox(
        "**Target Metric:**",
        ["Total Return [%]", "Sharpe Ratio", "Calmar Ratio", "Win Rate [%]", 
         "Profit Factor", "Sortino Ratio"]
    )
    
    min_trades = st.number_input("**Minimum Trades:**", min_value=0, value=5)
    max_drawdown_limit = st.number_input("**Maximum Drawdown (%):**", min_value=0, value=50)
    st.markdown('</div>', unsafe_allow_html=True)

# --- STRATEGY PARAMETERS SECTION ---
st.markdown("## ⚙️ STRATEGY PARAMETERS")

# Initialize parameter lists
fasts = []
slows = []
rsi_periods = []
rsi_oversold = []
rsi_overbought = []
bb_periods = []
bb_stds = []
macd_fast = []
macd_slow = []
macd_signal = []

if strategy_type == "MA Cross":
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="config-header">📈 MOVING AVERAGE CROSSOVER PARAMETERS</div>', unsafe_allow_html=True)
    
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        fast_options = list(range(5, 51, 2))
        fast_defaults = [x for x in [9, 11, 13, 15, 19] if x in fast_options]
        if not fast_defaults:
            fast_defaults = [fast_options[2]]
        fasts = st.multiselect("**Fast MA Periods:**", fast_options, default=fast_defaults)
        
    with param_col2:
        slow_options = list(range(20, 201, 5))
        slow_defaults = [x for x in [30, 50, 100, 150] if x in slow_options]
        if not slow_defaults:
            slow_defaults = [slow_options[2]]
        slows = st.multiselect("**Slow MA Periods:**", slow_options, default=slow_defaults)
    st.markdown('</div>', unsafe_allow_html=True)

# --- OPTIMIZATION BUTTON ---
optimize_btn = st.button("🚀 START ADVANCED OPTIMIZATION", type="primary", use_container_width=True)

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=600)
def load_price_data(asset, sym, timeframe, start, end):
    try:
        if asset == "Crypto":
            # Sử dụng hàm mới với multiple exchanges thay vì chỉ Binance
            df = get_crypto_data_simple(sym, timeframe, 2000)
            
            if df is not None and not df.empty:
                # Lọc theo ngày
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                df = df.loc[start_dt:end_dt]
                
                if df.empty:
                    # Fallback to yfinance
                    symbol_map = {
                        'BTC/USDT': 'BTC-USD',
                        'ETH/USDT': 'ETH-USD',
                        'SOL/USDT': 'SOL-USD',
                        'XRP/USDT': 'XRP-USD',
                        'DOGE/USDT': 'DOGE-USD'
                    }
                    yahoo_symbol = symbol_map.get(sym, sym.replace('/USDT', '-USD'))
                    data = yf.download(yahoo_symbol, start=start, end=end, interval='1d', progress=False, auto_adjust=True)
                    if data.empty:
                        data = yf.download(sym, start=start, end=end, interval='1d', progress=False, auto_adjust=True)
                    df = data
            else:
                # Fallback to yfinance
                symbol_map = {
                    'BTC/USDT': 'BTC-USD',
                    'ETH/USDT': 'ETH-USD',
                    'SOL/USDT': 'SOL-USD',
                    'XRP/USDT': 'XRP-USD',
                    'DOGE/USDT': 'DOGE-USD'
                }
                yahoo_symbol = symbol_map.get(sym, sym.replace('/USDT', '-USD'))
                data = yf.download(yahoo_symbol, start=start, end=end, interval='1d', progress=False, auto_adjust=True)
                if data.empty:
                    data = yf.download(sym, start=start, end=end, interval='1d', progress=False, auto_adjust=True)
                df = data
        else:
            yf_timeframe_map = {"1w": "1wk"}
            interval = yf_timeframe_map.get(timeframe, timeframe)
            data = yf.download(sym, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).capitalize() for col in data.columns]
            df = data
        
        if df.empty: 
            return None
            
        df['Returns'] = df['Close'].pct_change()
        return df
        
    except Exception as e:
        st.error(f"Error loading data for {sym}: {e}")
        return None

# --- MA CROSS STRATEGY FUNCTION ---
def run_ma_cross_strategy(price, fast_windows, slow_windows):
    """Run MA Cross strategy with all fast/slow combinations"""
    all_entries = []
    all_exits = []
    param_combinations = []
    
    progress_bar = st.progress(0)
    total_combinations = len(fast_windows) * len(slow_windows)
    processed = 0
    
    for fast, slow in product(fast_windows, slow_windows):
        if fast >= slow:
            continue
            
        # Calculate MA for each parameter pair
        fast_ma = vbt.MA.run(price, window=fast)
        slow_ma = vbt.MA.run(price, window=slow)
        
        # Create signals
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        
        all_entries.append(entries)
        all_exits.append(exits)
        param_combinations.append((fast, slow))
        
        processed += 1
        progress_bar.progress(processed / total_combinations)
    
    progress_bar.empty()
    
    if not all_entries:
        return None, None, []
    
    # Combine all results
    entries_combined = pd.concat(all_entries, axis=1, keys=param_combinations)
    exits_combined = pd.concat(all_exits, axis=1, keys=param_combinations)
    
    return entries_combined, exits_combined, param_combinations

# --- PERFORMANCE HELPER FUNCTIONS ---
def get_total_trades(pf_instance):
    """Get total trades from portfolio instance"""
    try:
        stats = pf_instance.stats()
        if 'Total Trades' in stats:
            return stats['Total Trades']
        elif 'Total Closed Trades' in stats:
            return stats['Total Closed Trades']
        else:
            return 0
    except:
        return 0

def get_performance_metric(pf_instance, metric_name):
    """Get performance metric value from portfolio instance"""
    try:
        stats = pf_instance.stats()
        
        metric_map = {
            "Total Return [%]": "Total Return [%]",
            "Sharpe Ratio": "Sharpe Ratio",
            "Calmar Ratio": "Calmar Ratio", 
            "Win Rate [%]": "Win Rate [%]",
            "Profit Factor": "Profit Factor",
            "Sortino Ratio": "Sortino Ratio"
        }
        
        stat_key = metric_map.get(metric_name, "Total Return [%]")
        return stats.get(stat_key, 0)
    except:
        return 0

def get_drawdown_data(pf_instance):
    """Get detailed drawdown data"""
    try:
        if hasattr(pf_instance, 'drawdowns') and hasattr(pf_instance.drawdowns, 'records_readable'):
            drawdown_data = pf_instance.drawdowns.records_readable
            if not drawdown_data.empty:
                column_map = {}
                if 'drawdown' in drawdown_data.columns:
                    column_map['drawdown'] = 'drawdown_pct'
                elif 'Drawdown' in drawdown_data.columns:
                    column_map['Drawdown'] = 'drawdown_pct'
                
                if column_map:
                    drawdown_data = drawdown_data.rename(columns=column_map)
                
                if 'drawdown_pct' not in drawdown_data.columns:
                    if 'drawdown' in drawdown_data.columns:
                        drawdown_data['drawdown_pct'] = drawdown_data['drawdown']
                    else:
                        drawdown_data['drawdown_pct'] = 0
                
                return drawdown_data
        
        # Fallback: calculate drawdown from equity curve
        equity_data = get_equity_curve(pf_instance)
        if len(equity_data) > 1:
            running_max = equity_data.cummax()
            drawdown_series = (equity_data - running_max) / running_max * 100
            
            # Find drawdown periods
            drawdown_records = []
            in_drawdown = False
            current_peak = None
            current_valley = None
            peak_date = None
            valley_date = None
            
            for i, (date, dd_value) in enumerate(drawdown_series.items()):
                if not in_drawdown and dd_value < -1:
                    in_drawdown = True
                    current_peak = running_max.iloc[i]
                    current_valley = equity_data.iloc[i]
                    peak_date = date
                    valley_date = date
                elif in_drawdown:
                    if equity_data.iloc[i] < current_valley:
                        current_valley = equity_data.iloc[i]
                        valley_date = date
                        current_drawdown = (current_valley - current_peak) / current_peak * 100
                    elif dd_value >= -0.5:
                        in_drawdown = False
                        drawdown_records.append({
                            'peak_date': peak_date,
                            'valley_date': valley_date,
                            'recovery_date': date,
                            'drawdown_pct': abs(current_drawdown)
                        })
            
            if drawdown_records:
                return pd.DataFrame(drawdown_records)
        
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

def get_equity_curve(pf_instance):
    """Get equity curve data from portfolio"""
    try:
        if hasattr(pf_instance, 'value'):
            equity = pf_instance.value
            if hasattr(equity, '__call__'):
                equity = equity()
            return equity
        
        elif hasattr(pf_instance, 'get_cumulative_returns'):
            equity = pf_instance.get_cumulative_returns()
            if hasattr(equity, '__call__'):
                equity = equity()
            return equity
        
        elif hasattr(pf_instance, 'cumulative_returns'):
            equity = pf_instance.cumulative_returns
            if hasattr(equity, '__call__'):
                equity = equity()
            return (equity + 1) * pf_instance.initial_cash if hasattr(pf_instance, 'initial_cash') else (equity + 1) * 100
        
        else:
            return pd.Series([100], index=[pd.Timestamp.now()])
            
    except Exception as e:
        return pd.Series([100], index=[pd.Timestamp.now()])

# --- MAIN OPTIMIZATION LOGIC ---
if optimize_btn:
    if start_date_input >= end_date_input:
        st.error("❌ Error: Start date must be before end date.")
    else:
        data = load_price_data(asset_class, symbol, tf, start_date_input, end_date_input)
        
        if data is not None and not data.empty:
            price = data['Close']
            
            if strategy_type == "MA Cross":
                if not fasts or not slows:
                    st.warning("⚠️ Please select at least one value for Fast MA and Slow MA.")
                else:
                    total_combinations = len(fasts) * len(slows)
                    st.info(f"🔄 Optimizing {len(fasts)}x{len(slows)} = {total_combinations} scenarios...")
                    
                    # Run strategy with product
                    entries, exits, param_combinations = run_ma_cross_strategy(price, fasts, slows)
                    
                    if entries is not None and len(param_combinations) > 0:
                        with st.spinner("📊 Creating portfolio and calculating performance..."):
                            try:
                                # Create portfolio
                                pf = vbt.Portfolio.from_signals(
                                    price, 
                                    entries, 
                                    exits,
                                    fees=0.001,
                                    freq=tf.upper().replace('M', 'T').replace('1W', 'W-MON')
                                )
                                
                                # --- CALCULATE AND FILTER RESULTS ---
                                st.success(f"✅ Successfully created {len(param_combinations)} portfolios!")
                                
                                # Collect results
                                results = []
                                for idx in param_combinations:
                                    try:
                                        pf_instance = pf[idx]
                                        trades_count = get_total_trades(pf_instance)
                                        perf_value = get_performance_metric(pf_instance, target_metric)
                                        stats = pf_instance.stats()
                                        
                                        results.append({
                                            'fast': idx[0],
                                            'slow': idx[1],
                                            'trades': trades_count,
                                            'performance': perf_value,
                                            'total_return': stats.get('Total Return [%]', 0),
                                            'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                                            'win_rate': stats.get('Win Rate [%]', 0),
                                            'max_drawdown': stats.get('Max Drawdown [%]', 0),
                                            'profit_factor': stats.get('Profit Factor', 0),
                                            'pf_instance': pf_instance
                                        })
                                    except Exception as e:
                                        continue
                                
                                if not results:
                                    st.error("❌ Could not calculate results for any parameter pairs.")
                                    st.stop()
                                
                                # Convert to DataFrame
                                results_df = pd.DataFrame(results)
                                
                                # Filter by minimum trades
                                filtered_results = results_df[results_df['trades'] >= min_trades]
                                
                                if len(filtered_results) == 0:
                                    st.warning(f"⚠️ No results found with at least {min_trades} trades. Showing all results.")
                                    filtered_results = results_df
                                
                                # Sort by performance metric
                                sorted_results = filtered_results.sort_values('performance', ascending=False)
                                
                                # --- DISPLAY RESULTS ---
                                st.markdown("## 🏆 OPTIMIZATION RESULTS")
                                
                                # Get best result
                                best_result = sorted_results.iloc[0]
                                best_fast = best_result['fast']
                                best_slow = best_result['slow']
                                best_performance = best_result['performance']
                                best_pf = best_result['pf_instance']
                                best_stats = best_pf.stats()
                                
                                # Display metrics in cards
                                col1, col2, col3, col4 = st.columns(4)
                                col1.markdown(f"""
                                <div class="status-card">
                                    <div class="status-icon">🎯</div>
                                    <div class="metric-value">{int(best_fast)}/{int(best_slow)}</div>
                                    <div class="metric-label">Best MA Pair</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col2.markdown(f"""
                                <div class="status-card">
                                    <div class="status-icon">📊</div>
                                    <div class="metric-value">{best_performance:.2f}</div>
                                    <div class="metric-label">{target_metric}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col3.markdown(f"""
                                <div class="status-card">
                                    <div class="status-icon">💰</div>
                                    <div class="metric-value">{best_result['total_return']:.1f}%</div>
                                    <div class="metric-label">Total Return</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col4.markdown(f"""
                                <div class="status-card">
                                    <div class="status-icon">📈</div>
                                    <div class="metric-value">{best_result['trades']:.0f}</div>
                                    <div class="metric-label">Total Trades</div>
                                </div>
                                """, unsafe_allow_html=True)

                                # --- ADVANCED ANALYSIS TABS ---
                                tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Heatmap", "📈 Equity Curve", "🔍 Top Results", "📋 Risk Analysis"])
                                
                                with tab1:
                                    st.markdown('<div class="config-section">', unsafe_allow_html=True)
                                    st.markdown('<div class="config-header">📈 PERFORMANCE HEATMAP</div>', unsafe_allow_html=True)
                                    try:
                                        # Create heatmap from results
                                        heatmap_data = []
                                        for _, row in results_df.iterrows():
                                            heatmap_data.append({
                                                'Fast MA': row['fast'],
                                                'Slow MA': row['slow'], 
                                                'Value': row['performance']
                                            })
                                        
                                        heatmap_df = pd.DataFrame(heatmap_data)
                                        pivot_df = heatmap_df.pivot(index='Slow MA', columns='Fast MA', values='Value')
                                        
                                        fig = px.imshow(
                                            pivot_df,
                                            title=f"Performance Heatmap - {symbol} | {target_metric}",
                                            color_continuous_scale='RdYlGn',
                                            aspect='auto',
                                            labels={'color': target_metric}
                                        )
                                        fig.update_layout(height=500, template="plotly_dark")
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"❌ Error creating heatmap: {e}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with tab2:
                                    st.markdown('<div class="config-section">', unsafe_allow_html=True)
                                    st.markdown('<div class="config-header">📊 EQUITY CURVE ANALYSIS</div>', unsafe_allow_html=True)
                                    try:
                                        equity_data = get_equity_curve(best_pf)
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=equity_data.index,
                                            y=equity_data.values,
                                            name="Equity Curve",
                                            line=dict(color="#00D4AA", width=3)
                                        ))
                                        fig.update_layout(
                                            title=f"Equity Curve - {symbol} (MA{best_fast}/{best_slow})",
                                            xaxis_title="Time",
                                            yaxis_title="Portfolio Value",
                                            template="plotly_dark",
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"❌ Error creating equity curve: {e}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with tab3:
                                    st.markdown('<div class="config-section">', unsafe_allow_html=True)
                                    st.markdown('<div class="config-header">🔝 TOP 5 OPTIMIZATION RESULTS</div>', unsafe_allow_html=True)
                                    top_5_results = sorted_results.head(5)
                                    
                                    display_df = top_5_results[['fast', 'slow', 'performance', 'total_return', 'sharpe_ratio', 'trades']].copy()
                                    display_df.columns = ['Fast MA', 'Slow MA', target_metric, 'Total Return %', 'Sharpe Ratio', 'Total Trades']
                                    
                                    st.dataframe(
                                        display_df.style.format({
                                            'Total Return %': '{:.2f}%',
                                            'Sharpe Ratio': '{:.2f}',
                                            target_metric: '{:.2f}'
                                        }),
                                        use_container_width=True
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with tab4:
                                    st.markdown('<div class="config-section">', unsafe_allow_html=True)
                                    st.markdown('<div class="config-header">📋 DETAILED RISK ANALYSIS</div>', unsafe_allow_html=True)
                                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                                    
                                    with risk_col1:
                                        st.metric("Max Drawdown", f"{best_result['max_drawdown']:.2f}%")
                                    with risk_col2:
                                        st.metric("Sharpe Ratio", f"{best_result['sharpe_ratio']:.2f}")
                                    with risk_col3:
                                        st.metric("Win Rate", f"{best_result['win_rate']:.2f}%")
                                    with risk_col4:
                                        st.metric("Profit Factor", f"{best_result['profit_factor']:.2f}")
                                    
                                    # Detailed drawdown analysis
                                    st.subheader("📉 Detailed Drawdown Analysis")
                                    try:
                                        drawdown_data = get_drawdown_data(best_pf)
                                        if not drawdown_data.empty and 'drawdown_pct' in drawdown_data.columns:
                                            # Show top 3 largest drawdowns
                                            worst_drawdowns = drawdown_data.nlargest(3, 'drawdown_pct')
                                            
                                            # Prepare display data
                                            display_dd = worst_drawdowns[['peak_date', 'valley_date', 'drawdown_pct']].copy()
                                            display_dd['drawdown_pct'] = display_dd['drawdown_pct'].round(2)
                                            display_dd.columns = ['Peak Date', 'Valley Date', 'Drawdown %']
                                            
                                            st.dataframe(display_dd, use_container_width=True)
                                            
                                            # Draw drawdown chart
                                            st.subheader("📊 Drawdown Chart")
                                            equity_data = get_equity_curve(best_pf)
                                            running_max = equity_data.cummax()
                                            drawdown_series = (equity_data - running_max) / running_max * 100
                                            
                                            fig_dd = go.Figure()
                                            fig_dd.add_trace(go.Scatter(
                                                x=drawdown_series.index,
                                                y=drawdown_series.values,
                                                name="Drawdown",
                                                line=dict(color="#FF6B6B", width=2),
                                                fill='tozeroy'
                                            ))
                                            fig_dd.update_layout(
                                                title=f"Drawdown Analysis - {symbol}",
                                                xaxis_title="Time",
                                                yaxis_title="Drawdown (%)",
                                                template="plotly_dark",
                                                height=400
                                            )
                                            st.plotly_chart(fig_dd, use_container_width=True)
                                        else:
                                            st.info("ℹ️ No detailed drawdown data available for display")
                                    except Exception as e:
                                        st.info("ℹ️ No detailed drawdown data available for display")
                                    st.markdown('</div>', unsafe_allow_html=True)

                                # --- DOWNLOAD RESULTS ---
                                with st.expander("💾 DOWNLOAD OPTIMIZATION RESULTS"):
                                    try:
                                        csv = results_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download All Results (CSV)",
                                            data=csv,
                                            file_name=f"optimization_results_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                            mime="text/csv"
                                        )
                                    except Exception as e:
                                        st.error(f"❌ Error exporting results: {e}")

                            except Exception as e:
                                st.error(f"❌ Error creating portfolio: {e}")
                    else:
                        st.warning("⚠️ No valid parameter combinations found (Fast MA must be less than Slow MA).")
            else:
                st.warning(f"⚠️ {strategy_type} strategy not fully implemented in this version.")
        else:
            st.warning("⚠️ No data available for optimization.")
else:
    st.info("👆 Please configure parameters and click 'START ADVANCED OPTIMIZATION' to begin.")

# --- Footer ---
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Strategy Optimization Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Advanced Strategy Optimizer v2.0</p>
</div>
""", unsafe_allow_html=True)