import streamlit as st
import pandas as pd
import yfinance as yf
import time
import ccxt
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from datetime import datetime, timedelta
import requests
import re
from textblob import TextBlob

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
    .badge-stock { background: linear-gradient(135deg, #28a745, #7ae582); color: white; }
    .badge-crypto { background: linear-gradient(135deg, #f7931a, #ffc46c); color: black; }
    .badge-forex { background: linear-gradient(135deg, #007bff, #6cb2ff); color: white; }
    .badge-profit { background: linear-gradient(135deg, #00ff88, #00cc6a); color: white; }
    .badge-loss { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
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
    .position-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .position-card:hover {
        transform: translateY(-2px);
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    .position-card.buy { border-left: 4px solid #00ff88; }
    .position-card.sell { border-left: 4px solid #ff4444; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header-gradient">🚀 Live Trading Pro</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #8898aa; font-size: 1.2rem; margin-bottom: 3rem;">Professional Algorithmic Trading Platform</div>', unsafe_allow_html=True)

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
        data = yf.download(yahoo_symbol, period='6mo', interval='1h')
        return data
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def load_data_for_live(symbol, asset_type):
    try:
        if asset_type == "Crypto":
            # Sử dụng hàm mới với multiple exchanges
            df = get_crypto_data_simple(symbol, '1d', 60)
            
            if df is not None and not df.empty:
                return df
            else:
                # Fallback to yfinance
                formatted_symbol = f"{symbol.replace('USD', '')}-USD"
                data = yf.download(formatted_symbol, period="60d", interval='1d', progress=False, auto_adjust=True)
                if data.empty:
                    # Thử định dạng khác
                    data = yf.download(symbol, period="60d", interval='1d', progress=False, auto_adjust=True)
                return data
        else:
            data = yf.download(symbol, period="2y", interval='1d', progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).capitalize() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu cho {symbol}: {e}")
        return None

# --- Lớp AlpacaTrader ĐÃ SỬA LỖI FRACTIONAL ORDERS ---
class AlpacaTrader:
    def __init__(self, api_key, api_secret, paper=True):
        self.api, self.account, self.connected = None, None, False
        try:
            self.api = TradingClient(api_key, api_secret, paper=paper)
            self.account = self.api.get_account()
            self.connected = True
        except Exception as e:
            st.error(f"Error connecting to Alpaca: {e}")

    def get_account_info(self): 
        return self.api.get_account()
    
    def get_positions(self): 
        return self.api.get_all_positions()
    
    def get_portfolio_history(self, period="1M"):
        """Lấy lịch sử portfolio từ Alpaca"""
        try:
            params = GetPortfolioHistoryRequest(period=period)
            history = self.api.get_portfolio_history(params)
            return history
        except Exception as e:
            st.error(f"Error getting portfolio history: {e}")
            return None

    def place_order(self, symbol, qty, side, asset_type):
        try:
            # Định dạng symbol theo loại tài sản
            formatted_symbol = self._format_symbol(symbol, asset_type)
            
            st.info(f"🔄 Placing {side.upper()} order for {qty} {formatted_symbol}...")
            
            # SỬA LỖI: Fractional orders must be DAY orders
            # Xác định xem có phải fractional order không
            is_fractional = qty != int(qty)
            
            # Sử dụng time_in_force phù hợp
            time_in_force = TimeInForce.DAY
            
            market_order_data = MarketOrderRequest(
                symbol=formatted_symbol, 
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=time_in_force  # SỬA LỖI Ở ĐÂY
            )
            order = self.api.submit_order(order_data=market_order_data)
            st.success(f"✅ Successfully submitted {side.upper()} order for {qty} units of {formatted_symbol}!")
            return order
        except Exception as e:
            st.error(f"❌ Error placing order: {e}")
            st.error(f"Symbol attempted: {formatted_symbol}, Asset Type: {asset_type}, Quantity: {qty}")
            return None

    def _format_symbol(self, symbol, asset_type):
        """Định dạng symbol theo loại tài sản"""
        symbol = symbol.upper().strip()
        
        if asset_type == "Crypto":
            # Xử lý các định dạng symbol crypto phổ biến
            if '/' in symbol:
                # Định dạng: BTC/USD -> BTCUSD
                return symbol.replace('/', '')
            elif symbol.endswith('USDT'):
                # Định dạng: BTCUSDT -> BTCUSD
                return symbol.replace('USDT', 'USD')
            elif symbol.endswith('USD'):
                # Đã đúng định dạng
                return symbol
            else:
                # Thêm USD nếu chưa có
                return f"{symbol}USD"
                
        elif asset_type == "Forex":
            if '/' in symbol:
                # Định dạng: EUR/USD -> EURUSD
                return symbol.replace('/', '')
            elif len(symbol) == 6:
                # Định dạng: EURUSD -> EURUSD (giữ nguyên)
                return symbol
            else:
                # Giả sử là cặp forex chuẩn
                return symbol
                
        else:  # Stocks
            # Giữ nguyên cho stocks
            return symbol

# --- ADVANCED RISK MANAGEMENT DASHBOARD ---
class RiskManager:
    def __init__(self, trader):
        self.trader = trader
        
    def calculate_var(self, positions, confidence_level=0.95, periods=252):
        """Tính Value at Risk"""
        try:
            portfolio_value = float(self.trader.get_account_info().portfolio_value)
            annual_volatility = 0.20
            daily_volatility = annual_volatility / np.sqrt(periods)
            
            var_1d = portfolio_value * daily_volatility * 2.33
            var_1w = var_1d * np.sqrt(5)
            var_1m = var_1d * np.sqrt(21)
            
            return {
                '1d': var_1d,
                '1w': var_1w,
                '1m': var_1m
            }
        except Exception as e:
            st.error(f"Error calculating VaR: {e}")
            return {'1d': 0, '1w': 0, '1m': 0}
    
    def calculate_max_drawdown(self, portfolio_history):
        """Tính maximum drawdown"""
        try:
            if portfolio_history and hasattr(portfolio_history, 'equity'):
                equity = portfolio_history.equity
                if equity and len(equity) > 0:
                    peak = equity[0]
                    max_dd = 0
                    
                    for value in equity:
                        if value > peak:
                            peak = value
                        dd = (peak - value) / peak
                        if dd > max_dd:
                            max_dd = dd
                    
                    return max_dd * 100
            return 0
        except Exception as e:
            st.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def calculate_sharpe_ratio(self, returns_series=None):
        """Tính Sharpe ratio"""
        try:
            annual_return = 0.08
            risk_free_rate = 0.02
            annual_volatility = 0.20
            
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return sharpe
        except Exception as e:
            st.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def calculate_drawdown_series(self, portfolio_history):
        """Tính drawdown series cho biểu đồ"""
        try:
            if portfolio_history and hasattr(portfolio_history, 'equity'):
                equity = portfolio_history.equity
                if equity and len(equity) > 0:
                    peak = equity[0]
                    drawdowns = []
                    
                    for value in equity:
                        if value > peak:
                            peak = value
                        dd = (peak - value) / peak * 100
                        drawdowns.append(dd)
                    
                    return drawdowns
            return [0]
        except Exception as e:
            st.error(f"Error calculating drawdown series: {e}")
            return [0]

def create_risk_dashboard(trader):
    """Create comprehensive risk management dashboard"""
    
    risk_manager = RiskManager(trader)
    
    portfolio_var = risk_manager.calculate_var(trader.get_positions())
    portfolio_history = trader.get_portfolio_history()
    max_drawdown = risk_manager.calculate_max_drawdown(portfolio_history)
    sharpe_ratio = risk_manager.calculate_sharpe_ratio()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Value at Risk', 'Portfolio Drawdown', 'Risk Metrics', 'Position Concentration'),
        specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "indicator"}, {"type": "pie"}]]
    )
    
    fig.add_trace(go.Bar(
        x=['1 Day', '1 Week', '1 Month'],
        y=[portfolio_var['1d'], portfolio_var['1w'], portfolio_var['1m']],
        name='Value at Risk',
        marker_color=['#ff6b6b', '#ffa726', '#66bb6a']
    ), row=1, col=1)
    
    drawdown_series = risk_manager.calculate_drawdown_series(portfolio_history)
    fig.add_trace(go.Scatter(
        y=drawdown_series,
        name='Portfolio Drawdown',
        line=dict(color='red'),
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.1)'
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = sharpe_ratio,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sharpe Ratio"},
        gauge = {
            'axis': {'range': [None, 3]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 2], 'color': "gray"},
                {'range': [2, 3], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.5}}
    ), row=2, col=1)
    
    positions = trader.get_positions()
    if positions:
        symbols = [p.symbol for p in positions]
        values = [float(p.market_value) for p in positions]
        fig.add_trace(go.Pie(
            labels=symbols,
            values=values,
            name="Position Concentration"
        ), row=2, col=2)
    
    fig.update_layout(
        title='Risk Management Dashboard',
        height=600,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig, {
        'var_1d': portfolio_var['1d'],
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

# --- PERFORMANCE ANALYTICS & REPORTING ---
class TradeJournal:
    def __init__(self):
        self.trades = []
    
    def add_trade(self, trade_data):
        self.trades.append(trade_data)
    
    def get_timeline(self):
        return [t['timestamp'] for t in self.trades]
    
    def get_equity_curve(self):
        equity = 10000
        curve = []
        for trade in self.trades:
            equity += trade.get('pnl', 0)
            curve.append(equity)
        return curve

class PerformanceAnalytics:
    def __init__(self, trader):
        self.trader = trader
        self.trade_journal = TradeJournal()
    
    def calculate_daily_pnl(self):
        try:
            positions = self.trader.get_positions()
            daily_pnl = sum(float(p.unrealized_pl) for p in positions)
            return daily_pnl
        except:
            return 0
    
    def calculate_win_rate(self):
        if len(self.trade_journal.trades) == 0:
            return 0
        winning_trades = sum(1 for trade in self.trade_journal.trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(self.trade_journal.trades)) * 100
    
    def calculate_profit_factor(self):
        gross_profit = sum(trade.get('pnl', 0) for trade in self.trade_journal.trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in self.trade_journal.trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf')
        return gross_profit / gross_loss
    
    def calculate_avg_trade_duration(self):
        if len(self.trade_journal.trades) == 0:
            return "N/A"
        
        durations = []
        for trade in self.trade_journal.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration)
        
        if durations:
            avg_hours = sum(durations) / len(durations)
            return f"{avg_hours:.1f} hours"
        return "N/A"
    
    def identify_best_strategy(self):
        strategies = {}
        for trade in self.trade_journal.trades:
            strategy = trade.get('strategy', 'Unknown')
            pnl = trade.get('pnl', 0)
            if strategy not in strategies:
                strategies[strategy] = {'total_pnl': 0, 'count': 0}
            strategies[strategy]['total_pnl'] += pnl
            strategies[strategy]['count'] += 1
        
        if strategies:
            best_strategy = max(strategies.items(), key=lambda x: x[1]['total_pnl'])
            return best_strategy[0]
        return "No strategies"

    def generate_daily_report(self):
        report = {
            'daily_pnl': self.calculate_daily_pnl(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'avg_trade_duration': self.calculate_avg_trade_duration(),
            'sharpe_ratio': RiskManager(self.trader).calculate_sharpe_ratio(),
            'max_drawdown': RiskManager(self.trader).calculate_max_drawdown(self.trader.get_portfolio_history()),
            'best_performing_strategy': self.identify_best_strategy(),
            'total_trades': len(self.trade_journal.trades)
        }
        
        return report
    
    def create_performance_charts(self, report):
        fig_equity = go.Figure()
        if self.trade_journal.get_equity_curve():
            fig_equity.add_trace(go.Scatter(
                x=self.trade_journal.get_timeline(),
                y=self.trade_journal.get_equity_curve(),
                name='Equity Curve',
                line=dict(color='#00ff00', width=3)
            ))
        
        fig_gauges = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=('Win Rate', 'Profit Factor', 'Sharpe Ratio')
        )
        
        fig_gauges.add_trace(go.Indicator(
            mode="gauge+number",
            value=report['win_rate'],
            title={'text': "Win Rate %"},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "green"},
                  'steps': [{'range': [0, 50], 'color': "red"},
                           {'range': [50, 75], 'color': "yellow"},
                           {'range': [75, 100], 'color': "green"}]}
        ), row=1, col=1)
        
        fig_gauges.add_trace(go.Indicator(
            mode="number",
            value=report['profit_factor'],
            title={'text': "Profit Factor"}
        ), row=1, col=2)
        
        fig_gauges.add_trace(go.Indicator(
            mode="number",
            value=report['sharpe_ratio'],
            title={'text': "Sharpe Ratio"}
        ), row=1, col=3)
        
        fig_gauges.update_layout(height=300, template="plotly_dark")
        
        return fig_equity, fig_gauges

# --- SOCIAL SENTIMENT INTEGRATION ---
class SocialSentimentAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def analyze_twitter_sentiment(self, symbol):
        try:
            base_score = np.random.normal(0.6, 0.3)
            score = max(-1, min(1, base_score))
            
            return {
                'score': score,
                'volume': np.random.randint(100, 10000),
                'positive_ratio': max(0, min(1, (score + 1) / 2))
            }
        except Exception as e:
            return {'score': 0, 'volume': 0, 'positive_ratio': 0.5}
    
    def analyze_reddit_sentiment(self, symbol):
        try:
            base_score = np.random.normal(0.4, 0.4)
            score = max(-1, min(1, base_score))
            
            return {
                'score': score,
                'volume': np.random.randint(50, 5000),
                'positive_ratio': max(0, min(1, (score + 1) / 2))
            }
        except Exception as e:
            return {'score': 0, 'volume': 0, 'positive_ratio': 0.5}
    
    def calculate_sentiment_trend(self, symbol):
        trends = ['📈 Improving', '📉 Declining', '➡️ Stable']
        return np.random.choice(trends)
    
    def get_social_sentiment(self, symbol):
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        twitter_sentiment = self.analyze_twitter_sentiment(symbol)
        reddit_sentiment = self.analyze_reddit_sentiment(symbol)
        
        combined_sentiment = (
            twitter_sentiment['score'] * 0.6 + 
            reddit_sentiment['score'] * 0.4
        )
        
        result = {
            'combined_score': combined_sentiment,
            'twitter_volume': twitter_sentiment['volume'],
            'reddit_volume': reddit_sentiment['volume'],
            'sentiment_trend': self.calculate_sentiment_trend(symbol),
            'sentiment_label': 'Bullish' if combined_sentiment > 0.1 else 'Bearish' if combined_sentiment < -0.1 else 'Neutral',
            'confidence': min(100, abs(combined_sentiment) * 100)
        }
        
        self.cache[cache_key] = result
        return result

# --- Các hàm hỗ trợ ---
@st.cache_resource
def train_model_on_the_fly(data):
    if data is None or len(data) < 100:
        return None
    
    try:
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        
        data['Future_Return'] = data['Close'].shift(-5) / data['Close'] - 1
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
        st.error(f"Lỗi training model: {e}")
        return None

def get_ml_signal(data, model):
    if data is None or model is None:
        return "NO_SIGNAL", 0
    
    try:
        latest = data.iloc[-1].copy()
        latest['SMA_20'] = data['Close'].rolling(20).mean().iloc[-1]
        latest['SMA_50'] = data['Close'].rolling(50).mean().iloc[-1]
        latest['RSI'] = ta.rsi(data['Close'], length=14).iloc[-1]
        latest['Volume_SMA'] = data['Volume'].rolling(20).mean().iloc[-1]
        
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
        st.error(f"Lỗi get ML signal: {e}")
        return "NO_SIGNAL", 0

def get_asset_badge(asset_type):
    """Trả về badge CSS class cho từng loại tài sản"""
    if asset_type == "Crypto":
        return "badge-crypto"
    elif asset_type == "Forex":
        return "badge-forex"
    else:
        return "badge-stock"

def display_position(position):
    symbol = position.symbol
    qty = float(position.qty)
    avg_entry = float(position.avg_entry_price)
    current_price = float(position.current_price)
    unrealized_pl = float(position.unrealized_pl)
    pl_percent = (unrealized_pl / (avg_entry * qty)) * 100 if avg_entry * qty != 0 else 0
    
    pl_class = "" if unrealized_pl >= 0 else "sell"
    badge_class = "badge-profit" if unrealized_pl >= 0 else "badge-loss"
    
    st.markdown(f"""
    <div class="position-item {pl_class}">
        <div style="display: flex; justify-content: between; align-items: center;">
            <div>
                <strong>{symbol}</strong>
                <span class="badge {badge_class}">{qty:.4f} shares</span>
            </div>
            <div style="text-align: right;">
                <div style="color: {'#00ff88' if unrealized_pl >= 0 else '#ff4444'}; font-weight: bold;">
                    ${unrealized_pl:+.2f} ({pl_percent:+.1f}%)
                </div>
                <div style="font-size: 0.8em; color: #8898aa;">
                    Avg: ${avg_entry:.4f} • Current: ${current_price:.4f}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Session State ---
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'performance_analytics' not in st.session_state:
    st.session_state.performance_analytics = None
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SocialSentimentAnalyzer()
if 'selected_position' not in st.session_state:
    st.session_state.selected_position = None
if 'suggested_qty' not in st.session_state:
    st.session_state.suggested_qty = 1.0
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>🚀</h1>
        <h2 style='color: white; font-size: 1.2rem; margin: 0;'>Live Trading Pro</h2>
        <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>Professional Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="dashboard-card">
        <h3>🔌 Platform Connection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    account_type = st.radio("Account Type:", ["Paper Trading", "Live Trading"])
    api_key = st.text_input("API Key", type="password", key="api_key_input")
    api_secret = st.text_input("API Secret", type="password", key="api_secret_input")

    if st.button("Connect", use_container_width=True, type="primary"):
        if api_key and api_secret:
            with st.spinner("Connecting..."):
                st.session_state.trader = AlpacaTrader(api_key.strip(), api_secret.strip(), paper=(account_type == 'Paper Trading'))
                if st.session_state.trader.connected:
                    st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                    st.success(f"✅ Connected to {account_type}!")
                else:
                    st.error("❌ Connection failed")
        else:
            st.warning("Please enter API credentials")

    if st.session_state.trader and st.session_state.trader.connected:
        st.success(f"✅ Connected to {account_type}!")
        
        st.markdown("""
        <div class="dashboard-card">
            <h3>⚡ Quick Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    else:
        st.info("Enter API credentials to begin")

# --- Main Content ---
if st.session_state.trader and st.session_state.trader.connected:
    trader = st.session_state.trader
    performance_analytics = st.session_state.performance_analytics
    sentiment_analyzer = st.session_state.sentiment_analyzer

    # Status Cards
    try:
        account = trader.get_account_info()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="feature-icon">💰</div>
                <div class="metric-value">${float(account.portfolio_value):,.2f}</div>
                <div class="metric-label">Portfolio Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="feature-icon">⚡</div>
                <div class="metric-value">${float(account.buying_power):,.2f}</div>
                <div class="metric-label">Buying Power</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            positions = trader.get_positions()
            total_positions = len(positions) if positions else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="feature-icon">📈</div>
                <div class="metric-value">{total_positions}</div>
                <div class="metric-label">Open Positions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            daily_pnl = performance_analytics.calculate_daily_pnl()
            st.markdown(f"""
            <div class="metric-card">
                <div class="feature-icon">💹</div>
                <div class="metric-value">${daily_pnl:,.2f}</div>
                <div class="metric-label">Daily P&L</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Cannot load account information: {e}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Tabs với UI/UX đồng bộ - ĐÃ SỬA LỖI HIỂN THỊ
    tab_titles = ["📊 Overview", "📈 Positions", "🛠️ Manual Trading", "🤖 Automated Trading", 
                  "📉 Risk Dashboard", "📊 Performance Analytics", "🎭 Social Sentiment"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_titles)

    # Tab 1: Overview
    with tab1:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📊 Account Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            account = trader.get_account_info()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${float(account.portfolio_value):,.2f}</div>
                    <div class="metric-label">Portfolio Value</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${float(account.buying_power):,.2f}</div>
                    <div class="metric-label">Buying Power</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${float(account.cash):,.2f}</div>
                    <div class="metric-label">Available Cash</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                status_color = "#00ff88" if account.status.value == "ACTIVE" else "#ff4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {status_color}">{account.status.value.upper()}</div>
                    <div class="metric-label">Account Status</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick Stats
            st.markdown("""
            <div class="dashboard-card">
                <h3>📈 Quick Stats</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            try:
                positions = trader.get_positions()
                total_positions = len(positions) if positions else 0
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_positions}</div>
                        <div class="metric-label">Total Positions</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    daily_pnl = performance_analytics.calculate_daily_pnl()
                    pnl_color = "#00ff88" if daily_pnl >= 0 else "#ff4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {pnl_color}">${daily_pnl:,.2f}</div>
                        <div class="metric-label">Daily P&L</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_trades = len(performance_analytics.trade_journal.trades)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_trades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error loading quick stats: {e}")
                
        except Exception as e:
            st.error(f"Cannot load account information: {e}")

    # Tab 2: Positions
    with tab2:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📈 Current Positions - Click to Trade</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Positions", key="refresh_positions", use_container_width=True):
            st.rerun()
            
        try:
            positions = trader.get_positions()
            if positions:
                st.info("💡 Click on any position below to automatically fill trading form")
                
                # Group positions by profit/loss
                profitable_positions = [p for p in positions if float(p.unrealized_pl) > 0]
                losing_positions = [p for p in positions if float(p.unrealized_pl) <= 0]
                
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
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"📈 Buy More", key=f"buy_{position.symbol}", use_container_width=True):
                                    st.session_state.selected_position = {
                                        'symbol': position.symbol,
                                        'action': 'buy',
                                        'current_qty': float(position.qty),
                                        'asset_type': "Crypto" if "USD" in position.symbol and len(position.symbol) in [6,7] else "Stocks"
                                    }
                                    st.success(f"Selected {position.symbol} for BUY - check Manual Trading tab!")
                            with col2:
                                if st.button(f"📉 Sell", key=f"sell_{position.symbol}", use_container_width=True):
                                    st.session_state.selected_position = {
                                        'symbol': position.symbol,
                                        'action': 'sell', 
                                        'current_qty': float(position.qty),
                                        'asset_type': "Crypto" if "USD" in position.symbol and len(position.symbol) in [6,7] else "Stocks"
                                    }
                                    st.success(f"Selected {position.symbol} for SELL - check Manual Trading tab!")
                
                with col_pos2:
                    if losing_positions:
                        st.markdown("""
                        <div class="position-group">
                            <h4>🔴 Losing Positions</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for position in losing_positions:
                            display_position(position)
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"📈 Buy More", key=f"buy_l_{position.symbol}", use_container_width=True):
                                    st.session_state.selected_position = {
                                        'symbol': position.symbol,
                                        'action': 'buy',
                                        'current_qty': float(position.qty),
                                        'asset_type': "Crypto" if "USD" in position.symbol and len(position.symbol) in [6,7] else "Stocks"
                                    }
                                    st.success(f"Selected {position.symbol} for BUY - check Manual Trading tab!")
                            with col2:
                                if st.button(f"📉 Sell", key=f"sell_l_{position.symbol}", use_container_width=True):
                                    st.session_state.selected_position = {
                                        'symbol': position.symbol,
                                        'action': 'sell', 
                                        'current_qty': float(position.qty),
                                        'asset_type': "Crypto" if "USD" in position.symbol and len(position.symbol) in [6,7] else "Stocks"
                                    }
                                    st.success(f"Selected {position.symbol} for SELL - check Manual Trading tab!")
                
            else:
                st.info("💰 No open positions. Start trading to see your positions here!")
                
        except Exception as e:
            st.error(f"Cannot load positions list: {e}")

    # Tab 3: Manual Trading - ĐÃ SỬA LỖI STREAMLIT MIXED NUMERIC TYPES
    with tab3:
        st.markdown("""
        <div class="dashboard-card">
            <h3>🛠️ Manual Trading</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị thông báo nếu có position được chọn
        if st.session_state.selected_position:
            selected = st.session_state.selected_position
            badge_class = get_asset_badge(selected.get('asset_type', 'Stocks'))
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="margin: 0; color: white;">🎯 Trading {selected['symbol']} - Current: {selected['current_qty']:.4f} shares - Action: {selected['action'].upper()}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Tự động điền thông tin
            default_symbol = selected['symbol']
            default_action = selected['action']
            default_asset_type = selected.get('asset_type', 'Stocks')
            suggested_qty = selected['current_qty'] * 0.1  # 10% của position hiện tại
        else:
            default_symbol = "BTCUSD"
            default_action = "buy"
            default_asset_type = "Crypto"
            suggested_qty = 0.01
        
        # Trading form với UI đồng bộ
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="dashboard-card">
                <h4>🎯 Trade Configuration</h4>
            </div>
            """, unsafe_allow_html=True)
            
            asset_type = st.radio("Asset Type:", ["Stocks", "Crypto", "Forex"], 
                                 index=["Stocks", "Crypto", "Forex"].index(default_asset_type) if default_asset_type in ["Stocks", "Crypto", "Forex"] else 1,
                                 horizontal=True, key="manual_asset")
            
            symbol = st.text_input("Symbol:", value=default_symbol, key="manual_symbol").upper()
            
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <h4>📊 Order Details</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # SỬA LỖI: StreamlitMixedNumericTypesError - Đảm bảo tất cả giá trị cùng kiểu
            if asset_type == "Crypto":
                min_qty = 0.00001
                step = 0.001
                format_str = "%.5f"
                default_qty = float(0.01)  # Đảm bảo là float
            elif asset_type == "Forex":
                min_qty = 0.01
                step = 0.01
                format_str = "%.2f"
                default_qty = float(1.0)  # Đảm bảo là float
            else:  # Stocks
                min_qty = 1  # Stocks phải là số nguyên
                step = 1
                format_str = "%d"  # Định dạng số nguyên
                default_qty = int(1)  # Đảm bảo là int
            
            current_qty = st.session_state.selected_position['current_qty'] if st.session_state.selected_position else 0
            
            if current_qty > 0:
                st.write(f"📊 **Current Position**: {current_qty:.4f} shares")
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    if st.button("25%", use_container_width=True):
                        if asset_type == "Stocks":
                            st.session_state.suggested_qty = int(current_qty * 0.25)
                        else:
                            st.session_state.suggested_qty = current_qty * 0.25
                with col_q2:
                    if st.button("50%", use_container_width=True):
                        if asset_type == "Stocks":
                            st.session_state.suggested_qty = int(current_qty * 0.5)
                        else:
                            st.session_state.suggested_qty = current_qty * 0.5
                with col_q3:
                    if st.button("100%", use_container_width=True):
                        if asset_type == "Stocks":
                            st.session_state.suggested_qty = int(current_qty)
                        else:
                            st.session_state.suggested_qty = current_qty
            
            # SỬA LỖI QUAN TRỌNG: Xử lý kiểu dữ liệu phù hợp
            current_suggested_qty = st.session_state.get('suggested_qty', suggested_qty)
            
            if asset_type == "Stocks":
                # Đảm bảo giá trị là số nguyên cho Stocks
                if isinstance(current_suggested_qty, float):
                    current_suggested_qty = int(current_suggested_qty)
                qty = st.number_input("Quantity:", 
                                    min_value=min_qty, 
                                    value=current_suggested_qty, 
                                    step=step, 
                                    format=format_str, 
                                    key="manual_qty")
            else:
                # Crypto và Forex dùng số thập phân
                qty = st.number_input("Quantity:", 
                                    min_value=min_qty, 
                                    value=float(current_suggested_qty), 
                                    step=step, 
                                    format=format_str, 
                                    key="manual_qty")
        
        # Hiển thị thông tin thời gian thực
        current_price = None
        if symbol:
            try:
                with st.spinner("Loading real-time data..."):
                    data = load_data_for_live(symbol, asset_type)
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        if len(data) > 1:
                            price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
                        else:
                            price_change = 0
                        
                        # Hiển thị thông tin giá với UI đồng bộ
                        col_price, col_change, col_value = st.columns(3)
                        with col_price:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">${current_price:.4f}</div>
                                <div class="metric-label">Current Price</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_change:
                            change_color = "#00ff88" if price_change >= 0 else "#ff4444"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {change_color}">{price_change:+.2f}%</div>
                                <div class="metric-label">24h Change</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_value:
                            order_value = current_price * qty
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">${order_value:.2f}</div>
                                <div class="metric-label">Order Value</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
            except Exception as e:
                st.error(f"Could not load data for {symbol}: {e}")
        
        # Trading buttons với UI đồng bộ
        st.markdown("""
        <div class="dashboard-card">
            <h3>🎯 Execute Order</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            if st.button("🟢 BUY / ADD MORE", 
                        use_container_width=True, 
                        type="primary",
                        key="manual_buy"):
                
                if not symbol:
                    st.error("❌ Please enter a symbol")
                elif qty <= 0:
                    st.error("❌ Quantity must be greater than 0")
                else:
                    result = trader.place_order(symbol=symbol, qty=qty, side="buy", asset_type=asset_type)
                    if result:
                        # Log trade
                        trade_data = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'side': 'BUY',
                            'quantity': qty,
                            'price': current_price if current_price else 0,
                            'strategy': 'Manual',
                            'asset_type': asset_type,
                            'order_value': current_price * qty if current_price else 0
                        }
                        performance_analytics.trade_journal.add_trade(trade_data)
                        st.balloons()
                        
                        # Clear selected position sau khi trade
                        st.session_state.selected_position = None
                        st.session_state.suggested_qty = default_qty
                        time.sleep(2)
                        st.rerun()
        
        with col_sell:
            if st.button("🔴 SELL / REDUCE", 
                        use_container_width=True, 
                        type="primary",
                        key="manual_sell"):
                
                if not symbol:
                    st.error("❌ Please enter a symbol")
                elif qty <= 0:
                    st.error("❌ Quantity must be greater than 0")
                else:
                    result = trader.place_order(symbol=symbol, qty=qty, side="sell", asset_type=asset_type)
                    if result:
                        # Log trade
                        trade_data = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'side': 'SELL',
                            'quantity': qty,
                            'price': current_price if current_price else 0,
                            'strategy': 'Manual',
                            'asset_type': asset_type,
                            'order_value': current_price * qty if current_price else 0
                        }
                        performance_analytics.trade_journal.add_trade(trade_data)
                        st.balloons()
                        
                        # Clear selected position sau khi trade
                        st.session_state.selected_position = None
                        st.session_state.suggested_qty = default_qty
                        time.sleep(2)
                        st.rerun()
        
        # Clear selection button
        if st.session_state.selected_position:
            if st.button("🧹 Clear Selection", use_container_width=True):
                st.session_state.selected_position = None
                st.session_state.suggested_qty = default_qty
                st.rerun()

    # Các tab còn lại giữ nguyên
    with tab4:
        st.markdown("""
        <div class="dashboard-card">
            <h3>🤖 Automated Trading Bot</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Configure your automated trading strategy below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_asset_type = st.selectbox("Asset Type:", ["Stocks", "Crypto", "Forex"], key="auto_asset")
            auto_symbol = st.text_input("Trading Symbol:", "BTCUSD", key="auto_symbol").upper()
            auto_timeframe = st.selectbox("Timeframe:", ["1h", "4h", "1d"], key="auto_timeframe")
            
        with col2:
            risk_per_trade = st.slider("Risk per Trade (%):", 0.1, 5.0, 1.0, 0.1, key="risk_slider")
            max_positions = st.number_input("Max Simultaneous Positions:", 1, 10, 3, key="max_pos")
            auto_capital = st.number_input("Capital Allocation ($):", 1000, 100000, 5000, key="auto_capital")
        
        st.subheader("🤖 ML Strategy Settings")
        ml_enabled = st.checkbox("Enable ML-based Signals", value=True, key="ml_enable")
        min_confidence = st.slider("Minimum Confidence Level:", 0.5, 0.95, 0.7, 0.05, key="min_conf")
        
        st.subheader("🎮 Trading Controls")
        
        col_start, col_stop, col_status = st.columns(3)
        
        bot_running = st.session_state.get('bot_running', False)
        
        with col_start:
            if not bot_running:
                if st.button("🚀 Start Bot", use_container_width=True, type="primary", key="start_bot"):
                    st.session_state.bot_running = True
                    st.success("🤖 Trading bot started!")
                    st.rerun()
            else:
                if st.button("⏸️ Pause Bot", use_container_width=True, key="pause_bot"):
                    st.session_state.bot_running = False
                    st.warning("⏸️ Trading bot paused!")
                    st.rerun()
        
        with col_stop:
            if st.button("🛑 Stop Bot", use_container_width=True, type="secondary", key="stop_bot"):
                st.session_state.bot_running = False
                st.error("🛑 Bot stopped!")
                st.rerun()
        
        with col_status:
            status_color = "🟢" if bot_running else "🔴"
            st.metric("Bot Status", "RUNNING" if bot_running else "STOPPED", delta=status_color)
        
        if bot_running:
            st.subheader("📊 Real-time Monitoring")
            
            with st.spinner("🤖 Bot is monitoring markets..."):
                time.sleep(2)
                
                data = load_data_for_live(auto_symbol, auto_asset_type)
                if data is not None:
                    model = train_model_on_the_fly(data)
                    signal, confidence = get_ml_signal(data, model)
                    
                    current_price = data['Close'].iloc[-1]
                    if len(data) > 1:
                        price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
                    else:
                        price_change = 0
                    
                    col_price, col_signal, col_confidence = st.columns(3)
                    
                    with col_price:
                        st.metric("Current Price", f"${current_price:.4f}", f"{price_change:.2f}%")
                    
                    with col_signal:
                        signal_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"
                        st.metric("ML Signal", signal, delta_color="normal" if signal == "BUY" else "inverse")
                    
                    with col_confidence:
                        conf_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
                        st.metric("Confidence", f"{confidence:.1%}", conf_level)
                    
                    if ml_enabled and confidence >= min_confidence and signal != "HOLD":
                        st.warning(f"🚨 HIGH CONFIDENCE SIGNAL: {signal} {auto_symbol} (Confidence: {confidence:.1%})")
                        
                        position_size = (auto_capital * risk_per_trade / 100) / current_price
                        
                        col_exec, col_info = st.columns([1, 2])
                        with col_exec:
                            if st.button(f"Execute {signal} Order", type="primary", key="auto_execute"):
                                side = "buy" if signal == "BUY" else "sell"
                                result = trader.place_order(auto_symbol, position_size, side, auto_asset_type)
                                
                                if result:
                                    trade_data = {
                                        'timestamp': datetime.now(),
                                        'symbol': auto_symbol,
                                        'side': signal,
                                        'quantity': position_size,
                                        'price': current_price,
                                        'strategy': 'ML-Auto',
                                        'confidence': confidence,
                                        'asset_type': auto_asset_type
                                    }
                                    performance_analytics.trade_journal.add_trade(trade_data)
                                    st.success(f"✅ Automated {signal} order executed!")
                                    st.rerun()
                        
                        with col_info:
                            st.info(f"**Position Size**: {position_size:.4f} units | **Risk**: ${auto_capital * risk_per_trade / 100:.2f}")
                else:
                    st.error("❌ Cannot load market data for automated trading")
        
        st.subheader("📈 Strategy Performance")
        if performance_analytics.trade_journal.trades:
            auto_trades = [t for t in performance_analytics.trade_journal.trades if t.get('strategy') == 'ML-Auto']
            if auto_trades:
                win_count = sum(1 for t in auto_trades if t.get('pnl', 0) > 0)
                total_auto = len(auto_trades)
                win_rate = (win_count / total_auto * 100) if total_auto > 0 else 0
                
                col_win, col_total, col_rate = st.columns(3)
                col_win.metric("Winning Trades", win_count)
                col_total.metric("Total Trades", total_auto)
                col_rate.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.info("No automated trades yet. Start the bot to see performance metrics.")
        else:
            st.info("No trading history available. Start trading to see performance metrics.")

    # Tab 5: Risk Dashboard
    with tab5:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📉 Risk Management Dashboard</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            risk_fig, risk_metrics = create_risk_dashboard(trader)
            st.plotly_chart(risk_fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_color = "inverse" if risk_metrics['var_1d'] > 1000 else "off" if risk_metrics['var_1d'] > 500 else "normal"
                st.metric(
                    "1-Day Value at Risk", 
                    f"${risk_metrics['var_1d']:,.2f}",
                    delta=f"{(risk_metrics['var_1d']/float(trader.get_account_info().portfolio_value)*100):.1f}% of portfolio",
                    delta_color=var_color
                )
            
            with col2:
                dd_color = "inverse" if risk_metrics['max_drawdown'] > 20 else "off" if risk_metrics['max_drawdown'] > 10 else "normal"
                st.metric(
                    "Max Drawdown", 
                    f"{risk_metrics['max_drawdown']:.1f}%",
                    delta_color=dd_color
                )
            
            with col3:
                sharpe_color = "normal" if risk_metrics['sharpe_ratio'] > 1.5 else "off" if risk_metrics['sharpe_ratio'] > 0.5 else "inverse"
                st.metric(
                    "Sharpe Ratio", 
                    f"{risk_metrics['sharpe_ratio']:.2f}",
                    delta="Good" if risk_metrics['sharpe_ratio'] > 1 else "Poor" if risk_metrics['sharpe_ratio'] < 0 else "Average",
                    delta_color=sharpe_color
                )
                
        except Exception as e:
            st.error(f"Error loading risk dashboard: {e}")

    # Tab 6: Performance Analytics
    with tab6:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📊 Performance Analytics & Reporting</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Daily Report", key="generate_report"):
            try:
                report = performance_analytics.generate_daily_report()
                fig_equity, fig_gauges = performance_analytics.create_performance_charts(report)
                
                st.subheader("Daily Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Daily P&L", f"${report['daily_pnl']:,.2f}")
                col2.metric("Win Rate", f"{report['win_rate']:.1f}%")
                col3.metric("Profit Factor", f"{report['profit_factor']:.2f}")
                col4.metric("Total Trades", report['total_trades'])
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Avg Trade Duration", report['avg_trade_duration'])
                col6.metric("Sharpe Ratio", f"{report['sharpe_ratio']:.2f}")
                col7.metric("Max Drawdown", f"{report['max_drawdown']:.1f}%")
                col8.metric("Best Strategy", report['best_performing_strategy'])
                
                st.plotly_chart(fig_gauges, use_container_width=True)
                if fig_equity.data:
                    st.plotly_chart(fig_equity, use_container_width=True)
                else:
                    st.info("No equity curve data available yet. Start trading to see performance metrics.")
                    
            except Exception as e:
                st.error(f"Error generating performance report: {e}")

    # Tab 7: Social Sentiment
    with tab7:
        st.markdown("""
        <div class="dashboard-card">
            <h3>🎭 Social Sentiment Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        sentiment_symbol = st.text_input("Enter symbol for sentiment analysis:", "BTCUSD", key="sentiment_symbol")
        
        if st.button("Analyze Social Sentiment", key="analyze_sentiment"):
            try:
                sentiment_data = sentiment_analyzer.get_social_sentiment(sentiment_symbol)
                
                st.subheader(f"Social Sentiment for {sentiment_symbol}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment_color = "normal" if sentiment_data['combined_score'] > 0.1 else "inverse" if sentiment_data['combined_score'] < -0.1 else "off"
                    st.metric(
                        "Overall Sentiment",
                        sentiment_data['sentiment_label'],
                        delta=f"{sentiment_data['combined_score']:.2f}",
                        delta_color=sentiment_color
                    )
                
                with col2:
                    st.metric("Confidence Level", f"{sentiment_data['confidence']:.1f}%")
                
                with col3:
                    st.metric("Twitter Volume", f"{sentiment_data['twitter_volume']:,}")
                
                with col4:
                    st.metric("Reddit Volume", f"{sentiment_data['reddit_volume']:,}")
                
                sentiment_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sentiment_data['combined_score'],
                    title = {'text': "Sentiment Score"},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "red"},
                            {'range': [-0.1, 0.1], 'color': "yellow"},
                            {'range': [0.1, 1], 'color': "green"}],
                    }
                ))
                
                sentiment_gauge.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(sentiment_gauge, use_container_width=True)
                
                st.info(f"📊 Sentiment Trend: {sentiment_data['sentiment_trend']}")
                
            except Exception as e:
                st.error(f"Error analyzing social sentiment: {e}")

else:
    # Welcome Screen với design đồng bộ
    st.markdown("""
    <div class="dashboard-card">
        <h2 style="text-align: center; margin-bottom: 2rem;">🚀 Welcome to Live Trading Pro</h2>
        <p style="text-align: center; color: #8898aa; font-size: 1.1rem;">
            Connect your Alpaca account to start trading with advanced features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="feature-icon">🔌</div>
            <div class="metric-value">Connect</div>
            <div class="metric-label">Alpaca API Integration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="feature-icon">📈</div>
            <div class="metric-value">Trade</div>
            <div class="metric-label">Multiple Asset Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="feature-icon">🛡️</div>
            <div class="metric-value">Manage</div>
            <div class="metric-label">Advanced Risk Analytics</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Trading Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Live Trading Pro v2.0</p>
</div>
""", unsafe_allow_html=True)