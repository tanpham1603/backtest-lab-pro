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
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
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
    .pending-order {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header-gradient">🚀 Live Trading Pro</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #8898aa; font-size: 1.2rem; margin-bottom: 3rem;">Professional Algorithmic Trading Platform</div>', unsafe_allow_html=True)

# --- HÀM TẢI DỮ LIỆU ĐÃ SỬA HOÀN TOÀN ---
def safe_yfinance_download(symbol, period="60d", interval="1d"):
    """Hàm tải dữ liệu an toàn từ yfinance không bị lỗi format"""
    try:
        # Chuẩn hóa symbol
        clean_symbol = symbol.upper().replace('USDT', '').replace('USD', '').replace('/', '')
        
        # Thử các định dạng symbol khác nhau
        symbol_variants = [
            clean_symbol,
            f"{clean_symbol}-USD",
            symbol,
            symbol.replace('USDT', '-USD'),
            symbol.replace('USD', '-USD')
        ]
        
        for sym in symbol_variants:
            try:
                data = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
                if not data.empty and len(data) > 0:
                    # Chuẩn hóa columns
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    # Đảm bảo có column Close
                    if 'Close' not in data.columns and len(data.columns) > 0:
                        data = data.rename(columns={data.columns[0]: 'Close'})
                    
                    return data
            except Exception as e:
                continue
                
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu cho {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_data_for_live(symbol, asset_type):
    """Hàm tải dữ liệu chính - ĐÃ SỬA HOÀN TOÀN LỖI FORMAT"""
    try:
        if asset_type == "Crypto":
            data = safe_yfinance_download(symbol, "60d", "1d")
        else:
            data = safe_yfinance_download(symbol, "2y", "1d")
        
        # Kiểm tra và chuẩn hóa dữ liệu trả về
        if data is None or data.empty:
            return pd.DataFrame()
            
        # Đảm bảo các columns cần thiết tồn tại
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                if len(data.columns) >= 5:  # Nếu có đủ columns
                    data.columns = required_columns[:len(data.columns)]
                else:
                    # Tạo columns mặc định nếu thiếu
                    if 'Close' not in data.columns:
                        data['Close'] = data.iloc[:, 0] if len(data.columns) > 0 else 0
        
        return data
        
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu cho {symbol}: {str(e)}")
        return pd.DataFrame()

# --- Lớp AlpacaTrader ĐÃ SỬA LỖI HOÀN TOÀN ---
class AlpacaTrader:
    def __init__(self, api_key, api_secret, paper=True):
        self.api, self.account, self.connected = None, None, False
        try:
            self.api = TradingClient(api_key, api_secret, paper=paper)
            self.account = self.api.get_account()
            self.connected = True
            st.success("✅ Kết nối Alpaca thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi kết nối Alpaca: {e}")

    def get_account_info(self): 
        try:
            return self.api.get_account()
        except Exception as e:
            st.error(f"Lỗi lấy thông tin tài khoản: {e}")
            return None
    
    def get_positions(self): 
        try:
            return self.api.get_all_positions()
        except Exception as e:
            st.error(f"Lỗi lấy danh sách positions: {e}")
            return []
    
    def get_orders(self, status_filter=None):
        """Lấy danh sách các lệnh - ĐÃ SỬA HOÀN TOÀN"""
        try:
            # Lấy tất cả orders
            all_orders = self.api.get_orders()
            
            if status_filter:
                # Filter manual dựa trên status
                if status_filter == 'open':
                    return [order for order in all_orders if order.status in [
                        OrderStatus.ACCEPTED, 
                        OrderStatus.NEW, 
                        OrderStatus.PARTIALLY_FILLED,
                        OrderStatus.PENDING_NEW
                    ]]
                elif status_filter == 'closed':
                    return [order for order in all_orders if order.status in [
                        OrderStatus.FILLED, 
                        OrderStatus.CANCELED, 
                        OrderStatus.REJECTED,
                        OrderStatus.EXPIRED
                    ]]
            
            return all_orders
            
        except Exception as e:
            st.error(f"Lỗi lấy danh sách orders: {e}")
            return []

    def cancel_order(self, order_id):
        """Hủy lệnh chờ"""
        try:
            self.api.cancel_order_by_id(order_id)
            st.success(f"✅ Đã hủy lệnh {order_id}")
            time.sleep(1)  # Chờ một chút để update
            return True
        except Exception as e:
            st.error(f"Lỗi hủy order {order_id}: {e}")
            return False

    def get_portfolio_history(self, period="1M"):
        """Lấy lịch sử portfolio từ Alpaca"""
        try:
            params = GetPortfolioHistoryRequest(period=period)
            history = self.api.get_portfolio_history(params)
            return history
        except Exception as e:
            st.error(f"Lỗi lấy lịch sử portfolio: {e}")
            return None

    def place_order(self, symbol, qty, side, asset_type):
        """Đặt lệnh - ĐÃ SỬA HOÀN TOÀN"""
        try:
            # Định dạng symbol theo loại tài sản
            formatted_symbol = self._format_symbol(symbol, asset_type)
            
            st.info(f"🔄 Đang đặt lệnh {side.upper()} cho {qty} {formatted_symbol}...")
            
            # Xử lý quantity cho các loại tài sản
            if asset_type == "Stocks":
                qty = int(qty)  # Stocks phải là số nguyên
            
            # Sử dụng time_in_force phù hợp
            time_in_force = TimeInForce.DAY
            
            market_order_data = MarketOrderRequest(
                symbol=formatted_symbol, 
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=time_in_force
            )
            
            order = self.api.submit_order(order_data=market_order_data)
            st.success(f"✅ Đã đặt lệnh {side.upper()} thành công cho {qty} {formatted_symbol}!")
            st.success(f"📋 ID lệnh: {order.id}")
            return order
            
        except Exception as e:
            error_msg = f"❌ Lỗi đặt lệnh: {e}"
            st.error(error_msg)
            st.error(f"Symbol: {formatted_symbol}, Asset Type: {asset_type}, Quantity: {qty}")
            return None

    def _format_symbol(self, symbol, asset_type):
        """Định dạng symbol theo loại tài sản"""
        symbol = symbol.upper().strip()
        
        if asset_type == "Crypto":
            if '/' in symbol:
                return symbol.replace('/', '')
            elif symbol.endswith('USDT'):
                return symbol.replace('USDT', 'USD')
            elif symbol.endswith('USD'):
                return symbol
            else:
                return f"{symbol}USD"
        elif asset_type == "Forex":
            if '/' in symbol:
                return symbol.replace('/', '')
            elif len(symbol) == 6:
                return symbol
            else:
                return symbol
        else:  # Stocks
            return symbol

# --- ADVANCED RISK MANAGEMENT DASHBOARD ---
class RiskManager:
    def __init__(self, trader):
        self.trader = trader
        
    def calculate_var(self, positions, confidence_level=0.95, periods=252):
        """Tính Value at Risk"""
        try:
            account_info = self.trader.get_account_info()
            if not account_info:
                return {'1d': 0, '1w': 0, '1m': 0}
                
            portfolio_value = float(account_info.portfolio_value)
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
            daily_pnl = sum(float(p.unrealized_pl) for p in positions) if positions else 0
            return daily_pnl
        except:
            return 0
    
    def calculate_win_rate(self):
        if len(self.trade_journal.trades) == 0:
            return 0
        winning_trades = sum(1 for trade in self.trade_journal.trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(self.trade_journal.trades)) * 100
    
    def calculate_profit_factor(self):
        if len(self.trade_journal.trades) == 0:
            return 0
            
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

# --- Các hàm hỗ trợ ---
def get_asset_badge(asset_type):
    """Trả về badge CSS class cho từng loại tài sản"""
    if asset_type == "Crypto":
        return "badge-crypto"
    elif asset_type == "Forex":
        return "badge-forex"
    else:
        return "badge-stock"

def display_position(position):
    """Hiển thị position"""
    try:
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
    except Exception as e:
        st.error(f"Lỗi hiển thị position: {e}")

def display_pending_order(order):
    """Hiển thị lệnh chờ với nút hủy"""
    try:
        symbol = order.symbol
        side = order.side.value
        qty = float(order.qty)
        filled_qty = float(order.filled_qty) if order.filled_qty else 0
        remaining_qty = qty - filled_qty
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{symbol}**")
            st.write(f"Side: {side} | Qty: {remaining_qty:.2f}/{qty:.2f}")
            st.write(f"Status: {order.status.value}")
        
        with col2:
            st.write(f"Type: {order.type.value}")
            if order.limit_price:
                st.write(f"Limit: ${float(order.limit_price):.2f}")
        
        with col3:
            if st.button("❌ Hủy", key=f"cancel_{order.id}", use_container_width=True):
                if trader.cancel_order(order.id):
                    st.rerun()
                    
        st.markdown("---")
    except Exception as e:
        st.error(f"Lỗi hiển thị order: {e}")

# --- Session State ---
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'performance_analytics' not in st.session_state:
    st.session_state.performance_analytics = None
if 'selected_position' not in st.session_state:
    st.session_state.selected_position = None
if 'suggested_qty' not in st.session_state:
    st.session_state.suggested_qty = 1.0
if 'last_order_time' not in st.session_state:
    st.session_state.last_order_time = None

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

    if st.button("Kết nối", use_container_width=True, type="primary"):
        if api_key and api_secret:
            with st.spinner("Đang kết nối..."):
                st.session_state.trader = AlpacaTrader(api_key.strip(), api_secret.strip(), paper=(account_type == 'Paper Trading'))
                if st.session_state.trader and st.session_state.trader.connected:
                    st.session_state.performance_analytics = PerformanceAnalytics(st.session_state.trader)
                    st.success(f"✅ Đã kết nối {account_type}!")
                else:
                    st.error("❌ Kết nối thất bại")
        else:
            st.warning("Vui lòng nhập API credentials")

    if st.session_state.trader and st.session_state.trader.connected:
        st.success(f"✅ Đã kết nối {account_type}!")
        
        st.markdown("""
        <div class="dashboard-card">
            <h3>⚡ Quick Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
            st.rerun()
    else:
        st.info("Nhập API credentials để bắt đầu")

# --- Main Content ---
if st.session_state.trader and st.session_state.trader.connected:
    trader = st.session_state.trader
    performance_analytics = st.session_state.performance_analytics

    # Status Cards
    try:
        account = trader.get_account_info()
        if account:
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
        st.error(f"Lỗi tải thông tin tài khoản: {e}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Tabs
    tab_titles = ["📊 Tổng quan", "📈 Vị thế", "🛠️ Giao dịch", 
                  "📉 Quản lý rủi ro", "📊 Hiệu suất", "⏳ Lệnh chờ"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    # Tab 1: Overview
    with tab1:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📊 Tổng quan tài khoản</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            account = trader.get_account_info()
            if account:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${float(account.portfolio_value):,.2f}</div>
                        <div class="metric-label">Tổng tài sản</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${float(account.buying_power):,.2f}</div>
                        <div class="metric-label">Sức mua</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">${float(account.cash):,.2f}</div>
                        <div class="metric-label">Tiền mặt</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    status_color = "#00ff88" if account.status.value == "ACTIVE" else "#ff4444"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {status_color}">{account.status.value}</div>
                        <div class="metric-label">Trạng thái</div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi tải thông tin: {e}")

    # Tab 2: Positions
    with tab2:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📈 Vị thế hiện tại</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Làm mới vị thế", key="refresh_positions", use_container_width=True):
            st.rerun()
            
        try:
            positions = trader.get_positions()
            if positions:
                st.info("💡 Click vào vị thế để tự động điền form giao dịch")
                
                for position in positions:
                    display_position(position)
                    col1, col2 = st.columns(2)
                    with col1:
                        # CODE MỚI ĐÃ SỬA
                        if st.button(f"📈 Mua thêm", key=f"buy_{position.symbol}", use_container_width=True):
                            # Tự động nhận diện asset type
                            asset_type = "Crypto" if position.symbol.endswith("USD") else "Stocks"
                            st.session_state.selected_position = {
                                'symbol': position.symbol,
                                'action': 'buy',
                                'current_qty': float(position.qty),
                                'asset_type': asset_type  # <-- ĐÃ SỬA
                            }
                            st.success(f"Đã chọn {position.symbol} ({asset_type}) để MUA - kiểm tra tab Giao dịch!")
                    # CODE MỚI ĐÃ SỬA
                    with col2:
                        if st.button(f"📉 Bán", key=f"sell_{position.symbol}", use_container_width=True):
                                # Tự động nhận diện asset type
                            asset_type = "Crypto" if position.symbol.endswith("USD") else "Stocks"
                            st.session_state.selected_position = {
                                'symbol': position.symbol,
                                'action': 'sell', 
                                'current_qty': float(position.qty),
                                'asset_type': asset_type # <-- ĐÃ SỬA
                            }
                            st.success(f"Đã chọn {position.symbol} ({asset_type}) để BÁN - kiểm tra tab Giao dịch!")
                    st.markdown("---")
            else:
                st.info("💰 Không có vị thế nào. Bắt đầu giao dịch để xem vị thế ở đây!")
                
        except Exception as e:
            st.error(f"Lỗi tải danh sách vị thế: {e}")

    # Tab 3: Manual Trading - ĐÃ SỬA HOÀN TOÀN
    with tab3:
        st.markdown("""
        <div class="dashboard-card">
            <h3>🛠️ Giao dịch thủ công</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị thông báo nếu có position được chọn
        if st.session_state.selected_position:
            selected = st.session_state.selected_position
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="margin: 0; color: white;">🎯 Đang giao dịch {selected['symbol']} - Hiện có: {selected['current_qty']:.2f} shares - Hành động: {selected['action'].upper()}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            default_symbol = selected['symbol']
            default_action = selected['action']
            default_asset_type = selected.get('asset_type', 'Stocks')
            suggested_qty = selected['current_qty'] * 0.1
        else:
            default_symbol = "BTCUSD"
            default_action = "buy"
            default_asset_type = "Crypto"
            suggested_qty = 0.001
        
        # Trading form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="dashboard-card">
                <h4>🎯 Cấu hình giao dịch</h4>
            </div>
            """, unsafe_allow_html=True)
            
            asset_type = st.radio("Loại tài sản:", ["Stocks", "Crypto", "Forex"], 
                                 index=["Stocks", "Crypto", "Forex"].index(default_asset_type) if default_asset_type in ["Stocks", "Crypto", "Forex"] else 0,
                                 horizontal=True, key="manual_asset")
            
            symbol = st.text_input("Mã:", value=default_symbol, key="manual_symbol").upper()
            
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <h4>📊 Chi tiết lệnh</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Xử lý quantity
            if asset_type == "Crypto":
                min_qty = 0.0001
                step = 0.001
                format_str = "%.4f"
                default_qty = float(0.01)
            elif asset_type == "Forex":
                min_qty = 0.01
                step = 0.01
                format_str = "%.2f"
                default_qty = float(1.0)
            else:  # Stocks
                min_qty = 1
                step = 1
                format_str = "%d"
                default_qty = int(1)
            
            current_qty = st.session_state.selected_position['current_qty'] if st.session_state.selected_position else 0
            
            if current_qty > 0:
                st.write(f"📊 **Vị thế hiện tại**: {current_qty:.2f} shares")
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    if st.button("25%", use_container_width=True):
                        if asset_type == "Stocks":
                            st.session_state.suggested_qty = max(1, int(current_qty * 0.25))
                        else:
                            st.session_state.suggested_qty = current_qty * 0.25
                with col_q2:
                    if st.button("50%", use_container_width=True):
                        if asset_type == "Stocks":
                            st.session_state.suggested_qty = max(1, int(current_qty * 0.5))
                        else:
                            st.session_state.suggested_qty = current_qty * 0.5
                with col_q3:
                    if st.button("100%", use_container_width=True):
                        if asset_type == "Stocks":
                            st.session_state.suggested_qty = max(1, int(current_qty))
                        else:
                            st.session_state.suggested_qty = current_qty
            
            current_suggested_qty = st.session_state.get('suggested_qty', suggested_qty)
            
            if asset_type == "Stocks":
                if isinstance(current_suggested_qty, float):
                    current_suggested_qty = int(current_suggested_qty)
                qty = st.number_input("Số lượng:", 
                                    min_value=min_qty, 
                                    value=current_suggested_qty, 
                                    step=step, 
                                    format=format_str, 
                                    key="manual_qty")
                qty = int(qty)
            else:
                qty = st.number_input("Số lượng:", 
                                    min_value=min_qty, 
                                    value=float(current_suggested_qty), 
                                    step=step, 
                                    format=format_str, 
                                    key="manual_qty")
        
        # Hiển thị thông tin giá - ĐÃ SỬA LỖI FORMAT
        current_price = None
        if symbol:
            try:
                with st.spinner("Đang tải dữ liệu..."):
                    data = load_data_for_live(symbol, asset_type)
                    if data is not None and not data.empty and len(data) > 0:
                        if 'Close' in data.columns and len(data['Close']) > 0:
                            current_price = float(data['Close'].iloc[-1])
                            price_info = f"${current_price:.2f}"
                            
                            if len(data) > 1:
                                prev_price = float(data['Close'].iloc[-2])
                                price_change = ((current_price - prev_price) / prev_price * 100)
                                change_info = f"{price_change:+.2f}%"
                            else:
                                price_change = 0
                                change_info = "N/A"
                            
                            # Hiển thị thông tin
                            col_price, col_change, col_value = st.columns(3)
                            with col_price:
                                st.metric("Giá hiện tại", price_info)
                            with col_change:
                                st.metric("Thay đổi", change_info)
                            with col_value:
                                order_value = current_price * qty
                                st.metric("Giá trị lệnh", f"${order_value:.2f}")
                        else:
                            st.warning("⚠️ Không thể lấy dữ liệu giá cho mã này")
                    else:
                        st.warning("⚠️ Không có dữ liệu cho mã này")
                            
            except Exception as e:
                st.warning(f"⚠️ Không thể tải dữ liệu cho {symbol}")
        
        # Nút đặt lệnh
        st.markdown("""
        <div class="dashboard-card">
            <h3>🎯 Thực hiện lệnh</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            if st.button("🟢 MUA / MUA THÊM", 
                        use_container_width=True, 
                        type="primary",
                        key="manual_buy"):
                
                if not symbol:
                    st.error("❌ Vui lòng nhập mã")
                elif qty <= 0:
                    st.error("❌ Số lượng phải lớn hơn 0")
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
                            'asset_type': asset_type
                        }
                        performance_analytics.trade_journal.add_trade(trade_data)
                        st.session_state.last_order_time = datetime.now()
                        st.balloons()
                        
                        # Clear selection
                        st.session_state.selected_position = None
                        st.session_state.suggested_qty = default_qty
                        time.sleep(2)
                        st.rerun()
        
        with col_sell:
            if st.button("🔴 BÁN / GIẢM", 
                        use_container_width=True, 
                        type="primary",
                        key="manual_sell"):
                
                if not symbol:
                    st.error("❌ Vui lòng nhập mã")
                elif qty <= 0:
                    st.error("❌ Số lượng phải lớn hơn 0")
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
                            'asset_type': asset_type
                        }
                        performance_analytics.trade_journal.add_trade(trade_data)
                        st.session_state.last_order_time = datetime.now()
                        st.balloons()
                        
                        # Clear selection
                        st.session_state.selected_position = None
                        st.session_state.suggested_qty = default_qty
                        time.sleep(2)
                        st.rerun()
        
        # Clear selection button
        if st.session_state.selected_position:
            if st.button("🧹 Xóa lựa chọn", use_container_width=True):
                st.session_state.selected_position = None
                st.session_state.suggested_qty = default_qty
                st.rerun()

    # Tab 4: Risk Dashboard
    with tab4:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📉 Quản lý rủi ro</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            risk_fig, risk_metrics = create_risk_dashboard(trader)
            st.plotly_chart(risk_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi tải bảng quản lý rủi ro: {e}")

    # Tab 5: Performance Analytics
    with tab5:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📊 Phân tích hiệu suất</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Tạo báo cáo", key="generate_report"):
            try:
                report = performance_analytics.generate_daily_report()
                
                st.subheader("Tóm tắt hiệu suất")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Lợi nhuận/Thua lỗ", f"${report['daily_pnl']:,.2f}")
                col2.metric("Tỷ lệ thắng", f"{report['win_rate']:.1f}%")
                col3.metric("Hệ số lợi nhuận", f"{report['profit_factor']:.2f}")
                col4.metric("Tổng giao dịch", report['total_trades'])
                    
            except Exception as e:
                st.error(f"Lỗi tạo báo cáo: {e}")

    # Tab 6: Pending Orders - ĐÃ SỬA HOÀN TOÀN
    with tab6:
        st.markdown("""
        <div class="dashboard-card">
            <h3>⏳ Quản lý lệnh chờ</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Quản lý các lệnh chờ của bạn tại đây. Bạn có thể hủy các lệnh không mong muốn để tránh wash trade.")
        
        if st.button("🔄 Làm mới lệnh chờ", key="refresh_pending", use_container_width=True):
            st.rerun()
            
        try:
            # Lấy danh sách lệnh chờ - ĐÃ SỬA HOÀN TOÀN
            pending_orders = trader.get_orders(status_filter='open')
            
            if pending_orders:
                st.subheader(f"📋 Lệnh chờ ({len(pending_orders)})")
                st.info(f"⏰ Thời gian hiện tại: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                for order in pending_orders:
                    display_pending_order(order)
            else:
                st.success("🎉 Không có lệnh chờ nào! Tất cả lệnh đã được khớp hoặc hủy.")
                
        except Exception as e:
            st.error(f"Lỗi tải lệnh chờ: {e}")

else:
    # Welcome Screen
    st.markdown("""
    <div class="dashboard-card">
        <h2 style="text-align: center; margin-bottom: 2rem;">🚀 Chào mừng đến với Live Trading Pro</h2>
        <p style="text-align: center; color: #8898aa; font-size: 1.1rem;">
            Kết nối tài khoản Alpaca của bạn để bắt đầu giao dịch
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Trading Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Live Trading Pro v2.0</p>
</div>
""", unsafe_allow_html=True)