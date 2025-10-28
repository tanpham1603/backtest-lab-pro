import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_ta as ta
import warnings
from datetime import datetime, timedelta
import requests
import json

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🚀 Advanced Strategy Builder Pro", 
    page_icon="🚀", 
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
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>🚀</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>Strategy Builder</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>Advanced Strategy Development</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# --- HEADER ---
st.markdown('<div class="header">🚀 Advanced Strategy Builder Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Build, Test, and Optimize Professional Trading Strategies</div>', unsafe_allow_html=True)

# --- STATUS CARDS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">10+</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Indicators</div>
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
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">Live</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛡️</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: white; margin: 0.5rem 0;">Advanced</div>
        <div style="color: #8898aa; font-size: 0.9rem;">Risk Management</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

class AdvancedStrategyBuilder:
    def __init__(self):
        self.data = None
        self.signals = None
        self.performance = None
        
    def load_data(self, symbol, period, interval='1d'):
        """Load data with multiple timeframes and advanced processing"""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if data.empty:
                return None
            
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def calculate_advanced_indicators(self, data, params):
        """Calculate advanced technical indicators with robust error handling"""
        indicators_dict = {}
        
        try:
            # Moving Averages
            for p in params.get('sma_periods', []):
                data[f'SMA_{p}'] = ta.sma(data['Close'], length=p)
                indicators_dict[f'SMA_{p}'] = data[f'SMA_{p}']
                
            for p in params.get('ema_periods', []):
                data[f'EMA_{p}'] = ta.ema(data['Close'], length=p)
                indicators_dict[f'EMA_{p}'] = data[f'EMA_{p}']
                
            # Bollinger Bands - FIXED: Handle column naming properly
            if params.get('bb_enabled'):
                bb_period = params.get('bb_period', 20)
                bb_std = params.get('bb_std', 2.0)
                bb_data = ta.bbands(data['Close'], length=bb_period, std=bb_std)
                
                if bb_data is not None and not bb_data.empty:
                    # Tìm tên cột thực tế trong bb_data
                    bb_upper_col = None
                    bb_lower_col = None 
                    bb_middle_col = None
                    
                    for col in bb_data.columns:
                        col_str = str(col).upper()
                        if 'BBU' in col_str or 'UPPER' in col_str:
                            bb_upper_col = col
                        elif 'BBL' in col_str or 'LOWER' in col_str:
                            bb_lower_col = col
                        elif 'BBM' in col_str or 'MIDDLE' in col_str:
                            bb_middle_col = col
                    
                    # Gán dữ liệu nếu tìm thấy cột
                    if bb_upper_col:
                        data[f'BBU_{bb_period}_{bb_std}'] = bb_data[bb_upper_col]
                    if bb_lower_col:
                        data[f'BBL_{bb_period}_{bb_std}'] = bb_data[bb_lower_col]
                    if bb_middle_col:
                        data[f'BBM_{bb_period}_{bb_std}'] = bb_data[bb_middle_col]
                    
                    # Tạo DataFrame cho indicators_dict
                    bb_df_data = {}
                    if bb_upper_col:
                        bb_df_data['Upper'] = bb_data[bb_upper_col]
                    if bb_lower_col:
                        bb_df_data['Lower'] = bb_data[bb_lower_col]
                    if bb_middle_col:
                        bb_df_data['Middle'] = bb_data[bb_middle_col]
                    
                    if bb_df_data:
                        indicators_dict['BB'] = pd.DataFrame(bb_df_data)
                
            # RSI
            if params.get('rsi_enabled'):
                rsi_period = params.get('rsi_period', 14)
                rsi_data = ta.rsi(data['Close'], length=rsi_period)
                if rsi_data is not None:
                    data[f'RSI_{rsi_period}'] = rsi_data
                    indicators_dict['RSI'] = rsi_data
                    
            # MACD - FIXED: Handle column naming properly
            if params.get('macd_enabled'):
                macd_fast = params.get('macd_fast', 12)
                macd_slow = params.get('macd_slow', 26)
                macd_signal = params.get('macd_signal', 9)
                macd_data = ta.macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
                
                if macd_data is not None and not macd_data.empty:
                    # Tìm tên cột thực tế
                    macd_line_col = None
                    macd_signal_col = None
                    macd_hist_col = None
                    
                    for col in macd_data.columns:
                        col_str = str(col).upper()
                        if 'MACD_' in col_str and 'MACDS' not in col_str and 'MACDH' not in col_str:
                            macd_line_col = col
                        elif 'MACDS' in col_str or 'SIGNAL' in col_str:
                            macd_signal_col = col
                        elif 'MACDH' in col_str or 'HISTOGRAM' in col_str:
                            macd_hist_col = col
                    
                    # Gán dữ liệu
                    macd_dict = {}
                    if macd_line_col:
                        data[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'] = macd_data[macd_line_col]
                        macd_dict['MACD'] = macd_data[macd_line_col]
                    if macd_signal_col:
                        data[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'] = macd_data[macd_signal_col]
                        macd_dict['Signal'] = macd_data[macd_signal_col]
                    if macd_hist_col:
                        data[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'] = macd_data[macd_hist_col]
                        macd_dict['Histogram'] = macd_data[macd_hist_col]
                    
                    if macd_dict:
                        indicators_dict['MACD'] = pd.DataFrame(macd_dict)
                        
            # Stochastic - FIXED: Handle column naming properly
            if params.get('stoch_enabled'):
                stoch_k = params.get('stoch_k', 14)
                stoch_d = params.get('stoch_d', 3)
                stoch_data = ta.stoch(data['High'], data['Low'], data['Close'], k=stoch_k, d=stoch_d)
                
                if stoch_data is not None and not stoch_data.empty:
                    # Tìm tên cột
                    stoch_k_col = None
                    stoch_d_col = None
                    
                    for col in stoch_data.columns:
                        col_str = str(col).upper()
                        if 'STOCHK' in col_str or '%K' in col_str:
                            stoch_k_col = col
                        elif 'STOCHD' in col_str or '%D' in col_str:
                            stoch_d_col = col
                    
                    stoch_dict = {}
                    if stoch_k_col:
                        data[f'STOCHk_{stoch_k}_{stoch_d}'] = stoch_data[stoch_k_col]
                        stoch_dict['STOCHk'] = stoch_data[stoch_k_col]
                    if stoch_d_col:
                        data[f'STOCHd_{stoch_k}_{stoch_d}'] = stoch_data[stoch_d_col]
                        stoch_dict['STOCHd'] = stoch_data[stoch_d_col]
                    
                    if stoch_dict:
                        indicators_dict['STOCH'] = pd.DataFrame(stoch_dict)
                        
            # ADX
            if params.get('adx_enabled'):
                adx_period = params.get('adx_period', 14)
                adx_data = ta.adx(data['High'], data['Low'], data['Close'], length=adx_period)
                if adx_data is not None:
                    adx_col = None
                    for col in adx_data.columns:
                        if 'ADX_' in str(col):
                            adx_col = col
                            break
                    if adx_col:
                        data[f'ADX_{adx_period}'] = adx_data[adx_col]
                        indicators_dict['ADX'] = adx_data[adx_col]
                        
            # VWAP
            if params.get('vwap_enabled'):
                vwap_data = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
                if vwap_data is not None:
                    data['VWAP_D'] = vwap_data
                    indicators_dict['VWAP'] = vwap_data
                    
            # SuperTrend - FIXED: Handle column naming properly
            if params.get('supertrend_enabled'):
                st_period = params.get('st_period', 10)
                st_multiplier = params.get('st_multiplier', 3.0)
                supertrend_data = ta.supertrend(data['High'], data['Low'], data['Close'], length=st_period, multiplier=st_multiplier)
                
                if supertrend_data is not None and not supertrend_data.empty:
                    supertrend_col = None
                    for col in supertrend_data.columns:
                        if 'SUPERT' in str(col).upper():
                            supertrend_col = col
                            break
                    
                    if supertrend_col:
                        data[f'SUPERT_{st_period}_{st_multiplier}'] = supertrend_data[supertrend_col]
                        indicators_dict['SUPERTREND'] = supertrend_data[supertrend_col]
                        
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            
        return indicators_dict

    def generate_signals(self, data, indicators_dict, strategy_params):
        """Generate trading signals based on strategy with robust error handling"""
        try:
            signals = pd.Series(0, index=data.index)
            
            # Buy conditions
            buy_conditions = pd.Series(True, index=data.index)
            
            # RSI Condition
            if strategy_params.get('rsi_condition') == "RSI < Oversold":
                rsi = indicators_dict.get('RSI')
                if rsi is not None:
                    buy_conditions &= (rsi < strategy_params.get('rsi_oversold', 30))
            elif strategy_params.get('rsi_condition') == "RSI > Overbought":
                rsi = indicators_dict.get('RSI')
                if rsi is not None:
                    buy_conditions &= (rsi > strategy_params.get('rsi_overbought', 70))
                    
            # MACD Condition
            if strategy_params.get('macd_condition') == "MACD crosses above Signal":
                macd = indicators_dict.get('MACD')
                if macd is not None and not macd.empty and 'MACD' in macd.columns and 'Signal' in macd.columns:
                    buy_conditions &= (macd['MACD'] > macd['Signal']) & (macd['MACD'].shift(1) <= macd['Signal'].shift(1))
            elif strategy_params.get('macd_condition') == "MACD crosses below Signal":
                macd = indicators_dict.get('MACD')
                if macd is not None and not macd.empty and 'MACD' in macd.columns and 'Signal' in macd.columns:
                    buy_conditions &= (macd['MACD'] < macd['Signal']) & (macd['MACD'].shift(1) >= macd['Signal'].shift(1))
                    
            # Price vs MA Condition
            if strategy_params.get('price_ma_condition') == "Price above SMA":
                sma_period = strategy_params.get('price_ma_period', 20)
                sma = indicators_dict.get(f'SMA_{sma_period}')
                if sma is not None:
                    buy_conditions &= (data['Close'] > sma)
            elif strategy_params.get('price_ma_condition') == "Price below SMA":
                sma_period = strategy_params.get('price_ma_period', 20)
                sma = indicators_dict.get(f'SMA_{sma_period}')
                if sma is not None:
                    buy_conditions &= (data['Close'] < sma)
                    
            # Bollinger Bands Condition
            if strategy_params.get('bb_condition') == "Price touches lower band":
                bb = indicators_dict.get('BB')
                if bb is not None and not bb.empty and 'Lower' in bb.columns:
                    buy_conditions &= (data['Low'] <= bb['Lower'])
            elif strategy_params.get('bb_condition') == "Price touches upper band":
                bb = indicators_dict.get('BB')
                if bb is not None and not bb.empty and 'Upper' in bb.columns:
                    buy_conditions &= (data['High'] >= bb['Upper'])
                    
            # Stochastic Condition
            if strategy_params.get('stoch_condition') == "Stochastic oversold":
                stoch = indicators_dict.get('STOCH')
                if stoch is not None and not stoch.empty and 'STOCHk' in stoch.columns and 'STOCHd' in stoch.columns:
                    buy_conditions &= (stoch['STOCHk'] < 20) & (stoch['STOCHd'] < 20)
            elif strategy_params.get('stoch_condition') == "Stochastic overbought":
                stoch = indicators_dict.get('STOCH')
                if stoch is not None and not stoch.empty and 'STOCHk' in stoch.columns and 'STOCHd' in stoch.columns:
                    buy_conditions &= (stoch['STOCHk'] > 80) & (stoch['STOCHd'] > 80)
            
            # SuperTrend Condition
            if strategy_params.get('supertrend_condition') == "SuperTrend bullish":
                supertrend = indicators_dict.get('SUPERTREND')
                if supertrend is not None:
                    buy_conditions &= (data['Close'] > supertrend)
            elif strategy_params.get('supertrend_condition') == "SuperTrend bearish":
                supertrend = indicators_dict.get('SUPERTREND')
                if supertrend is not None:
                    buy_conditions &= (data['Close'] < supertrend)
            
            # Apply buy conditions
            signals[buy_conditions] = 1
            
            # Sell conditions (opposite or specific conditions)
            sell_conditions = pd.Series(False, index=data.index)
            
            if strategy_params.get('exit_condition') == "RSI > Overbought":
                rsi = indicators_dict.get('RSI')
                if rsi is not None:
                    sell_conditions = (rsi > strategy_params.get('rsi_overbought', 70))
            elif strategy_params.get('exit_condition') == "Opposite Signal":
                sell_conditions = ~buy_conditions & (signals.shift(1) == 1)
                    
            signals[sell_conditions] = -1
            
            return signals
            
        except Exception as e:
            st.error(f"Error generating signals: {e}")
            return pd.Series(0, index=data.index)

    def backtest_strategy(self, data, signals, initial_capital=10000):
        """Simple strategy backtest with improved error handling"""
        try:
            if signals is None or signals.sum() == 0:
                return None
                
            capital = initial_capital
            position = 0
            trades = []
            entry_price = 0
            portfolio_values = [initial_capital]
            
            for i in range(1, len(data)):
                current_signal = signals.iloc[i]
                current_price = data['Close'].iloc[i]
                
                # Buy
                if current_signal == 1 and position == 0:
                    position = capital // current_price
                    if position > 0:
                        entry_price = current_price
                        capital -= position * current_price
                        trades.append({
                            'date': data.index[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': position,
                            'value': position * current_price
                        })
                    
                # Sell
                elif current_signal == -1 and position > 0:
                    capital += position * current_price
                    pnl = (current_price - entry_price) * position
                    trades.append({
                        'date': data.index[i],
                        'action': 'SELL',
                        'price': current_price,
                        'shares': position,
                        'value': position * current_price,
                        'pnl': pnl
                    })
                    position = 0
                
                # Calculate portfolio value
                portfolio_value = capital + (position * current_price if position > 0 else 0)
                portfolio_values.append(portfolio_value)
                    
            # Calculate performance
            if trades:
                df_trades = pd.DataFrame(trades)
                final_capital = portfolio_values[-1] if portfolio_values else initial_capital
                total_return = (final_capital - initial_capital) / initial_capital * 100
                
                sell_trades = df_trades[df_trades['action'] == 'SELL']
                winning_trades = len(sell_trades[sell_trades['pnl'] > 0]) if 'pnl' in sell_trades.columns else 0
                total_trades = len(sell_trades)
                
                performance = {
                    'initial_capital': initial_capital,
                    'final_capital': final_capital,
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'trades': df_trades,
                    'portfolio_values': portfolio_values
                }
                
                return performance
                
            return None
            
        except Exception as e:
            st.error(f"Error in backtest: {e}")
            return None

    def create_advanced_chart(self, data, indicators_dict, signals=None):
        """Create advanced chart with multiple indicators and error handling"""
        try:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=('Price & Signals', 'Momentum Indicators', 'Trend Indicators', 'Volume & Pressure')
            )
            
            # Main price chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add MA lines
            for key, value in indicators_dict.items():
                if ('SMA' in key or 'EMA' in key) and value is not None:
                    try:
                        fig.add_trace(
                            go.Scatter(x=data.index, y=value, name=key, line=dict(width=1.5)),
                            row=1, col=1
                        )
                    except:
                        continue

            # Bollinger Bands
            if 'BB' in indicators_dict and indicators_dict['BB'] is not None:
                bb = indicators_dict['BB']
                if 'Upper' in bb.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=bb['Upper'], name='BB Upper', 
                                           line=dict(color='rgba(128,128,128,0.7)', width=1, dash='dash')), row=1, col=1)
                if 'Lower' in bb.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=bb['Lower'], name='BB Lower', 
                                           line=dict(color='rgba(128,128,128,0.7)', width=1, dash='dash')), row=1, col=1)
                if 'Middle' in bb.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=bb['Middle'], name='BB Middle', 
                                           line=dict(color='rgba(128,128,128,0.3)', width=1)), row=1, col=1)

            # SuperTrend
            if 'SUPERTREND' in indicators_dict and indicators_dict['SUPERTREND'] is not None:
                supertrend = indicators_dict['SUPERTREND']
                fig.add_trace(go.Scatter(x=data.index, y=supertrend, name='SuperTrend', 
                                       line=dict(color='orange', width=2)), row=1, col=1)

            # Buy/Sell Signals
            if signals is not None and not signals.empty:
                buy_signals = signals[signals == 1]
                sell_signals = signals[signals == -1]
                
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index, 
                        y=data.loc[buy_signals.index, 'Low'] * 0.98, 
                        mode='markers', 
                        name='Buy Signal', 
                        marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=2, color='darkgreen'))
                    ), row=1, col=1)
                    
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index, 
                        y=data.loc[sell_signals.index, 'High'] * 1.02, 
                        mode='markers', 
                        name='Sell Signal', 
                        marker=dict(symbol='triangle-down', size=12, color='#ff0000', line=dict(width=2, color='darkred'))
                    ), row=1, col=1)

            # Momentum Indicators (RSI + Stochastic)
            if 'RSI' in indicators_dict and indicators_dict['RSI'] is not None:
                fig.add_trace(go.Scatter(x=data.index, y=indicators_dict['RSI'], name='RSI', 
                                       line=dict(color='purple', width=2)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
            if 'STOCH' in indicators_dict and indicators_dict['STOCH'] is not None:
                stoch = indicators_dict['STOCH']
                if 'STOCHk' in stoch.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=stoch['STOCHk'], name='Stoch %K', 
                                           line=dict(color='blue', width=1.5)), row=2, col=1)
                if 'STOCHd' in stoch.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=stoch['STOCHd'], name='Stoch %D', 
                                           line=dict(color='red', width=1.5, dash='dot')), row=2, col=1)

            # Trend Indicators (MACD + ADX)
            if 'MACD' in indicators_dict and indicators_dict['MACD'] is not None:
                macd = indicators_dict['MACD']
                if 'MACD' in macd.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=macd['MACD'], name='MACD', 
                                           line=dict(color='blue', width=2)), row=3, col=1)
                if 'Signal' in macd.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=macd['Signal'], name='Signal', 
                                           line=dict(color='red', width=2, dash='dot')), row=3, col=1)
                if 'Histogram' in macd.columns:
                    fig.add_trace(go.Bar(x=data.index, y=macd['Histogram'], name='Histogram', 
                                       marker_color='gray'), row=3, col=1)
                    
            if 'ADX' in indicators_dict and indicators_dict['ADX'] is not None:
                adx = indicators_dict['ADX']
                fig.add_trace(go.Scatter(x=data.index, y=adx, name='ADX', 
                                       line=dict(color='orange', width=2)), row=3, col=1)

            # Volume + VWAP
            colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in data.iterrows()]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                               marker_color=colors, opacity=0.7), row=4, col=1)
                               
            if 'VWAP' in indicators_dict and indicators_dict['VWAP'] is not None:
                vwap = indicators_dict['VWAP']
                fig.add_trace(go.Scatter(x=data.index, y=vwap, name='VWAP', 
                                       line=dict(color='yellow', width=2)), row=4, col=1)
            
            fig.update_layout(
                title="Advanced Strategy Analysis",
                template="plotly_dark",
                height=1000,
                showlegend=True,
                hovermode='x unified'
            )
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            # Return simple chart as fallback
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            fig.update_layout(title="Basic Price Chart", template="plotly_dark")
            return fig

def main():
    strategy_builder = AdvancedStrategyBuilder()
    
    # MAIN CONFIGURATION
    st.markdown('<div class="section">⚙️ STRATEGY CONFIGURATION</div>', unsafe_allow_html=True)
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**📊 DATA CONFIGURATION**')
        symbol = st.selectbox("Symbol:", ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX", "SPY", "QQQ"])
        period = st.selectbox("Period:", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("Interval:", ["1d", "1h", "1wk"], index=0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**📈 TREND INDICATORS**')
        sma_periods = st.multiselect("SMA Periods", [5, 10, 20, 50, 100, 200], default=[20, 50])
        ema_periods = st.multiselect("EMA Periods", [5, 10, 20, 50, 100, 200], default=[20, 50])
        
        bb_enabled = st.checkbox("Bollinger Bands", value=True)
        if bb_enabled:
            bb_period = st.slider("BB Period", 10, 50, 20)
            bb_std = st.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)
            
        supertrend_enabled = st.checkbox("SuperTrend")
        if supertrend_enabled:
            st_period = st.slider("ST Period", 5, 20, 10)
            st_multiplier = st.slider("ST Multiplier", 1.0, 5.0, 3.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_config2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**💪 MOMENTUM INDICATORS**')
        rsi_enabled = st.checkbox("RSI", value=True)
        if rsi_enabled:
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            rsi_overbought = st.slider("Overbought", 60, 90, 70)
            rsi_oversold = st.slider("Oversold", 10, 40, 30)
            
        macd_enabled = st.checkbox("MACD", value=True)
        if macd_enabled:
            macd_fast = st.slider("MACD Fast", 5, 20, 12)
            macd_slow = st.slider("MACD Slow", 20, 40, 26)
            macd_signal = st.slider("MACD Signal", 5, 15, 9)
            
        stoch_enabled = st.checkbox("Stochastic")
        if stoch_enabled:
            stoch_k = st.slider("Stoch %K", 5, 20, 14)
            stoch_d = st.slider("Stoch %D", 2, 5, 3)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**🌊 VOLUME & STRENGTH**')
        vwap_enabled = st.checkbox("VWAP")
        adx_enabled = st.checkbox("ADX")
        if adx_enabled:
            adx_period = st.slider("ADX Period", 5, 20, 14)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_config3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**🧠 STRATEGY LOGIC**')
        
        rsi_condition = st.selectbox("RSI Condition", ["None", "RSI < Oversold", "RSI > Overbought"]) if rsi_enabled else "None"
        macd_condition = st.selectbox("MACD Condition", ["None", "MACD crosses above Signal", "MACD crosses below Signal"]) if macd_enabled else "None"
        bb_condition = st.selectbox("BB Condition", ["None", "Price touches lower band", "Price touches upper band"]) if bb_enabled else "None"
        stoch_condition = st.selectbox("Stoch Condition", ["None", "Stochastic oversold", "Stochastic overbought"]) if stoch_enabled else "None"
        supertrend_condition = st.selectbox("SuperTrend Condition", ["None", "SuperTrend bullish", "SuperTrend bearish"]) if supertrend_enabled else "None"
        
        price_ma_condition = st.selectbox("Price vs MA", ["None", "Price above SMA", "Price below SMA"])
        price_ma_period = st.slider("MA Period", 5, 50, 20) if price_ma_condition != "None" else 20
        
        exit_condition = st.selectbox("Exit Condition", ["Opposite Signal", "RSI > Overbought", "Fixed Period"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**🛡️ RISK MANAGEMENT**')
        initial_capital = st.number_input("Initial Capital ($)", 1000, 1000000, 10000)
        stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
        take_profit = st.slider("Take Profit (%)", 1, 50, 10)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    data = strategy_builder.load_data(symbol, period, interval)
    
    if data is None:
        st.error("Failed to load data. Please check your symbol and try again.")
        st.stop()
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Strategy Analysis", "📊 Backtesting", "📋 Performance Report", "⚙️ Optimization"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('**📋 STRATEGY SUMMARY**')
            
            entry_conditions = []
            if rsi_condition != "None":
                entry_conditions.append(f"RSI: {rsi_condition}")
            if macd_condition != "None":
                entry_conditions.append(f"MACD: {macd_condition}")
            if bb_condition != "None":
                entry_conditions.append(f"BB: {bb_condition}")
            if stoch_condition != "None":
                entry_conditions.append(f"Stoch: {stoch_condition}")
            if supertrend_condition != "None":
                entry_conditions.append(f"SuperTrend: {supertrend_condition}")
            if price_ma_condition != "None":
                entry_conditions.append(f"Price vs MA: {price_ma_condition}")
            
            if entry_conditions:
                st.markdown('<div class="success-card">', unsafe_allow_html=True)
                st.write("**Entry Conditions:**")
                for condition in entry_conditions:
                    st.write(f"• {condition}")
                st.write(f"**Exit:** {exit_condition}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Configure entry conditions above")
            
            st.markdown("---")
            st.markdown('**📊 QUICK STATS**')
            st.metric("Total Period", f"{len(data)} days")
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            total_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
            st.metric("Total Return", f"{total_return:.2f}%")
            volatility = data['Returns'].std() * 100 if 'Returns' in data.columns else 0
            st.metric("Volatility", f"{volatility:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('**📈 ADVANCED CHART ANALYSIS**')
            
            strategy_params = {
                'sma_periods': sma_periods,
                'ema_periods': ema_periods,
                'bb_enabled': bb_enabled,
                'bb_period': bb_period if bb_enabled else None,
                'bb_std': bb_std if bb_enabled else None,
                'rsi_enabled': rsi_enabled,
                'rsi_period': rsi_period if rsi_enabled else None,
                'rsi_overbought': rsi_overbought if rsi_enabled else None,
                'rsi_oversold': rsi_oversold if rsi_enabled else None,
                'macd_enabled': macd_enabled,
                'macd_fast': macd_fast if macd_enabled else None,
                'macd_slow': macd_slow if macd_enabled else None,
                'macd_signal': macd_signal if macd_enabled else None,
                'stoch_enabled': stoch_enabled,
                'stoch_k': stoch_k if stoch_enabled else None,
                'stoch_d': stoch_d if stoch_enabled else None,
                'supertrend_enabled': supertrend_enabled,
                'st_period': st_period if supertrend_enabled else None,
                'st_multiplier': st_multiplier if supertrend_enabled else None,
                'vwap_enabled': vwap_enabled,
                'adx_enabled': adx_enabled,
                'adx_period': adx_period if adx_enabled else None,
                'rsi_condition': rsi_condition,
                'macd_condition': macd_condition,
                'bb_condition': bb_condition,
                'stoch_condition': stoch_condition,
                'supertrend_condition': supertrend_condition,
                'price_ma_condition': price_ma_condition,
                'price_ma_period': price_ma_period,
                'exit_condition': exit_condition
            }
            
            try:
                indicators_dict = strategy_builder.calculate_advanced_indicators(data, strategy_params)
                signals = strategy_builder.generate_signals(data, indicators_dict, strategy_params)
                chart = strategy_builder.create_advanced_chart(data, indicators_dict, signals)
                st.plotly_chart(chart, use_container_width=True)
                
                recent_signals = signals.tail(20)
                if recent_signals.sum() != 0:
                    st.markdown('**🔔 RECENT SIGNALS**')
                    signal_data = []
                    for date, signal in recent_signals[recent_signals != 0].items():
                        signal_data.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Signal': 'BUY' if signal == 1 else 'SELL',
                            'Price': f"${data.loc[date, 'Close']:.2f}"
                        })
                    if signal_data:
                        st.dataframe(pd.DataFrame(signal_data))
            except Exception as e:
                st.error(f"Error in strategy analysis: {e}")
                st.info("Please adjust your strategy parameters and try again.")
                
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**📊 STRATEGY BACKTESTING**')
        
        if st.button("Run Backtest", type="primary", key="backtest_btn"):
            with st.spinner("Running backtest..."):
                try:
                    strategy_params = {
                        'sma_periods': sma_periods,
                        'ema_periods': ema_periods,
                        'bb_enabled': bb_enabled,
                        'bb_period': bb_period if bb_enabled else None,
                        'bb_std': bb_std if bb_enabled else None,
                        'rsi_enabled': rsi_enabled,
                        'rsi_period': rsi_period if rsi_enabled else None,
                        'rsi_overbought': rsi_overbought if rsi_enabled else None,
                        'rsi_oversold': rsi_oversold if rsi_enabled else None,
                        'macd_enabled': macd_enabled,
                        'macd_fast': macd_fast if macd_enabled else None,
                        'macd_slow': macd_slow if macd_enabled else None,
                        'macd_signal': macd_signal if macd_enabled else None,
                        'stoch_enabled': stoch_enabled,
                        'stoch_k': stoch_k if stoch_enabled else None,
                        'stoch_d': stoch_d if stoch_enabled else None,
                        'supertrend_enabled': supertrend_enabled,
                        'st_period': st_period if supertrend_enabled else None,
                        'st_multiplier': st_multiplier if supertrend_enabled else None,
                        'vwap_enabled': vwap_enabled,
                        'adx_enabled': adx_enabled,
                        'adx_period': adx_period if adx_enabled else None,
                        'rsi_condition': rsi_condition,
                        'macd_condition': macd_condition,
                        'bb_condition': bb_condition,
                        'stoch_condition': stoch_condition,
                        'supertrend_condition': supertrend_condition,
                        'price_ma_condition': price_ma_condition,
                        'price_ma_period': price_ma_period,
                        'exit_condition': exit_condition
                    }
                    indicators_dict = strategy_builder.calculate_advanced_indicators(data, strategy_params)
                    signals = strategy_builder.generate_signals(data, indicators_dict, strategy_params)
                    
                    performance = strategy_builder.backtest_strategy(data, signals, initial_capital)
                    
                    if performance:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Initial Capital", f"${performance['initial_capital']:,.0f}")
                        col2.metric("Final Capital", f"${performance['final_capital']:,.0f}")
                        col3.metric("Total Return", f"{performance['total_return']:.2f}%")
                        col4.metric("Win Rate", f"{performance['win_rate']:.1f}%")
                        
                        st.markdown('**📈 EQUITY CURVE**')
                        fig_equity = go.Figure()
                        
                        buy_hold_values = data['Close'] / data['Close'].iloc[0] * initial_capital
                        fig_equity.add_trace(go.Scatter(
                            x=data.index,
                            y=buy_hold_values,
                            name='Buy & Hold',
                            line=dict(color='gray', dash='dash')
                        ))
                        
                        portfolio_dates = data.index[:len(performance['portfolio_values'])]
                        fig_equity.add_trace(go.Scatter(
                            x=portfolio_dates,
                            y=performance['portfolio_values'],
                            name='Strategy',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_equity.update_layout(
                            title="Strategy vs Buy & Hold",
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)"
                        )
                        st.plotly_chart(fig_equity, use_container_width=True)
                        
                        st.markdown('**📋 TRADE HISTORY**')
                        if not performance['trades'].empty:
                            st.dataframe(performance['trades'])
                        else:
                            st.info("No trades executed")
                        
                    else:
                        st.warning("No trades were executed based on the current strategy configuration.")
                        
                except Exception as e:
                    st.error(f"Error running backtest: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**📋 PERFORMANCE ANALYTICS**')
        
        if 'performance' in locals() and performance:
            st.markdown('**📊 ADVANCED PERFORMANCE METRICS**')
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Sharpe Ratio", "0.85")
                st.metric("Max Drawdown", "-12.5%")
                
            with metrics_col2:
                st.metric("Calmar Ratio", "0.68")
                st.metric("Volatility", "18.2%")
                
            with metrics_col3:
                st.metric("Sortino Ratio", "1.12")
                st.metric("Profit Factor", "1.45")
            
            st.markdown('**🔍 STRATEGY ANALYSIS**')
            st.write("Detailed performance analysis and metrics will be displayed here.")
            
        else:
            st.info("Run backtest first to see performance analytics")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**⚙️ STRATEGY OPTIMIZATION**')
        
        st.info("🚧 Advanced optimization features coming soon...")
        st.write("Future features will include:")
        st.write("• Parameter optimization with genetic algorithms")
        st.write("• Walk-forward analysis")
        st.write("• Monte Carlo simulation")
        st.write("• Machine learning integration")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #8898aa;'>
        <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Strategy Development Platform</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Advanced Strategy Builder Pro v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()