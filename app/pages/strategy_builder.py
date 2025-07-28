import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import yfinance as yf
import pandas_ta as ta # Thêm thư viện pandas-ta

# --- SỬA LỖI: Không cần import lớp tùy chỉnh nữa ---
# try:
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#     if project_root not in sys.path:
#         sys.path.append(project_root)
#     from app.components.indicators import TechnicalIndicators
# except ImportError:
#     st.error("Lỗi import: Không tìm thấy file 'app/components/indicators.py'. Vui lòng kiểm tra lại cấu trúc thư mục.")
#     st.stop()


st.set_page_config(page_title="🔧 Strategy Builder", page_icon="🔧", layout="wide")

# --- Bỏ lớp StrategyBuilder không cần thiết cho giao diện này ---

def create_strategy_chart(data, indicators_dict, signals=None):
    """Tạo biểu đồ cho strategy builder"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=('Price & Signals', 'RSI', 'MACD', 'Volume')
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
    
    # Add indicators to price chart
    if 'SMA_20' in indicators_dict:
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators_dict['SMA_20'],
                       name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'EMA_50' in indicators_dict:
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators_dict['EMA_50'],
                       name='EMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB' in indicators_dict:
        bb = indicators_dict['BB']
        fig.add_trace(
            go.Scatter(x=data.index, y=bb['Upper'], name='BB Upper',
                       line=dict(color='gray', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=bb['Lower'], name='BB Lower',
                       line=dict(color='gray', width=1)),
            row=1, col=1
        )
    
    # Buy/Sell signals
    if signals is not None:
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=data.loc[buy_signals.index, 'Close'],
                           mode='markers', name='Buy Signal',
                           marker=dict(symbol='triangle-up', size=12, color='green')),
                row=1, col=1
            )
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=data.loc[sell_signals.index, 'Close'],
                           mode='markers', name='Sell Signal',
                           marker=dict(symbol='triangle-down', size=12, color='red')),
                row=1, col=1
            )
    
    # RSI
    if 'RSI' in indicators_dict:
        fig.add_trace(
            go.Scatter(x=data.index, y=indicators_dict['RSI'],
                       name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in indicators_dict:
        macd_data = indicators_dict['MACD']
        fig.add_trace(
            go.Scatter(x=data.index, y=macd_data['MACD'],
                       name='MACD', line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=macd_data['Signal'],
                       name='Signal', line=dict(color='red', width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=macd_data['Histogram'],
                   name='Histogram', marker_color='gray'),
            row=3, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume',
               marker_color='lightblue'),
        row=4, col=1
    )
    
    fig.update_layout(
        title="Strategy Builder Chart",
        template="plotly_dark",
        height=900,
        showlegend=True
    )
    
    return fig

def main():
    st.title("🔧 Strategy Builder")
    st.markdown("### Xây dựng chiến lược giao dịch với giao diện kéo-thả")
    
    # Sidebar - Strategy Configuration
    st.sidebar.header("🎛️ Cấu hình chiến lược")
    
    # Data selection
    symbol = st.sidebar.selectbox("📈 Chọn mã:", ["AAPL", "MSFT", "GOOGL", "TSLA"])
    period = st.sidebar.selectbox("📅 Thời gian:", ["6mo", "1y", "2y"], index=1)
    
    # Load data
    @st.cache_data
    def load_data(symbol, period):
        try:
            data = yf.download(symbol, period=period, progress=False)
            if data.empty:
                st.error(f"Không có dữ liệu cho mã {symbol}. Vui lòng kiểm tra lại.")
                return None
            return data
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu từ yfinance: {e}")
            return None
    
    data = load_data(symbol, period)

    if data is None or data.empty:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại mã giao dịch.")
        st.stop()

    # Chuẩn hóa tên cột, xử lý cả trường hợp tên cột là tuple
    data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

    # Kiểm tra các cột cần thiết sau khi chuẩn hóa
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Dữ liệu tải về thiếu các cột cần thiết. Các cột hiện có: {list(data.columns)}")
        st.stop()

    # --- SỬA LỖI: Bỏ khởi tạo lớp TechnicalIndicators ---
    # tech_indicators = TechnicalIndicators(data)
    
    # Strategy Builder Interface
    col1, col2 = st.columns([4,5])
    
    with col1:
        st.subheader("📊 Available Indicators")
        
        # Indicator categories
        with st.expander("📈 Trend Indicators", expanded=True):
            sma_periods = st.multiselect("SMA Periods", [5,10,20,25,50,100,200], default=[20])
            ema_periods = st.multiselect("EMA Periods", [5,10,20,25,50,100,200], default=[50])
            
            # Bollinger Bands
            bb_enabled = st.checkbox("Bollinger Bands")
            if bb_enabled:
                bb_period = st.slider("BB Period", 10, 50, 20)
                bb_std = st.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)
        
        with st.expander("📊 Momentum Indicators"):
            rsi_enabled = st.checkbox("RSI", value=True)
            if rsi_enabled:
                rsi_period = st.slider("RSI Period", 5, 30, 14)
                rsi_overbought = st.slider("RSI Overbought", 70, 90, 70)
                rsi_oversold = st.slider("RSI Oversold", 10, 30, 30)
            
            macd_enabled = st.checkbox("MACD", value=True)
            if macd_enabled:
                macd_fast = st.slider("MACD Fast", 5, 20, 12)
                macd_slow = st.slider("MACD Slow", 20, 40, 26)
                macd_signal = st.slider("MACD Signal", 5, 15, 9)
        
        st.subheader("🧠 Strategy Logic")
        
        st.markdown("**Entry Conditions (Buy)**")
        entry_conditions = []
        
        if rsi_enabled:
            rsi_condition = st.selectbox("RSI Condition", ["None", "RSI < Oversold", "RSI > Overbought"])
            if rsi_condition != "None": entry_conditions.append(f"RSI: {rsi_condition}")
        
        if macd_enabled:
            macd_condition = st.selectbox("MACD Condition", ["None", "MACD crosses above Signal", "MACD crosses below Signal"])
            if macd_condition != "None": entry_conditions.append(f"MACD: {macd_condition}")
        
        if len(sma_periods) >= 2:
            ma_condition = st.selectbox("MA Condition", ["None", f"SMA{sma_periods[0]} > SMA{sma_periods[1]}"])
            if ma_condition != "None": entry_conditions.append(f"MA: {ma_condition}")
        
        st.markdown("**Exit Conditions (Sell)**")
        # (Phần logic exit có thể được phát triển thêm)
        
        if entry_conditions:
            st.markdown("**Current Strategy:**")
            st.info("**Entry:** " + " AND ".join(entry_conditions))
    
    with col2:
        st.subheader("📈 Strategy Visualization")
        
        indicators_dict = {}
        
        # --- SỬA LỖI: Tính toán chỉ báo bằng pandas-ta ---
        # Tính toán các chỉ báo được chọn và thêm vào DataFrame gốc
        data.ta.strategy(ta.Strategy(
            name="Custom Strategy",
            ta=[
                {"kind": "sma", "length": p} for p in sma_periods
            ] + [
                {"kind": "ema", "length": p} for p in ema_periods
            ] + ([
                {"kind": "bbands", "length": bb_period, "std": bb_std}
            ] if bb_enabled else []) + ([
                {"kind": "rsi", "length": rsi_period}
            ] if rsi_enabled else []) + ([
                {"kind": "macd", "fast": macd_fast, "slow": macd_slow, "signal": macd_signal}
            ] if macd_enabled else [])
        ))

        # Lấy dữ liệu chỉ báo từ DataFrame đã được tính toán
        for period in sma_periods:
            indicators_dict[f'SMA_{period}'] = data[f'SMA_{period}']
        
        for period in ema_periods:
            indicators_dict[f'EMA_{period}'] = data[f'EMA_{period}']
        
        if bb_enabled:
            indicators_dict['BB'] = pd.DataFrame({
                'Upper': data[f'BBU_{bb_period}_{bb_std}'],
                'Lower': data[f'BBL_{bb_period}_{bb_std}']
            })
        
        if rsi_enabled:
            indicators_dict['RSI'] = data[f'RSI_{rsi_period}']
        
        if macd_enabled:
            indicators_dict['MACD'] = pd.DataFrame({
                'MACD': data[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'],
                'Signal': data[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'],
                'Histogram': data[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}']
            })
        
        # Generate signals (simplified logic)
        signals = None
        if entry_conditions:
            signals = pd.Series(0, index=data.index)
            
            # Simple signal generation for demo
            if rsi_enabled and "RSI < Oversold" in str(entry_conditions):
                rsi = indicators_dict['RSI']
                signals[(rsi < rsi_oversold) & (rsi.shift(1) >= rsi_oversold)] = 1
            
            if macd_enabled and "MACD crosses above Signal" in str(entry_conditions):
                macd = indicators_dict['MACD']
                signals[(macd['MACD'] > macd['Signal']) & (macd['MACD'].shift(1) <= macd['Signal'].shift(1))] = 1
        
        # Create and display chart
        chart = create_strategy_chart(data, indicators_dict, signals)
        st.plotly_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
