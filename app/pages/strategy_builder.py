import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_ta as ta

# --- Cấu hình trang ---
st.set_page_config(page_title="🔧 Strategy Builder", page_icon="🔧", layout="wide")

# --- Tùy chỉnh CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .stButton>button {
            width: 100%;
        }
        .stExpander {
            border: 1px solid #30363D;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)


def create_strategy_chart(data, indicators_dict, signals=None):
    """Create strategy builder chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('Price & Signals', 'RSI', 'MACD', 'Volume')
    )
    
    # Biểu đồ giá chính
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
    
    # Thêm các chỉ báo vào biểu đồ giá
    for key, value in indicators_dict.items():
        if 'SMA' in key or 'EMA' in key:
            fig.add_trace(
                go.Scatter(x=data.index, y=value, name=key, line=dict(width=1)),
                row=1, col=1
            )

    # Bollinger Bands
    if 'BB' in indicators_dict and indicators_dict['BB'] is not None:
        bb = indicators_dict['BB']
        fig.add_trace(go.Scatter(x=data.index, y=bb['Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=bb['Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)

    # Tín hiệu Mua/Bán
    if signals is not None:
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=data.loc[buy_signals.index, 'Low'] * 0.98, mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='#28a745')), row=1, col=1)
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=data.loc[sell_signals.index, 'High'] * 1.02, mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='#dc3545')), row=1, col=1)

    # RSI
    if 'RSI' in indicators_dict and indicators_dict['RSI'] is not None:
        fig.add_trace(go.Scatter(x=data.index, y=indicators_dict['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if 'MACD' in indicators_dict and indicators_dict['MACD'] is not None:
        macd = indicators_dict['MACD']
        fig.add_trace(go.Scatter(x=data.index, y=macd['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=macd['Signal'], name='Signal', line=dict(color='red', dash='dot')), row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=macd['Histogram'], name='Histogram', marker_color='gray'), row=3, col=1)

    # Volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'), row=4, col=1)
    
    fig.update_layout(title="Strategy Builder Chart", template="plotly_dark", height=800, showlegend=True)
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def main():
    st.title("🔧 Strategy Builder")
    st.markdown("### Build and visualize your trading strategy.")
    
    with st.sidebar:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
        st.header("🎛️ Strategy Configuration")
        symbol = st.selectbox("📈 Select Symbol:", ["AAPL", "MSFT", "GOOGL", "TSLA"])
        period = st.selectbox("📅 Time Period:", ["6mo", "1y", "2y"], index=1)

    @st.cache_data
    def load_data(symbol, period):
        try:
            data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if data.empty:
                st.error(f"Dont have data for {symbol}.")
                return None
            return data
        except Exception as e:
            st.error(f"Error loading data from yfinance: {e}")
            return None
            
    data = load_data(symbol, period)

    if data is None:
        st.stop()

    data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Downloaded data is missing required columns. Current columns: {list(data.columns)}")
        st.stop()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📊 Indicators")

        with st.expander("📈 Trend", expanded=True):
            sma_periods = st.multiselect("SMA Periods", [10, 20, 50, 100, 200], default=[20, 50])
            ema_periods = st.multiselect("EMA Periods", [10, 20, 50, 100, 200], default=[])
            bb_enabled = st.checkbox("Bollinger Bands")
            bb_period, bb_std = (st.slider("BB Period", 10, 50, 20), st.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)) if bb_enabled else (None, None)
        
        with st.expander("Momentum"):
            rsi_enabled = st.checkbox("RSI", value=True)
            rsi_period, rsi_overbought, rsi_oversold = (st.slider("RSI Period", 5, 30, 14), 70, 30) if rsi_enabled else (None, None, None)
            
            macd_enabled = st.checkbox("MACD", value=True)
            macd_fast, macd_slow, macd_signal = (st.slider("MACD Fast", 5, 20, 12), st.slider("MACD Slow", 20, 40, 26), st.slider("MACD Signal", 5, 15, 9)) if macd_enabled else (None, None, None)
        
        st.subheader("🧠 Strategy logic")
        
        entry_conditions = []
        if rsi_enabled:
            rsi_condition = st.selectbox("RSI Condition", ["None", "RSI < Oversold", "RSI > Overbought"])
            if rsi_condition != "None": entry_conditions.append(f"RSI: {rsi_condition}")
        
        if macd_enabled:
            macd_condition = st.selectbox("MACD Condition", ["None", "MACD crosses above Signal"])
            if macd_condition != "None": entry_conditions.append(f"MACD: {macd_condition}")
        
        if entry_conditions:
            st.info("**Entry:** " + " AND ".join(entry_conditions))
    
    with col2:
        st.subheader("📈 Visualization")

        indicators_dict = {}
        
        strategy_list = []
        if sma_periods: strategy_list.extend([{"kind": "sma", "length": p} for p in sma_periods])
        if ema_periods: strategy_list.extend([{"kind": "ema", "length": p} for p in ema_periods])
        if bb_enabled: strategy_list.append({"kind": "bbands", "length": bb_period, "std": bb_std})
        if rsi_enabled: strategy_list.append({"kind": "rsi", "length": rsi_period})
        if macd_enabled: strategy_list.append({"kind": "macd", "fast": macd_fast, "slow": macd_slow, "signal": macd_signal})
        
        if strategy_list:
            data.ta.strategy(ta.Strategy(name="Custom Strategy", ta=strategy_list))

        for p in sma_periods: indicators_dict[f'SMA_{p}'] = data.get(f'SMA_{p}')
        for p in ema_periods: indicators_dict[f'EMA_{p}'] = data.get(f'EMA_{p}')
        if bb_enabled: indicators_dict['BB'] = pd.DataFrame({'Upper': data.get(f'BBU_{bb_period}_{bb_std}'), 'Lower': data.get(f'BBL_{bb_period}_{bb_std}')})
        if rsi_enabled: indicators_dict['RSI'] = data.get(f'RSI_{rsi_period}')
        if macd_enabled: indicators_dict['MACD'] = pd.DataFrame({'MACD': data.get(f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'), 'Signal': data.get(f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'), 'Histogram': data.get(f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}')})
        
        signals = pd.Series(0, index=data.index)
        if rsi_enabled and "RSI < Oversold" in str(entry_conditions):
            rsi = indicators_dict.get('RSI')
            if rsi is not None:
                signals[(rsi < rsi_oversold) & (rsi.shift(1) >= rsi_oversold)] = 1
        
        if macd_enabled and "MACD crosses above Signal" in str(entry_conditions):
            macd = indicators_dict.get('MACD')
            if macd is not None and not macd.isnull().all().all():
                signals[(macd['MACD'] > macd['Signal']) & (macd['MACD'].shift(1) <= macd['Signal'].shift(1))] = 1
        
        chart = create_strategy_chart(data, indicators_dict, signals)
        st.plotly_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
