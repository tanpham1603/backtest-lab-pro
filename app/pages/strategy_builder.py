import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.indicators import TechnicalIndicators, AdvancedIndicators
from data.data_loader import DataLoader

st.set_page_config(page_title="🔧 Strategy Builder", page_icon="🔧", layout="wide")

class StrategyBuilder:
    """Lớp xây dựng chiến lược tương tác"""
    
    def __init__(self):
        self.indicators = {}
        self.conditions = []
        self.signals = pd.Series()
    
    def add_indicator(self, name, indicator_data):
        """Thêm chỉ báo vào strategy"""
        self.indicators[name] = indicator_data
    
    def add_condition(self, condition):
        """Thêm điều kiện giao dịch"""
        self.conditions.append(condition)
    
    def generate_signals(self, data):
        """Tạo tín hiệu giao dịch dựa vào điều kiện"""
        signals = pd.Series(0, index=data.index)
        
        # Logic sẽ được thêm vào sau
        return signals

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
    
    # Initialize session state
    if 'strategy_conditions' not in st.session_state:
        st.session_state.strategy_conditions = []
    if 'selected_indicators' not in st.session_state:
        st.session_state.selected_indicators = []
    
    # Sidebar - Strategy Configuration
    st.sidebar.header("🎛️ Cấu hình chiến lược")
    
    # Data selection
    symbol = st.sidebar.selectbox("📈 Chọn mã:", ["AAPL", "MSFT", "GOOGL", "TSLA"])
    period = st.sidebar.selectbox("📅 Thời gian:", ["6mo", "1y", "2y"], index=1)
    
    # Load data
    @st.cache_data
    def load_data(symbol, period):
        loader = DataLoader()
        return loader.download_stock_data(symbol, period=period, save=False)
    
    try:
        data = load_data(symbol, period)
        tech_indicators = TechnicalIndicators(data)
        advanced_indicators = AdvancedIndicators(data)
    except:
        st.error("Không thể tải dữ liệu. Sử dụng dữ liệu demo.")
        # Demo data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(252).cumsum() + 150,
            'High': np.random.randn(252).cumsum() + 155,
            'Low': np.random.randn(252).cumsum() + 145,
            'Close': np.random.randn(252).cumsum() + 150,
            'Volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)
        tech_indicators = TechnicalIndicators(data)
        advanced_indicators = AdvancedIndicators(data)
    
    # Strategy Builder Interface
    col1, col2 = st.columns([4,5])
    
    with col1:
        st.subheader("📊 Available Indicators")
        
        # Indicator categories
        with st.expander("📈 Trend Indicators", expanded=True):
            sma_periods = st.multiselect("SMA Periods", [5,10,20,25,50,100,200], default=[5])
            ema_periods = st.multiselect("EMA Periods", [5,10,20,25,50,100,200], default=[10])
            
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
        
        with st.expander("🔄 Oscillators"):
            stoch_enabled = st.checkbox("Stochastic")
            if stoch_enabled:
                stoch_k = st.slider("Stochastic K", 5, 20, 14)
                stoch_d = st.slider("Stochastic D", 3, 10, 3)
            
            williams_enabled = st.checkbox("Williams %R")
            cci_enabled = st.checkbox("CCI")
        
        with st.expander("💹 Volume Indicators"):
            volume_sma = st.checkbox("Volume SMA")
            if volume_sma:
                vol_period = st.slider("Volume SMA Period", 10, 50, 20)
        
        # Strategy Logic Builder
        st.subheader("🧠 Strategy Logic")
        
        st.markdown("**Entry Conditions (Buy)**")
        entry_conditions = []
        
        # RSI condition
        if rsi_enabled:
            rsi_condition = st.selectbox(
                "RSI Condition",
                ["None", "RSI < Oversold", "RSI > Overbought", "RSI crosses above 50", "RSI crosses below 50"]
            )
            if rsi_condition != "None":
                entry_conditions.append(f"RSI: {rsi_condition}")
        
        # MACD condition
        if macd_enabled:
            macd_condition = st.selectbox(
                "MACD Condition", 
                ["None", "MACD > Signal", "MACD < Signal", "MACD crosses above Signal", "MACD crosses below Signal"]
            )
            if macd_condition != "None":
                entry_conditions.append(f"MACD: {macd_condition}")
        
        # Moving Average condition
        if len(sma_periods) >= 2:
            ma_condition = st.selectbox(
                "MA Condition",
                ["None", f"SMA{sma_periods[0]} > SMA{sma_periods[1]}", f"SMA{sma_periods[0]} < SMA{sma_periods[1]}"]
            )
            if ma_condition != "None":
                entry_conditions.append(f"MA: {ma_condition}")
        
        st.markdown("**Exit Conditions (Sell)**")
        exit_conditions = []
        
        # Simple exit conditions
        exit_type = st.radio("Exit Type", ["Opposite Signal", "Fixed %", "Trailing Stop"])
        
        if exit_type == "Fixed %":
            stop_loss = st.slider("Stop Loss %", 1, 20, 5)
            take_profit = st.slider("Take Profit %", 1, 50, 10)
            exit_conditions.append(f"Stop Loss: {stop_loss}%, Take Profit: {take_profit}%")
        
        # Display current strategy
        if entry_conditions or exit_conditions:
            st.markdown("**Current Strategy:**")
            st.info("**Entry:** " + " AND ".join(entry_conditions) if entry_conditions else "No entry conditions")
            st.info("**Exit:** " + " AND ".join(exit_conditions) if exit_conditions else "No exit conditions")
    
    with col2:
        st.subheader("📈 Strategy Visualization")
        
        # Calculate selected indicators
        indicators_dict = {}
        
        # Calculate indicators based on user selection
        for period in sma_periods:
            indicators_dict[f'SMA_{period}'] = tech_indicators.sma(period)
        
        for period in ema_periods:
            indicators_dict[f'EMA_{period}'] = tech_indicators.ema(period)
        
        if bb_enabled:
            indicators_dict['BB'] = tech_indicators.bollinger_bands(bb_period, bb_std)
        
        if rsi_enabled:
            indicators_dict['RSI'] = tech_indicators.rsi(rsi_period)
        
        if macd_enabled:
            indicators_dict['MACD'] = tech_indicators.macd(macd_fast, macd_slow, macd_signal)
        
        if stoch_enabled:
            indicators_dict['Stochastic'] = tech_indicators.stochastic(stoch_k, stoch_d)
        
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
        
        # Performance summary
        if signals is not None:
            st.subheader("📊 Quick Performance")
            
            buy_signals = signals[signals == 1]
            total_signals = len(buy_signals)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Signals", total_signals)
            with col2:
                if total_signals > 0:
                    avg_days_between = (data.index[-1] - data.index).days / total_signals if total_signals > 0 else 0
                    st.metric("Avg Days Between Signals", f"{avg_days_between:.0f}")
            with col3:
                signal_strength = "High" if total_signals > 10 else "Medium" if total_signals > 5 else "Low"
                st.metric("Signal Frequency", signal_strength)
        
        # Advanced Analysis
        with st.expander("🔬 Advanced Analysis"):
            if len(indicators_dict) > 0:
                st.markdown("**Indicator Correlation Matrix**")
                
                # Create correlation matrix
                corr_data = {}
                for name, indicator in indicators_dict.items():
                    if isinstance(indicator, pd.Series):
                        corr_data[name] = indicator
                    elif isinstance(indicator, pd.DataFrame) and 'MACD' in name:
                        corr_data[f"{name}_line"] = indicator['MACD']
                
                if len(corr_data) > 1:
                    corr_df = pd.DataFrame(corr_data).corr()
                    st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
        
        # Save Strategy
        st.subheader("💾 Save Strategy")
        
        strategy_name = st.text_input("Strategy Name", value="My Custom Strategy")
        
        if st.button("💾 Save Strategy"):
            # In a real app, this would save to database
            strategy_config = {
                'name': strategy_name,
                'symbol': symbol,
                'period': period,
                'indicators': {
                    'sma_periods': sma_periods,
                    'ema_periods': ema_periods,
                    'rsi_enabled': rsi_enabled,
                    'macd_enabled': macd_enabled,
                    'bb_enabled': bb_enabled
                },
                'entry_conditions': entry_conditions,
                'exit_conditions': exit_conditions
            }
            
            st.success(f"✅ Strategy '{strategy_name}' saved successfully!")
            st.json(strategy_config)

if __name__ == "__main__":
    main()
