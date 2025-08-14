import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Cấu hình trang ---
st.set_page_config(page_title="Risk Manager", page_icon="🛡️", layout="wide")

# --- Tùy chỉnh CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)


# --- Lớp quản lý rủi ro ---
class RiskManager:
    """Risk management system"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    def calculate_position_size(self, price, stop_loss_price, risk_percent=2):
        """Calculate position size based on % risk"""
        if price <= stop_loss_price:
            return 0, 0 # Avoid division by zero or invalid risk

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
        
        # Giới hạn Kelly không quá 25% để giảm rủi ro
        return max(0, min(kelly_percent, 0.25))
    
    def simulate_portfolio_monte_carlo(self, expected_return, volatility, days=252, simulations=1000):
        """Run Monte Carlo simulation for portfolio value."""
        daily_returns = np.random.normal(expected_return/days, volatility/np.sqrt(days), (days, simulations))
        price_paths = np.cumprod(1 + daily_returns, axis=0) * self.initial_capital
        return price_paths

# --- Streamlit Interface ---
def main():
    st.title("🛡️ Risk Management System")
    st.markdown("### Analyze risk, calculate position size, and simulate portfolio.")

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
        st.header("🎛️ Risk Parameters")
        initial_capital = st.number_input("💰 Initial Capital ($)", 10000, 1000000, 100000)

    risk_manager = RiskManager(initial_capital)

    # --- Tabs Interface ---
    tab1, tab2, tab3 = st.tabs(["🎯 Position Sizing", "🎲 Monte Carlo Simulation", "📊 Portfolio Analysis"])

    # --- Tab 1: Position Sizing ---
    with tab1:
        st.subheader("🎯 Position Sizing Tool")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Trade Setup**")
            entry_price = st.number_input("Entry Price ($)", 0.01, 10000.0, 150.0, 0.01)
            stop_loss_price = st.number_input("Stop Loss Price ($)", 0.01, 10000.0, 140.0, 0.01)
            sizing_method = st.selectbox("Sizing Method:", ["Fixed Risk %", "Kelly Criterion"])

            if sizing_method == "Fixed Risk %":
                risk_percent = st.slider("📉 Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1)
            else: # Kelly Criterion
                st.info("Enter parameters from your backtest results.")
                win_rate = st.slider("Win Rate (%)", 1, 100, 60) / 100
                avg_win = st.number_input("Average Win ($)", 1.0, 1000.0, 50.0, 0.1)
                avg_loss = st.number_input("Average Loss ($)", 1.0, 1000.0, 30.0, 0.1)

        with col2:
            st.markdown("**Analysis Results**")

            position_size = 0
            risk_amount_val = 0
            
            if sizing_method == "Fixed Risk %":
                position_size, risk_amount_val = risk_manager.calculate_position_size(entry_price, stop_loss_price, risk_percent)
            elif sizing_method == "Kelly Criterion":
                kelly_fraction = risk_manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                risk_amount_val = initial_capital * kelly_fraction
                position_size = int(risk_amount_val / entry_price) if entry_price > 0 else 0
            
            position_value = position_size * entry_price

            st.metric("Position Size (Shares)", f"{position_size:,}")
            st.metric("Position Value", f"${position_value:,.2f}")
            st.metric("Risk Amount", f"${risk_amount_val:,.2f} ({(risk_amount_val/initial_capital):.2%})")

    # --- Tab 2: Monte Carlo Simulation ---
    with tab2:
        st.subheader("🎲 Monte Carlo Simulation")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Simulation Parameters**")
            expected_return = st.slider("Expected Annual Return (%)", -20, 50, 12) / 100
            annual_volatility = st.slider("Annual Volatility (%)", 5, 80, 20) / 100
            time_horizon = st.slider("Time Horizon (Days)", 30, 1000, 252)
            num_simulations = st.select_slider("Number of Simulations", [100, 500, 1000, 2000], 1000)

            if st.button("🎲 Run Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    paths = risk_manager.simulate_portfolio_monte_carlo(expected_return, annual_volatility, time_horizon, num_simulations)
                    st.session_state.mc_results = paths
                    st.session_state.mc_params = {'sims': num_simulations, 'initial': initial_capital}

        with col2:
            if 'mc_results' in st.session_state:
                paths = st.session_state.mc_results
                params = st.session_state.mc_params
                final_values = paths[-1, :]

                fig = go.Figure()
                # Chỉ vẽ 100 đường để không bị lag
                for i in range(min(100, params['sims'])):
                    fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(100,149,237,0.2)'), showlegend=False))

                fig.update_layout(title=f'{params["sims"]} Scenarios for Portfolio', template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

                mc_col1, mc_col2, mc_col3 = st.columns(3)
                mc_col1.metric("Final Average Value", f"${np.mean(final_values):,.2f}")
                mc_col2.metric("Best case", f"${np.max(final_values):,.2f}")
                mc_col3.metric("Worst case", f"${np.min(final_values):,.2f}")
                st.metric("Probability of Loss (Value < Initial Capital)", f"{(final_values < params['initial']).mean():.2%}")

    # --- Tab 3: Portfolio Risk ---
    with tab3:
        st.subheader("📊 Portfolio Risk Analysis (Demo)")

        portfolio_data = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
            'Position_Value': [25000, 20000, 15000, 18000, 12000],
            'Volatility': [0.25, 0.22, 0.28, 0.45, 0.30]
        }
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df['Weight'] = portfolio_df['Position_Value'] / portfolio_df['Position_Value'].sum()
        portfolio_df['Risk_Contribution'] = portfolio_df['Weight'] * portfolio_df['Volatility']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Portfolio Structure**")
            st.dataframe(portfolio_df.style.format({
                "Position_Value": "${:,.0f}", "Weight": "{:.2%}",
                "Volatility": "{:.1%}", "Risk_Contribution": "{:.2%}"
            }))
        with col2:
            st.markdown("**Risk Contribution**")
            fig_pie = px.pie(
                portfolio_df, values='Risk_Contribution', names='Symbol',
                title='Risk Contribution of Each Position', template='plotly_dark'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
