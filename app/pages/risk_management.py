import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Thêm đường dẫn để import các module từ thư mục app
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # Lưu ý: Trang này hiện đang dùng dữ liệu mẫu, chưa cần đến loader
    # from loaders.data_loader import DataLoader 
except ImportError:
    st.error("Lỗi import. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(page_title="Risk Manager", page_icon="🛡️", layout="wide")

# --- Lớp quản lý rủi ro (tận dụng từ code cũ) ---
class RiskManager:
    """Hệ thống quản lý rủi ro nâng cao"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    def calculate_position_size(self, price, stop_loss_price, risk_percent=2):
        """Tính kích thước vị thế dựa trên % rủi ro"""
        risk_per_share = abs(price - stop_loss_price)
        risk_amount = self.current_capital * (risk_percent / 100)
        
        if risk_per_share == 0:
            return 0
            
        position_size = int(risk_amount / risk_per_share)
        max_position_value = self.current_capital * 0.1  # Giới hạn tối đa 10% vốn cho mỗi vị thế
        
        if position_size * price > max_position_value:
            position_size = int(max_position_value / price)
            
        return position_size
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Tính Kelly Criterion cho position sizing"""
        if avg_loss == 0 or avg_win <= 0:
            return 0
            
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_percent = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Giới hạn Kelly không quá 25% để giảm rủi ro
        return max(0, min(kelly_percent, 0.25))
    
    def simulate_portfolio_monte_carlo(self, expected_return, volatility, days=252, simulations=1000):
        """Chạy mô phỏng Monte Carlo cho giá trị danh mục."""
        daily_returns = np.random.normal(expected_return/days, volatility/np.sqrt(days), (days, simulations))
        price_paths = np.cumprod(1 + daily_returns, axis=0) * self.initial_capital
        return price_paths

# --- Giao diện Streamlit ---
def main():
    st.title("🛡️ Hệ thống Quản lý Rủi ro")
    st.markdown("### Phân tích rủi ro, tính toán kích thước vị thế và mô phỏng danh mục.")
    
    # Sidebar
    st.sidebar.header("🎛️ Tham số Rủi ro")
    initial_capital = st.sidebar.number_input("💰 Vốn ban đầu ($)", 10000, 1000000, 100000)
    max_risk_per_trade = st.sidebar.slider("📉 Rủi ro tối đa mỗi giao dịch (%)", 0.5, 10.0, 2.0, 0.1)
    
    # Khởi tạo đối tượng RiskManager
    risk_manager = RiskManager(initial_capital)

    # Tạo các tab chức năng
    tab1, tab2, tab3 = st.tabs(["🎯 Tính toán Vị thế", "🎲 Mô phỏng Monte Carlo", "📊 Phân tích Danh mục"])

    # --- Tab 1: Position Sizing ---
    with tab1:
        st.subheader("🎯 Công cụ Tính toán Kích thước Vị thế")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Thiết lập Giao dịch**")
            entry_price = st.number_input("Giá vào lệnh ($)", 0.01, 10000.0, 150.0, 0.01)
            stop_loss_price = st.number_input("Giá dừng lỗ ($)", 0.01, 10000.0, 140.0, 0.01)
            sizing_method = st.selectbox("Phương pháp tính:", ["Fixed Risk %", "Kelly Criterion"])
            
            if sizing_method == "Kelly Criterion":
                win_rate = st.slider("Tỷ lệ thắng (%)", 30, 90, 60) / 100
                avg_win = st.number_input("Lợi nhuận trung bình ($)", 1, 1000, 50)
                avg_loss = st.number_input("Lỗ trung bình ($)", 1, 1000, 30)

        with col2:
            st.markdown("**Kết quả Phân tích**")
            
            if sizing_method == "Fixed Risk %":
                position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price, max_risk_per_trade)
            elif sizing_method == "Kelly Criterion":
                kelly_fraction = risk_manager.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
                position_size = int((initial_capital * kelly_fraction) / entry_price)
            
            position_value = position_size * entry_price
            
            st.metric("Kích thước vị thế (Số cổ phiếu)", f"{position_size:,}")
            st.metric("Giá trị vị thế", f"${position_value:,.2f}")
            st.metric("% Vốn sử dụng", f"{(position_value/initial_capital)*100:.2f}%")

    # --- Tab 2: Monte Carlo Simulation ---
    with tab2:
        st.subheader("🎲 Mô phỏng Monte Carlo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Tham số Mô phỏng**")
            expected_return = st.slider("Lợi nhuận kỳ vọng hàng năm (%)", -20, 50, 12) / 100
            annual_volatility = st.slider("Độ biến động hàng năm (%)", 5, 80, 20) / 100
            time_horizon = st.slider("Số ngày mô phỏng", 30, 1000, 252)
            num_simulations = st.select_slider("Số lần mô phỏng", [100, 500, 1000, 2000], 1000)

            if st.button("🎲 Chạy Mô phỏng"):
                with st.spinner("Đang chạy mô phỏng..."):
                    paths = risk_manager.simulate_portfolio_monte_carlo(expected_return, annual_volatility, time_horizon, num_simulations)
                    st.session_state.mc_results = paths

        with col2:
            if 'mc_results' in st.session_state:
                paths = st.session_state.mc_results
                final_values = paths[-1, :]

                fig = go.Figure()
                for i in range(min(100, num_simulations)): # Chỉ vẽ 100 đường để không bị lag
                    fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(100,149,237,0.2)'), showlegend=False))
                
                fig.update_layout(title=f'{num_simulations} Kịch bản cho Danh mục', template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                st.metric("Giá trị trung bình cuối kỳ", f"${np.mean(final_values):,.2f}")
                st.metric("Xác suất lỗ", f"{(final_values < initial_capital).mean():.2%}")

    # --- Tab 3: Portfolio Risk ---
    with tab3:
        st.subheader("📊 Phân tích Rủi ro Danh mục (Demo)")
        
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
            st.markdown("**Cấu trúc Danh mục**")
            st.dataframe(portfolio_df.style.format({
                "Position_Value": "${:,.0f}", "Weight": "{:.2%}",
                "Volatility": "{:.1%}", "Risk_Contribution": "{:.2%}"
            }))
        with col2:
            st.markdown("**Phân bổ Rủi ro**")
            fig_pie = px.pie(
                portfolio_df, values='Risk_Contribution', names='Symbol',
                title='Đóng góp vào Rủi ro của từng Vị thế', template='plotly_dark'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
