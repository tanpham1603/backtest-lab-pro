import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Cấu hình trang (Nên đặt ở đầu) ---
st.set_page_config(
    page_title="🚀 Backtest Lab Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/user/backtest-lab',
        'Report a bug': 'https://github.com/user/backtest-lab/issues',
        'About': '''
        # Backtest Lab Pro 🚀
        **Phần mềm backtesting chuyên nghiệp**
        
        - Dashboard tương tác
        - Tối ưu hóa chiến lược
        - Quản lý rủi ro
        - Tích hợp Live Trading
        '''
    }
)

# --- Sidebar ---
# Các thành phần trong sidebar chính sẽ hiển thị trên tất cả các trang
st.sidebar.title("🚀 Backtest Lab Pro")
st.sidebar.markdown("---")

# Hướng dẫn điều hướng
st.sidebar.success("Chọn một chức năng ở trên ⬆️")

# Các thông tin phụ có thể giữ lại ở sidebar
with st.sidebar.expander("👤 Thông tin tài khoản (Demo)"):
    st.success("✅ **Demo User**")
    st.info("💰 Virtual Balance: $100,000")
    st.metric("📊 Strategies Created", "23")

# --- Nội dung chính của trang chủ ---

st.title("🚀 Backtest Lab Professional")
st.markdown("### Hệ thống backtesting và phân tích chiến lược giao dịch chuyên nghiệp")
st.markdown("---")

# Giới thiệu các tính năng chính
st.subheader("Các tính năng chính")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("""
    📊 **Dashboard**
    - Biểu đồ tương tác
    - Real-time metrics
    - Multi-timeframe
    """)

with col2:
    st.success("""
    🔧 **Strategy Builder**
    - Drag & Drop
    - 15+ Indicators
    - Custom logic
    """)

with col3:
    st.warning("""
    ⚡ **Optimizer**
    - Grid Search
    - Random Search
    - Bayesian Opt
    """)

with col4:
    st.error("""
    🛡️ **Risk Manager**
    - Stop Loss/Profit
    - Position Sizing
    - Portfolio Risk
    """)

st.markdown("---")

# Demo nhanh một biểu đồ
st.subheader("🎯 Demo nhanh Biểu đồ")

# Tạo dữ liệu mẫu
dates = pd.date_range('2023-01-01', periods=100, freq='D')
price = 100 + np.cumsum(np.random.randn(100) * 0.5)

# Vẽ biểu đồ
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=price, name='Price', line=dict(color='cyan')))
fig.update_layout(
    title="Biểu đồ giá mẫu",
    template="plotly_dark",
    height=400
)
st.plotly_chart(fig, use_container_width=True)
