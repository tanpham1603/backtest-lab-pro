import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Cấu hình trang (Nên đặt ở đầu) ---
st.set_page_config(
    page_title="🚀 Backtest Lab Pro with TanPham",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.your-repo-link.com', # Thay link repo của bạn vào đây
        'Report a bug': 'https://www.your-repo-link.com/issues', # Thay link repo của bạn vào đây
        'About': '''
         # Backtest Lab Pro 🚀
         **Hệ thống backtesting, tối ưu hóa và giao dịch thuật toán chuyên nghiệp.**
         
         Phát triển để cung cấp một môi trường mạnh mẽ cho các nhà giao dịch.
         '''
    }
)

# --- Sidebar ---
# Các thành phần trong sidebar chính sẽ hiển thị trên tất cả các trang
st.sidebar.title("🚀 Backtest Lab Pro with TanPham")
st.sidebar.markdown("---")
# Hướng dẫn điều hướng
st.sidebar.success("Choose a function from the sidebar ⬆️")

# --- Nội dung chính của trang chủ ---

st.title("Welcome to Backtest Lab Pro 🚀")
st.markdown("### Comprehensive platform for building and testing your trading strategies.")
st.markdown("---")

# --- Kiểm tra trạng thái kết nối API ---
st.subheader("Connection Status")
try:
    # Kiểm tra xem các key cần thiết có trong secrets không
    if "ALPACA_API_KEY" in st.secrets and st.secrets["ALPACA_API_KEY"] and \
       "ALPACA_API_SECRET" in st.secrets and st.secrets["ALPACA_API_SECRET"]:
        st.success("✅ Found API information in `secrets.toml`. Function pages are ready.")
    else:
        st.warning("⚠️ API information not found. Please check the `.streamlit/secrets.toml` file.")
        st.code("""
# Add to .streamlit/secrets.toml
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_API_SECRET = "YOUR_SECRET_KEY_HERE"
        """, language="toml")
except Exception:
    st.error("❌ `secrets.toml` file not found. Please create the file in the `.streamlit` directory.")
    st.code("""
# Create the .streamlit/secrets.toml file and add the following content:
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_API_SECRET = "YOUR_SECRET_KEY_HERE"
    """, language="toml")

st.markdown("---")


# Giới thiệu các tính năng chính
st.subheader("Main functions")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 Analysis & Backtest
    - Test strategies on historical data.
    - View detailed performance metrics (Sharpe, Drawdown, Winrate...).
    - Visualize results intuitively.
    """)

with col2:
    st.markdown("""
    #### ⚡ Optimization
    - Fine-tune strategy parameters.
    - Use Grid Search to find the best parameter set.
    - Compare the performance between different parameter sets.
    """)

with col3:
    st.markdown("""
    #### 🤖 Live Trading
    - Integrate with Alpaca for paper/live trading.
    - Automatically execute trades based on signals.
    - Monitor account and open positions.
    """)

st.info("Start by selecting a function from the left navigation pane.", icon="👈")

