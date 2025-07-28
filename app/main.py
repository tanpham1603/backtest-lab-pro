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
st.sidebar.title("🚀 Backtest Lab Pro")
st.sidebar.markdown("---")
# Hướng dẫn điều hướng
st.sidebar.success("Chọn một chức năng ở trên ⬆️")

# --- Nội dung chính của trang chủ ---

st.title("Chào mừng đến với Backtest Lab Pro 🚀")
st.markdown("### Nền tảng toàn diện cho việc xây dựng và kiểm thử chiến lược giao dịch của bạn.")
st.markdown("---")

# --- Kiểm tra trạng thái kết nối API ---
st.subheader("Trạng thái kết nối")
try:
    # Kiểm tra xem các key cần thiết có trong secrets không
    if "ALPACA_API_KEY" in st.secrets and st.secrets["ALPACA_API_KEY"] and \
       "ALPACA_API_SECRET" in st.secrets and st.secrets["ALPACA_API_SECRET"]:
        st.success("✅ Đã tìm thấy thông tin API trong `secrets.toml`. Các trang chức năng đã sẵn sàng.")
    else:
        st.warning("⚠️ Không tìm thấy thông tin API. Vui lòng kiểm tra lại tệp `.streamlit/secrets.toml`.")
        st.code("""
# Thêm vào file .streamlit/secrets.toml
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_API_SECRET = "YOUR_SECRET_KEY_HERE"
        """, language="toml")
except Exception:
    st.error("❌ Chưa có tệp `secrets.toml`. Vui lòng tạo tệp trong thư mục `.streamlit`.")
    st.code("""
# Tạo tệp .streamlit/secrets.toml và thêm vào nội dung sau:
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_API_SECRET = "YOUR_SECRET_KEY_HERE"
    """, language="toml")

st.markdown("---")


# Giới thiệu các tính năng chính
st.subheader("Các tính năng chính")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 Phân tích & Backtest
    - Kiểm thử chiến lược trên dữ liệu lịch sử.
    - Xem các chỉ số hiệu suất chi tiết (Sharpe, Drawdown, Winrate...).
    - Biểu đồ hóa kết quả một cách trực quan.
    """)

with col2:
    st.markdown("""
    #### ⚡ Tối ưu hóa
    - Tinh chỉnh các tham số của chiến lược.
    - Sử dụng Grid Search để tìm ra bộ tham số tốt nhất.
    - So sánh hiệu quả giữa các bộ tham số.
    """)

with col3:
    st.markdown("""
    #### 🤖 Giao dịch Live
    - Tích hợp với Alpaca cho paper/live trading.
    - Tự động thực thi giao dịch dựa trên tín hiệu.
    - Theo dõi tài khoản và các vị thế đang mở.
    """)

st.info("Bắt đầu bằng cách chọn một chức năng từ thanh điều hướng bên trái.", icon="👈")

