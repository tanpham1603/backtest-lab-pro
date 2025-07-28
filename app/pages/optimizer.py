import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import vectorbt as vbt
from itertools import product
import sys
import os

# Thêm đường dẫn để import các module từ thư mục app
try:
    # Điều chỉnh đường dẫn để linh hoạt hơn
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from app.loaders.crypto_loader import CryptoLoader
    from app.loaders.forex_loader import ForexLoader
except ImportError:
    st.error("Lỗi import: Không tìm thấy các file loader. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(page_title="Optimizer", page_icon="⚡", layout="wide")
st.title("⚡ Grid-Search Tối ưu hóa MA-Cross")

# --- Sidebar để người dùng tùy chỉnh ---
st.sidebar.header("🎛️ Cấu hình Tối ưu hóa")

asset_class = st.sidebar.selectbox("Loại tài sản:", ["Crypto", "Forex", "Stocks"])

if asset_class == "Crypto":
    symbol = st.sidebar.text_input("Mã giao dịch:", "BTC/USDT")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0)
else:
    symbol = st.sidebar.text_input("Mã giao dịch:", "EURUSD=X" if asset_class == "Forex" else "AAPL")
    tf = st.sidebar.selectbox("Khung thời gian:", ["1d"], index=0)

st.sidebar.subheader("Dải tham số")
fasts = st.sidebar.multiselect("Danh sách MA Nhanh:", [5, 10, 15, 20, 25, 30], default=[10, 20])
slows = st.sidebar.multiselect("Danh sách MA Chậm:", [40, 50, 60, 100, 150, 200], default=[50, 100])

target_metric = st.sidebar.selectbox(
    "Chỉ số mục tiêu:",
    ["Sharpe", "Return", "Win Rate", "Profit Factor"]
)

# --- Hàm tải dữ liệu với cache (ĐÃ CẬP NHẬT THEO HƯỚNG DẪN) ---
@st.cache_data(ttl=600)
def load_price_data(asset, sym, timeframe):
    """Tải về chuỗi giá đóng cửa cho backtest một cách an toàn."""
    try:
        if asset == "Crypto":
            data = CryptoLoader().fetch(sym, timeframe, 2000)
        else:
            data = ForexLoader().fetch(sym, timeframe, "730d")
        
        # --- KIỂM TRA AN TOÀN ---
        if data is None or data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
            return None
        
        if 'Close' not in data.columns:
            st.error(f"Dữ liệu trả về cho {sym} không chứa cột 'Close'.")
            return None
        
        return data["Close"]
        # ----------------------

    except Exception as e:
        st.error(f"Lỗi hệ thống khi tải dữ liệu: {e}")
        return None

# --- Hàm trợ giúp để lấy giá trị số từ kết quả của vectorbt ---
def get_scalar(value):
    """Trích xuất một giá trị số từ một scalar hoặc một Series."""
    if isinstance(value, pd.Series):
        if not value.empty:
            return value.iloc[0]
        return np.nan # Trả về NaN nếu Series rỗng
    return value # Trả về chính nó nếu đã là scalar

# --- Chạy tối ưu hóa khi người dùng nhấn nút ---
if st.sidebar.button("🚀 Chạy Tối ưu hóa", type="primary"):
    price = load_price_data(asset_class, symbol, tf)
    
    if price is not None and not price.empty:
        results = []
        
        param_combinations = [p for p in product(fasts, slows) if p[0] < p[1]]
        
        if not param_combinations:
            st.warning("Không có cặp tham số hợp lệ nào (MA Nhanh phải < MA Chậm).")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (f, s) in enumerate(param_combinations):
                status_text.text(f"Đang kiểm tra: MA({f}, {s})... ({i+1}/{len(param_combinations)})")
                
                fast_ma = price.rolling(f).mean()
                slow_ma = price.rolling(s).mean()
                entries = fast_ma > slow_ma
                exits = fast_ma < slow_ma
                
                pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.001, freq=tf)
                
                results.append({
                    "Fast": f,
                    "Slow": s,
                    "Sharpe": get_scalar(pf.sharpe_ratio()),
                    "Return": get_scalar(pf.total_return()),
                    "Win Rate": get_scalar(pf.trades.win_rate()),
                    "Profit Factor": get_scalar(pf.trades.profit_factor()),
                    "Trades": get_scalar(pf.trades.count())
                })
                
                progress_bar.progress((i + 1) / len(param_combinations))

            progress_bar.empty()
            status_text.empty()

            if results:
                df = pd.DataFrame(results)

                df['Profit Factor'] = df['Profit Factor'].replace([np.inf, -np.inf], np.nan)

                df.sort_values(target_metric, ascending=False, na_position='last', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                st.subheader("📊 Kết quả Tối ưu hóa")
                st.dataframe(df.style.format({
                    "Sharpe": "{:.2f}",
                    "Return": "{:.2%}",
                    "Win Rate": "{:.2%}",
                    "Profit Factor": "{:.2f}"
                }))
                
                best_df = df.dropna(subset=[target_metric])
                if not best_df.empty:
                    best = best_df.iloc[0]
                    st.success(f"🏆 **Tốt nhất:** Fast={int(best.Fast)}, Slow={int(best.Slow)} | {target_metric}={best[target_metric]:.2f}")

                st.subheader("Trực quan hóa Heatmap")
                try:
                    heatmap_df = df.dropna(subset=[target_metric]).pivot(index='Slow', columns='Fast', values=target_metric)
                    if not heatmap_df.empty:
                        fig = go.Figure(data=go.Heatmap(
                            z=heatmap_df.values, x=heatmap_df.columns, y=heatmap_df.index, colorscale='Viridis'
                        ))
                        fig.update_layout(title=f'Heatmap của {target_metric}', template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Không có đủ dữ liệu để vẽ heatmap.")
                except Exception as e:
                    st.warning(f"Không thể vẽ heatmap: {e}")
            else:
                st.warning("Không có kết quả nào được tạo ra.")
else:
    st.info("👈 Vui lòng cấu hình các tham số và nhấn 'Chạy Tối ưu hóa' ở thanh bên trái.")
