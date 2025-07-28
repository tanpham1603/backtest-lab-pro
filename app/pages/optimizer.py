import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import vectorbt as vbt
from itertools import product
import sys
import os
import ccxt
import yfinance as yf

# --- Cấu hình trang ---
st.set_page_config(page_title="Optimizer", page_icon="⚡", layout="wide")

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


# --- Sidebar ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("🎛️ Cấu hình Tối ưu hóa")

    asset_class = st.selectbox("Loại tài sản:", ["Crypto", "Forex", "Stocks"], key="optimizer_asset")

    if asset_class == "Crypto":
        symbol = st.text_input("Mã giao dịch:", "BTC/USDT", key="optimizer_crypto_symbol")
        tf = st.selectbox("Khung thời gian:", ["1h", "4h", "1d"], index=0, key="optimizer_crypto_tf")
    else:
        symbol = st.text_input("Mã giao dịch:", "EURUSD=X" if asset_class == "Forex" else "AAPL", key="optimizer_stock_symbol")
        tf = st.selectbox("Khung thời gian:", ["1d"], index=0, key="optimizer_stock_tf")

    st.subheader("Dải tham số")
    fasts = st.multiselect("Danh sách MA Nhanh:", [5, 10, 15, 20, 25, 30], default=[10, 20], key="optimizer_fasts")
    slows = st.multiselect("Danh sách MA Chậm:", [40, 50, 60, 100, 150, 200], default=[50, 100], key="optimizer_slows")

    target_metric = st.selectbox(
        "Chỉ số mục tiêu:",
        ["Sharpe", "Return", "Win Rate", "Profit Factor"],
        key="optimizer_metric"
    )

# --- Hàm tải dữ liệu an toàn ---
@st.cache_data(ttl=600)
def load_price_data(asset, sym, timeframe):
    """Tải về chuỗi giá đóng cửa cho backtest một cách an toàn."""
    try:
        if asset == "Crypto":
            exchange = ccxt.kucoin()
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=2000)
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else: # Forex và Stocks
            period = "5y" # Tải dữ liệu 5 năm để tối ưu hóa
            data = yf.download(sym, period=period, interval=timeframe, progress=False, auto_adjust=True)
            data.columns = [col[0].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in data.columns]

        if data is None or data.empty:
            st.error(f"Không nhận được dữ liệu cho mã {sym}. API có thể đã bị lỗi hoặc mã không hợp lệ.")
            return None
        
        if 'Close' not in data.columns:
            st.error(f"Dữ liệu trả về cho {sym} không chứa cột 'Close'.")
            return None
            
        return data["Close"]

    except Exception as e:
        st.error(f"Lỗi hệ thống khi tải dữ liệu cho {sym}: {e}")
        return None

# --- Hàm trợ giúp để lấy giá trị số ---
def get_scalar(value):
    if isinstance(value, pd.Series):
        return value.iloc[0] if not value.empty else np.nan
    return value

# --- Giao diện chính ---
st.title("⚡ Grid-Search Tối ưu hóa MA-Cross")
st.markdown("### Tìm ra bộ tham số hiệu quả nhất cho chiến lược giao cắt đường trung bình động.")

if st.sidebar.button("🚀 Chạy Tối ưu hóa", type="primary"):
    price = load_price_data(asset_class, symbol, tf)
    
    if price is not None and not price.empty:
        param_combinations = [p for p in product(fasts, slows) if p[0] < p[1]]
        
        if not param_combinations:
            st.warning("Không có cặp tham số hợp lệ nào (MA Nhanh phải < MA Chậm).")
        else:
            results = []
            progress_bar = st.progress(0, text="Đang xử lý...")
            status_text = st.empty()

            for i, (f, s) in enumerate(param_combinations):
                status_text.text(f"Đang kiểm tra: MA({f}, {s})... ({i+1}/{len(param_combinations)})")
                
                fast_ma = price.rolling(f).mean()
                slow_ma = price.rolling(s).mean()
                entries = fast_ma > slow_ma
                exits = fast_ma < slow_ma
                
                pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.001, freq=tf)
                
                results.append({
                    "Fast": f, "Slow": s,
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
                
                st.header("🏆 Kết quả Tốt nhất")
                best_df = df.dropna(subset=[target_metric])
                if not best_df.empty:
                    best = best_df.iloc[0]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Cặp MA Tốt nhất", f"{int(best.Fast)} / {int(best.Slow)}")
                    col2.metric(f"Chỉ số {target_metric}", f"{best[target_metric]:.2f}")
                    col3.metric("Tổng Lợi nhuận", f"{best['Return']:.2%}")
                    col4.metric("Tổng số Giao dịch", f"{best['Trades']:.0f}")

                st.subheader("📈 Trực quan hóa Heatmap")
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

                with st.expander("🔬 Xem Bảng kết quả chi tiết"):
                    st.dataframe(df.style.format({
                        "Sharpe": "{:.2f}", "Return": "{:.2%}",
                        "Win Rate": "{:.2%}", "Profit Factor": "{:.2f}"
                    }))
            else:
                st.warning("Không có kết quả nào được tạo ra.")
else:
    st.info("👈 Vui lòng cấu hình các tham số và nhấn 'Chạy Tối ưu hóa' ở thanh bên trái.")
