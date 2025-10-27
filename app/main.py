import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Cấu hình trang ---
st.set_page_config(
    page_title="🚀 Backtest Lab Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS tùy chỉnh ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    .welcome-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #8898aa;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        height: 280px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: 0.5s;
    }
    .feature-card:hover::before {
        left: 100%;
    }
    .feature-card:hover {
        transform: translateY(-10px);
        border-color: #667eea;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
    }
    .feature-desc {
        color: #8898aa;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .status-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #8898aa;
        font-size: 0.9rem;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 3rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='color: #667eea; font-size: 1.8rem; margin-bottom: 0.5rem;'>🚀</h1>
    <h2 style='color: white; font-size: 1.2rem; margin: 0;'>Backtest Lab Pro</h2>
    <p style='color: #8898aa; font-size: 0.8rem; margin: 0;'>by TanPham</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("✨ Select a function to begin")

# --- Nội dung chính ---
st.markdown('<div class="welcome-header">🚀 Backtest Lab Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Algorithmic Trading Platform</div>', unsafe_allow_html=True)

# --- Status Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🔌</div>
        <div class="metric-value">Connected</div>
        <div class="metric-label">API Status</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">📊</div>
        <div class="metric-value">Live</div>
        <div class="metric-label">Data Feed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">⚡</div>
        <div class="metric-value">Ready</div>
        <div class="metric-label">Trading Engine</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="status-card">
        <div class="status-icon">🛡️</div>
        <div class="metric-value">Active</div>
        <div class="metric-label">Risk Management</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Features Grid ---
st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h2 style='color: white; font-size: 2rem; font-weight: 700;'>Core Features</h2>
    <p style='color: #8898aa; font-size: 1.1rem;'>Everything you need for professional trading</p>
</div>
""", unsafe_allow_html=True)

# Hàng 1
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Strategy Builder</div>
        <div class="feature-desc">Create and visualize custom trading strategies with advanced technical indicators and real-time backtesting.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Backtest Engine</div>
        <div class="feature-desc">Comprehensive historical testing with detailed performance metrics and visual analytics.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Live Trading</div>
        <div class="feature-desc">Seamless Alpaca integration for paper and live trading with real-time execution.</div>
    </div>
    """, unsafe_allow_html=True)

# Hàng 2
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">ML Signals</div>
        <div class="feature-desc">AI-powered trading signals and predictive analytics using machine learning models.</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Optimizer</div>
        <div class="feature-desc">Advanced parameter optimization with genetic algorithms and walk-forward analysis.</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <div class="feature-title">Risk Management</div>
        <div class="feature-desc">Comprehensive risk analysis, position sizing, and portfolio protection tools.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8898aa;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with ❤️ using Streamlit • Professional Trading Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>Backtest Lab Pro v2.0</p>
</div>
""", unsafe_allow_html=True)