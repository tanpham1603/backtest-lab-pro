import streamlit as st

def init():
    if "broker"   not in st.session_state: st.session_state.broker = None
    if "api_ok"   not in st.session_state: st.session_state.api_ok = False
    if "ml_model" not in st.session_state: st.session_state.ml_model = None
