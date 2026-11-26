import streamlit as st
import normal_dist_module as normal_dist

st.set_page_config(page_title="NCEA Level 2 Maths", page_icon="📊", layout="centered")

MODULES = {
    "Normal Distributions": normal_dist.run_app,
    # Future modules (e.g. "Algebra", "Calculus", "Trigonometry") plug in here
}

with st.sidebar:
    st.title("NCEA Level 2 Maths 📐")
    selected_module = st.selectbox("Select Topic", list(MODULES.keys()))

MODULES[selected_module]()