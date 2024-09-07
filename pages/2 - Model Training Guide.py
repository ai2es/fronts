import os
import streamlit as st

cwd = 'I:/PycharmProjects/fronts'  # TODO: be sure to change this once the xCITE instance is started

with open(cwd + '/README.md', 'r') as f:
    md = f.read()
    
st.markdown(md, unsafe_allow_html=True)