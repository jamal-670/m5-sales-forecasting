import streamlit as st
import pandas as pd

st.set_page_config(page_title="M5 Forecasting Demo", page_icon="📈", layout="wide")

st.title("M5 Sales Forecasting")
st.write("A simple forecasting demo for the Walmart M5 sales dataset.")

st.markdown("### What this app shows")
st.write("- Time-based and lag-based forecasting concepts")
st.write("- A sample forecasting demo")
st.write("- A simple interface for exploring predictions")

feature_1 = st.number_input("Feature 1", value=4.0)
feature_2 = st.number_input("Feature 2", value=8.0)

if st.button("Generate sample prediction"):
    sample = pd.DataFrame({"feature_1": [feature_1], "feature_2": [feature_2]})
    prediction = 0.8 * feature_1 + 1.2 * feature_2 + 2.0
    st.metric("Predicted sales", round(float(prediction), 2))

st.info("This is a lightweight Streamlit demo for presentation and deployment.")
