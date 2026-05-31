import streamlit as st

st.title("Bitcoin Price and Transaction Fee Forecasting")

st.header("Bitcoin Price Prediction")

value1 = st.number_input("Previous Bitcoin Price 1")
value2 = st.number_input("Previous Bitcoin Price 2")
value3 = st.number_input("Previous Bitcoin Price 3")

if st.button("Predict Bitcoin Price"):
    st.success("Prediction functionality will be connected to LSTM model next.")

st.header("Transaction Fee Prediction")

fee1 = st.number_input("Previous Transaction Fee 1")
fee2 = st.number_input("Previous Transaction Fee 2")
fee3 = st.number_input("Previous Transaction Fee 3")

if st.button("Predict Transaction Fee"):
    st.success("Prediction functionality will be connected to LSTM model next.")
