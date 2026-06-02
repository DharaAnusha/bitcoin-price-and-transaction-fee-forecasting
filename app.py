import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load Models
bitcoin_model = load_model("bitcoin_lstm_model.keras")
fee_model = load_model("transaction_fee_lstm_model.keras")

# Load Scalers
with open("bitcoin_scaler.pkl", "rb") as f:
    bitcoin_scaler = pickle.load(f)

with open("transaction_fee_scaler.pkl", "rb") as f:
    fee_scaler = pickle.load(f)

# App Title
st.title("Bitcoin Price and Transaction Fee Forecasting Using LSTM")

st.write("Enter the previous 3 values to predict future Bitcoin Price and Transaction Fee.")

# BITCOIN PRICE PREDICTION

st.header("Bitcoin Price Prediction")

btc1 = st.number_input("Previous Bitcoin Price 1")
btc2 = st.number_input("Previous Bitcoin Price 2")
btc3 = st.number_input("Previous Bitcoin Price 3")

if st.button("Predict Bitcoin Price"):

    btc_input = np.array([[btc1], [btc2], [btc3]])

    btc_scaled = bitcoin_scaler.transform(btc_input)
    btc_scaled = btc_scaled.reshape(1, 3, 1)

    prediction_scaled = bitcoin_model.predict(btc_scaled, verbose=0)

    prediction = bitcoin_scaler.inverse_transform(
        prediction_scaled
    )

    st.success(
        f"Predicted Bitcoin Price: ${prediction[0][0]:,.2f}"
    )

    # 5-Year Forecast

    future_prices = []

    current_sequence = btc_scaled.reshape(3)

    for i in range(5):

        input_seq = current_sequence.reshape(1, 3, 1)

        next_pred = bitcoin_model.predict(
            input_seq,
            verbose=0
        )

        future_prices.append(next_pred[0][0])

        current_sequence = np.append(
            current_sequence[1:],
            next_pred[0][0]
        )

    future_prices = bitcoin_scaler.inverse_transform(
        np.array(future_prices).reshape(-1, 1)
    )

    years = [
        "Year 1",
        "Year 2",
        "Year 3",
        "Year 4",
        "Year 5"
    ]

    forecast_df = pd.DataFrame({
        "Year": years,
        "Bitcoin Price": future_prices.flatten()
    })

    st.subheader("5-Year Bitcoin Forecast")

    st.dataframe(forecast_df)

    fig, ax = plt.subplots()

    ax.plot(
        years,
        future_prices.flatten(),
        marker="o"
    )

    ax.set_title("Bitcoin Price Forecast")

    st.pyplot(fig)

# TRANSACTION FEE PREDICTION

st.header("Transaction Fee Prediction")

fee1 = st.number_input("Previous Transaction Fee 1")
fee2 = st.number_input("Previous Transaction Fee 2")
fee3 = st.number_input("Previous Transaction Fee 3")

if st.button("Predict Transaction Fee"):

    fee_input = np.array([[fee1], [fee2], [fee3]])

    fee_scaled = fee_scaler.transform(fee_input)

    fee_scaled = fee_scaled.reshape(1, 3, 1)

    prediction_scaled = fee_model.predict(
        fee_scaled,
        verbose=0
    )

    prediction = fee_scaler.inverse_transform(
        prediction_scaled
    )

    st.success(
        f"Predicted Transaction Fee: {prediction[0][0]:,.2f}"
    )

    # 5-Year Forecast

    future_fees = []

    current_sequence = fee_scaled.reshape(3)

    for i in range(5):

        input_seq = current_sequence.reshape(1, 3, 1)

        next_pred = fee_model.predict(
            input_seq,
            verbose=0
        )

        future_fees.append(next_pred[0][0])

        current_sequence = np.append(
            current_sequence[1:],
            next_pred[0][0]
        )

    future_fees = fee_scaler.inverse_transform(
        np.array(future_fees).reshape(-1, 1)
    )

    years = [
        "Year 1",
        "Year 2",
        "Year 3",
        "Year 4",
        "Year 5"
    ]

    fee_df = pd.DataFrame({
        "Year": years,
        "Transaction Fee": future_fees.flatten()
    })

    st.subheader("5-Year Transaction Fee Forecast")

    st.dataframe(fee_df)

    fig, ax = plt.subplots()

    ax.plot(
        years,
        future_fees.flatten(),
        marker="o"
    )

    ax.set_title("Transaction Fee Forecast")

    st.pyplot(fig)
