import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved system
data = joblib.load("fraud_full_system.joblib")

pipeline = data["pipeline"]
features = data["features"]
encoders = data["encoders"]   # ✅ ADD THIS

# Page setup
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and click Predict")

# ✅ ONLY 4 INPUTS (clean UI)

# Amount
amount = st.number_input("Amount", value=100.0)

# Merchant ID
merchant_id = st.number_input("Merchant ID", value=1)

# Transaction Type (Dropdown)
transaction_type = st.selectbox(
    "Transaction Type",
    options=encoders['TransactionType'].classes_
)

# Location (Dropdown)
location = st.selectbox(
    "Location",
    options=encoders['Location'].classes_
)

# Encode categorical values
transaction_type_encoded = encoders['TransactionType'].transform([transaction_type])[0]
location_encoded = encoders['Location'].transform([location])[0]

# Create full input (fill missing features with 0)
input_data = dict.fromkeys(features, 0)

input_data['Amount'] = amount
input_data['MerchantID'] = merchant_id
input_data['TransactionType'] = transaction_type_encoded
input_data['Location'] = location_encoded

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected!\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Legit Transaction\nProbability: {probability:.2f}")