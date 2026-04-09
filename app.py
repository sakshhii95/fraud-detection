

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved system
data = joblib.load("fraud_full_system.joblib")

pipeline = data["pipeline"]
features = data["features"]

# Page setup
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and click Predict")

# Create input fields
user_input = {}

for col in features:
    user_input[col] = st.number_input(col, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected!\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Legit Transaction\nProbability: {probability:.2f}")
