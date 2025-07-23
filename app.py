import streamlit as st
import numpy as np
import joblib

# Load model and feature names
model = joblib.load("california_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🏠 California Housing Price Predictor")

st.write("Enter the values for each feature:")

# Input fields
input_values = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_values.append(val)

# Predict
if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"🏡 Predicted Median House Price: ${prediction * 100000:.2f}")
