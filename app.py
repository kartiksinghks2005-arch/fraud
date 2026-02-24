import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Page settings
st.set_page_config(page_title="Fraud Detection", page_icon="💳")

# Title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction behaviour details to check fraud probability")

st.divider()

# Friendly feature names
feature_names = [
    "Transaction Behaviour Score 1",
    "Transaction Behaviour Score 2",
    "Transaction Behaviour Score 3",
    "Transaction Behaviour Score 4",
    "Transaction Behaviour Score 5",
    "Transaction Behaviour Score 6",
    "Transaction Behaviour Score 7",
    "Transaction Behaviour Score 8",
    "Transaction Behaviour Score 9",
    "Transaction Behaviour Score 10",
    "Transaction Behaviour Score 11",
    "Transaction Behaviour Score 12",
    "Transaction Behaviour Score 13",
    "Transaction Behaviour Score 14",
    "Transaction Behaviour Score 15",
    "Transaction Behaviour Score 16",
    "Transaction Behaviour Score 17",
    "Transaction Behaviour Score 18",
    "Transaction Behaviour Score 19",
    "Transaction Behaviour Score 20",
    "Transaction Behaviour Score 21",
    "Transaction Behaviour Score 22",
    "Transaction Behaviour Score 23",
    "Transaction Behaviour Score 24",
    "Transaction Behaviour Score 25",
    "Transaction Behaviour Score 26",
    "Transaction Behaviour Score 27",
    "Transaction Behaviour Score 28",
    "Transaction Time (seconds)",
    "Transaction Amount"
]

# Create input boxes
features = []
for name in feature_names:
    val = st.number_input(name, value=0.0)
    features.append(val)

st.divider()

# Predict button
if st.button("🔍 Predict Fraud"):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    prob = model.predict_proba(features_array)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error(f"🚨 Fraud Transaction Detected\n\nProbability: {prob:.2%}")
    else:
        st.success(f"✅ Genuine Transaction\n\nProbability of Fraud: {prob:.2%}")