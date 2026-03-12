import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------- CLEAN CSS (WORKS IN LIGHT & DARK MODE) ----------------
st.markdown("""
<style>

/* spacing */
.block-container {
    padding-top: 2rem;
}

/* input styling */
.stNumberInput input {
    border-radius: 8px;
}

/* button styling */
.stButton button {
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
}

/* metric cards */
[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 10px;
}

/* header styling */
.main-title {
    text-align:center;
    margin-bottom:5px;
}

.sub-title {
    text-align:center;
    opacity:0.7;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<h1 class="main-title">💳 AI Credit Card Fraud Detection</h1>
<p class="sub-title">
Analyze transaction behaviour and detect fraudulent activity using Machine Learning
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- MODEL FEATURES ----------------
feature_names = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
"V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
"V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# ---------------- INPUT PANEL ----------------
st.markdown("## 🧾 Transaction Input Panel")
st.divider()

features = []
cols = st.columns(3)

for i, name in enumerate(feature_names):

    if name == "Time":
        label = "Transaction Time"
    elif name == "Amount":
        label = "Transaction Amount"
    else:
        label = f"Behaviour Score {name}"

    with cols[i % 3]:
        val = st.number_input(label, value=0.0)
        features.append(val)

st.divider()

# ---------------- PREDICT BUTTON ----------------
predict = st.button("🚀 Run Fraud Detection", use_container_width=True)

# ---------------- PREDICTION ----------------
if predict:

    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)
    prob = model.predict_proba(features_array)[0][1]

    st.markdown("## 📊 Prediction Results")
    st.divider()

    if prediction[0] == 1:
        st.error(f"🚨 Fraud Transaction Detected | Probability: {prob:.2%}")
    else:
        st.success(f"✅ Genuine Transaction | Fraud Probability: {prob:.2%}")

    # risk meter
    st.write("### Fraud Risk Level")
    st.progress(float(prob))

    # dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Transaction Amount", features[-1])
    col2.metric("Transaction Time", features[0])
    col3.metric("Fraud Probability", f"{prob:.2%}")

    # feature visualization
    st.write("### Behaviour Feature Visualization")

    values = features[1:11]

    fig, ax = plt.subplots()

    ax.bar(range(len(values)), values)

    ax.set_title("Transaction Behaviour Indicators")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Score")

    st.pyplot(fig)

st.divider()

# ---------------- CSV FRAUD SCANNER ----------------
st.markdown("## 📂 Bulk Transaction Fraud Scanner")
st.divider()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.dataframe(df)

    try:

        # remove class column if present
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        df = df[feature_names]

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:,1]

        df["Fraud Prediction"] = predictions
        df["Fraud Probability"] = probabilities

        st.write("### Prediction Results")
        st.dataframe(df)

        fraud_count = df["Fraud Prediction"].sum()

        st.metric("Total Fraud Transactions Detected", fraud_count)

    except Exception as e:
        st.error(f"CSV format error: {e}")
