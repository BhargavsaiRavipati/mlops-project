import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="ML App", layout="centered")

st.title("⚡ ML Prediction App")
st.markdown("### Enter feature values below")

# Get number of features
num_features = model.n_features_in_

inputs = []

# Input fields
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# Convert input
input_array = np.array(inputs).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Prediction button
if st.button("Predict"):

    prediction = model.predict(input_scaled)

    st.subheader("🔍 Prediction Result")
    st.success(f"Predicted Class: {prediction[0]}")

    # 🔥 Add probability (IMPORTANT)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)
        confidence = np.max(probs)

        st.subheader("📊 Confidence Score")
        st.info(f"{confidence:.2f}")

    # 🔥 Interpretation
    st.subheader("🧠 Interpretation")

    if prediction[0] == 1:
        st.error("⚠️ High Risk / Positive Case")
    else:   
        st.success("✅ Low Risk / Negative Case")