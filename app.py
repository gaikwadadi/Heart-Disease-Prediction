import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Page settings
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Prediction App")

# User input form
def get_user_input():
    st.subheader("Enter Patient Details")

    age = st.number_input("Age (years)", min_value=20, max_value=100)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, step=0.1)
    slope = st.selectbox("Slope of Peak ST Segment", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    return pd.DataFrame([input_data])

# Prediction section
with st.form("prediction_form"):
    input_df = get_user_input()
    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        proba = model.predict_proba(input_df)[0][1]
        prediction = 1 if proba >= 0.5 else 0

        st.subheader("🧾 Prediction Result")
        if prediction == 1:
            st.error("🚨 Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

        st.info(f"📊 Confidence: {proba:.2%}")
        st.progress(int(proba * 100))  # Confidence progress bar
