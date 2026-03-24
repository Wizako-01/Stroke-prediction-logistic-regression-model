import streamlit as st
import joblib
import json
import numpy as np

# Load model and files
model = joblib.load("stroke_logistic_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# Title
st.title("Stroke Risk Prediction App")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", 0.0, 120.0, 50.0)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0, 100.0)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encode inputs
gender = label_encoders["gender"].transform([gender])[0]
ever_married = label_encoders["ever_married"].transform([ever_married])[0]
work_type = label_encoders["work_type"].transform([work_type])[0]
residence_type = label_encoders["Residence_type"].transform([residence_type])[0]
smoking_status = label_encoders["smoking_status"].transform([smoking_status])[0]

# Prediction
if st.button("Predict"):
    input_data = np.array([[gender, age, hypertension, heart_disease,
                            ever_married, work_type, residence_type,
                            avg_glucose_level, bmi, smoking_status]])

    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= threshold)

    st.subheader("Result")
    st.write(f"Stroke Probability: {probability*100:.1f}%")

    if prediction == 1:
        st.error("High Stroke Risk")
    else:
        st.success("Low Stroke Risk")
