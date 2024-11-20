import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model_compressed.pkl')
    scaler = joblib.load('scaler_compressed.pkl')
    return model, scaler

model, scaler = load_model()

# Define prediction function
def predict_loan_status(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Scaling the data
    prediction = model.predict(input_data)  # Predict using the model
    return prediction[0]

# Streamlit App
st.title("Loan Status Prediction App")
st.write("Provide details about the applicant and loan to predict the loan status.")

# Input fields
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Person Gender", ["Male", "Female"], index=0)
person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Doctorate"], index=1)
person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=5)
person_home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"], index=1)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=15000)
loan_intent = st.selectbox("Loan Intent", ["Car", "Education", "Home Improvement", "Medical", "Personal", "Vacation"], index=2)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=12.5)
loan_percent_income = st.number_input("Loan Percent of Income (%)", min_value=0.0, value=30.0)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["No", "Yes"], index=0)

# Convert inputs to numerical values for model
gender_map = {"Male": 1, "Female": 0}
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3}
home_map = {"Rent": 0, "Own": 1, "Mortgage": 2}
intent_map = {"Car": 0, "Education": 1, "Home Improvement": 2, "Medical": 3, "Personal": 4, "Vacation": 5}
default_map = {"No": 0, "Yes": 1}

input_data = [
    person_age,
    gender_map[person_gender],
    education_map[person_education],
    person_income,
    person_emp_exp,
    home_map[person_home_ownership],
    loan_amnt,
    intent_map[loan_intent],
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    default_map[previous_loan_defaults_on_file]
]

# Predict button
if st.button("Predict Loan Status"):
    prediction = predict_loan_status(input_data)
    status = "Approved" if prediction == 1 else "Rejected"
    st.success(f"The loan status is: {status}")
