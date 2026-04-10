import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/loan_model.joblib")
st.write(type(model))
st.title("Loan Default Prediction System")
st.write("Model Used: Logistic Regression (Optimized for Recall)")

loan_amnt = st.number_input("Loan Amount")
int_rate = st.number_input("Interest Rate (%)") / 100
revol_util = st.number_input("Revolving Utilization (%)") / 100
annual_inc = st.number_input("Annual Income")
dti = st.number_input("Debt to Income Ratio")
revol_bal = st.number_input("Revolving Balance")
installment = st.number_input("Installment")
total_acc = st.number_input("Total Accounts")
open_acc = st.number_input("Open Accounts")
credit_history_years = st.number_input("Credit History (years)")

if st.button("Predict Risk"):

    input_dict = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "revol_util": revol_util,
        "revol_bal": revol_bal,
        "installment": installment,
        "total_acc": total_acc,
        "open_acc": open_acc,
        "credit_history_years": credit_history_years
    }

    input_data = pd.DataFrame([input_dict])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    
    st.write(input_data)
    st.write("Default Probability:", round(probability,3))

    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
