import joblib
import pandas as pd

model = joblib.load("../models/loan_default_model.pkl")
features = joblib.load("../models/model_features.pkl")

def predict_default(input_data):

    df = pd.DataFrame([input_data])
    df = df[features]

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    if prob < 0.3:
        risk = "Low Risk"
    elif prob < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return pred, prob, risk