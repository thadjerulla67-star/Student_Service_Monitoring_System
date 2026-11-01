from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category"
]

@st.cache_resource
def load_model():
    path = Path(__file__).resolve().parents[1] / "models" / "trained_model.joblib"
    return joblib.load(path)

def predict_one(model, values):
    row = pd.DataFrame([{key: values[key] for key in FEATURES}], columns=FEATURES)
    label = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][list(model.classes_).index(1)])
    return label, probability
