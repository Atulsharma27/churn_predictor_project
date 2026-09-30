"""Step 3: Streamlit web app. Run with:  streamlit run app.py"""
import json
import joblib
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.caption("Random Forest with GridSearchCV hyperparameter tuning | AI Internship Project")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "outputs", "churn_model.joblib"))
tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    st.subheader("Enter customer details")
    col1, col2 = st.columns(2)
    tenure = col1.slider("Tenure (months)", 1, 72, 12)
    charges = col2.slider("Monthly charges ($)", 20.0, 120.0, 70.0)
    calls = col1.slider("Support calls", 0, 10, 2)
    contract = col2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = col1.selectbox("Internet service", ["DSL", "Fiber", "None"])

    if st.button("Predict"):
        row = pd.DataFrame([{"tenure_months": tenure, "monthly_charges": charges,
                             "contract": contract, "internet_service": internet,
                             "support_calls": calls}])
        p = model.predict_proba(row)[0, 1]
        st.progress(float(p))
        if p >= 0.5:
            st.error(f"⚠️ Likely to CHURN (probability {p:.0%})")
        else:
            st.success(f"✅ Likely to STAY (churn probability {p:.0%})")

with tab2:
    m = json.load(open(os.path.join(BASE_DIR, "outputs", "metrics.json")))
    st.subheader("Baseline vs Tuned model")
    st.dataframe(pd.DataFrame({"Baseline": m["baseline"], "Tuned": m["tuned"]}))
    st.write("**Best parameters:**", m["best_params"])
    c1, c2 = st.columns(2)
    c1.image("outputs/confusion_matrix.png")
    c2.image("outputs/feature_importance.png")
