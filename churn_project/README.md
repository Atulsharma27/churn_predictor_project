# Customer Churn Predictor – Random Forest + Hyperparameter Tuning
Project for the Artificial Intelligence Internship at Laugh Logic Labs LLP.

## What it does
Predicts whether a customer will leave (churn) based on tenure, monthly charges,
contract type, internet service and number of support calls.

## Workflow
1. `generate_data.py`  – creates the dataset (`data/customer_churn.csv`, with a few missing values)
2. `train_model.py`    – preprocessing -> baseline Random Forest -> GridSearchCV tuning -> evaluation -> saves model
3. `app.py`            – Streamlit web app for live predictions

## Run
```
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

## Results (test set)
| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 0.7125 | 0.7275 |
| ROC-AUC | 0.7689 | 0.8203 |

Best parameters: n_estimators=300, max_depth=5, min_samples_split=2 (5-fold CV, scoring = ROC-AUC).

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib
