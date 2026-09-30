"""Step 1: Create a sample customer dataset (so the project runs without downloads).
You can replace data/customer_churn.csv with any Kaggle churn dataset that has the same columns."""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "tenure_months": np.random.randint(1, 72, n),
    "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
    "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
    "internet_service": np.random.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.45, 0.20]),
    "support_calls": np.random.poisson(2, n),
})

# Churn rule: short tenure, high bill, month-to-month contract and many support calls -> more churn
score = (
    -0.04 * df["tenure_months"]
    + 0.02 * df["monthly_charges"]
    + 0.35 * df["support_calls"]
    + np.where(df["contract"] == "Month-to-month", 1.2, np.where(df["contract"] == "One year", 0.0, -1.0))
    + np.where(df["internet_service"] == "Fiber", 0.4, 0)
    - 0.8
)
prob = 1 / (1 + np.exp(-score))
df["churn"] = (np.random.rand(n) < prob).astype(int)

# Add a few missing values on purpose to show data cleaning
df.loc[np.random.choice(n, 40, replace=False), "monthly_charges"] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/customer_churn.csv", index=False)
print("Dataset saved:", df.shape, "| churn rate:", round(df["churn"].mean(), 2))
