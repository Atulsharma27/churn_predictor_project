"""Step 2: Train a baseline Random Forest, tune it with GridSearchCV, evaluate and save it."""
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---- 1. Load data ----
df = pd.read_csv("data/customer_churn.csv")
X = df.drop(columns="churn")
y = df["churn"]

num_cols = ["tenure_months", "monthly_charges", "support_calls"]
cat_cols = ["contract", "internet_service"]

# ---- 2. Preprocessing: fill missing values + encode text columns ----
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---- 3. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate(model):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

# ---- 4. Baseline model (default settings) ----
baseline = Pipeline([("prep", preprocess), ("rf", RandomForestClassifier(random_state=42))])
baseline.fit(X_train, y_train)
baseline_scores = evaluate(baseline)
print("Baseline:", baseline_scores)

# ---- 5. Hyperparameter tuning with GridSearchCV ----
param_grid = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [5, 10, None],
    "rf__min_samples_split": [2, 5, 10],
}
grid = GridSearchCV(
    Pipeline([("prep", preprocess), ("rf", RandomForestClassifier(random_state=42))]),
    param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
tuned_scores = evaluate(best)
print("Best parameters:", grid.best_params_)
print("Tuned:", tuned_scores)

# ---- 6. Detailed evaluation ----
pred = best.predict(X_test)
print("\n", classification_report(y_test, pred, target_names=["Stay", "Churn"]))

os.makedirs("outputs", exist_ok=True)

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["rf"].feature_importances_, index=names).sort_values()
plt.figure(figsize=(6, 4))
imp.plot(kind="barh", color="#4f6ef7")
plt.title("Feature Importance"); plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()

# ---- 7. Save model + results ----
joblib.dump(best, "outputs/churn_model.joblib")
with open("outputs/metrics.json", "w") as f:
    json.dump({"baseline": baseline_scores, "tuned": tuned_scores,
               "best_params": grid.best_params_}, f, indent=2)
print("\nModel saved to outputs/churn_model.joblib")
