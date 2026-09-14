"""
train_model.py
---------------
Trains a logistic regression model to predict 10-year coronary heart
disease (CHD) risk using the Framingham Heart Study dataset.

Run this once before launching the Streamlit app:
    python train_model.py

It produces:
    model.pkl   -> trained scikit-learn pipeline (scaler + logistic regression)
    metrics.txt -> quick evaluation summary, for your project report
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import joblib

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------
df = pd.read_csv("framingham.csv")

# Keep only the features we ask the user for in the app:
#   male (sex), age, currentSmoker, BMI  ->  predicting TenYearCHD
FEATURES = ["male", "age", "currentSmoker", "BMI"]
TARGET = "TenYearCHD"

df = df[FEATURES + [TARGET]].dropna()
print(f"Rows after dropping missing values: {len(df)}")

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Pipeline: scale features, then logistic regression
#    class_weight="balanced" because CHD-positive cases are a minority
#    (about 15% of the dataset), so the model doesn't just predict "no risk"
#    for everyone.
# ---------------------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)),
])

model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

summary = f"""Cardiovascular Risk Model — Evaluation Summary
================================================
Features used: {FEATURES}
Training rows: {len(X_train)}
Test rows:     {len(X_test)}

Accuracy: {acc:.3f}
ROC AUC:  {auc:.3f}

Classification report:
{report}
Confusion matrix (rows=actual, cols=predicted):
{cm}

Note: accuracy alone is misleading here because CHD-positive cases are
a minority class (~15%). ROC AUC is the more meaningful number for a
project write-up, since it reflects how well the model ranks
higher-risk patients above lower-risk ones regardless of threshold.
"""

print(summary)
with open("metrics.txt", "w") as f:
    f.write(summary)

# ---------------------------------------------------------------------
# 5. Save the trained pipeline
# ---------------------------------------------------------------------
joblib.dump(model, "model.pkl")
print("Saved trained pipeline to model.pkl")
