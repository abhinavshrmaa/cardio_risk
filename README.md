# Cardiovascular Risk Detector (Python-only prototype)

A 10-year coronary heart disease (CHD) risk estimator, built entirely in
Python with `scikit-learn` and `Streamlit`. No HTML/CSS/JavaScript required —
Streamlit turns a plain Python script into a web app.

## What's in this folder

| File               | What it does                                                        |
|---------------------|----------------------------------------------------------------------|
| `framingham.csv`   | The dataset — Framingham Heart Study, ~4,240 patients                |
| `train_model.py`   | Trains a logistic regression model and saves it as `model.pkl`       |
| `app.py`           | The Streamlit app — collects inputs, shows the prediction and explanation |
| `requirements.txt` | Python packages needed                                               |
| `metrics.txt`      | Generated after training — accuracy/ROC AUC for your report          |

## How it works

1. **Data**: the [Framingham Heart Study dataset](https://en.wikipedia.org/wiki/Framingham_Heart_Study),
   a long-running cardiovascular cohort study. It includes a `TenYearCHD`
   column (1 if the patient developed CHD within 10 years, 0 if not) — this
   is what the model learns to predict.
2. **Features used**: sex, age, current smoking status, and BMI — matching
   the inputs from your original spec (age, height, gender, smoking, BMI).
   Height and weight are collected separately in the app and combined into
   BMI, since that's what the dataset provides.
3. **Model**: a `scikit-learn` `Pipeline` of `StandardScaler` +
   `LogisticRegression` (with `class_weight="balanced"`, since only ~15% of
   patients in the dataset developed CHD). Logistic regression was chosen
   over something like a random forest because it's easy to explain in a
   final-year project defense — each coefficient has a direct, readable
   effect on risk.
4. **Explainability**: `SHAP` (`shap.LinearExplainer`) shows, for each
   individual prediction, how much each input pushed the score up or down.
   This is the same interpretability approach used in your fertility
   predictor project.
5. **AQI**: optional. If you enter a city, the app calls Open-Meteo's free
   geocoding and air-quality APIs (no key needed) to fetch the current US
   AQI, and adds a small transparent adjustment on top of the model's
   output. This adjustment is a heuristic, not something the model learned
   — Framingham has no air-quality data — and the app labels it as such.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates model.pkl and metrics.txt)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

Streamlit will open the app in your browser automatically (usually at
`http://localhost:8501`).

## Notes for your report

- Current test-set performance: **ROC AUC ≈ 0.70**, accuracy ≈ 0.65 (see
  `metrics.txt` after training — exact numbers vary slightly by random
  seed). ROC AUC is the more meaningful number to quote, since CHD-positive
  cases are a minority class and accuracy alone is misleading for
  imbalanced data.
- This is intentionally a small, interpretable feature set. If you want to
  push accuracy higher for a stronger results section, `train_model.py` can
  easily be extended to include more Framingham columns (`sysBP`, `totChol`,
  `diabetes`, etc.) — the trade-off is that the app would need to collect
  more inputs from the user.
- The model is a genuine machine-learning classifier trained on real
  clinical data, not a hand-tuned rule system — so it's a fair one to
  describe and evaluate as a "predictive model" in a written report.

## Disclaimer

This is an educational prototype, not a validated clinical tool. It should
not be used for real medical decisions.
