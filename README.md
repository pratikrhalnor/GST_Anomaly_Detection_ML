# GST Anomaly (Fraud) Detection — README

A compact end-to-end example to detect fraudulent GST transactions using a Random Forest classifier. This repository contains a Colab-exported Python script (`trainingCode/gst_anomoly_detection.py`) that demonstrates data preprocessing, exploratory data analysis (EDA), model training, evaluation, feature-importance analysis, and a small interactive UI (for Colab) for prediction. 🚀

---

## Table of Contents
- Project Overview
- What the script does
- Dataset
- Requirements
- How to run
- Outputs / Artifacts
- Model usage / Predicting new samples
- Notes & Next steps
- License

---

## Project Overview
This project shows a simple machine learning pipeline for detecting potentially fraudulent GST transactions using a Random Forest classifier. The script was authored in Google Colab and includes comments and interactive widgets for quick experimentation. 🔍

Key goals:
- Clean and impute missing values 🧹
- Explore dataset distributions & correlations 📊
- Train a Random Forest classifier with balanced class handling 🌲
- Evaluate with a custom probability threshold and confusion matrix ✅❌
- Save model and scaler for later inference 💾

---

## What the script does
- Loads a CSV dataset (example filename used in the script: `gst_fraud_dataset (2).csv`)
- Fills missing numeric values with median values
- Performs basic EDA: class distribution and correlation heatmap
- Splits into train/test (stratified), scales features using `StandardScaler`
- Trains `RandomForestClassifier(n_estimators=200, class_weight="balanced")`
- Uses predicted probabilities and applies a custom decision threshold (example: 0.3) to increase recall for fraud detection
- Prints accuracy and a full classification report; plots a confusion matrix and feature importance chart
- Provides an interactive Colab widget (ipywidgets) for single-record prediction
- Persists trained `model` and `scaler` with `joblib` as `gst_fraud_model.pkl` and `scaler_.pkl`

---

## Dataset
A synthetic dataset (example filename in the script): `gst_fraud_dataset (2).csv`.  
Typical columns used in the script:
- InvoiceValue, IGST, CGST, SGST
- InputTaxCredit, Turnover, InvoiceCount
- BusinessAge, FilingDelay
- Fraudulent (target: 0 → normal, 1 → fraud)

Note: Replace the CSV path in the script with your dataset path when running locally or in Colab. 🗂️

---

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- ipywidgets (if using the interactive Colab UI)

Install with pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib ipywidgets
```

---

## How to run
1. Open the script in Google Colab or run locally in a Python environment. ☁️ or 🖥️  
2. Update the dataset path in the script:
   ```python
   df = pd.read_csv("/path/to/gst_fraud_dataset.csv")
   ```
3. Execute cells or run the script; plots will appear inline in Colab / Jupyter.  
4. To use the interactive UI, run the widget section in Colab or a Jupyter environment with ipywidgets enabled.

If running headless (no interactive display), replace `plt.show()` with `plt.savefig(...)` to store charts as files.

---

## Outputs / Artifacts
After training, the script saves:
- gst_fraud_model.pkl — trained Random Forest model
- scaler_.pkl — StandardScaler used for preprocessing

Load them for inference:
```python
from joblib import load
model = load("gst_fraud_model.pkl")
scaler = load("scaler_.pkl")
```

---

## Model usage / Predicting new samples
Prepare a single-record DataFrame with the same columns used for training (names and order must match), scale it, then call `model.predict_proba` or `model.predict`.

Example:
```python
from joblib import load
import pandas as pd

model = load("gst_fraud_model.pkl")
scaler = load("scaler_.pkl")

new_record = pd.DataFrame([{
  'InvoiceValue': 10000,
  'IGST': 0.0,
  'CGST': 90.0,
  'SGST': 90.0,
  'InputTaxCredit': 270.0,
  'Turnover': 500000,
  'InvoiceCount': 120,
  'BusinessAge': 5,
  'FilingDelay': 2
}])

new_scaled = scaler.transform(new_record)
prob = model.predict_proba(new_scaled)[0, 1]
threshold = 0.3
prediction = int(prob > threshold)  # 1 => Fraud, 0 => Normal
```

---

## Notes & Next steps
- The dataset in the script is synthetic — for production use, train on real-world GST/transaction data and validate thoroughly. ⚠️  
- Consider advanced models (XGBoost / LightGBM), hyperparameter tuning, cross-validation, and stronger imbalance handling (SMOTE, class weighting, focal loss). ⚙️  
- Add explainability tools (SHAP/LIME) and monitoring for deployed models.  
- To serve predictions in production, wrap inference in a Flask/FastAPI/Streamlit app and include input validation and authentication. 🛠️

---

## License
This project is provided as an example template — adapt and extend for your use. Add a LICENSE file to set explicit license terms. 📜
