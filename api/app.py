from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
model = joblib.load("../model/gst_fraud_model.pkl")
scaler = joblib.load("../model/scaler_.pkl")

# Feature columns expected by the model
FEATURE_COLUMNS = [
    "InvoiceValue", "IGST", "CGST", "SGST",
    "InputTaxCredit", "Turnover", "InvoiceCount",
    "BusinessAge", "FilingDelay"
]

# Folder to save prediction results
RESULTS_FOLDER = "../results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    # Check if all required columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

    # Extract features and apply scaler
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    # Predict anomalies
    predictions = model.predict(X_scaled)

    # Attach predictions to dataframe
    df["is_anomaly"] = predictions

    # Save result CSV
    result_file = os.path.join(RESULTS_FOLDER, f"predictions_{datetime.today().strftime('%Y-%m-%d')}.csv")
    df.to_csv(result_file, index=False)

    # Return JSON
    return df.to_dict(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
