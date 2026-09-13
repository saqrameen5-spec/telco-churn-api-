# =====================================================================
# PART 7 — (2) Flask API for churn prediction
# =====================================================================
# Run locally:  python churn_api.py
# Deploy (Render/Railway): gunicorn churn_api:app
from flask import Flask, request, jsonify
import joblib
import pandas as pd

MODEL_FILE = "churn_model_pipeline.pkl"
model = joblib.load(MODEL_FILE)

# The 19 raw input features (exactly as in training, WITHOUT customerID/Churn)
FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "ok",
                    "service": "Telco Customer Churn Prediction API",
                    "endpoints": {"GET /": "health check",
                                  "POST /predict": "single or list of customers"}})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    records = [data] if isinstance(data, dict) else data

    try:
        df = pd.DataFrame(records)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        df = df[FEATURES].copy()
        # robust type handling (API callers may send TotalCharges as string)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0)

        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1]
        out = [{"customer_index": i,
                "churn_prediction": "Yes" if int(p) == 1 else "No",
                "churn_probability": round(float(pr), 4),
                "risk_level": "HIGH" if pr >= 0.75 else ("MEDIUM" if pr >= 0.5 else "LOW")}
               for i, (p, pr) in enumerate(zip(preds, probas))]
        return jsonify({"predictions": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
