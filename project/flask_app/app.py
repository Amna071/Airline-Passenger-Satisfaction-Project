"""
Airline Passenger Satisfaction Predictor — Flask app.

Loads the trained model + preprocessor produced by the training notebook
(model_training/Airline_Passenger_Satisfaction_Training.ipynb) and serves a
form-driven prediction UI.
"""
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
bundle = joblib.load(PREPROCESSOR_PATH)

encoders = bundle["encoders"]
target_encoder = bundle["target_encoder"]
feature_order = bundle["feature_order"]
categorical_cols = bundle["categorical_cols"]
numeric_cols = bundle["numeric_cols"]
scaler = bundle["scaler"]
uses_scaling = bundle["uses_scaling"]
model_name = bundle["model_name"]

# Only the highest feature-importance ratings are shown on the form.
# (Full ranked list lives in model_training/model_comparison_results.csv /
# the notebook's feature-importance step.)
VISIBLE_RATING_FIELDS = [
    "Online boarding", "Inflight wifi service", "Seat comfort",
    "Inflight entertainment", "Ease of Online booking",
]

# Categorical fields shown on the form (also top-importance).
VISIBLE_CATEGORICAL_FIELDS = ["Customer Type", "Type of Travel", "Class"]

# Everything else the model still needs, filled with neutral/typical
# defaults instead of asking the user. 3 = middle-of-the-road rating,
# 0 minutes = no delay, values below are dataset-typical.
HIDDEN_DEFAULTS = {
    "Gender": "Female",
    "Age": 35,
    "Flight Distance": 800,
    "Departure/Arrival time convenient": 3,
    "Gate location": 3,
    "Food and drink": 3,
    "On-board service": 3,
    "Leg room service": 3,
    "Baggage handling": 3,
    "Checkin service": 3,
    "Inflight service": 3,
    "Cleanliness": 3,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0,
}

CATEGORY_OPTIONS = {col: list(le.classes_) for col, le in encoders.items()}


@app.route("/")
def index():
    return render_template(
        "index.html",
        rating_fields=VISIBLE_RATING_FIELDS,
        visible_categorical_fields=VISIBLE_CATEGORICAL_FIELDS,
        category_options=CATEGORY_OPTIONS,
        model_name=model_name,
    )


def build_feature_row(payload: dict) -> pd.DataFrame:
    row = {}
    for col in categorical_cols:
        raw_value = payload.get(col, HIDDEN_DEFAULTS.get(col, ""))
        le = encoders[col]
        if raw_value not in list(le.classes_):
            raise ValueError(f"Unrecognized value '{raw_value}' for '{col}'")
        row[col] = le.transform([raw_value])[0]

    for col in numeric_cols:
        raw_value = payload.get(col, HIDDEN_DEFAULTS.get(col, 0))
        try:
            row[col] = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for '{col}'")

    return pd.DataFrame([row])[feature_order]


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        X = build_feature_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    X_input = scaler.transform(X) if uses_scaling else X

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    label = target_encoder.inverse_transform([pred])[0]
    confidence = float(np.max(proba))

    satisfied_index = list(target_encoder.classes_).index("satisfied")

    return jsonify({
        "prediction": label,
        "is_satisfied": bool(label == "satisfied"),
        "confidence": round(confidence * 100, 1),
        "satisfied_probability": round(float(proba[satisfied_index]) * 100, 1),
        "model_name": model_name,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)