import pandas as pd
import pickle
import sys
import os
from autogluon.tabular import TabularPredictor, TabularDataset
from flask import Flask, request, jsonify


# Load the model
try:
    predictor = TabularPredictor.load("test_model_ag")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = Flask("prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        data_df = TabularDataset([data])
        pred = predictor.predict(data_df)[0]
        result = {"cost": round(float(pred), 2)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
