import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from flask import Flask, request, jsonify
from dotenv import load_dotenv, find_dotenv
import os


# parametrize target prediction column
load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True))
target = os.getenv("TARGET_COLUMN", "cost")


# Load the model
try:
    predictor = TabularPredictor.load("AutoGluon_mlflow_best_quality_deployment")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = Flask("prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Endpoint for making predictions.

    This endpoint accepts a POST request with JSON data containing the input for prediction.
    It uses the `predictor` object to make the prediction and returns the result as a JSON response.

    Returns:
        A JSON response containing the predicted result or an error message if an exception occurs.
    """
    try:
        data = request.get_json()
        data_df = TabularDataset([data])
        pred = predictor.predict(data_df)[0]
        result = {target: round(float(pred), 2)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
