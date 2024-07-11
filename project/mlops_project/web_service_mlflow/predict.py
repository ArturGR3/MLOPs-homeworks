import pandas as pd
import os
from autogluon.tabular import TabularPredictor, TabularDataset
from flask import Flask, request, jsonify
import boto3
from download_folder_s3 import download_s3_folder, list_files_in_s3_folder
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=True))

# Specify your bucket name and model file path
AWS_BUCKER_NAME = os.getenv("AWS_BUCKET_NAME")
experiment_id = os.getenv("EXPERIMENT_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# Initialize the S3 client
s3 = boto3.client("s3")

# Download the folder from s3

s3_model_folder = f"1/{experiment_id}/artifacts/AutoGluon_mlflow_best_quality_deployment/artifacts/AutoGluon_mlflow_best_quality_deployment/"
local_model_path = "model_ag_deployment"
# Ensure the local folder path exists
os.makedirs(local_model_path, exist_ok=True)

# download the model from s3
download_s3_folder(
    bucket_name=AWS_BUCKER_NAME, s3_folder=s3_model_folder, local_dir=local_model_path
)

# Load the model from the local file
try:
    predictor = TabularPredictor.load("model_ag_deployment")
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
