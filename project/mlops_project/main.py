import pandas as pd
import os
import sys
import mlflow 
from mlflow.tracking import MlflowClient
from autogluon.tabular import TabularPredictor
# import subprocess
from dotenv import load_dotenv, find_dotenv

# import mlflow
# from autogluon.core.metrics import make_scorer

# Make sure that correct .env is loaded
load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True))
project_path = os.getenv("PROJECT_PATH")
print(f"project path {project_path}")
# Make sure that the project path is in the sys.path to be able to import modules
if project_path not in sys.path:
    sys.path.append(project_path)

from modules.kaggle_client import KaggleClient
from modules.data_preprocesing import DataPreprocessor
from modules.feature_engineering import FeatureEnginering
from modules.mlflow_client import MLflowAutoGluon
from modules.download_folder_s3 import download_s3_folder, list_files_in_s3_folder
import logging
# from autogluon.tabular import TabularPredictor


competition_name = "playground-series-s3e11"
run_time = 1
target = "cost"

# Initialize KaggleClient to be able to talk to kaggle API
kaggle_client = KaggleClient(competition_name=competition_name, target_column=target)

# Downloads raw data for a given competition locally to the data folder
### potentially adjust to AWS or GCS storage as an option
### make sure that the path to the data is sourced from .env
kaggle_client.download_data()

# Preprocess data to adjust data types and reduce memory usage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
data_preprocessor = DataPreprocessor(competition_name)
train = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "train.csv"))
test = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "test.csv"))
submission = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "sample_submission.csv"))
train = data_preprocessor.optimize_dtypes(train)
test = data_preprocessor.optimize_dtypes(test)

# Feature engineering with OpenFE
feature_engineering = FeatureEnginering(competition_name, target_column="cost")
# openfe_transform method does:
## train-test split (80-20) from initial train data
## stratified sampling on (80% of train data) to get stratified samples
## using openfe to transform the data
## returns transformed train and test data
train_transformed, test_transformend = feature_engineering.openfe_transform(train, test)

# Saving transformed data
train_transformed = pd.read_pickle(
    filepath_or_buffer=f"{project_path}/data/{competition_name}/feature_engineered/train_transformed.pkl"
)
test_transformed = pd.read_pickle(
    filepath_or_buffer=f"{project_path}/data/{competition_name}/feature_engineered/test_transformed.pkl"
)

# Create autogloun model
# You need to make sure that EC2 instance is running the server
tracking_server = os.getenv("TRACKING_SERVER_HOST")
artifact_path_s3 = os.getenv("AWS_BUCKET_NAME")

mlflow_autogluon_remote_aws_ec2 = MLflowAutoGluon(
    tracking_server="remote",
    backend_store=tracking_server,
    artifact_location=f"s3://{artifact_path_s3}",
    experiment_name="test",
    competition_name=competition_name,
    target_name=target,
)
mlflow_autogluon_remote_aws_ec2.train_and_log_model(
    presets=["best_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
)

# Submit the model to Kaggle
model_name = "Autogluon_test"
message = "Integration Test Submission"
submission_file = f"{project_path}/data/{competition_name}/raw/sample_submission.csv"
# Import the model from MLflow from s3 bucket 
# and make predictions on the test data

experiment_id = 1 
run_id = '5610ecacb173409498fd1ed3281a1982'
preset = 'best_quality'
s3_folder = f"{experiment_id}/{run_id}/artifacts/AutoGluon_mlflow_best_quality_deployment/artifacts/AutoGluon_mlflow_best_quality_deployment/"
list_files_in_s3_folder(bucket_name=artifact_path_s3, s3_folder=s3_folder)
download_s3_folder(bucket_name=artifact_path_s3, 
                   s3_folder=s3_folder, 
                   local_dir=f"{project_path}/model/{competition_name}")

predictor = TabularPredictor.load(f"{project_path}/model/{competition_name}")

submission_path = f"data/{competition_name}/submission_files"
os.makedirs(submission_path, exist_ok=True)

submission = pd.read_csv(
    f"data/{competition_name}/raw/sample_submission.csv"
)
submission[target] = predictor.predict(test_transformed)
submission_file = f"{submission_path}/sub_{run_time}_{preset}.csv"
submission.to_csv(submission_file, index=False)

kaggle_score = kaggle_client.submit(
    submission_file=submission_file,
    model_name=preset,
    message=f"AutoGluon {preset} {run_time} min",
)

client = MlflowClient('http://ec2-13-60-23-26.eu-north-1.compute.amazonaws.com:5000')
# list all the experiments 
experiments = client.search_experiments()

for experiment in experiments:
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    print("---")

experiment_name = "test"
experiment = client.get_experiment_by_name(experiment_name)
run_name = 'best_quality'

# Search for runs with the specified run name
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{run_name}'"
)

# Extract and print the run ID
if runs:
    run_id = runs[0].info.run_id
    print(f"Run ID: {run_id}")
else:
    print("No run found with the specified name.")

# Log the new metric to the existing run
metric_name = "kaggle_score"
metric_value = kaggle_score

# Log the new metric to the existing run
client.log_metric(run_id, metric_name, metric_value)

predictor.predict(test_transformed).to_csv(submission_file, index=False)


# Example usage
# Scenario 1:
## Tracking server: no
## Backend store: local filesystem
## Artifacts store: local filesystem

mlflow_autogluon_local = MLflowAutoGluon(
    tracking_server="no",
    backend_store=f"{project_path}/mlruns",
    artifact_location=f"{project_path}/mlruns",
    experiment_name="test",
    competition_name=competition_name,
    target_name=target,
)
mlflow_autogluon_local.train_and_log_model(
    presets=["medium_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
    for_deployment=True,
    for_kaggle_submission=False,
)



# Scenario 2:
## tracking server: yes, local server
## backend store: sqlite database
## artifacts store: local filesystem

mlflow_autogluon_local_server = MLflowAutoGluon(
    tracking_server="local",
    backend_store=f"{project_path}/backend.db",
    artifact_location=f"{project_path}/mlruns",
    experiment_name="test_32",
    competition_name=competition_name,
    target_name=target,
)
mlflow_autogluon_local_server.train_and_log_model(
    presets=["best_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
)

# Scenario 3: Remote server with PostgreSQL and S3
# You need to make sure that EC2 instance is running the server (per module2 instructions)
tracking_server = os.getenv("TRACKING_SERVER_HOST")
artifact_path_s3 = os.getenv("AWS_BUCKET_NAME")

mlflow_autogluon_remote_aws_ec2 = MLflowAutoGluon(
    tracking_server="remote",
    backend_store=tracking_server,
    artifact_location=f"s3://{artifact_path_s3}",
    experiment_name="test",
    competition_name=competition_name,
    target_name=target,
)
mlflow_autogluon_remote_aws_ec2.train_and_log_model(
    presets=["best_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
)

