import pandas as pd
import os
import sys
import subprocess
from dotenv import load_dotenv, find_dotenv

import mlflow
from autogluon.core.metrics import make_scorer

# Make sure that correct .env is loaded
load_dotenv(find_dotenv(filename="mlops_project.env", usecwd=True, raise_error_if_not_found=True))
project_path = os.getenv("PROJECT_PATH")
print(f"project path {project_path}")
if project_path not in sys.path:
    sys.path.append(project_path)

from modules.kaggle_client import KaggleClient
from modules.data_preprocesing import DataPreprocessor
from modules.feature_engineering import FeatureEnginering
from modules.autogloun_model_class import AutogluonModel
from modules.metrics import root_mean_squared_log_error
from modules.mlflow_client import MLflowAutoGluon
from autogluon.tabular import TabularPredictor


competition_name = "playground-series-s3e11"
run_time = 1
target = "cost"

# Initialize KaggleClient to be able to talk to kaggle API
kaggle_client = KaggleClient()

# Downloads raw data for a given competition locally to the data folder
### potentially adjust to AWS or GCS storage as an option
### make sure that the path to the data is sourced from .env
kaggle_client.download_data(competition_name)

# Preprocess data to adjust data types and reduce memory usage
data_preprocessor = DataPreprocessor(competition_name)
train = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "train.csv"))
test = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "test.csv"))
submission = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "sample_submission.csv"))
### make sure that loading happens using class methods
train = data_preprocessor.optimize_dtypes(train)
test = data_preprocessor.optimize_dtypes(test)


# feature_engineering.openfe_fit(train)
# Feature engineering with OpenFE
feature_engineering = FeatureEnginering(competition_name, target_column="cost")
train_transformed, test_transformend = feature_engineering.openfe_transform(train, test)

train_transformed = pd.read_pickle(
    filepath_or_buffer=f"{project_path}/data/{competition_name}/feature_engineered/train_transformed.pkl"
)
test_transformed = pd.read_pickle(
    filepath_or_buffer=f"{project_path}/data/{competition_name}/feature_engineered/test_transformed.pkl"
)

# Deploy it as a batch, streaming or online service

# Create a visualization in Looker

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
)
mlflow_autogluon_local.train_and_log_model(
    presets=["medium_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
)

# Scenario 2:
## tracking server: yes, local server
## backend store: sqlite database
## artifacts store: local filesystem

mlflow_autogluon_local_server = MLflowAutoGluon(
    tracking_server="local",
    backend_store=f"{project_path}/backend.db",
    artifact_location=f"{project_path}/mlruns",
    experiment_name="test",
)
mlflow_autogluon_local_server.train_and_log_model(
    presets=["medium_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
)

# Scenario 3: Remote server with PostgreSQL and S3
tracking_server = os.getenv("TRACKING_SERVER_HOST")
artifact_path_s3 = os.getenv("AWS_BUCKET_NAME")

mlflow_autogluon_remote_aws_ec2 = MLflowAutoGluon(
    tracking_server="remote",
    backend_store=tracking_server,
    artifact_location=f"s3://{artifact_path_s3}",
    experiment_name="test",
)
mlflow_autogluon_remote_aws_ec2.train_and_log_model(
    presets=["medium_quality"],
    target=target,
    train_transformed=train_transformed,
    test_transformed=test_transformed,
    run_time=1,
)
