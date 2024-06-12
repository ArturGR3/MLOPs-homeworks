import os
import zipfile
import time

import numpy as np
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

import lightgbm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_log_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import torch
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

# Define MLflow experiment name
experiment_name = "Kaggle_Competition_s3e11"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_name)

# Kaggle competition settings
competition_name = "playground-series-s3e11"
target_name = "cost"
problem_type = "reg"
metric = "rmsle"
loss_metric = "rmsle"
training_time = 10  # minutes

api = KaggleApi()
api.authenticate()

def load_data(comp_name: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Load data from a Kaggle competition zip file.
    Parameters:
        comp_name (str): Name of the Kaggle competition.
    Returns:
        tuple: A tuple containing three DataFrames: train, test, and submission.
    """
    # Download the competition files
    api.competition_download_files(comp_name, path=".", force=True)

    # Unzip the downloaded files
    with zipfile.ZipFile(f"{comp_name}.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    # Load data into DataFrames
    submission = pd.read_csv("sample_submission.csv")
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")

    return train, test, submission

def adjust_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust data types for the DataFrame columns.
    Parameters:
        df (pd.DataFrame): The DataFrame to adjust.
    Returns:
        pd.DataFrame: The DataFrame with adjusted data types.
    """
    int_columns = df.select_dtypes(include=["int64"]).columns
    float_columns = df.select_dtypes(include=["float64"]).columns

    # Change integer columns to the smallest type that fits the data
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Change float columns to the smallest float type that fits the data
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df

def kaggle_submition(
    submission_output: pd.DataFrame,
    model_name: str,
    competition_name: str,
    version_number: str,
) -> float:
    """
    Submits a DataFrame to a Kaggle competition and retrieves the submission score.

    Args:
        submission_output (pd.DataFrame): The DataFrame to submit.
        model_name (str): The name of the model.
        competition_name (str): The name of the Kaggle competition.
        version_number (str): The version number of the submission.

    Returns:
        float: The public score of the submission if successful, otherwise None.
    """
    # Save submission to a CSV file
    submission_file = f"submission_{model_name}.csv"
    submission_output.to_csv(submission_file, index=False)

    # Submit to Kaggle competition
    submission_message = f"{model_name} version {version_number}"

    try:
        kaggle.api.competition_submit(
            submission_file, submission_message, competition_name
        )
    except kaggle.rest.ApiException as e:
        print(f"failed to submit to Kaggle: {e}")
        return None

    # Get Kaggle submission score
    polling_interval = 5  # seconds between checks
    max_wait_time = 60  # maximum time to wait
    start_time = time.time()

    submission_score = None  # Initialize submission_score

    while (time.time() - start_time) < max_wait_time:
        try:
            submissions = kaggle.api.competitions_submissions_list(competition_name)
            for sub in submissions:
                if sub["description"] == submission_message:
                    public_score = sub.get("publicScore", "")
                    if public_score != "":
                        submission_score = round(np.float32(public_score), 4)
                        return submission_score
                    else:
                        print("Public score not yet available, waiting...")
        except kaggle.rest.ApiException as e:
            print(f"Failed to get submission score: {e}")

        time.sleep(polling_interval)

    print("Failed to retrieve submission score within the maximum wait time.")
    return submission_score

# Load data
train, test, submission = load_data(competition_name)

# Adjust data types
train = adjust_dtypes(train)
test = adjust_dtypes(test)
submission = adjust_dtypes(submission)

# Split data into features and target
x_train_full = train.drop(columns=[target_name])
y_train_full = train[target_name]

# Split data into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

x_train = train_df.drop(columns=[target_name])
y_train = train_df[target_name]

x_val = val_df.drop(columns=[target_name])
y_val = val_df[target_name]

x_test = test

# AutoML parameters
torch.set_num_threads(4)

automl_params = {
    "task": Task(problem_type, loss=loss_metric, metric=metric),
    "timeout": training_time * 60,
    "reader_params": {"n_jobs": 4, "cv": 3, "random_state": 42},
    "cpu_limit": 4,
}


def fit_and_predict_automl(
    model, df_full: pd.DataFrame, x_test: pd.DataFrame, target_name
):
    # Fit the model and make predictions
    oof_pred = model.fit_predict(df_full, roles={"target": target_name}, verbose=1)
    score_oof = root_mean_squared_log_error(df_full[target_name], oof_pred.data[:, 0])
    y_pred_test = model.predict(x_test).data[:, 0]
    print(f"{model} oof_pred: {score_oof:.4f}")
    mlflow.log_metric("oof_score", round(score_oof, 4))
    return y_pred_test


# Train models and record R2 scores
models = {
    "Linear_Regression": LinearRegression(),
    "Ridge": Ridge(),
    # "LightGBM": lightgbm.LGBMRegressor(),
    # "LightAutoML": TabularAutoML(**automl_params)
}

def fit_and_predict(model, x_train, y_train, x_val, x_test, x_train_full, y_train_full):
    # Enable autologging
    mlflow.autolog()

    # Fit the model and make predictions
    model.fit(x_train, y_train)
    y_pred_val = model.predict(x_val)

    # Fit the model for the whole set to predict on the test set
    model.fit(x_train_full, y_train_full)
    y_pred_test = model.predict(x_test)

    return y_pred_val, y_pred_test


def log_metrics_and_params(metric_score, model_name, competition_name, score):
    # Log R^2 score
    mlflow.log_metric(f"{metric} validation set", round(metric_score, 4), step=0)

    # Log parameters and artifacts
    mlflow.log_param("Model Name", model_name)
    mlflow.log_param("Competition Name", competition_name)

    submission_score = 0.2 if score is None else score
    mlflow.log_metric("Kaggle Score", round(submission_score, 5), step=0)


with mlflow.start_run(run_name="Base sumbitions with Autologing") as parent_run:
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            y_pred_val, y_pred_test = fit_and_predict(
                model, x_train, y_train, x_val, x_test, x_train_full, y_train_full
            )

            metric_score = root_mean_squared_log_error(y_val, y_pred_val)

            # Update submission DataFrame with predictions
            submission[target_name] = y_pred_test

            score = kaggle_submition(submission, model_name, competition_name, 0)

            log_metrics_and_params(metric_score, model_name, competition_name, score)
