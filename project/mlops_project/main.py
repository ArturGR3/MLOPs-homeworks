import pandas as pd
import os
import sys
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
train_transforment, test_transformend = feature_engineering.openfe_transform(train, test)

# Apply AutoGluon training
predictor = TabularPredictor(label="cost")
predictor.fit(
    train_transforment, presets="medium_quality", time_limit=run_time * 60, infer_limit=0.001
)

leaderboard = predictor.leaderboard(silent=True)
best_model = leaderboard.loc[leaderboard["score_val"].idxmax()]
score_val = round(-best_model["score_val"], 4)
inf_time = round(best_model["pred_time_val"], 4)
fit_time = round(best_model["fit_time"], 4)

print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

# Log it with mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("test")
presets = ["medium_quality"]


## define a metric in a separate file
rmsle = make_scorer(
    "rmsle", root_mean_squared_log_error, greater_is_better=False, needs_proba=False
)

# Iterate over the presets
for preset in presets:
    # Start a parent run for the preset
    with mlflow.start_run(run_name=f"{preset}") as parent_run:
        # Train AutoGluon with the preset
        predictor = TabularPredictor(
            label=target, path=f"AutoGloun_mlflow_{preset}", eval_metric=rmsle
        )
        predictor.fit(
            train_data=train,
            time_limit=run_time * 60,
            presets=preset,
            excluded_model_types=["KNN", "NN"],
        )

        # Predict on the test set for subission
        test_pred = predictor.predict(test)
        submission[target] = test_pred

        # Submit to kaggle using kaggle_submition
        kaggle_score = kaggle_client.submit(
            submission,
            competition_name=competition_name,
            model_name=f"AutoGloun_{preset}",
            message="test",
        )

        # Get the leaderboard
        leaderboard = predictor.leaderboard(silent=True)

        # Select the row with the best validation score
        best_model = leaderboard.loc[leaderboard["score_val"].idxmax()]
        score_val = round(-best_model["score_val"], 4)
        inf_time = round(best_model["pred_time_val"], 4)
        fit_time = round(best_model["fit_time"], 4)

        # Log best model to MLflow
        mlflow.log_metric("score_val", score_val)
        mlflow.log_metric("inference_time", inf_time)
        mlflow.log_metric("fit_time", fit_time)
        mlflow.log_metric("kaggle_score", round(kaggle_score, 4))

        # Iterate over the models in the leaderboard
        for index, row in leaderboard.iterrows():
            model_name = row["model"]
            model_path = f"AutoGloun_mlflow_{preset}/models/{model_name}/"

            # Start a child run for the model
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                # Log the model's score and inference time
                mlflow.set_tag("model_name", model_name)
                mlflow.log_metric("score_val", round(-row["score_val"], 4))
                mlflow.log_metric("inference_time", round(row["pred_time_val"], 4))
                mlflow.log_metric("fit_time", round(row["fit_time"], 4))

                # Log the model in MLflow
                model = AutogluonModel()
                artifacts = {"predictor_path": model_path}
                mlflow.pyfunc.log_model(
                    artifact_path="artifacts", python_model=model, artifacts=artifacts
                )


# Deploy it as a batch, streaming or online service

# Create a visualization in Looker
