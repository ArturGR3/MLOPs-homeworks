import os
import zipfile

import numpy as np
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi 
from metrics import root_mean_squared_log_error
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
from kaggle_submition import kaggle_submition

# Set competition name
competition_name = "playground-series-s3e11"

# Load data 
def load_data(competition_name)-> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Load data from the specified Kaggle competition.

    Parameters:
        competition_name (str): The name of the Kaggle competition.

    Returns:
        train (pandas.DataFrame): The training data.
        test (pandas.DataFrame): The test data.
        submission (pandas.DataFrame): The submission data.
    """
    data_folder = f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data"
    train = pd.read_pickle(filepath_or_buffer=f"{data_folder}/train.pkl")
    test = pd.read_pickle(filepath_or_buffer=f"{data_folder}/test.pkl")
    submission = pd.read_pickle(filepath_or_buffer=f"{data_folder}/submission.pkl")
    return train, test, submission


experiment_name = f"Kaggle_Competition_{competition_name}"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_name)

train, test, submission = load_data(competition_name)
target = 'cost'
rmsle = make_scorer('rmsle', root_mean_squared_log_error, greater_is_better=False, needs_proba=False)
# Define your presets
presets = ['best_quality', 'high_quality' , 'good_quality', 'medium_quality']

class AutogluonModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

    def predict(self, context, model_input):
        return self.predictor.predict(model_input)

# Iterate over the presets
for preset in presets:
    # Start a parent run for the preset
    with mlflow.start_run(run_name=f"{preset}") as parent_run:
        # Train AutoGluon with the preset
        predictor = TabularPredictor(label=target, path=f'AutoGloun_mlflow_{preset}', eval_metric=rmsle)
        predictor.fit(train_data=train.drop(columns=['id']), time_limit=15*60, presets=preset, save_bag_folds=True)
        
        # Predict on the test set for subission
        test_pred = predictor.predict(test)
        submission['cost'] = test_pred
        
        # Submit to kaggle using kaggle_submition
        kaggle_score = kaggle_submition(submission, model_name=f'AutoGloun_{preset}', 
                                        competition_name=competition_name, version_number='test')
        
        # Get the leaderboard
        leaderboard = predictor.leaderboard(silent=True)

        # Select the row with the best validation score
        best_model = leaderboard.loc[leaderboard['score_val'].idxmax()]
        score_val = round(-best_model['score_val'],4)
        inf_time = round(best_model['pred_time_val'],4)
        fit_time = round(best_model['fit_time'],4)
        
        # Log best model to MLflow
        mlflow.log_metric("score_val", score_val)
        mlflow.log_metric("inference_time", inf_time)
        mlflow.log_metric("fit_time", fit_time)
        mlflow.log_metric("kaggle_score", round(kaggle_score,4))
       
        # Iterate over the models in the leaderboard
        for index, row in leaderboard.iterrows():
            model_name = row['model']
            model_path = f"AutoGloun_mlflow_{preset}/models/{model_name}/"
            
            # Start a child run for the model
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                # Log the model's score and inference time
                mlflow.set_tag("model_name", model_name)
                mlflow.log_metric("score_val", round(-row['score_val'],4))
                mlflow.log_metric("inference_time", round(row['pred_time_val'],4))
                mlflow.log_metric("fit_time", round(row['fit_time'],4))

                # Log the model in MLflow
                model = AutogluonModel()
                artifacts = {"predictor_path": model_path}
                mlflow.pyfunc.log_model(artifact_path="artifacts", python_model=model, artifacts=artifacts)


