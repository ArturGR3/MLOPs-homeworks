import mlflow
import os
import subprocess
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from mlflow.pyfunc import PythonModel

# from metrics import root_mean_squared_log_error

# metrics.py
from sklearn.metrics import mean_squared_log_error
import numpy as np


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


class MLflowAutoGluon:
    def __init__(self, tracking_server, backend_store, artifact_location, experiment_name):
        self.tracking_server = tracking_server
        self.backend_store = backend_store
        self.artifact_location = artifact_location
        self.experiment_name = experiment_name
        self.setup_mlflow()

    class AutogluonModel(PythonModel):
        def load_context(self, context):
            self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

        def predict(self, context, model_input):
            return self.predictor.predict(model_input)

    def check_port_in_use(self, port):
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Port {port} is in use: {result.stdout.strip()}")
        else:
            print(f"Port {port} is not in use")
        return result

    def kill_process_on_port(self, port=5000):
        result = self.check_port_in_use(port)
        if result.returncode == 0:
            # If the port is in use, get the process IDs and kill them
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                subprocess.run(["kill", "-9", pid])
            print(f"Killed the process on port {port}")

    def start_mlflow_server(self, port=5000):
        self.kill_process_on_port(port)
        with open(os.devnull, "wb") as devnull:
            process = subprocess.Popen(
                ["mlflow", "server", "--backend-store-uri", f"sqlite:///{self.backend_store}"],
                stdout=devnull,
                stderr=devnull,
                stdin=devnull,
                close_fds=True,
            )

    def create_mlflow_experiment(self):
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
        except AttributeError:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=f"{self.artifact_location}",
            )
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Experiment Name: {experiment.name}")
        print(f"Experiment_id: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Creation timestamp: {experiment.creation_time}")

    def setup_mlflow(self):
        if self.tracking_server == "no":
            print("Local setup with no tracking server")
            self.create_mlflow_experiment()
            mlflow.set_tracking_uri(f"file://{self.backend_store}")

        elif self.tracking_server == "local":
            print("Local setup with Local tracking server")
            self.start_mlflow_server()
            self.create_mlflow_experiment()
            mlflow.set_tracking_uri(f"sqlite:///{self.backend_store}")

        elif self.tracking_server == "remote":
            print("Remote setup with remote tracking server")
            self.kill_process_on_port()
            self.create_mlflow_experiment()
            mlflow.set_tracking_uri(f"http://{self.backend_store}:5000")

        tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_experiment(self.experiment_name)
        print(f"Current tracking uri: {tracking_uri}")

    def train_and_log_model(self, presets, target, train_transformed, test_transformed, run_time):

        rmsle = make_scorer(
            "rmsle", root_mean_squared_log_error, greater_is_better=False, needs_proba=False
        )

        for preset in presets:
            with mlflow.start_run(run_name=f"{preset}") as parent_run:
                predictor = TabularPredictor(
                    label=target, path=f"AutoGluon_mlflow_{preset}", eval_metric=rmsle
                )
                predictor.fit(
                    train_data=train_transformed,
                    time_limit=run_time * 60,
                    presets=preset,
                    excluded_model_types=["KNN", "NN"],
                )

                test_pred = predictor.predict(test_transformed)

                leaderboard = predictor.leaderboard(silent=True)
                best_model = leaderboard.loc[leaderboard["score_val"].idxmax()]
                score_val = round(-best_model["score_val"], 4)
                inf_time = round(best_model["pred_time_val"], 4)
                fit_time = round(best_model["fit_time"], 4)

                mlflow.log_metric("score_val", score_val)
                mlflow.log_metric("inference_time", inf_time)
                mlflow.log_metric("fit_time", fit_time)

                for index, row in leaderboard.iterrows():
                    model_name = row["model"]
                    model_path = f"AutoGluon_mlflow_{preset}/models/{model_name}/"

                    with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_metric("score_val", round(-row["score_val"], 4))
                        mlflow.log_metric("inference_time", round(row["pred_time_val"], 4))
                        mlflow.log_metric("fit_time", round(row["fit_time"], 4))

                        model = self.AutogluonModel()
                        artifacts = {"predictor_path": model_path}
                        mlflow.pyfunc.log_model(
                            artifact_path="artifacts", python_model=model, artifacts=artifacts
                        )

            mlflow.end_run()
