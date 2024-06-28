import os
import zipfile
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import time
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(usecwd=True))


class KaggleClient:
    def __init__(self):

        os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

        self.api = KaggleApi()
        self.api.authenticate()

    def download_data(self, competition_name):

        competition_name = "playground-series-s3e11"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, f"data/{competition_name}/raw")
        os.makedirs(data_path, exist_ok=True)
        self.api.competition_download_files(competition_name, path=data_path)
        zip_file = os.path.join(data_path, f"{competition_name}.zip")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_file)
        print(f"Data downloaded and extracted to {data_path}")

    def submit(self, submission_file, competition_name, message):
        self.api.competition_submit(submission_file, message, competition_name)
        print(f"Submission {submission_file} for {competition_name}: '{message}'")
        polling_interval = 5
        max_wait_time = 60
        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            submissions = self.api.competitions_submissions_list(competition_name)
            for sub in submissions:
                if sub["description"] == message:
                    public_score = sub.get("publicScore", "")
                    if public_score:
                        print(f"Submission score: {public_score}")
                        return public_score
            time.sleep(polling_interval)
        print("Failed to retrieve submission score within the maximum wait time.")
        return None
