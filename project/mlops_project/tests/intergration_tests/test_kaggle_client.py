import os
from module_name.kaggle_client import KaggleClient


def test_kaggle_integration():
    client = KaggleClient()
    competition_name = "playground-series-s3e11"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, f"data/{competition_name}/raw")

    # Test download_data method
    client.download_data(competition_name)
    assert os.path.exists(f"{data_path}/sample_submission.csv")

    # Test submit method
    submission_file = f"{data_path}/sample_submission.csv"
    message = "Integration Test Submission"
    score = client.submit(submission_file, competition_name, message)
    assert score is not None
