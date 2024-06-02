import pandas as pd
import numpy as np
import time

import kaggle.api

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