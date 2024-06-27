"""Integration test module for data processing."""

# pylint: disable=import-error

import os
from datetime import datetime

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from batch import get_input_path, get_output_path

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# S3 endpoint URL from environment
s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

# S3 options
options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}


def datetime_for_test(hour, minute, second=0):
    """Create a datetime object for 2023-01-01 with specified time."""
    return datetime(2023, 1, 1, hour, minute, second)


# Data setup
data = [
    (None, None, datetime_for_test(1, 1), datetime_for_test(1, 10)),
    (1, 1, datetime_for_test(1, 2), datetime_for_test(1, 10)),
    (1, None, datetime_for_test(1, 2, 0), datetime_for_test(1, 2, 59)),
    (3, 4, datetime_for_test(1, 2, 0), datetime_for_test(2, 2, 1)),
]

columns = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]
df = pd.DataFrame(data, columns=columns)

# File paths
input_file = get_input_path(2023, 1)
output_file = get_output_path(2023, 1)

# Load and save data to S3
df.to_parquet(
    input_file, engine="pyarrow", compression=None, index=False, storage_options=options
)
os.system("python batch.py 2023 1")

# Load saved data and validate
df_save = pd.read_parquet(output_file, storage_options=options)
print(f"Sum of predicted duration: {df_save.predicted_duration.sum():.2f}")

# Assertion for validation
assert df_save.predicted_duration.sum().round(1) == 36.3, "Predicted duration sum mismatch"
