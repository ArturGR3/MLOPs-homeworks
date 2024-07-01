import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv, find_dotenv
import os
import logging


def access_s3_bucket() -> tuple[str, str, boto3.client]:
    # Load environment variables from .env file, add a try with usecwd
    # load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))
    load_dotenv(find_dotenv(usecwd=True))
    # Get the AWS credentials and bucket name from environment variables
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("AWS_BUCKET_NAME")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        # Check if the bucket exists
        s3.head_bucket(Bucket=bucket_name)
    except NoCredentialsError:
        logging.error("Credentials not available")
    except PartialCredentialsError:
        logging.error("Incomplete credentials provided")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    return bucket_name, s3


# test the function
bucket_name, s3 = access_s3_bucket()
