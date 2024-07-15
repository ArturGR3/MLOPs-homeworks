import mlflow
import pickle
from sklearn.pipeline import Pipeline
import os 
# from AWS_S3 import access_s3_bucket
from dotenv import load_dotenv, find_dotenv 

# Load environment variables
load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))
TRACKING_SERVER_HOST = os.environ.get("TRACKING_SERVER_HOST")

# Connect to MLFLOW server
# access_s3_bucket()
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.get_tracking_uri()

# List what experiments are available
mlflow.search_experiments()

mlflow.start_run(run_name='run_from_script')

# load binary file to store in mlflow
with open('lin_reg.bin', 'rb') as f:
    (dv, model) = pickle.load(f)
    
# combine dv and model in pipeline to avoid artifacts pulling
pipeline = Pipeline([
    ('data_vectorizer', dv),
    ('model', model)
])

# store pipeline in mlflow
mlflow.sklearn.log_model(pipeline, 'model')
mlflow.end_run()

