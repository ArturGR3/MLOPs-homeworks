import mlflow
import pickle
from sklearn.pipeline import Pipeline
import os 


mlflow_db_password = os.environ.get("MLFLOW_GCP_DB_PASSWORD")
mlflow_db_name = os.environ.get("MLFLOW_GCP_DB_NAME")
mlflow_cloud_sql_ip = os.environ.get("MLFLOW_GCP_SQL_IP")
mlflow_bucket_name = os.environ.get("MLFLOW_GCS_BUCKET_NAME")

tracking_uri = f"postgresql://mlflow_user:{mlflow_db_password}@{mlflow_cloud_sql_ip}/{mlflow_db_name}"
# connect to URI
mlflow.set_tracking_uri("http://localhost:5000")
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

# # create artifact for dv
# with open('dv.bin', 'wb') as f:
#     pickle.dump(dv, f)
    
# mlflow.log_artifact('dv.bin')

mlflow.end_run()

