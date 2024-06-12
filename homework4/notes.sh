export MLFLOW_TRACKING_URI="postgresql://mlflow_user:qwerty12345@104.155.25.51/mlflow"
export MLFLOW_S3_ENDPOINT_URL="https://storage.googleapis.com"


echo 'MLFLOW_GCP_DB_PASSWORD="qwerty12345"' >> ~/.bashrc
echo 'MLFLOW_GCP_DB_NAME="mlflow_db"' >> ~/.bashrc
echo 'MLFLOW_GCP_SQL_IP="104.155.25.51"' >> ~/.bashrc
echo 'MLFLOW_GCS_BUCKET_NAME="mlflow_bucket_ag3' >> ~/.bashrc

export MLFLOW_GCP_DB_PASSWORD="qwerty12345"
export MLFLOW_GCP_DB_NAME="mlflow_db"
export MLFLOW_GCP_SQL_IP="104.155.25.51"
export MLFLOW_GCS_BUCKET_NAME="mlflow_bucket_ag3"

mlflow server \
  --backend-store-uri postgresql://mlflow_user:$MLFLOW_GCP_DB_PASSWORD@$MLFLOW_GCP_SQK_IP/$MLFLOW_GCP_DB_NAME \
  --default-artifact-root gs://$MLFLOW_GCS_BUCKET_NAME/mlflow/ \
  --host 0.0.0.0