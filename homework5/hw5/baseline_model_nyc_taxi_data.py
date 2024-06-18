import requests
import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric,  ColumnQuantileMetric, RegressionPerformanceMetrics

from joblib import load, dump
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

year = 2024
month = 3

def download_taxi_data(year, month):
    files = [(f'green_tripdata_{year}-{month}.parquet', './data')]

    for file, path in files:
        url=f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
        resp=requests.get(url, stream=True)
        save_path=f"{path}/{file}"
        with open(save_path, "wb") as handle:
            for data in tqdm(resp.iter_content(),
                            desc=f"{file}",
                            postfix=f"save to {save_path}",
                            total=int(resp.headers["Content-Length"])):
                handle.write(data)

def data_preprocessing(data_path = 'data/green_tripdata_2024-03.parquet'):
    
    march_data = pd.read_parquet(data_path)
    print(f'Nrows: {march_data.shape[0]}')
    
    # Filter only March 2024
    march_data = march_data[(march_data.lpep_pickup_datetime >= '2024-03-01') & (march_data.lpep_pickup_datetime < '2024-04-01')]
    
    # Create a column that finds daily median (lpep_pickup_datetime) for fare_amount
    march_data['median_fare_amount'] = march_data.groupby(march_data.lpep_pickup_datetime.dt.date).fare_amount.transform('median')
    
    # Find the maximum median fare amount
    print(f"Max daily median for fare amount is: {march_data.median_fare_amount.max():.2f}") 
    
    # create target
    march_data["duration_min"] = march_data.lpep_dropoff_datetime - march_data.lpep_pickup_datetime
    march_data.duration_min = march_data.duration_min.apply(lambda td : float(td.total_seconds())/60)
    
    # filter out outliers
    march_data = march_data[(march_data.duration_min >= 0) & (march_data.duration_min <= 60)]
    march_data = march_data[(march_data.passenger_count > 0) & (march_data.passenger_count <= 8)]

    # data labeling
    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    train_data = march_data[:30000]
    val_data = march_data[30000:]
    
    return train_data, val_data, target, num_features, cat_features

train_data, val_data, target, num_features, cat_features = data_preprocessing()

def model_building(train_data, val_data, target, num_features, cat_features):
    
    # train model
    model = LinearRegression().fit(train_data[num_features + cat_features], train_data[target])

    # evaluate model
    train_preds = model.predict(train_data[num_features + cat_features])
    train_data['prediction'] = train_preds

    val_preds = model.predict(val_data[num_features + cat_features])
    val_data['prediction'] = val_preds

    # Find the mean absolute error for train and validation data
    print(mean_absolute_error(train_data.duration_min, train_data.prediction))
    print(mean_absolute_error(val_data.duration_min, val_data.prediction))

    # Save the model
    with open('models/lin_reg.bin', 'wb') as f_out:
        dump(model, f_out)

    val_data.to_parquet('data/reference.parquet')

