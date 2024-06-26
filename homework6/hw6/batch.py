#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))    

def prepare_data(data_frame, categorical_columns):
    """
    Prepares data by calculating duration and filtering based on it.
    """
    categorical_columns = ['PULocationID', 'DOLocationID']
    
    data_frame['duration'] = data_frame.tpep_dropoff_datetime - data_frame.tpep_pickup_datetime
    data_frame['duration'] = data_frame.duration.dt.total_seconds() / 60

    data_frame = data_frame[(data_frame.duration >= 1) & (data_frame.duration <= 60)].copy()

    data_frame[categorical_columns] = data_frame[categorical_columns].fillna(-1).astype('int').astype('str')    
    return data_frame

def read_data(filename, categorical_columns):
    """
    Reads data from a file, supporting reading from S3 if configured.
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    
    if s3_endpoint_url is not None:
        options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
        data_frame = pd.read_parquet(filename, storage_options=options)
        print('read data from s3')
    else:
        data_frame = pd.read_parquet(filename)
        print('read data from local')

    return prepare_data(data_frame, categorical_columns)

def save_data(data_frame, filename):
    """
    Saves data to a file, supporting saving to S3 if configured.
    """
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    
    if s3_endpoint_url is not None:
        options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
        data_frame.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
        print('saved data to s3')
    else:
        data_frame.to_parquet(filename, engine='pyarrow', index=False)
        print('saved data to local')

def get_input_path(year, month):
    """
    Generates the input file path based on the year and month.
    """
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    """
    Generates the output file path based on the year and month.
    """
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year, month):
    """
    Main function to process taxi trip data.
    """
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    
    # load the model 
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    # prepare the data 
    categorical_columns = ['PULocationID', 'DOLocationID']

    data_frame = read_data(input_file, categorical_columns)
    data_frame['ride_id'] = f'{year:04d}/{month:02d}_' + data_frame.index.astype('str')

    dicts = data_frame[categorical_columns].to_dict(orient='records')
    x_val = dv.transform(dicts)
    y_pred = lr.predict(x_val)

    print('predicted mean duration:', y_pred.mean().round(2))

    df_result = pd.DataFrame()
    df_result['ride_id'] = data_frame['ride_id']
    df_result['predicted_duration'] = y_pred
        
    save_data(df_result, output_file)

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    main(year=year, month=month)
