import pickle
import pandas as pd
import sklearn 
import sys 
import click
import os 
import numpy as np

import os

year = int(os.getenv('year'))
month = os.getenv('month')

training_data = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet"
output_file = f"yellow_tripdata_{year}-{month}_prediction.parquet"

print('--------')
print(f"Yellow taxi data for {year}-{month} is loaded from {training_data}\n and the prediction is saved to {output_file}")
print('--------')

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Describing the linear regression model
feature_names = dv.get_feature_names_out()
non_zero_indices = model.coef_ != 0

print("Linear Regression Model Formula:\ny = ", end="")
for i, coef in enumerate(model.coef_[non_zero_indices]):
    if i > 0:
        print(" + ", end="")
    print(f"{coef} * {feature_names[non_zero_indices][i]}", end="")
print(f" + {model.intercept_}")

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(training_data)

# find the share of DOlocationID=177
# print(df['DOLocationID'].value_counts(normalize=True).loc['177'])

df['ride_id'] = f'{year:04d}/{int(month):02d}_' + df.index.astype('str')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Create a prediction file with ride_id and y_pred
df['prediction'] = y_pred

# Show example of the data 
print('--------')
print(df.head(1).T)

df_result = df[['ride_id', 'prediction']]

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
standard_deviation = y_pred.std()
mean_duration = y_pred.mean()

# Find the file of the output_file 
# $ du -sh {output_file}

# make this executable with printing standard deviation of y_pred
if __name__ == '__main__':
    print('--------')
    print(f"std: {standard_deviation:.3f}, mean is: {mean_duration:.3f}")
