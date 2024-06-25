from batch import prepare_data
from datetime import datetime
import pandas as pd 
# Add the directory containing batch.py to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    actual_result = prepare_data(df, categorical=['PULocationID', 'DOLocationID']) 
    actuals_result_dict = actual_result[['PULocationID', 'DOLocationID', 'duration']].to_dict(orient='records')
    
    expected_result_dict = [{'PULocationID': '-1', 'DOLocationID': '-1', 'duration': 9.0},
    {'PULocationID': '1', 'DOLocationID': '1', 'duration': 8.0}]
        
    assert actuals_result_dict == expected_result_dict
    
