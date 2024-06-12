
# Run the code with data type adjustment
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import time as time 
from sklearn.metrics import root_mean_squared_log_error
from common_files.data_loader import adjust_dtypes



subprocess.call([
    "python", 
    "/home/artur/MLOPs-homeworks/project/common_files/data_loader.py", 
    "--competition_name", 
    "playground-series-s3e11", 
    "--optimize_dtypes", 
    "False", 
    "--original_data_set", 
    "https://www.kaggle.com/datasets/gauravduttakiit/media-campaign-cost-prediction"
])


competition_name = "playground-series-s3e11" # needs to be feed for automation 
data_folder = f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data"

target = 'cost' # needs to be feed with parameter


def train_model(train: pd.DataFrame, ad_dtypes: bool = True) -> None:
    if ad_dtypes:
        # Adjust data types of the train DataFrame
        train = adjust_dtypes(train, verbose=True, convert_float_to_int=True)
    else:
        train = train 
        
    # Split train DataFrame into train and validation sets
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=1)
    print(train_df.dtypes)
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]
    start_time = time.time()
    # Train LightGBM model
    lg = lgb.LGBMRegressor(n_estimators=3000).fit(X_train, y_train)
    y_pred = lg.predict(X_val)
    score = root_mean_squared_log_error(y_val, y_pred)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed: {time_elapsed}")
    print(f"Score: {score}")


train = pd.read_pickle(filepath_or_buffer=f"{data_folder}/train.pkl")
train.dtypes 
train_model(train, ad_dtypes=True)
train.dtypes
train = pd.read_pickle(filepath_or_buffer=f"{data_folder}/train.pkl")
train_model(train, ad_dtypes=False)
train.dtypes

train = adjust_dtypes(train, verbose=True)
train = train.drop(columns=['id'])