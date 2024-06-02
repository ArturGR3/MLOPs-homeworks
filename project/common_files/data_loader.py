import os
import zipfile
import numpy as np
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import argparse
import click

api = KaggleApi()
api.authenticate()

def rename_columns(data: pd.DataFrame):
    data = data.copy()
    old_columns = data.columns
    data.columns = data.columns.str.replace(r'[.\(\) ]', '_', regex=True)
    # create a dictionary to store the old and new column names
    column_dict = dict(zip(old_columns, data.columns))
    return data, column_dict

def adjust_dtypes(df: pd.DataFrame, verbose: bool = True, convert_float_to_int: bool = True) -> pd.DataFrame:
    """
    Adjust data types for the DataFrame columns to the smallest possible types.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to adjust.
        verbose (bool): If True, prints memory usage before and after adjustment.
        convert_float_to_int (bool): If True, attempts to convert float columns to int if possible.
        
    Returns:
        pd.DataFrame: The DataFrame with adjusted data types.
    """
    def memory_usage_in_gb(df: pd.DataFrame) -> float:
        return df.memory_usage(deep=True).sum() / (1024 ** 3)  # Convert bytes to GB

    if verbose:
        initial_memory_gb = memory_usage_in_gb(df)
        print(f"Initial memory usage: {initial_memory_gb:.6f} GB")

    # Find int and float columns in the DataFrame
    int_columns = df.select_dtypes(include=["int64", "int32"]).columns
    float_columns = df.select_dtypes(include=["float64", "float32"]).columns
    object_columns = df.select_dtypes(include=["object"]).columns

    # Change integer columns to the smallest type that fits the data
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Change float columns to the smallest float type that fits the data
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Convert object columns to category if they have a small number of unique values
    for col in object_columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            # Convert to category if the column has relatively few unique values
            df[col] = df[col].astype("category")
        else:
            try:
                # Attempt to convert to datetime
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                # If conversion fails, continue without changes
                continue

    # Convert float32 columns to int if all values are whole numbers and within int range
    if convert_float_to_int:
        float_columns = df.select_dtypes(include=["float32"]).columns
        for col in float_columns:
            if np.all(np.modf(df[col])[0] == 0):  # Check if all values are whole numbers
                min_val, max_val = df[col].min(), df[col].max()
                if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

    if verbose:
        final_memory_gb = memory_usage_in_gb(df)
        print(f"Final memory usage: {final_memory_gb:.6f} GB")
        print(f"Memory reduced from {initial_memory_gb:.6f} GB to {final_memory_gb:.6f} GB, "
              f"which is a reduction of {initial_memory_gb - final_memory_gb:.6f} GB "
              f"({100 * (initial_memory_gb - final_memory_gb) / initial_memory_gb:.2f}%).")
    
    return df

# Create a function to download original file from the link provided (eg. https://www.kaggle.com/datasets/gauravduttakiit/media-campaign-cost-prediction)
def download_original_data(original_data_set: str):
    """
    Download the original data from the link provided.
    Parameters:
        link (str): The link to the original data.
    """
    dataset_owner = original_data_set.split('/')[4]
    dataset_name = original_data_set.split('/')[5]
    file_name = 'train_dataset.csv'
    api.dataset_download_file(f'{dataset_owner}/{dataset_name}', file_name, path='.')
    # unzip the downloaded file
    with zipfile.ZipFile(f"{file_name}.zip", "r") as zip_ref:
        zip_ref.extractall(".")
        # Load data into DataFrames
        train = pd.read_csv("train_dataset.csv")
    return train

# Kaggle competition settings
@click.command()
@click.option(
    '--competition_name',
    type=str, 
    help='Name of the Kaggle competition')
@click.option(
    '--optimize_dtypes',
    type = bool,
    # is_flag=True, 
    help='Whether to optimize DataFrame dtypes')
@click.option(
    '--original_data_set',
    type = str, 
    help='link to the original data')

def load_data(competition_name: str, optimize_dtypes: bool, original_data_set: str): 
    """
    Load data from a Kaggle competition zip file.
    Parameters:
        comp_name (str): Name of the Kaggle competition.
        optimize_dtypes (bool): Whether to optimize DataFrame dtypes.
    Returns:
        tuple: A tuple containing three DataFrames: train, test, and submission.
    """
    
    # Set up a directory to store the data 
    if not os.path.exists(f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data"):
        os.makedirs(f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data")
    
    # Change the working directory to the competition folder
    os.chdir(f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data")
    
    # Download the competition files
    api.competition_download_files(competition_name, path=".", force=True)

    # Unzip the downloaded files
    with zipfile.ZipFile(f"{competition_name}.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    
    # Load data into DataFrames
    submission = pd.read_csv("sample_submission.csv")
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    original = download_original_data(original_data_set)
    
    # Rename columns
    train, column_dict = rename_columns(train)
    test, _ = rename_columns(test)
    original, _ = rename_columns(original) 
    
    # Create a list of names and data types for train
    train_dtypes_old = train.dtypes

    if optimize_dtypes:
        train = adjust_dtypes(train)
        test = adjust_dtypes(test)
        submission = adjust_dtypes(submission)
        original = adjust_dtypes(original)
  
    # Create a list of names and data types for train after changes
    train_dtypes_new = train.dtypes
   
    # Create a dataframe from column_dict with 2 columns: old name and new name 
    column_df = pd.DataFrame(list(column_dict.items()), columns=["old_name", "new_name"])
    column_df['old_dtype'] = train_dtypes_old[column_df['new_name']].values
    column_df['new_dtype'] = train_dtypes_new[column_df['new_name']].values  
       
    # Create a readme.txt in the same folder outlining old data names and their data types and new names and their data type
    with open("column_dtype_rename_disc.txt", "w") as f:
        f.write(f"Data types for train before and after changes:\n\n")
        f.write(f"{column_df}")
        # print full path for readme file
        print(f"column_dtype_rename_disc.txt file stored in {os.getcwd()}")
           
    # Store the DataFrames as pickle files in the data folder for later use
    train.to_pickle("train.pkl")
    test.to_pickle("test.pkl")
    submission.to_pickle("submission.pkl")
    original.to_pickle("original.pkl")

    # Remove the zip and csv files to safe space
    os.remove(f"{competition_name}.zip")
    os.remove("sample_submission.csv")
    os.remove("test.csv")
    os.remove("train.csv")
    os.remove("train_dataset.csv")
    os.remove("train_dataset.csv.zip")
    
    # Print path where the data is stored
    print(f"Data stored in {os.getcwd()}")

if __name__ == "__main__":
    load_data()


    

