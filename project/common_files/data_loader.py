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

def adjust_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust data types for the DataFrame columns.
    Parameters:
        df (pd.DataFrame): The DataFrame to adjust.
    Returns:
        pd.DataFrame: The DataFrame with adjusted data types.
    """
    int_columns = df.select_dtypes(include=["int64"]).columns
    float_columns = df.select_dtypes(include=["float64"]).columns

    # Change integer columns to the smallest type that fits the data
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Change float columns to the smallest float type that fits the data
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df;

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
    
    if optimize_dtypes:
        train = adjust_dtypes(train)
        test = adjust_dtypes(test)
        submission = adjust_dtypes(submission)
        original = adjust_dtypes(original)

    # Create a folder based on the competition name if it doesn't exist
    folder_name = competition_name.replace(" ", "_")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Create a folder data to store the data if it doesn't exist
    if not os.path.exists(f"{folder_name}/data"):
        os.makedirs(f"{folder_name}/data")
    
    # Store the DataFrames as pickle files in the data folder for later use
    train.to_pickle(f"{folder_name}/data/train.pkl")
    test.to_pickle(f"{folder_name}/data/test.pkl")
    submission.to_pickle(f"{folder_name}/data/submission.pkl")
    original.to_pickle(f"{folder_name}/data/original.pkl")

    # Remove the zip and csv files to safe space
    os.remove(f"{competition_name}.zip")
    os.remove("sample_submission.csv")
    os.remove("test.csv")
    os.remove("train.csv")
    os.remove("train_dataset.csv")

if __name__ == "__main__":
    load_data()


    

