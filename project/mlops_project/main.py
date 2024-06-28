import pandas as pd
import os
from module_name.kaggle_client import KaggleClient
from module_name.data_preprocesing import DataPreprocessor
from module_name.feature_engineering import FeatureEnginering

competition_name = "playground-series-s3e11"
# root directory:
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Download data from Kaggle

kaggle_client = KaggleClient()

kaggle_client.download_data(competition_name)

# Preprocess data
data_preprocessor = DataPreprocessor(competition_name)
train = pd.read_csv(os.path.join(data_preprocessor.df_raw_path, "train.csv"))

train = data_preprocessor.optimize_dtypes(train)

feature_engineering = FeatureEnginering(competition_name, target_column="cost")
# feature_engineering.openfe_fit(train)
# Feature engineering with OpenFE
train_transforment = feature_engineering.openfe_transform(train)
