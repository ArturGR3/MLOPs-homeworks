from openfe import OpenFE, transform, tree_to_formula
import pandas as pd 
import contextlib
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
import matplotlib.pyplot as plt

competition_name = "playground-series-s3e11" # needs to be feed for automation 
data_folder = f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data"

target = 'cost' # needs to be feed with parameter

def stratified_sample(data, target, size_per_bin, bins)-> (pd.DataFrame, pd.Series):
    data = data.copy()
    data['bins'] = pd.qcut(data[target], q=bins, labels=False)
    sample = data.groupby('bins').apply(lambda x: x.sample(n=size_per_bin, random_state=1))
    sample = sample.reset_index(drop=True)
    sample = sample.drop(columns=['bins'])
    X_train = sample.drop(columns=[target])    
    y_train = sample[target]
    return X_train, y_train

# Load data 
def load_data(data_folder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training data from the specified Kaggle competition.

    Parameters:
        data_folder (str): The path to the data folder where the training and original data is stored.

    Returns:
        X_train_fe (pandas.DataFrame): The sample of the training data for feature engineering.
        y_train_fe (pandas.Series): The target of the sample of the training data for feature engineering.
        train_original_combined (pandas.DataFrame): The combined training and original data.
        val_df (pandas.DataFrame): The validation data to assess the implication of feature engineering.
    """
    # Load initial data 
    train = pd.read_pickle(filepath_or_buffer=f"{data_folder}/train.pkl")
    train = train.drop(columns=['id'])
    original = pd.read_pickle(filepath_or_buffer=f"{data_folder}/original.pkl")
    test = pd.read_pickle(filepath_or_buffer=f"{data_folder}/test.pkl")
    
    # Combine train and original data
    train_original_combined = pd.concat([train, original], axis=0)
    
    # Create a validation set from train data
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=1)
    
    # Combine original with train_df for feature engineering 
    combined_df = pd.concat([train_df, original], axis=0)
    
    # create a sample of 1% of the data, stratified by cost, with 10 bins
    X_train_fe, y_train_fe = stratified_sample(combined_df, target=target, size_per_bin=1000, bins=15) 
    
    return X_train_fe, y_train_fe, train_original_combined, combined_df, val_df, test

X_train_fe, y_train_fe, train_original_combined, combined_df, val_df, test = load_data(data_folder)
# Create a function to rename columns automatically replacing spaces, brackets and dots with underscores using regex


# Create a function to run openfe for a given data
def openfe_fit(X_train, y_train):
    """
    Fit OpenFE on the given data.

    Parameters:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        ofe (OpenFE): The fitted OpenFE object.
    """
    ofe = OpenFE()
    with contextlib.redirect_stdout(None):
        ofe.fit(data=X_train, label=y_train, n_jobs=4)  
    topk = 10
    print(f'The top {topk} generated features are:')
    for feature in ofe.new_features_list[:topk]:
        print(tree_to_formula(feature))
    return ofe

# Fit OpenFE on the training data
ofe = openfe_fit(X_train_fe, y_train_fe)

rmsle = make_scorer('rmsle', root_mean_squared_log_error, greater_is_better=False, needs_proba=False) # needs automation 

# Train AutoGluon with the preset
def get_AutoGluon_score(train, val, target, metric = rmsle, preset='medium_quality', time_min=1):
    predictor = TabularPredictor(label=target, eval_metric=rmsle)
    predictor.fit(train_data=train, time_limit=time_min*60, presets = preset, excluded_model_types=['KNN', 'NN'])
    score = predictor.evaluate(val)
    metric = list(score.keys())[0]
    return -score[metric]
   
# Understanding the impact of additional features 
scores = {}
for topk in [0,5]:
    X_train_ofe, X_val_ofe = transform(combined_df, val_df, ofe.new_features_list[:topk], n_jobs=4)
    scores[topk] = get_AutoGluon_score(X_train_ofe, X_val_ofe, target, metric = rmsle, preset='medium_quality', time_min=1)   
    print(f'Top {topk} features score: {scores[topk]}')
# myltiply by -1 to get the RMSLE

# Create a plot that shows improvements in score based on additional features, compared to the baseline
plt.plot(list(scores.keys()), list(scores.values()))
plt.xlabel('Top k features')
plt.ylabel('RMSLE')
plt.title('RMSLE vs Top k features')   

# Create final feature engineered data
train_final_transformed, test_transformed = transform(train_original_combined, test, ofe.new_features_list[:30], n_jobs=4)

# Save the feature engineered data in the data folder
data_folder = f"/home/artur/MLOPs-homeworks/project/common_files/{competition_name}/data"
train_final_transformed.to_pickle(f"{data_folder}/train_final.pkl")
test_transformed.to_pickle(f"{data_folder}/test_final.pkl")

 

 