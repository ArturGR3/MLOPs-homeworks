# from typing import Dict, List, Optional, Tuple

# import pandas as pd
# import scipy
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LinearRegression

# if 'transformer' not in globals():
#     from mage_ai.data_preparation.decorators import transformer
# if 'test' not in globals():
#     from mage_ai.data_preparation.decorators import test


# @transformer
# def transform(
#     data: pd.DataFrame,
#  *args, **kwargs):
#     """
#     Template code for a transformer block.

#     Add more parameters to this function if this block has multiple parent blocks.
#     There should be one parameter for each output variable from each parent block.

#     Args:
#         data: The output from the upstream parent block
#         args: The output from any additional upstream blocks (if applicable)

#     Returns:
#         Anything (e.g. data frame, dictionary, array, int, str, etc.)
#     """
#     categorical = ['PULocationID', 'DOLocationID']
#     dv = DictVectorizer()

#     train_dicts = data[categorical].to_dict(orient='records')
#     X_train = dv.fit_transform(train_dicts)
#     y_train = data['duration']

#     ln = LinearRegression()
#     model = ln.fit(X_train,y_train)
#     print(model.intercept_)

#     return dv, model 


from typing import Dict, List, Optional, Tuple

import pandas as pd
import pickle
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(
    data: pd.DataFrame,
 *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    
    # Connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("homework_03")
    
    categorical = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()
    train_dicts = data[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = data['duration']

    # Save the model using MLflow autologging
    # mlflow.sklearn.autolog()
     
    ln = LinearRegression().fit(X_train, y_train)
    print(ln.intercept_)

    mlflow.sklearn.log_model(ln, artifact_path="models") 
    # Save the DictVectorizer as an artifact
        # Save the DictVectorizer to a file
    dv_path = "dict_vectorizer.pkl"
    with open(dv_path, "wb") as f:
        pickle.dump(dv, f)

    # Log the DictVectorizer file as an artifact
    mlflow.log_artifact(dv_path, artifact_path="artifacts")

    return dv, ln
