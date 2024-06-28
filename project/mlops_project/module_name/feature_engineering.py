import os
import pandas as pd
from openfe import OpenFE, transform, tree_to_formula
import contextlib
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
from sklearn.model_selection import train_test_split


class FeatureEnginering:
    def __init__(self, competition_name, target_column):
        """
        Initialize the FeatureEngineering class with competition name and target column.

        Parameters:
        competition_name (str): The name of the competition.
        target_column (str): The name of the target column.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.preprocessed_data = os.path.join(project_root, f"data/{competition_name}/prepocessed")
        self.feature_eng_data = os.path.join(
            project_root, f"data/{competition_name}/feature_engineered"
        )
        os.makedirs(self.feature_eng_data, exist_ok=True)
        self.target_column = target_column

    def stratified_sample(self, data, size_per_bin=1000, bins=15):
        """
        Perform stratified sampling on the data.

        Parameters:
        data (pd.DataFrame): The input data.
        size_per_bin (int): The size per bin for stratified sampling.
        bins (int): The number of bins for stratified sampling.

        Returns:
        tuple: Stratified samples, train and validation data.
        """
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=1)
        data_s = train_df.copy()
        data_s["bins"] = pd.qcut(data_s[self.target_column], q=bins, labels=False)
        sample = (
            data_s.groupby("bins")
            .apply(lambda x: x.sample(n=size_per_bin, random_state=1))
            .reset_index(drop=True)
        )
        sample = sample.drop(columns=["bins"])
        X_train_stratified = sample.drop(columns=[self.target_column])
        y_train_stratified = sample[self.target_column]
        return X_train_stratified, y_train_stratified, train_df, val_df

    def train_quick_autogluon(
        self, train_df, val_df, preset="medium_quality", time_limit_minutes=1, metric="rmse"
    ):
        """
        Train a quick AutoGluon model.

        Parameters:
        train_df (pd.DataFrame): The training data.
        val_df (pd.DataFrame): The validation data.
        preset (str): The preset quality setting for AutoGluon.
        time_limit_minutes (int): The time limit for training in minutes.
        metric (str): The evaluation metric.

        Returns:
        float: The negative evaluation score.
        """
        predictor = TabularPredictor(label=self.target_column, eval_metric=metric)
        predictor.fit(train_data=train_df, time_limit=time_limit_minutes * 60, presets=preset)
        score = predictor.evaluate(val_df)
        metric = list(score.keys())[0]
        return -score[metric]

    def openfe_fit(self, data, number_of_features=5):
        """
        Fit OpenFE on the data.

        Parameters:
        data (pd.DataFrame): The input data.
        number_of_features (int): The number of features to generate.

        Returns:
        OpenFE: The fitted OpenFE object.
        """
        ofe = OpenFE()
        X_train_stratified, y_train_stratified, _, _ = self.stratified_sample(data)

        with contextlib.redirect_stdout(None):  # Suppress the output
            ofe.fit(data=X_train_stratified, label=y_train_stratified, n_jobs=4)

        print(f"The top {number_of_features} generated features are:")
        for feature in ofe.new_features_list[:number_of_features]:
            print(tree_to_formula(feature))

        return ofe

    def openfe_transform(self, data, number_of_features=5):
        """
        Transform the data using OpenFE.

        Parameters:
        data (pd.DataFrame): The input data.
        number_of_features (int): The number of features to generate.

        Returns:
        pd.DataFrame: The transformed data.
        """
        ofe = self.openfe_fit(data, number_of_features)
        train_transformed, test_transformed = transform(
            data, data, ofe.new_features_list[:number_of_features], n_jobs=4
        )
        # save the transformed data
        train_transformed.to_pickle(os.path.join(self.feature_eng_data, "train_transformed.pkl"))
        test_transformed.to_pickle(os.path.join(self.feature_eng_data, "test_transformed.pkl"))
        return train_transformed, test_transformed
