"""
Class file containing KNNImputer() for data imputation.
"""

import random
import numpy as np
from pandas.api.types import is_numeric_dtype
from utils import categorical_to_onehot_columns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class KNNImputer(object):
    """
    Class representing a data imputation method with K-Nearest Neighbours algorithm.

    @param (int) k: number of nearest neighbours to distinguish examples (default: 3)
    """
    def __init__(self, k=3):
        self.k = k
        self.impute_variables, self.ignore_features, self.features, self.models = None, None, {}, []

    def preprocess(self, df, target_column, get_null=True):
        """
        Function to preprocess data frame before fitting a KNN

        @param (pd.DataFrame) df: dataframe to apply KNN on for filling missing values
        @param (str) target_column: name of the column that is currently being imputed, using
               all the other available columns
        @param (bool) get_null: whether to get rows where the target column contains an empty
               value (prediction data) or not (training data) (default: True)
        """
        # Non-empty target variable -> training data, Empty target variable -> prediction data
        data = df[df[target_column].isnull()] if get_null else df[df[target_column].notnull()]
        # Remove ignored features
        if self.ignore_features is not None:
            for feature in self.ignore_features:
                if feature in data.columns.values.tolist():
                    data = data.drop(feature, axis=1)
        # One-hot encode categorical independent variables, setting NaNs to column means
        data = categorical_to_onehot_columns(df=data, target_column=target_column)
        # Set NaNs to column means for numerical independent variables
        data = data.fillna(data.drop(target_column, axis=1).mean())
        return data

    def fit(self, df, impute_variables='all', ignore_features=None):
        """
        Function to fit KNN algorithm.

        @param (pd.DataFrame) df: dataframe to apply KNN on for filling missing values
        @param (list/str) impute_variables: dependent variables, based on column names, to impute
               for; or you can specify 'all' to impute for all columns (default: 'all')
        @param (list) ignore_features: independent variables, based on column names, that will not
               be utilized when imputing for dependent variables (default: None)
        """
        if impute_variables == 'all':
            self.impute_variables = df.columns.values.tolist()
        else:
            self.impute_variables = impute_variables

        self.ignore_features = ignore_features
        self.models = [None] * len(self.impute_variables)

        for i, column in enumerate(self.impute_variables):
            # Check if the target variable really contains empty values
            if not df[column].isnull().values.any():
                continue
            # Preprocess data
            train_data = self.preprocess(df=df, target_column=column, get_null=False)
            # Separate training data into features and labels
            train_features, train_labels = train_data.drop(column, axis=1), train_data[column]
            # Save used features for matching the same features in prediction time
            # NOTE: This is necessary for situations where certain categorical values don't
            # exist in training or testing examples.
            self.features[column] = train_features.columns.values.tolist()
            # Decide on model type
            if is_numeric_dtype(train_labels):
                self.models[i] = KNeighborsRegressor(n_neighbors=self.k, n_jobs=-1)
            else:
                self.models[i] = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
            # Train
            self.models[i].fit(X=train_features, y=train_labels)

    def impute(self, df):
        """Function to impute missing values in data frame via the pretrained KNN model."""
        for i, column in enumerate(self.impute_variables):
            # Check if the target variable really contains empty values
            if not df[column].isnull().values.any():
                continue

            # Check if training data has no missing variables for this column,
            # numerical column -> gaussian random, categorical columns -> uniform random
            if column not in self.features.keys():
                if is_numeric_dtype(df[column]):
                    df.loc[df[column].isnull(), column] = df.loc[df[column].isnull(), column].apply(
                        lambda x: np.random.normal(loc=df[column].mean(),
                                                   scale=df[column].std())
                    )
                else:
                    all_possible_values = list(set(df[column].dropna().tolist()))
                    df.loc[df[column].isnull(), column] = df.loc[df[column].isnull(), column].apply(
                        lambda x: random.choice(all_possible_values)
                    )
                continue

            # Preprocess data
            pred_data = self.preprocess(df=df, target_column=column, get_null=True)
            # Separate prediction data into features only
            pred_features = pred_data.drop(column, axis=1)
            # Add features that were present in training, and remove features that weren't
            for base_feature in self.features[column]:
                if base_feature not in pred_features.columns.values.tolist():
                    pred_features[base_feature] = 0.0
            for extra_feature in pred_features.columns.values.tolist():
                if extra_feature not in self.features[column]:
                    pred_features.drop(extra_feature, axis=1, inplace=True)
            # Predict
            df.loc[df[column].isnull(), column] = self.models[i].predict(X=pred_features)

        return df
