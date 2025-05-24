import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def clean_data(df, target_col):
    """
    Remove rows from the DataFrame where the target column has NaN or empty values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_col (str): Name of the target column to check for missing values.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows where target is NaN or empty string
    cleaned_df = df[df[target_col].notna() & (df[target_col] != "")]
    return cleaned_df


def split_data(df, feature_cols, target_col, test_size=0.2, random_state=SEED):
    """
    Split the DataFrame into train/test sets.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - feature_cols (list): List of column names to use as features.
    - target_col (str): Name of the target column.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed.

    Returns:
    - X_train, X_test, y_train, y_test (tuple of pd.DataFrames/Series): Split data.
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
