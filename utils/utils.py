from models.classification.models import (
    train_logistic_regression_model,
    train_random_forest_model,
    train_multinomial_nb_model,
)
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from typing import Literal

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


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model and return precision, recall, and F1-score as a DataFrame.

    Parameters:
    - model: Trained scikit-learn classification model.
    - X_test (pd.DataFrame or np.ndarray): Test features.
    - y_test (pd.Series or np.ndarray): True labels.

    Returns:
    - pd.DataFrame: DataFrame with precision, recall, and F1-score per class.
    """
    # Generate predictions
    y_pred = model.predict(X_test)

    # Get classification report as a dictionary
    report = classification_report(y_test, y_pred, output_dict=True)

    # Filter out only the classes (exclude 'accuracy', 'macro avg', etc.)
    class_metrics = {
        label: values for label, values in report.items()
        if label not in ['accuracy', 'macro avg', 'weighted avg']
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(
        class_metrics,
        orient='index'
    )[['precision', 'recall', 'f1-score']]

    return metrics_df


def train_and_evaluate_model(
        task_type: Literal["classification", "regression"],
        model_name: str,
        dataset: pd.DataFrame,
        feature_cols: list,
        target_col: str
) -> pd.DataFrame:
    """
    Train and evaluate a classification model based on the provided model name.

    Parameters:
    - task_type (Literal["classification", "regression"]): Type of task to perform.
    - model_name (str): Name of the model to train.
    - dataset (pd.DataFrame): Input dataset.
    - feature_cols (list): List of feature column names.
    - target_col (str): Name of the target column.

    Returns:
    - pd.DataFrame: Evaluation metrics DataFrame.
    """
    # Clean data
    cleaned_data = clean_data(dataset, target_col)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        cleaned_data, feature_cols, target_col
    )

    metrics_df = pd.DataFrame()

    match task_type:
        case "classification":
            match model_name:
                case "Logistic Regression":
                    model = train_logistic_regression_model(X_train, y_train)
                case "Random Forest Classifier":
                    model = train_random_forest_model(X_train, y_train)
                case "Multinomial Naive Bayes":
                    model = train_multinomial_nb_model(X_train, y_train)
                case _:
                    raise ValueError(f"Unsupported model: {model_name}")

            # Evaluate the model
            metrics_df = evaluate_model(model, X_test, y_test)

        case "regression":
            match model_name:
                case "Linear Regression":
                    pass
                case "Random Forest Regressor":
                    pass
                case "Gaussian Naive Bayes":
                    pass
                case _:
                    raise ValueError(f"Unsupported model: {model_name}")

            # TODO Evaluate the model

    return metrics_df
