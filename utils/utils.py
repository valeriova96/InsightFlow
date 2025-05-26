import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal

SEED = 42


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Remove rows from the DataFrame where the target column has NaN or empty
    values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_col (str): Name of the target column to check for missing values.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows where target is NaN or empty string
    cleaned_df = df[df[target_col].notna() & (df[target_col] != "")]
    return cleaned_df


def find_and_convert_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify categorical columns in the DataFrame and convert them to
    numerical codes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with categorical columns converted.
    - list: List of categorical column names.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes

    return df


def split_data(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the DataFrame into train/test sets.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - feature_cols (list): List of column names to use as features.
    - target_col (str): Name of the target column.
    - test_size (float): Proportion of the dataset to include in the test
      split.
    - random_state (int): Random seed.

    Returns:
    - X_train, X_test, y_train, y_test (tuple of pd.DataFrames/Series): Split
      data.
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def evaluate_class_model(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate a classification model and return precision, recall, and F1-score
    as a DataFrame.

    Parameters:
    - model: Trained scikit-learn classification model.
    - X_test (pd.DataFrame or np.ndarray): Test features.
    - y_test (pd.Series or np.ndarray): True labels.

    Returns:
    - pd.DataFrame: DataFrame with precision, recall, and F1-score per class.
    """
    from sklearn.metrics import classification_report

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


def evaluate_regr_model(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate a regression model and return RMSE and R-squared as a DataFrame.

    Parameters:
    - model: Trained scikit-learn regression model.
    - X_test (pd.DataFrame or np.ndarray): Test features.
    - y_test (pd.Series or np.ndarray): True labels.

    Returns:
    - pd.DataFrame: DataFrame with RMSE and R-squared.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate RMSE and R-squared
    mse = round(mean_squared_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 2)

    # Create a DataFrame to hold the metrics
    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R-squared': [r2]
    })

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
    - task_type (Literal["classification", "regression"]): Type of task to
      perform.
    - model_name (str): Name of the model to train.
    - dataset (pd.DataFrame): Input dataset.
    - feature_cols (list): List of feature column names.
    - target_col (str): Name of the target column.

    Returns:
    - pd.DataFrame: Evaluation metrics DataFrame.
    """
    # Clean data
    cleaned_data = clean_data(dataset, target_col)

    # Convert categorical columns to numerical codes
    if task_type == "regression":
        dataset = find_and_convert_cat_cols(cleaned_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        cleaned_data, feature_cols, target_col
    )

    metrics_df = pd.DataFrame()

    match task_type:
        case "classification":
            match model_name:
                case "Logistic Regression":
                    from models.classification.models import (
                        train_logistic_regression_model
                    )
                    model = train_logistic_regression_model(X_train, y_train)
                case "Random Forest Classifier":
                    from models.classification.models import (
                        train_random_forest_model
                    )
                    model = train_random_forest_model(X_train, y_train)
                case "Support Vector Machine":
                    from models.classification.models import (
                        train_support_vector_machine
                    )
                    model = train_support_vector_machine(X_train, y_train)
                case _:
                    raise ValueError(f"Unsupported model: {model_name}")

            # Evaluate the model
            metrics_df = evaluate_class_model(model, X_test, y_test)

        case "regression":
            match model_name:
                case "Linear Regression":
                    from models.regression.models import (
                        train_linear_regression_model
                    )
                    model = train_linear_regression_model(X_train, y_train)
                case "Random Forest Regressor":
                    from models.regression.models import (
                        train_random_forest_model
                    )
                    model = train_random_forest_model(X_train, y_train)
                case "Support Vector Regression":
                    from models.regression.models import (
                        train_support_vector_regession
                    )
                    model = train_support_vector_regession(X_train, y_train)
                case _:
                    raise ValueError(f"Unsupported model: {model_name}")

            metrics_df = evaluate_regr_model(model, X_test, y_test)

    return metrics_df
