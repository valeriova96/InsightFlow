SEED = 42


def train_linear_regression_model(X_train, y_train):
    """
    Train a linear regression model on pre-split training data.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
    - y_train (pd.Series or np.ndarray): Target vector for training.

    Returns:
    - model (LinearRegression): Trained linear regression model.
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(X_train, y_train):
    """
    Train a random forest model on pre-split training data.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
    - y_train (pd.Series or np.ndarray): Target vector for training.

    Returns:
    - model (RandomForestRegressor): Trained random forest model.
    """
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model


def train_support_vector_regression(X_train, y_train):
    """
    Train a support vector regression model on pre-split training data.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
    - y_train (pd.Series or np.ndarray): Target vector for training.

    Returns:
    - model (SVR): Trained support vector regression model.
    """
    from sklearn.svm import SVR
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    return model
