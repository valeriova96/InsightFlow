SEED = 42


def train_logistic_regression_model(X_train, y_train):
    """
    Train a logistic regression model on pre-split training data.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
    - y_train (pd.Series or np.ndarray): Target vector for training.

    Returns:
    - model (LogisticRegression): Trained logistic regression model.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(X_train, y_train):
    """
    Train a random forest model on pre-split training data.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
    - y_train (pd.Series or np.ndarray): Target vector for training.

    Returns:
    - model (RandomForestClassifier): Trained random forest model.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model


def train_multinomial_nb_model(X_train, y_train):
    """
    Train a multinomial Naive Bayes model on pre-split training data.

    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
    - y_train (pd.Series or np.ndarray): Target vector for training.

    Returns:
    - model (MultinomialNB): Trained multinomial Naive Bayes model.
    """
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model
