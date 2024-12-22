from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train the random forest model
    :param X_train: Training features
    :param y_train: Training target values
    :param n_estimators: Number of trees in the forest
    :param max_depth: Maximum depth of a single tree
    :param random_state: Random seed
    :return: Trained model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model
