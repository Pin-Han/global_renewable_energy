from sklearn.linear_model import LinearRegression


def train_linear_model(X_train, y_train):
    """Train a linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

