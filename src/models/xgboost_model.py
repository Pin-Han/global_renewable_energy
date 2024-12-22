from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
    """
    訓練 XGBoost 模型
    :param X_train: 訓練特徵
    :param y_train: 訓練目標值
    :param n_estimators: 樹的數量
    :param learning_rate: 學習率
    :param max_depth: 樹的最大深度
    :param random_state: 隨機種子
    :return: 訓練好的 XGBoost 模型
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model