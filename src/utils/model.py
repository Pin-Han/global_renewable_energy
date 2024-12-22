import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def save_model(model, filepath):
    """
    保存模型到指定文件
    :param model: 要保存的模型
    :param filepath: 保存路徑
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """
    從文件加載模型
    :param filepath: 模型路徑
    :return: 加載的模型
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model, X_test, y_test):
    """
    評估模型性能
    :param model: 已訓練的模型
    :param X_test: 測試特徵
    :param y_test: 測試目標值
    :return: 評估指標 (RMSE, R^2)
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred