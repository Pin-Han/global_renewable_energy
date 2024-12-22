# Prediction 

# Use the Year、Energy_Source、Monthly_Usage_kWh to predict the future energy usage

# Train data 2020-2023
# Test data 2024


from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 

from models.linear_model import train_linear_model
from models.random_forest import train_random_forest
from utils.model import save_model, load_model, evaluate_model
import os
import numpy as np


file_path = '../data/raw/Renewable_Energy_Usage_Sampled.csv'
if os.path.exists(file_path):
    # Saving the cleaned data for further analysis
    data = pd.read_csv(file_path)

    # one hot encoding and drop the first column
    # Add more columns like 'Energy_Source_Geothermal', 'Energy_Source_Hydro', 'Energy_Source_Solar', 'Energy_Source_Wind'
    data_prepared = pd.get_dummies(data, columns=['Energy_Source'], drop_first=True)

    print(data_prepared)
    X = data_prepared[['Year'] + [col for col in data_prepared.columns if col.startswith('Energy_Source_')]]
    y = data_prepared['Monthly_Usage_kWh']


    # split the data into train and test
    X_train = X[data_prepared['Year'] < 2024]
    y_train = y[data_prepared['Year'] < 2024]
    X_test = X[data_prepared['Year'] == 2024]
    y_test = y[data_prepared['Year'] == 2024]

    # train the linear model
    linear_model = train_linear_model(X_train, y_train)
    save_model(linear_model, './models/saved_models/linear_model.pkl')

    # predict the test data [linear model]

    linear_rmse, linear_r2, linear_y_pred = evaluate_model(linear_model, X_test, y_test)
    print(f"[Linear Model] RMSE: {linear_rmse}, R2: {linear_r2}")

    predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': linear_y_pred})
    predictions.to_csv('./result/linear_predictions_2024.csv', index=False)


    # RMSE: 406.58781622461163
    # R2: -0.022576591298384185
    # actual value: [1043.49, 610.01, 1196.75, 629.67, 759.23]
    # predicted value: [795.21, 768.59, 776.11, 776.11, 768.59]

    # Random Forest Model
    random_forest_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=10)
    save_model(random_forest_model, './models/saved_models/random_forest_model.pkl')

    # predict the test data [random forest model]
    random_forest_rmse, random_forest_r2, random_forest_y_pred = evaluate_model(random_forest_model, X_test, y_test)
    print(f"[Random Forest Model] RMSE: {random_forest_rmse}, R2: {random_forest_r2}")

    predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': random_forest_y_pred})
    predictions.to_csv('./result/random_forest_predictions_2024.csv', index=False)
