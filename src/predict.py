# Prediction 

# Use the Year、Energy_Source、Monthly_Usage_kWh to predict the future energy usage

# Train data 2020-2023
# Test data 2024


from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 

from models.linear_model import train_linear_model
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from utils.model import save_model, load_model, evaluate_model
import os
import numpy as np

import matplotlib.pyplot as plt


file_path = '../data/raw/Renewable_Energy_Usage_Sampled.csv'
if os.path.exists(file_path):
    # Saving the cleaned data for further analysis
    data = pd.read_csv(file_path)

    # one hot encoding and drop the first column
    # Add more columns like 'Energy_Source_Geothermal', 'Energy_Source_Hydro', 'Energy_Source_Solar', 'Energy_Source_Wind'
    data_prepared = pd.get_dummies(data, columns=['Energy_Source'], drop_first=True)

    # add new column - Yearly_Change, for calculating the yearly change of energy usage
    data_prepared['Yearly_Change'] = data_prepared.groupby('Year')['Monthly_Usage_kWh'].diff().fillna(0)
    # add new column - Energy_Per_Household, for calculating the energy usage per household
    data_prepared['Energy_Per_Household'] = data_prepared['Monthly_Usage_kWh'] / data_prepared['Household_Size']


    print(data_prepared)
    X = data_prepared[['Year', 'Yearly_Change', 'Energy_Per_Household'] + [col for col in data_prepared.columns if col.startswith('Energy_Source_')]]
    y = data_prepared['Monthly_Usage_kWh']


    # split the data into train and test
    X_train = X[data_prepared['Year'] < 2024]
    y_train = y[data_prepared['Year'] < 2024]
    X_test = X[data_prepared['Year'] == 2024]
    y_test = y[data_prepared['Year'] == 2024]

    # train the linear model
    linear_model = train_linear_model(X_train, y_train)
    save_model(linear_model, './models/save_models/linear_model.pkl')

    # predict the test data [linear model]

    linear_rmse, linear_r2, linear_y_pred = evaluate_model(linear_model, X_test, y_test)
    print(f"[Linear Model] RMSE: {linear_rmse}, R2: {linear_r2}")

    predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': linear_y_pred})
    predictions.to_csv('./result/linear_predictions_2024.csv', index=False)
    
    # Random Forest Model
    random_forest_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=10)
    save_model(random_forest_model, './models/save_models/random_forest_model.pkl')

    # predict the test data [random forest model]
    random_forest_rmse, random_forest_r2, random_forest_y_pred = evaluate_model(random_forest_model, X_test, y_test)
    print(f"[Random Forest Model] RMSE: {random_forest_rmse}, R2: {random_forest_r2}")

    predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': random_forest_y_pred})
    predictions.to_csv('./result/random_forest_predictions_2024.csv', index=False)


    # Train XGBoost Model
    xgb_model = train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=6)
    save_model(xgb_model, './models/save_models/xgboost_model.pkl')
    xgb_rmse, xgb_r2, xgb_y_pred = evaluate_model(xgb_model, X_test, y_test)
    print(f"[XGBoost Model] RMSE: {xgb_rmse}, R2: {xgb_r2}")
    predictions = pd.DataFrame({
        'Actual': y_test.values,
        'Linear_Predicted': linear_y_pred,
        'Random_Forest_Predicted': random_forest_y_pred,
        'XGBoost_Predicted': xgb_y_pred
    })
    predictions.to_csv('./result/predictions_comparison_2024.csv', index=False)
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Get unique months for plotting
    unique_months = y_test.groupby(data_prepared[data_prepared['Year'] == 2024]['Month']).mean()
    
    # Plot actual values
    plt.plot(range(len(unique_months)), unique_months.values, 
            label="Actual", marker='o', linewidth=2, markersize=8)
    
    # Calculate monthly averages for predictions
    linear_monthly = pd.Series(linear_y_pred).groupby(data_prepared[data_prepared['Year'] == 2024]['Month']).mean()
    rf_monthly = pd.Series(random_forest_y_pred).groupby(data_prepared[data_prepared['Year'] == 2024]['Month']).mean()
    xgb_monthly = pd.Series(xgb_y_pred).groupby(data_prepared[data_prepared['Year'] == 2024]['Month']).mean()
    
    # Plot predictions with different styles
    plt.plot(range(len(linear_monthly)), linear_monthly, 
            label=f"Linear (R² = {linear_r2:.3f})", 
            linestyle='--', marker='^', alpha=0.7)
    plt.plot(range(len(rf_monthly)), rf_monthly, 
            label=f"Random Forest (R² = {random_forest_r2:.3f})", 
            linestyle='--', marker='s', alpha=0.7)
    plt.plot(range(len(xgb_monthly)), xgb_monthly, 
            label=f"XGBoost (R² = {xgb_r2:.3f})", 
            linestyle='--', marker='D', alpha=0.7)

    plt.title("Model Predictions vs Actual Monthly Usage (2024)", fontsize=12, pad=20)
    plt.xlabel("Month (2024)", fontsize=10)
    plt.ylabel("Monthly Usage (kWh)", fontsize=10)
    
    # Add month labels on x-axis
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(12), months, rotation=45)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig('./result/prediction_comparison_plot.png', 
                dpi=300, bbox_inches='tight')
    plt.show()