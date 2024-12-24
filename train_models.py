import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
import datetime


def encode_qualitative(df):
    # Encode the qualitative features
    le = LabelEncoder()
    
    qualitative_columns = df.select_dtypes(include=['object']).columns
    
    for features in qualitative_columns:
        df[features] = le.fit_transform(df[features].astype(str))


def train_regression_tree(X_train, X_val, y_train, y_val, 
                          param_grid_tree, cv, metric, verbose, random_state=42):
    # We create a Decision Tree Regressor
    reg_tree = DecisionTreeRegressor(random_state=random_state)

    # Grid Search (we use negative mean squared error as score since gridsearch want to maximize it)
    grid_tree = GridSearchCV(estimator=reg_tree, param_grid=param_grid_tree, 
                             cv=cv, scoring=metric, verbose=verbose)
    grid_tree.fit(X_train, y_train)

    # Best parameters and best model
    print("Best parameters for Regression Tree:", grid_tree.best_params_)
    best_tree = grid_tree.best_estimator_
    y_pred_tree = best_tree.predict(X_val)

    print("Regression Tree RMSE:", root_mean_squared_error(y_val, y_pred_tree))

    # Save the model
    # Define file paths
    base_path = 'models'
    file_path = os.path.join(base_path, 'regression_tree.joblib')

    # Check if the "models" directory exists, create it if not
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Check if the file already exists
    if os.path.exists(file_path):
        # Generate a new filename with the current date and time
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        new_file_path = os.path.join(base_path, f'regression_tree_{timestamp}.joblib')
        dump(best_tree, new_file_path)
        print(f"File already exists. Model saved as: {new_file_path}")
    else:
        # Save the model to the original file path
        dump(best_tree, file_path)
        print(f"Model saved as: {file_path}")
        dump(best_tree, 'models/regression_tree.joblib')


def train_xgboost(X_train, X_val, y_train, y_val, param_grid_xgb,
                cv, metric, verbose, random_state):
    # We create a XGBoost Regressor
    xgb_model = XGBRegressor(random_state=random_state)

    # Grid Search
    grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=cv, scoring=metric, verbose=verbose)
    grid_xgb.fit(X_train, y_train)

    # Best parameters and best model
    print("Best parameters for XGBoost:", grid_xgb.best_params_)
    best_xgb = grid_xgb.best_estimator_
    y_pred_xgb = best_xgb.predict(X_val)

    # RMSE
    print("XGBoost RMSE:", root_mean_squared_error(y_val, y_pred_xgb))

    # Save the model
    # Define file paths
    base_path = 'models'
    file_path = os.path.join(base_path, 'xgboost.joblib')

    # Check if the "models" directory exists, create it if not
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Check if the file already exists
    if os.path.exists(file_path):
        # Generate a new filename with the current date and time
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        new_file_path = os.path.join(base_path, f'xgboost_{timestamp}.joblib')
        dump(best_xgb, new_file_path)
        print(f"File already exists. Model saved as: {new_file_path}")
    else:
        # Save the model to the original file path
        dump(best_xgb, file_path)
        print(f"Model saved as: {file_path}")
        dump(best_xgb, 'models/xgboost.joblib')


if __name__ == "__main__":
    # Load the dataset
    print("Loading the processed datasets...")
    train = pd.read_csv('data/train_processed.csv')
    test = pd.read_csv('data/test_processed.csv')

    # Encode the qualitative features
    print("Encoding the qualitative features...")
    encode_qualitative(train)
    encode_qualitative(test)

    # Train Test the dataset
    X = train.drop(columns=['SalePrice'])  
    y = train['SalePrice']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameters Grid
    param_grid_tree = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1, 100, 'sqrt', 'log2']
    }
    param_grid_xgb = {
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    } 
    cv = 5
    metric = 'neg_mean_squared_error'
    verbose = 1

    # # Train the regression tree model
    # print("Training the regression tree model...")
    # train_regression_tree(X_train, X_val, y_train, y_val, 
    #                       param_grid_tree, cv, metric, verbose, random_state=42)
    
    # # Train the XGBoost model
    # print("Training the XGBoost model...")
    # train_xgboost(X_train, X_val, y_train, y_val,
    #               param_grid_xgb, cv, metric, verbose, random_state=42)
