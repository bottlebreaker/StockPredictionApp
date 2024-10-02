# xgboost_model.py

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_xgboost_model():
    """
    Load the saved XGBoost model from disk.
    """
    model_path = "models/xgb_model.json"
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)
    print(f"XGBoost model loaded from {model_path}")
    return xgb_model

def train_xgboost(X_train, y_train):
    """
    Train the XGBoost model using the preprocessed features including time-related features.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.7
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=500)
    
    # Save the XGBoost model
    model_path = "models/xgb_model.json"
    xgb_model.save_model(model_path)
    print(f"XGBoost model saved to {model_path}")

    return xgb_model


def predict_xgboost(model, X_test, days=10):
    """
    Predicts stock prices using the trained XGBoost model for the next 'days' days.
    """
    predictions = []
    input_data = X_test

    for _ in range(days):
        dtest = xgb.DMatrix(input_data)
        pred = model.predict(dtest)[0]  # Predict for the next day
        predictions.append(pred)

        # Update input_data for the next prediction by appending the predicted value
        input_data = np.roll(input_data, -1, axis=0)  # Shift data by one day
        input_data[-1, 0] = pred  # Replace last row with predicted 'close' price

    return predictions


